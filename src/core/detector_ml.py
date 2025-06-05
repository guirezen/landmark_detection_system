# -*- coding: utf-8 -*-
"""Módulo para detecção de landmarks usando Machine Learning básico (Random Forest)."""

import numpy as np
import trimesh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
import joblib
import os
import logging

from .landmarks import LANDMARK_NAMES

class MLDetector:
    """Detecta landmarks usando um modelo de Machine Learning (Random Forest)."""

    def __init__(self, model_dir="./models", feature_radius_multiplier=5, confidence_threshold=0.5):
        self.model_dir = model_dir
        self.feature_radius_multiplier = feature_radius_multiplier
        self.confidence_threshold = confidence_threshold
        os.makedirs(self.model_dir, exist_ok=True)
        # Dicionários para armazenar modelos e scalers carregados em memória
        self.models = {}
        self.scalers = {}
        logging.debug(f"MLDetector inicializado (dir modelos: {self.model_dir})")

    def _extract_features(self, mesh, vertex_indices=None):
        """Extrai um vetor de features locais para cada vértice especificado (ou todos)."""
        # Selecionar vértices de interesse
        if vertex_indices is None:
            vertices = mesh.vertices
            indices_out = np.arange(len(vertices))
        else:
            vertices = mesh.vertices[vertex_indices]
            indices_out = vertex_indices
        n = len(vertices)
        if n == 0:
            return None, None
        features = []
        try:
            # Feature 1: Coordenadas normalizadas (posição relativa ao centro, escalada pelo tamanho)
            bounds = mesh.bounds
            center = mesh.centroid
            extent = mesh.extents
            extent = np.where(extent == 0, 1.0, extent)  # evitar divisão por zero
            norm_coords = (vertices - center) / extent
            features.append(norm_coords)
            # Feature 2: Normais dos vértices
            try:
                if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) == len(mesh.vertices):
                    vertex_normals = mesh.vertex_normals[indices_out]
                else:
                    mesh.vertex_normals  # força cálculo das normais se não existem
                    vertex_normals = mesh.vertex_normals[indices_out]
                features.append(vertex_normals)
            except Exception as e:
                logging.warning(f"Erro ao obter normais dos vértices: {e}. Usando vetor zero.")
                features.append(np.zeros((n, 3)))
            # Feature 3: Curvatura Gaussiana local
            try:
                avg_edge = np.mean(mesh.edges_unique_length)
                radius = avg_edge * self.feature_radius_multiplier
                # Curvatura Gaussiana discreta para todos vértices, medir só nos de interesse
                all_curv = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius=radius)
                all_curv = np.nan_to_num(all_curv, nan=0.0, posinf=1.0, neginf=-1.0)
                curvatures = all_curv if vertex_indices is None else all_curv[indices_out]
                features.append(curvatures.reshape(-1, 1))
            except Exception as e:
                logging.warning(f"Falha ao calcular curvatura para features: {e}. Usando 0.")
                features.append(np.zeros((n, 1)))
            # Feature 4: Distância ao centro da malha (norma da posição)
            distances = np.linalg.norm(vertices - center, axis=1).reshape(-1, 1)
            features.append(distances)
            # Concatenar todas as features em um array 2D [n_vertices x n_features]
            X = np.hstack(features)
            return X, indices_out
        except Exception as e:
            logging.error(f"Erro na extração de features: {e}")
            return None, None

    def train(self, meshes, all_landmarks_gt, target_landmark_name):
        """Treina um modelo Random Forest para detectar um dado landmark."""
        if target_landmark_name not in LANDMARK_NAMES:
            logging.error(f"Landmark desconhecido para treinamento: {target_landmark_name}")
            return False
        logging.info(f"Iniciando treinamento do modelo ML para: {target_landmark_name}")
        X_all = []
        y_all = []
        # Montar conjunto de treino iterando por todas as malhas e seus GTs
        for i, mesh in enumerate(meshes):
            try:
                gt_landmarks = all_landmarks_gt[i]
                target_coord = gt_landmarks.get(target_landmark_name)
                if target_coord is None:
                    logging.debug(f"GT de {target_landmark_name} ausente na malha {i}, ignorando.")
                    continue
                # Encontrar vértice real mais próximo do ponto GT desejado
                kdtree = KDTree(mesh.vertices)
                dist, target_idx = kdtree.query(target_coord)
                if dist > 10.0:
                    logging.warning(f"Ponto GT de {target_landmark_name} muito distante de qualquer vértice na malha {i} (dist={dist:.2f}). Ignorado.")
                    continue
                # Extrair features para todos vértices desta malha
                X, indices = self._extract_features(mesh)
                if X is None or len(X) == 0:
                    logging.warning(f"Sem features extraídas da malha {i} para {target_landmark_name}.")
                    continue
                # Criar labels (1 para vértice alvo, 0 demais)
                labels = np.zeros(len(mesh.vertices), dtype=int)
                if target_idx < len(labels):
                    labels[target_idx] = 1
                else:
                    logging.error(f"Índice de vértice inválido durante treinamento (malha {i})")
                    continue
                # Balancear classes: limitar razão negativos:positivos em 20:1
                pos_idx = np.where(labels == 1)[0]
                neg_idx = np.where(labels == 0)[0]
                if len(pos_idx) == 0:
                    logging.warning(f"Malha {i} não contribuiu exemplos positivos para {target_landmark_name}.")
                    continue
                max_negativos = min(len(neg_idx), len(pos_idx) * 20)
                if max_negativos > 0:
                    chosen_neg_idx = np.random.choice(neg_idx, size=max_negativos, replace=False)
                    keep_idx = np.concatenate([pos_idx, chosen_neg_idx])
                else:
                    keep_idx = pos_idx
                X_sampled = X[keep_idx]
                y_sampled = labels[keep_idx]
                X_all.append(X_sampled)
                y_all.append(y_sampled)
            except Exception as e:
                logging.error(f"Erro preparando dados de treinamento na malha {i}: {e}")
                continue
        if not X_all:
            logging.error(f"Nenhum dado de treinamento compilado para {target_landmark_name}.")
            return False
        # Combinar dados de todas as malhas e treinar o modelo
        X = np.vstack(X_all)
        y = np.hstack(y_all)
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[target_landmark_name] = scaler
            clf = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced',
                n_jobs=-1, max_depth=10, min_samples_split=5, min_samples_leaf=2
            )
            logging.info(f"Treinando Random Forest para {target_landmark_name} com {len(y)} amostras...")
            clf.fit(X_scaled, y)
            # Avaliação simples via cross-val
            try:
                scores = cross_val_score(clf, X_scaled, y, cv=3, scoring='f1')
                logging.info(f"Desempenho CV (F1) p/ {target_landmark_name}: {scores.mean():.3f} ± {scores.std():.3f}")
            except Exception as e:
                logging.warning(f"Falha na validação cruzada: {e}")
            # Salvar modelo treinado e scaler
            model_path = os.path.join(self.model_dir, f"rf_model_{target_landmark_name}.joblib")
            scaler_path = os.path.join(self.model_dir, f"scaler_{target_landmark_name}.joblib")
            joblib.dump(clf, model_path)
            joblib.dump(scaler, scaler_path)
            self.models[target_landmark_name] = clf
            logging.info(f"Modelo {target_landmark_name} salvo em {model_path}")
            return True
        except Exception as e:
            logging.error(f"Erro durante treinamento do modelo {target_landmark_name}: {e}")
            return False

    def load_model(self, landmark_name):
        """Carrega em memória o modelo e scaler treinados de um determinado landmark."""
        model_path = os.path.join(self.model_dir, f"rf_model_{landmark_name}.joblib")
        scaler_path = os.path.join(self.model_dir, f"scaler_{landmark_name}.joblib")
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logging.warning(f"Arquivos de modelo não encontrados para '{landmark_name}'.")
            return False
        try:
            self.models[landmark_name] = joblib.load(model_path)
            self.scalers[landmark_name] = joblib.load(scaler_path)
            logging.info(f"Modelo '{landmark_name}' carregado com sucesso.")
            return True
        except Exception as e:
            logging.error(f"Erro ao carregar modelo '{landmark_name}': {e}")
            return False

    def predict(self, mesh, landmark_name):
        """Executa predição para localizar um landmark específico na malha fornecida."""
        # Assegurar que o modelo esteja carregado (tenta carregar se ainda não)
        if landmark_name not in self.models:
            if not self.load_model(landmark_name):
                logging.error(f"Modelo não disponível para {landmark_name}")
                return None, None
        try:
            model = self.models[landmark_name]
            scaler = self.scalers[landmark_name]
            # Extrair features de todos vértices da malha
            X, indices = self._extract_features(mesh)
            if X is None or X.shape[0] == 0:
                logging.error(f"Falha na extração de features para predição de {landmark_name}.")
                return None, None
            X_scaled = scaler.transform(X)
            probs = model.predict_proba(X_scaled)
            # Classe 1 corresponde ao landmark, obter probabilidade desta classe
            if probs.shape[1] < 2:
                logging.error(f"Modelo de {landmark_name} retornou formato inesperado de probabilidade.")
                return None, None
            landmark_probs = probs[:, 1]  # prob. de ser o landmark em cada índice
            best_idx = np.argmax(landmark_probs)
            best_prob = landmark_probs[best_idx]
            if best_prob < self.confidence_threshold:
                logging.info(f"{landmark_name}: nenhuma predição com confiança suficiente (max={best_prob:.2f}).")
                return None, None
            best_idx_global = best_idx if indices is None else indices[best_idx]
            predicted_coord = mesh.vertices[best_idx_global]
            logging.debug(f"{landmark_name} previsto no índice {best_idx_global} (confiança={best_prob:.3f})")
            return best_idx_global, predicted_coord
        except Exception as e:
            logging.error(f"Erro na predição de {landmark_name}: {e}")
            return None, None

    def detect(self, mesh):
        """Detecta todos os landmarks em uma malha usando os modelos treinados disponíveis."""
        if not isinstance(mesh, trimesh.Trimesh):
            logging.error("Entrada inválida para detecção ML (esperado Trimesh).")
            return None
        if len(mesh.vertices) == 0:
            logging.error("Malha vazia fornecida ao detector ML.")
            return None
        logging.info(f"Iniciando detecção ML - malha com {len(mesh.vertices)} vértices.")
        landmarks_found = {}
        for name in LANDMARK_NAMES:
            try:
                idx, point = self.predict(mesh, name)
                if point is not None:
                    landmarks_found[name] = point.tolist()
                else:
                    landmarks_found[name] = None
            except Exception as e:
                landmarks_found[name] = None
                logging.error(f"Erro ao detectar {name}: {e}")
        detected = sum(1 for coords in landmarks_found.values() if coords is not None)
        logging.info(f"Detecção ML finalizada: {detected}/{len(LANDMARK_NAMES)} landmarks detectados.")
        return landmarks_found
