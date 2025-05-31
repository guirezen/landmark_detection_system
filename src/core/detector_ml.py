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
        """Inicializa o detector ML.

        Args:
            model_dir (str): Diretório para salvar/carregar modelos treinados.
            feature_radius_multiplier (int): Multiplicador do comprimento médio da aresta
                                           para definir o raio da vizinhança na extração de features.
            confidence_threshold (float): Limiar de confiança para aceitar uma predição.
        """
        self.model_dir = model_dir
        self.feature_radius_multiplier = feature_radius_multiplier
        self.confidence_threshold = confidence_threshold
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.models = {}  # Dicionário para guardar modelos treinados
        self.scalers = {}  # Dicionário para guardar scalers
        
        logging.info(f"Detector ML inicializado. Modelos em: {self.model_dir}")

    def _extract_features(self, mesh, vertex_indices=None):
        """Extrai features geométricas locais para um conjunto de vértices.

        Args:
            mesh (trimesh.Trimesh): Malha de entrada.
            vertex_indices (np.ndarray, optional): Índices dos vértices para extração.

        Returns:
            tuple: (matriz de features, índices dos vértices)
        """
        if vertex_indices is None:
            vertices = mesh.vertices
            indices_out = np.arange(len(vertices))
        else:
            vertices = mesh.vertices[vertex_indices]
            indices_out = vertex_indices

        n_verts = len(vertices)
        logging.debug(f"Extraindo features para {n_verts} vértices...")

        features = []

        try:
            # Feature 1: Coordenadas normalizadas
            bounds = mesh.bounds
            center = mesh.centroid
            extent = mesh.extents
            # Evitar divisão por zero
            extent = np.where(extent == 0, 1.0, extent)
            norm_coords = (vertices - center) / extent
            features.append(norm_coords)

            # Feature 2: Normais dos vértices
            try:
                if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) == len(mesh.vertices):
                    vertex_normals = mesh.vertex_normals[indices_out]
                else:
                    # Calcular normais se não existirem
                    mesh.vertex_normals  # Isso força o cálculo
                    vertex_normals = mesh.vertex_normals[indices_out]
                features.append(vertex_normals)
            except Exception as e:
                logging.warning(f"Erro ao obter normais dos vértices: {e}. Usando zeros.")
                features.append(np.zeros((n_verts, 3)))

            # Feature 3: Curvatura Gaussiana
            try:
                avg_edge = np.mean(mesh.edges_unique_length)
                radius = avg_edge * self.feature_radius_multiplier
                
                all_curvatures = trimesh.curvature.discrete_gaussian_curvature_measure(
                    mesh, mesh.vertices, radius=radius
                )
                curvatures = all_curvatures[indices_out].reshape(-1, 1)
                # Normalizar curvatura para evitar valores extremos
                curvatures = np.nan_to_num(curvatures, nan=0.0, posinf=1.0, neginf=-1.0)
                features.append(curvatures)
            except Exception as e:
                logging.warning(f"Erro ao calcular curvatura: {e}. Usando zeros.")
                features.append(np.zeros((n_verts, 1)))

            # Feature 4: Distância ao centroide
            centroid_dist = np.linalg.norm(vertices - mesh.centroid, axis=1).reshape(-1, 1)
            features.append(centroid_dist)

            # Feature 5: Coordenadas esféricas relativas ao centroide
            try:
                relative_pos = vertices - mesh.centroid
                # Converter para coordenadas esféricas
                r = np.linalg.norm(relative_pos, axis=1)
                theta = np.arctan2(relative_pos[:, 1], relative_pos[:, 0])  # azimute
                phi = np.arccos(np.clip(relative_pos[:, 2] / (r + 1e-8), -1, 1))  # elevação
                
                spherical_coords = np.column_stack([r, theta, phi])
                spherical_coords = np.nan_to_num(spherical_coords)
                features.append(spherical_coords)
            except Exception as e:
                logging.warning(f"Erro ao calcular coordenadas esféricas: {e}. Usando zeros.")
                features.append(np.zeros((n_verts, 3)))

            # Combinar todas as features
            feature_matrix = np.hstack(features)
            
            # Tratar NaNs e infinitos
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
            
            logging.debug(f"Matriz de features criada: {feature_matrix.shape}")
            return feature_matrix, indices_out

        except Exception as e:
            logging.error(f"Erro durante extração de features: {e}")
            # Retornar features básicas em caso de erro
            basic_features = np.hstack([
                (vertices - mesh.centroid) / (mesh.extents + 1e-8),  # Coordenadas normalizadas
                np.linalg.norm(vertices - mesh.centroid, axis=1).reshape(-1, 1)  # Distância
            ])
            return basic_features, indices_out

    def train(self, meshes, all_landmarks_gt, target_landmark_name):
        """Treina um modelo Random Forest para um landmark específico.

        Args:
            meshes (list): Lista de malhas de treinamento.
            all_landmarks_gt (list): Lista de dicionários com landmarks ground truth.
            target_landmark_name (str): Nome do landmark para treinar.

        Returns:
            bool: True se o treinamento foi bem-sucedido.
        """
        if target_landmark_name not in LANDMARK_NAMES:
            logging.error(f"Landmark alvo desconhecido: {target_landmark_name}")
            return False

        logging.info(f"Iniciando treinamento para: {target_landmark_name}")

        all_features = []
        all_labels = []

        for i, mesh in enumerate(meshes):
            try:
                landmarks_gt = all_landmarks_gt[i]
                target_coord_gt = landmarks_gt.get(target_landmark_name)

                if target_coord_gt is None:
                    logging.warning(f"GT para {target_landmark_name} ausente na malha {i}")
                    continue

                # Encontrar vértice mais próximo do GT
                kdtree = KDTree(mesh.vertices)
                distance, target_vertex_idx = kdtree.query(target_coord_gt)
                
                if distance > 10.0:  # Limiar de distância em mm
                    logging.warning(f"GT muito distante dos vértices na malha {i} (dist={distance:.2f})")
                    continue

                # Extrair features
                features, vertex_indices = self._extract_features(mesh)
                if features is None or len(features) == 0:
                    logging.warning(f"Falha na extração de features para malha {i}")
                    continue

                # Criar labels
                labels = np.zeros(len(mesh.vertices), dtype=int)
                if target_vertex_idx < len(labels):
                    labels[target_vertex_idx] = 1
                else:
                    logging.error(f"Índice GT inválido para malha {i}")
                    continue

                # Balanceamento de classes (subamostragem da classe majoritária)
                positive_indices = np.where(labels == 1)[0]
                negative_indices = np.where(labels == 0)[0]

                n_positives = len(positive_indices)
                if n_positives == 0:
                    logging.warning(f"Nenhum exemplo positivo para {target_landmark_name} na malha {i}")
                    continue

                # Manter proporção positivo:negativo de 1:20 no máximo
                n_negatives_to_keep = min(len(negative_indices), n_positives * 20)
                
                if n_negatives_to_keep > 0:
                    sampled_negative_indices = np.random.choice(
                        negative_indices, size=n_negatives_to_keep, replace=False
                    )
                    indices_to_keep = np.concatenate([positive_indices, sampled_negative_indices])
                else:
                    indices_to_keep = positive_indices

                # Selecionar features e labels
                sampled_features = features[indices_to_keep]
                sampled_labels = labels[indices_to_keep]

                all_features.append(sampled_features)
                all_labels.append(sampled_labels)

                logging.debug(f"Malha {i}: {len(sampled_features)} amostras "
                            f"({np.sum(sampled_labels)} positivas)")

            except Exception as e:
                logging.error(f"Erro ao processar malha {i} para {target_landmark_name}: {e}")
                continue

        if not all_features:
            logging.error(f"Nenhum dado válido para treinar {target_landmark_name}")
            return False

        # Combinar dados de todas as malhas
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)

        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        logging.info(f"Dados de treinamento para {target_landmark_name}: "
                    f"{len(y)} amostras ({n_positive} positivas, {n_negative} negativas)")

        if n_positive == 0:
            logging.error(f"Nenhuma amostra positiva para {target_landmark_name}")
            return False

        try:
            # Escalar features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[target_landmark_name] = scaler

            # Treinar Random Forest
            rf_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                max_depth=10,  # Evitar overfitting
                min_samples_split=5,
                min_samples_leaf=2
            )

            logging.info(f"Treinando Random Forest para {target_landmark_name}...")
            rf_classifier.fit(X_scaled, y)

            # Validação cruzada rápida
            try:
                cv_scores = cross_val_score(rf_classifier, X_scaled, y, cv=3, scoring='f1')
                logging.info(f"CV F1-score para {target_landmark_name}: "
                           f"{np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            except Exception as e:
                logging.warning(f"Erro na validação cruzada: {e}")

            # Salvar modelo e scaler
            model_path = os.path.join(self.model_dir, f"rf_model_{target_landmark_name}.joblib")
            scaler_path = os.path.join(self.model_dir, f"scaler_{target_landmark_name}.joblib")

            joblib.dump(rf_classifier, model_path)
            joblib.dump(scaler, scaler_path)

            self.models[target_landmark_name] = rf_classifier

            logging.info(f"Modelo para {target_landmark_name} salvo em {model_path}")
            return True

        except Exception as e:
            logging.error(f"Erro durante treinamento de {target_landmark_name}: {e}")
            return False

    def load_model(self, landmark_name):
        """Carrega um modelo treinado para um landmark específico."""
        model_path = os.path.join(self.model_dir, f"rf_model_{landmark_name}.joblib")
        scaler_path = os.path.join(self.model_dir, f"scaler_{landmark_name}.joblib")
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            logging.warning(f"Arquivos de modelo não encontrados para {landmark_name}")
            return False

        try:
            self.models[landmark_name] = joblib.load(model_path)
            self.scalers[landmark_name] = joblib.load(scaler_path)
            logging.info(f"Modelo para {landmark_name} carregado com sucesso")
            return True
        except Exception as e:
            logging.error(f"Erro ao carregar modelo para {landmark_name}: {e}")
            return False

    def predict(self, mesh, landmark_name):
        """Prevê a localização de um landmark específico na malha.

        Args:
            mesh (trimesh.Trimesh): Malha onde procurar o landmark.
            landmark_name (str): Nome do landmark a ser previsto.

        Returns:
            tuple: (índice, coordenadas) do vértice previsto ou (None, None) se falhar.
        """
        if landmark_name not in self.models and not self.load_model(landmark_name):
            logging.error(f"Modelo para {landmark_name} não disponível")
            return None, None

        try:
            model = self.models[landmark_name]
            scaler = self.scalers[landmark_name]

            # Extrair features
            features, vertex_indices = self._extract_features(mesh)
            if features is None or len(features) == 0:
                logging.error(f"Falha na extração de features para predição de {landmark_name}")
                return None, None

            # Escalar features
            X_scaled = scaler.transform(features)

            # Prever probabilidades
            probabilities = model.predict_proba(X_scaled)
            
            # Verificar se o modelo tem classe positiva
            if probabilities.shape[1] < 2:
                logging.error(f"Modelo para {landmark_name} não tem classe positiva")
                return None, None
            
            positive_probs = probabilities[:, 1]  # Probabilidade da classe 1

            # Encontrar vértice com maior probabilidade
            best_idx_local = np.argmax(positive_probs)
            best_prob = positive_probs[best_idx_local]
            
            # Verificar limiar de confiança
            if best_prob < self.confidence_threshold:
                logging.warning(f"Baixa confiança para {landmark_name}: {best_prob:.3f}")
                return None, None

            best_idx_global = vertex_indices[best_idx_local]
            predicted_coord = mesh.vertices[best_idx_global]

            logging.info(f"{landmark_name} previsto no índice {best_idx_global} "
                        f"(confiança: {best_prob:.3f})")
            
            return best_idx_global, predicted_coord

        except Exception as e:
            logging.error(f"Erro durante predição de {landmark_name}: {e}")
            return None, None

    def detect(self, mesh):
        """Detecta todos os landmarks para os quais um modelo está disponível.

        Args:
            mesh (trimesh.Trimesh): Malha pré-processada.

        Returns:
            dict: Dicionário mapeando nomes de landmarks para coordenadas ou None.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            logging.error("Input para detecção ML não é um Trimesh válido")
            return None

        if len(mesh.vertices) == 0:
            logging.error("Malha vazia fornecida para detecção ML")
            return None

        logging.info(f"Iniciando detecção ML em malha com {len(mesh.vertices)} vértices")
        landmarks_found = {}

        for landmark_name in LANDMARK_NAMES:
            try:
                index, point = self.predict(mesh, landmark_name)
                if point is not None:
                    landmarks_found[landmark_name] = point.tolist()
                else:
                    landmarks_found[landmark_name] = None
            except Exception as e:
                landmarks_found[landmark_name] = None
                logging.error(f"Erro inesperado ao detectar {landmark_name}: {e}")

        # Estatísticas
        detected_count = sum(1 for coords in landmarks_found.values() if coords is not None)
        logging.info(f"Detecção ML concluída: {detected_count}/{len(LANDMARK_NAMES)} detectados")

        return landmarks_found

# Exemplo de uso e teste
if __name__ == "__main__":
    import os
    import sys
    
    # Adicionar path para imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from mesh_processor import MeshProcessor
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Criar dados dummy para teste
    os.makedirs("data/skulls", exist_ok=True)
    
    # Criar malhas dummy
    meshes_dummy = []
    landmarks_gt_dummy = []
    
    for i in range(2):
        center = [i*5, i*3, 50 + i*10]
        mesh = trimesh.primitives.Sphere(radius=50 + i*5, center=center, subdivisions=2)
        mesh_path = f"data/skulls/dummy_sphere_{i}.stl"
        mesh.export(mesh_path)
        meshes_dummy.append(mesh)
        
        # GT aproximado para a esfera
        gt = {
            "Glabela": [center[0], center[1] + 50 + i*5, center[2]],
            "Nasion": [center[0], center[1] + 45 + i*5, center[2] - 10],
            "Bregma": [center[0], center[1], center[2] + 50 + i*5],
            "Opisthocranion": [center[0], center[1] - 50 - i*5, center[2]],
            "Euryon_Esquerdo": [center[0] - 50 - i*5, center[1], center[2]],
            "Euryon_Direito": [center[0] + 50 + i*5, center[1], center[2]],
            "Vertex": [center[0], center[1], center[2] + 50 + i*5],
            "Inion": [center[0], center[1] - 45 - i*5, center[2] - 10]
        }
        landmarks_gt_dummy.append(gt)

    # Testar treinamento
    detector_ml = MLDetector(model_dir="./models_test")
    
    logging.info("=== Teste de Treinamento ML ===")
    success_count = 0
    
    for landmark_name in LANDMARK_NAMES[:3]:  # Testar apenas os primeiros 3
        success = detector_ml.train(meshes_dummy, landmarks_gt_dummy, landmark_name)
        if success:
            success_count += 1

    if success_count > 0:
        logging.info("=== Teste de Predição ML ===")
        # Testar predição na primeira malha
        test_mesh = meshes_dummy[0]
        detected_landmarks = detector_ml.detect(test_mesh)

        print("\n=== Landmarks Detectados (ML) ===")
        if detected_landmarks:
            for name, coords in detected_landmarks.items():
                if coords:
                    print(f"  {name}: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]")
                else:
                    print(f"  {name}: Não detectado")
        else:
            print("Falha geral na detecção ML.")
    else:
        logging.error("Nenhum modelo foi treinado com sucesso.")