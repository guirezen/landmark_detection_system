# -*- coding: utf-8 -*-
"""Módulo para detecção de landmarks usando Machine Learning básico (Random Forest)."""

import numpy as np
import trimesh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
import joblib # Para salvar/carregar modelos treinados
import os
import logging

from .landmarks import LANDMARK_NAMES, LANDMARK_MAP

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\")

class MLDetector:
    """Detecta landmarks usando um modelo de Machine Learning (Random Forest)."""

    def __init__(self, model_dir="./models", feature_radius_multiplier=5):
        """Inicializa o detector ML.

        Args:
            model_dir (str): Diretório para salvar/carregar modelos treinados.
            feature_radius_multiplier (int): Multiplicador do comprimento médio da aresta
                                           para definir o raio da vizinhança na extração de features.
        """
        self.model_dir = model_dir
        self.feature_radius_multiplier = feature_radius_multiplier
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {} # Dicionário para guardar modelos treinados (um por landmark)
        self.scalers = {} # Dicionário para guardar scalers (um por landmark)
        logging.info(f"Detector ML inicializado. Modelos em: {self.model_dir}")

    def _extract_features(self, mesh, vertex_indices=None):
        """Extrai features geométricas locais para um conjunto de vértices.

        Args:
            mesh (trimesh.Trimesh): Malha de entrada.
            vertex_indices (np.ndarray, optional): Índices dos vértices para os quais
                                                  extrair features. Se None, extrai para todos.

        Returns:
            np.ndarray: Matriz de features (n_vertices, n_features).
            np.ndarray: Índices dos vértices correspondentes às features extraídas.
        """
        if vertex_indices is None:
            vertices = mesh.vertices
            indices_out = np.arange(len(vertices))
        else:
            vertices = mesh.vertices[vertex_indices]
            indices_out = vertex_indices

        n_verts = len(vertices)
        logging.info(f"Extraindo features para {n_verts} vértices...")

        features = []

        # Feature 1: Coordenadas normalizadas (relativas ao bounding box)
        # Isso ajuda o modelo a ser um pouco mais invariante à posição/escala global
        bounds = mesh.bounding_box.bounds
        center = mesh.bounding_box.centroid
        extent = mesh.bounding_box.extents
        # Evitar divisão por zero se alguma extensão for nula
        extent[extent == 0] = 1.0
        norm_coords = (vertices - center) / extent
        features.append(norm_coords)

        # Feature 2: Normais dos vértices
        try:
            vertex_normals = mesh.vertex_normals[indices_out]
            features.append(vertex_normals)
        except Exception as e:
            logging.warning(f"Não foi possível obter normais dos vértices: {e}. Usando zeros.")
            features.append(np.zeros((n_verts, 3)))

        # Feature 3: Curvatura (Gaussiana como proxy, conforme detector geométrico)
        try:
            avg_edge = np.mean(mesh.edges_unique_length)
            radius = avg_edge * self.feature_radius_multiplier
            # Nota: Idealmente, calcularíamos curvaturas apenas para os índices necessários,
            # mas trimesh pode calcular para todos. Selecionamos depois.
            all_curvatures = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius=radius)
            curvatures = all_curvatures[indices_out].reshape(-1, 1)
            features.append(curvatures)
        except Exception as e:
            logging.warning(f"Não foi possível calcular curvatura: {e}. Usando zeros.")
            features.append(np.zeros((n_verts, 1)))

        # Feature 4: Distância ao centroide da malha
        centroid_dist = np.linalg.norm(vertices - mesh.centroid, axis=1).reshape(-1, 1)
        features.append(centroid_dist)

        # Feature 5: Shape Index / Curvedness (Mais avançado, requer cálculo de curvaturas principais k1, k2)
        # Por simplicidade (requisito do TCC), vamos omitir Shape Index/Curvedness por enquanto.
        # Se necessário, poderiam ser adicionados aqui.
        # Exemplo: k1, k2 = trimesh.curvature.principal_curvatures(mesh)
        # shape_index = (2 / np.pi) * np.arctan((k1 + k2) / (k1 - k2))
        # curvedness = np.sqrt((k1**2 + k2**2) / 2)

        # Combinar todas as features
        feature_matrix = np.hstack(features)
        logging.info(f"Matriz de features criada com shape: {feature_matrix.shape}")

        # Lidar com NaNs ou Infs que podem surgir dos cálculos
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_matrix, indices_out

    def train(self, meshes, all_landmarks_gt, target_landmark_name):
        """Treina um modelo Random Forest para um landmark específico.

        Args:
            meshes (list[trimesh.Trimesh]): Lista de malhas de treinamento.
            all_landmarks_gt (list[dict]): Lista de dicionários contendo as coordenadas
                                           ground truth dos landmarks para cada malha.
                                           Formato: [{'Glabela': [x,y,z], ...}, ...]
            target_landmark_name (str): Nome do landmark para o qual treinar o modelo.

        Returns:
            bool: True se o treinamento foi bem-sucedido, False caso contrário.
        """
        logging.info(f"Iniciando treinamento para o landmark: {target_landmark_name}")

        all_features = []
        all_labels = []

        if target_landmark_name not in LANDMARK_MAP:
            logging.error(f"Landmark alvo desconhecido: {target_landmark_name}")
            return False

        # Preparar dados de treinamento
        for i, mesh in enumerate(meshes):
            landmarks_gt = all_landmarks_gt[i]
            target_coord_gt = landmarks_gt.get(target_landmark_name)

            if target_coord_gt is None:
                logging.warning(f"Ground truth para {target_landmark_name} não encontrado na malha {i}. Pulando.")
                continue

            # Encontrar o vértice mais próximo da coordenada ground truth
            try:
                kdtree = KDTree(mesh.vertices)
                distance, target_vertex_idx = kdtree.query(target_coord_gt)
                logging.debug(f"Malha {i}: Vértice GT para {target_landmark_name} encontrado: índice {target_vertex_idx}, distância {distance:.4f}")
            except Exception as e:
                logging.error(f"Erro ao encontrar vértice GT para {target_landmark_name} na malha {i}: {e}")
                continue

            # Extrair features para todos os vértices da malha atual
            features, vertex_indices = self._extract_features(mesh)
            if features is None:
                logging.warning(f"Falha ao extrair features para a malha {i}. Pulando.")
                continue

            # Criar labels: 1 para o vértice alvo, 0 para os outros
            # NOTA: Esta é uma simplificação extrema! Em um cenário real, precisaríamos
            # de uma estratégia mais robusta (ex: considerar vizinhança, lidar com desbalanceamento).
            labels = np.zeros(len(mesh.vertices), dtype=int)
            if target_vertex_idx < len(labels):
                 labels[target_vertex_idx] = 1
            else:
                 logging.error(f"Índice GT {target_vertex_idx} fora dos limites para malha {i} com {len(labels)} vértices.")
                 continue # Pula esta malha se o índice for inválido

            # Lidar com desbalanceamento extremo (muitos 0s, poucos 1s)
            # Estratégia simples: Subamostragem dos negativos (undersampling)
            positive_indices = np.where(labels == 1)[0]
            negative_indices = np.where(labels == 0)[0]

            # Manter todos os positivos e um número similar de negativos (ex: 10x mais negativos)
            n_positives = len(positive_indices)
            n_negatives_to_keep = min(len(negative_indices), n_positives * 20) # Ajustar proporção

            if n_negatives_to_keep > 0:
                sampled_negative_indices = np.random.choice(negative_indices, size=n_negatives_to_keep, replace=False)
                indices_to_keep = np.concatenate([positive_indices, sampled_negative_indices])
            elif n_positives > 0:
                 indices_to_keep = positive_indices # Caso não haja negativos?
            else:
                 logging.warning(f"Nenhum vértice positivo encontrado para {target_landmark_name} na malha {i}. Pulando.")
                 continue # Pula se não houver positivos

            # Selecionar features e labels correspondentes aos índices mantidos
            # Precisamos mapear os indices_to_keep (índices globais da malha) para os índices da matriz de features
            # Se _extract_features retornou features para todos, o mapeamento é direto.
            if features.shape[0] == len(mesh.vertices):
                sampled_features = features[indices_to_keep]
                sampled_labels = labels[indices_to_keep]
            else:
                # Caso _extract_features tenha retornado um subconjunto (não deveria acontecer com a implementação atual)
                logging.error("Inconsistência no número de features extraídas.")
                continue

            all_features.append(sampled_features)
            all_labels.append(sampled_labels)

        if not all_features:
            logging.error(f"Nenhum dado de treinamento válido coletado para {target_landmark_name}.")
            return False

        # Combinar dados de todas as malhas
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)

        logging.info(f"Total de amostras para treinamento de {target_landmark_name}: {len(y)} (Positivos: {np.sum(y==1)}, Negativos: {np.sum(y==0)})")

        if np.sum(y==1) == 0:
            logging.error(f"Nenhuma amostra positiva encontrada para {target_landmark_name} após processamento. Treinamento cancelado.")
            return False

        # Escalar features (importante para muitos algoritmos de ML)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[target_landmark_name] = scaler # Salvar scaler

        # Dividir em treino/teste para avaliação interna (opcional, mas bom)
        # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        # Treinar o classificador Random Forest
        # Parâmetros podem ser otimizados (GridSearchCV, RandomizedSearchCV)
        # Usar class_weight=\'balanced\' para ajudar com desbalanceamento residual
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=\'balanced\
                                             , n_jobs=-1) # Usar todos os cores

        logging.info(f"Treinando RandomForest para {target_landmark_name}...")
        rf_classifier.fit(X_scaled, y) # Treinar com todos os dados coletados

        # Avaliação (ex: usando cross-validation)
        # scores = cross_val_score(rf_classifier, X_scaled, y, cv=5, scoring=\'accuracy\') # Usar métrica apropriada (f1, roc_auc)
        # logging.info(f"Cross-validation accuracy para {target_landmark_name}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

        # Salvar o modelo treinado
        model_filename = os.path.join(self.model_dir, f"rf_model_{target_landmark_name}.joblib")
        scaler_filename = os.path.join(self.model_dir, f"scaler_{target_landmark_name}.joblib")
        try:
            joblib.dump(rf_classifier, model_filename)
            joblib.dump(scaler, scaler_filename)
            self.models[target_landmark_name] = rf_classifier # Guardar na memória também
            logging.info(f"Modelo e scaler para {target_landmark_name} salvos em {self.model_dir}")
            return True
        except Exception as e:
            logging.error(f"Erro ao salvar modelo/scaler para {target_landmark_name}: {e}")
            return False

    def load_model(self, landmark_name):
        """Carrega um modelo treinado e scaler para um landmark específico."""
        model_filename = os.path.join(self.model_dir, f"rf_model_{landmark_name}.joblib")
        scaler_filename = os.path.join(self.model_dir, f"scaler_{landmark_name}.joblib")
        if os.path.exists(model_filename) and os.path.exists(scaler_filename):
            try:
                self.models[landmark_name] = joblib.load(model_filename)
                self.scalers[landmark_name] = joblib.load(scaler_filename)
                logging.info(f"Modelo e scaler para {landmark_name} carregados.")
                return True
            except Exception as e:
                logging.error(f"Erro ao carregar modelo/scaler para {landmark_name}: {e}")
                return False
        else:
            logging.warning(f"Arquivos de modelo/scaler não encontrados para {landmark_name} em {self.model_dir}")
            return False

    def predict(self, mesh, landmark_name):
        """Prevê a localização de um landmark específico na malha.

        Args:
            mesh (trimesh.Trimesh): Malha onde procurar o landmark.
            landmark_name (str): Nome do landmark a ser previsto.

        Returns:
            tuple (int, np.ndarray) or (None, None): Índice e coordenadas [x, y, z]
                                                    do vértice previsto como landmark,
                                                    ou (None, None) se falhar.
        """
        logging.info(f"Iniciando predição para {landmark_name}...")
        if landmark_name not in self.models or landmark_name not in self.scalers:
            # Tenta carregar se não estiver na memória
            if not self.load_model(landmark_name):
                logging.error(f"Modelo ou scaler para {landmark_name} não está carregado e não pôde ser encontrado.")
                return None, None

        model = self.models[landmark_name]
        scaler = self.scalers[landmark_name]

        # 1. Extrair features para todos os vértices da malha
        features, vertex_indices = self._extract_features(mesh)
        if features is None or len(features) == 0:
            logging.error("Falha ao extrair features para predição.")
            return None, None

        # 2. Escalar features usando o scaler carregado/treinado
        try:
            X_scaled = scaler.transform(features)
        except Exception as e:
            logging.error(f"Erro ao escalar features para predição: {e}")
            return None, None

        # 3. Prever probabilidades para cada vértice ser o landmark alvo
        try:
            # predict_proba retorna [prob_classe_0, prob_classe_1]
            probabilities = model.predict_proba(X_scaled)[:, 1] # Pegar a probabilidade da classe 1 (landmark)
        except Exception as e:
            logging.error(f"Erro durante a predição de probabilidades: {e}")
            return None, None

        # 4. Selecionar o vértice com a maior probabilidade
        if len(probabilities) > 0:
            best_vertex_idx_local = np.argmax(probabilities)
            # Mapear de volta para o índice global da malha (se necessário)
            # Na implementação atual de _extract_features, os índices são sequenciais 0..N-1
            best_vertex_idx_global = vertex_indices[best_vertex_idx_local]
            predicted_coord = mesh.vertices[best_vertex_idx_global]
            confidence = probabilities[best_vertex_idx_local]
            logging.info(f"Landmark {landmark_name} previsto no índice {best_vertex_idx_global} com confiança {confidence:.4f}")
            return best_vertex_idx_global, predicted_coord.tolist()
        else:
            logging.warning(f"Nenhuma probabilidade prevista para {landmark_name}.")
            return None, None

    def detect(self, mesh):
        """Detecta todos os landmarks para os quais um modelo foi treinado/carregado.

        Args:
            mesh (trimesh.Trimesh): Malha pré-processada.

        Returns:
            dict: Dicionário mapeando nomes de landmarks para suas coordenadas [x, y, z],
                  ou None se a detecção falhar ou o modelo não existir.
                  Formato: {"Glabela": [x,y,z], "Nasion": None, ...}
        """
        if not isinstance(mesh, trimesh.Trimesh):
            logging.error("Input para detecção ML não é um objeto Trimesh válido.")
            return None

        logging.info(f"Iniciando detecção ML de landmarks em malha com {len(mesh.vertices)} vértices.")
        landmarks_found = {}

        # Tentar carregar/prever para cada landmark na lista principal
        for landmark_name in LANDMARK_NAMES:
            if landmark_name in self.models or self.load_model(landmark_name):
                index, point = self.predict(mesh, landmark_name)
                landmarks_found[landmark_name] = point # point já é list ou None
            else:
                landmarks_found[landmark_name] = None
                logging.warning(f"Modelo para {landmark_name} não encontrado ou não pôde ser carregado.")

        logging.info("Detecção ML de landmarks concluída.")
        return landmarks_found

# Exemplo de uso (requer mesh_processor, arquivos STL dummy e dados GT dummy)
if __name__ == \"__main__\":
    from .mesh_processor import MeshProcessor # Import relativo

    # --- Configuração de Dados Dummy --- 
    # Criar malhas dummy (esferas deslocadas)
    meshes_dummy = []
    landmarks_gt_dummy = []
    if not os.path.exists("data/skulls"): os.makedirs("data/skulls")

    center1 = [0, 0, 50]
    mesh1 = trimesh.primitives.Sphere(radius=50, center=center1)
    mesh1.export("data/skulls/dummy_sphere1.stl")
    meshes_dummy.append(mesh1)
    # Landmarks GT para esfera 1 (aproximados)
    gt1 = {
        "Glabela": [center1[0], center1[1] + 50, center1[2]], # Frente
        "Nasion": [center1[0], center1[1] + 45, center1[2] - 10], # Frente, um pouco abaixo
        "Bregma": [center1[0], center1[1], center1[2] + 50], # Topo
        "Opisthocranion": [center1[0], center1[1] - 50, center1[2]], # Trás
        "Euryon_Esquerdo": [center1[0] - 50, center1[1], center1[2]], # Esquerda
        "Euryon_Direito": [center1[0] + 50, center1[1], center1[2]], # Direita
        "Vertex": [center1[0], center1[1], center1[2] + 50], # Topo
        "Inion": [center1[0], center1[1] - 45, center1[2] - 10] # Trás, um pouco abaixo
    }
    landmarks_gt_dummy.append(gt1)

    center2 = [10, 5, 60] # Esfera ligeiramente diferente
    mesh2 = trimesh.primitives.Sphere(radius=55, center=center2)
    mesh2.export("data/skulls/dummy_sphere2.stl")
    meshes_dummy.append(mesh2)
    gt2 = {
        "Glabela": [center2[0], center2[1] + 55, center2[2]],
        "Nasion": [center2[0], center2[1] + 50, center2[2] - 12],
        "Bregma": [center2[0], center2[1], center2[2] + 55],
        "Opisthocranion": [center2[0], center2[1] - 55, center2[2]],
        "Euryon_Esquerdo": [center2[0] - 55, center2[1], center2[2]],
        "Euryon_Direito": [center2[0] + 55, center2[1], center2[2]],
        "Vertex": [center2[0], center2[1], center2[2] + 55],
        "Inion": [center2[0], center2[1] - 50, center2[2] - 12]
    }
    landmarks_gt_dummy.append(gt2)

    # --- Treinamento --- 
    detector_ml = MLDetector(model_dir="./models_dummy") # Usar dir separado para teste
    logging.info("--- Iniciando Treinamento Dummy ---")
    training_successful = True
    for landmark_name in LANDMARK_NAMES:
        success = detector_ml.train(meshes_dummy, landmarks_gt_dummy, landmark_name)
        if not success:
            training_successful = False
            logging.error(f"Falha no treinamento para {landmark_name}")

    # --- Predição --- 
    if training_successful:
        logging.info("--- Iniciando Predição ML na Malha Dummy 1 ---")
        # Usar a primeira malha dummy para predição
        # Simplificar um pouco
        processor = MeshProcessor(data_dir="./data/skulls", cache_dir="./data/cache")
        simplified_mesh1 = processor.simplify(meshes_dummy[0], target_faces=2000, original_filename="dummy_sphere1.stl")

        if simplified_mesh1:
            detected_landmarks_ml = detector_ml.detect(simplified_mesh1)

            print("\n--- Landmarks Detectados (ML) na Malha 1 ---")
            if detected_landmarks_ml:
                for name, coords in detected_landmarks_ml.items():
                    if coords:
                        print(f"  {name}: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]")
                    else:
                        print(f"  {name}: Não detectado / Modelo não treinado")
            else:
                print("Falha geral na detecção ML.")
        else:
             logging.error("Falha ao simplificar malha 1 para predição ML.")

    else:
        logging.error("Treinamento ML falhou para um ou mais landmarks. Predição abortada.")

