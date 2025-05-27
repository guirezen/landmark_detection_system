# -*- coding: utf-8 -*-
"""Módulo para detecção de landmarks usando métodos geométricos simples."""

import numpy as np
import trimesh
from scipy.spatial import KDTree
import logging

from .landmarks import LANDMARK_NAMES, LANDMARK_MAP

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

class GeometricDetector:
    """Detecta landmarks em uma malha 3D usando propriedades geométricas."""

    def __init__(self, landmark_definitions=None):
        """Inicializa o detector geométrico.

        Args:
            landmark_definitions (dict, optional): Dicionário com definições ou
                                                  heurísticas específicas para cada landmark.
                                                  Defaults to None.
        """
        self.landmark_definitions = landmark_definitions if landmark_definitions else {}
        # Mapeamento de nomes para funções de detecção específicas
        self.detection_functions = {
            "Glabela": self._find_glabela,
            "Nasion": self._find_nasion,
            "Bregma": self._find_bregma_or_vertex, # Pode detectar Bregma ou Vertex
            "Opisthocranion": self._find_opisthocranion,
            "Euryon_Esquerdo": self._find_euryon_left,
            "Euryon_Direito": self._find_euryon_right,
            "Vertex": self._find_bregma_or_vertex, # Reutiliza a função
            "Inion": self._find_inion,
            # Adicionar funções para outros landmarks se necessário
        }
        logging.info("Detector Geométrico inicializado.")

    def _calculate_curvature(self, mesh, radius=5.0):
        """Calcula a curvatura média para cada vértice (simplificado).

        Args:
            mesh (trimesh.Trimesh): Malha de entrada.
            radius (float): Raio para considerar a vizinhança no cálculo.
                          (Nota: trimesh não tem um cálculo direto de curvatura
                           baseado em raio de forma simples. Usaremos a curvatura
                           principal disponível).

        Returns:
            np.ndarray: Array com a curvatura média (ou aproximação) para cada vértice.
        """
        try:
            # Trimesh calcula as curvaturas principais (k1, k2)
            curvatures = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius=radius)
            # A curvatura média é (k1+k2)/2. A Gaussiana é k1*k2.
            # Usaremos a curvatura Gaussiana como proxy inicial, ou curvatura média se disponível
            # Alternativa: Usar a curvatura média se implementada ou disponível em outra lib.
            # Aqui, vamos usar a curvatura Gaussiana como proxy, pois é mais direta no trimesh.
            # Valores altos (positivos) indicam picos, valores baixos (negativos) indicam vales/selas.
            logging.info(f"Curvatura Gaussiana calculada para {len(mesh.vertices)} vértices.")
            # Normalizar ou ajustar conforme necessário
            return curvatures
        except Exception as e:
            logging.error(f"Erro ao calcular curvatura: {e}")
            # Retorna um array de zeros ou NaN se o cálculo falhar
            return np.zeros(len(mesh.vertices))

    def _find_extreme_vertex(self, mesh, direction_vector):
        """Encontra o vértice mais extremo em uma determinada direção."""
        dots = mesh.vertices @ np.array(direction_vector)
        extreme_idx = np.argmax(dots)
        return extreme_idx, mesh.vertices[extreme_idx]

    def _find_glabela(self, mesh, kdtree, curvatures):
        """Encontra a Glabela (ponto mais proeminente frontal).
           Heurística: Ponto mais anterior (max +Y assumindo orientação padrão) na região superior frontal.
        """
        # Assumindo orientação: +Y é para frente, +Z é para cima
        # 1. Encontrar o ponto mais à frente (max Y)
        idx_max_y, _ = self._find_extreme_vertex(mesh, [0, 1, 0])

        # 2. Refinar: Buscar na vizinhança do ponto max Y por um ponto com alta curvatura convexa
        #    ou simplesmente o ponto mais anterior na metade superior da face.
        #    Simplificação: Usar o ponto mais anterior (max Y) que esteja acima do centroide Z.
        center_z = mesh.centroid[2]
        frontal_upper_indices = np.where((mesh.vertices[:, 1] > (mesh.bounds[1, 1] * 0.8)) & # Bem à frente
                                         (mesh.vertices[:, 2] > center_z))[0] # Na metade superior

        if len(frontal_upper_indices) == 0:
             logging.warning("Não foi possível encontrar vértices na região frontal superior para Glabela. Usando max Y geral.")
             return idx_max_y, mesh.vertices[idx_max_y]

        # Dentre os candidatos, pegar o mais à frente
        frontal_vertices = mesh.vertices[frontal_upper_indices]
        glabela_idx_local = np.argmax(frontal_vertices[:, 1])
        glabela_idx_global = frontal_upper_indices[glabela_idx_local]

        logging.info(f"Glabela detectada no índice: {glabela_idx_global}")
        return glabela_idx_global, mesh.vertices[glabela_idx_global]

    def _find_nasion(self, mesh, kdtree, curvatures):
        """Encontra o Nasion (depressão nasal).
           Heurística: Ponto de maior concavidade (menor curvatura) na linha média frontal,
           abaixo da glabela e acima de um certo limite.
        """
        # Assumindo orientação: +Y frente, +Z cima, X=0 linha média
        # 1. Definir uma região de interesse (ROI) na linha média frontal
        center_z = mesh.centroid[2]
        bounds = mesh.bounds
        roi_indices = np.where((np.abs(mesh.vertices[:, 0]) < (bounds[1, 0] - bounds[0, 0]) * 0.1) & # Próximo à linha média X=0
                               (mesh.vertices[:, 1] > bounds[0, 1] + (bounds[1, 1] - bounds[0, 1]) * 0.7) & # Na parte frontal
                               (mesh.vertices[:, 2] < center_z + (bounds[1, 2] - center_z) * 0.5) & # Abaixo da testa
                               (mesh.vertices[:, 2] > bounds[0, 2] + (bounds[1, 2] - bounds[0, 2]) * 0.3))[0] # Acima da base do nariz

        if len(roi_indices) == 0:
            logging.warning("Não foi possível encontrar vértices na ROI do Nasion. Tentando alternativa.")
            # Alternativa: Ponto mais recuado (min Y) na linha média frontal superior?
            # Ou ponto de menor curvatura próximo à glabela?
            # Por simplicidade, retornamos um ponto central da ROI esperada ou falhamos
            # Tentar ponto de menor curvatura na região frontal média
            frontal_mid_indices = np.where((np.abs(mesh.vertices[:, 0]) < (bounds[1, 0] - bounds[0, 0]) * 0.1) & 
                                           (mesh.vertices[:, 1] > (bounds[1, 1] * 0.7)))[0]
            if len(frontal_mid_indices) == 0: return None, None # Falha
            roi_indices = frontal_mid_indices

        # 2. Encontrar o ponto de menor curvatura (mais côncavo) dentro da ROI
        roi_curvatures = curvatures[roi_indices]
        nasion_idx_local = np.argmin(roi_curvatures)
        nasion_idx_global = roi_indices[nasion_idx_local]

        logging.info(f"Nasion detectado no índice: {nasion_idx_global}")
        return nasion_idx_global, mesh.vertices[nasion_idx_global]

    def _find_bregma_or_vertex(self, mesh, kdtree, curvatures):
        """Encontra o Bregma ou Vertex (ponto mais superior na linha média).
           Heurística: Ponto mais alto (max Z) próximo à linha média sagital (X=0).
        """
        # 1. Encontrar ponto mais alto (max Z)
        idx_max_z, _ = self._find_extreme_vertex(mesh, [0, 0, 1])

        # 2. Refinar: Considerar apenas pontos próximos à linha média
        bounds = mesh.bounds
        top_midline_indices = np.where((np.abs(mesh.vertices[:, 0]) < (bounds[1, 0] - bounds[0, 0]) * 0.1) & # Próximo à linha média X=0
                                       (mesh.vertices[:, 2] > bounds[0, 2] + (bounds[1, 2] - bounds[0, 2]) * 0.8))[0] # Na parte superior

        if len(top_midline_indices) == 0:
            logging.warning("Não foi possível encontrar vértices na linha média superior para Bregma/Vertex. Usando max Z geral.")
            return idx_max_z, mesh.vertices[idx_max_z]

        # Dentre os candidatos, pegar o mais alto
        top_vertices = mesh.vertices[top_midline_indices]
        bregma_idx_local = np.argmax(top_vertices[:, 2])
        bregma_idx_global = top_midline_indices[bregma_idx_local]

        logging.info(f"Bregma/Vertex detectado no índice: {bregma_idx_global}")
        return bregma_idx_global, mesh.vertices[bregma_idx_global]

    def _find_opisthocranion(self, mesh, kdtree, curvatures):
        """Encontra o Opisthocranion (ponto mais posterior na linha média).
           Heurística: Ponto mais posterior (min Y) próximo à linha média sagital (X=0).
        """
        # 1. Encontrar ponto mais posterior (min Y)
        idx_min_y, _ = self._find_extreme_vertex(mesh, [0, -1, 0])

        # 2. Refinar: Considerar apenas pontos próximos à linha média na parte posterior
        bounds = mesh.bounds
        rear_midline_indices = np.where((np.abs(mesh.vertices[:, 0]) < (bounds[1, 0] - bounds[0, 0]) * 0.1) & # Próximo à linha média X=0
                                        (mesh.vertices[:, 1] < bounds[1, 1] - (bounds[1, 1] - bounds[0, 1]) * 0.8))[0] # Na parte posterior

        if len(rear_midline_indices) == 0:
            logging.warning("Não foi possível encontrar vértices na linha média posterior para Opisthocranion. Usando min Y geral.")
            return idx_min_y, mesh.vertices[idx_min_y]

        # Dentre os candidatos, pegar o mais posterior
        rear_vertices = mesh.vertices[rear_midline_indices]
        opis_idx_local = np.argmin(rear_vertices[:, 1])
        opis_idx_global = rear_midline_indices[opis_idx_local]

        logging.info(f"Opisthocranion detectado no índice: {opis_idx_global}")
        return opis_idx_global, mesh.vertices[opis_idx_global]

    def _find_euryon_left(self, mesh, kdtree, curvatures):
        """Encontra o Euryon Esquerdo (ponto mais lateral esquerdo).
           Heurística: Ponto mais à esquerda (min X).
        """
        idx, point = self._find_extreme_vertex(mesh, [-1, 0, 0])
        logging.info(f"Euryon Esquerdo detectado no índice: {idx}")
        return idx, point

    def _find_euryon_right(self, mesh, kdtree, curvatures):
        """Encontra o Euryon Direito (ponto mais lateral direito).
           Heurística: Ponto mais à direita (max X).
        """
        idx, point = self._find_extreme_vertex(mesh, [1, 0, 0])
        logging.info(f"Euryon Direito detectado no índice: {idx}")
        return idx, point

    def _find_inion(self, mesh, kdtree, curvatures):
        """Encontra o Inion (protuberância occipital externa).
           Heurística: Ponto de alta curvatura convexa na região occipital inferior,
           próximo à linha média e ao Opisthocranion.
        """
        # 1. Definir ROI na região occipital inferior e linha média
        bounds = mesh.bounds
        center_z = mesh.centroid[2]
        opis_idx, _ = self._find_opisthocranion(mesh, kdtree, curvatures) # Usar como referência
        if opis_idx is None: return None, None # Precisa do Opisthocranion

        roi_indices = np.where((np.abs(mesh.vertices[:, 0]) < (bounds[1, 0] - bounds[0, 0]) * 0.15) & # Próximo linha média
                               (mesh.vertices[:, 1] < mesh.vertices[opis_idx, 1] + (bounds[1, 1]-bounds[0, 1])*0.1) & # Próximo ou um pouco à frente do Opisthocranion
                               (mesh.vertices[:, 2] < mesh.vertices[opis_idx, 2]) & # Abaixo do Opisthocranion
                               (mesh.vertices[:, 2] > bounds[0, 2] + (bounds[1, 2] - bounds[0, 2]) * 0.1))[0] # Acima da base

        if len(roi_indices) == 0:
            logging.warning("Não foi possível encontrar vértices na ROI do Inion.")
            # Alternativa: Ponto de maior curvatura próximo ao Opisthocranion?
            if opis_idx is not None:
                distances, indices = kdtree.query(mesh.vertices[opis_idx], k=100)
                roi_indices = indices
            else:
                return None, None

        # 2. Encontrar o ponto de maior curvatura (mais convexo) dentro da ROI
        roi_curvatures = curvatures[roi_indices]
        # Queremos curvatura convexa (positiva se usando Gaussiana, ou max k1+k2)
        # Como usamos Gaussiana (proxy), procuramos o máximo valor.
        inion_idx_local = np.argmax(roi_curvatures)
        inion_idx_global = roi_indices[inion_idx_local]

        logging.info(f"Inion detectado no índice: {inion_idx_global}")
        return inion_idx_global, mesh.vertices[inion_idx_global]


    def detect(self, mesh):
        """Detecta todos os landmarks definidos na malha fornecida.

        Args:
            mesh (trimesh.Trimesh): Malha pré-processada (idealmente simplificada).

        Returns:
            dict: Dicionário mapeando nomes de landmarks para suas coordenadas [x, y, z],
                  ou None se a detecção falhar para um landmark específico.
                  Formato: {"Glabela": [x,y,z], "Nasion": None, ...}
        """
        if not isinstance(mesh, trimesh.Trimesh):
            logging.error("Input para detecção não é um objeto Trimesh válido.")
            return None

        logging.info(f"Iniciando detecção geométrica de landmarks em malha com {len(mesh.vertices)} vértices.")
        landmarks_found = {}

        # 1. Pré-computar informações úteis
        try:
            kdtree = KDTree(mesh.vertices)
            logging.info("KDTree construída para consultas espaciais.")
        except Exception as e:
            logging.error(f"Erro ao construir KDTree: {e}. Algumas detecções podem falhar.")
            kdtree = None

        # Calcular curvatura (usando um raio razoável, ajustar conforme necessário)
        # O raio ideal pode depender da escala e resolução da malha.
        avg_edge = np.mean(mesh.edges_unique_length)
        curvature_radius = avg_edge * 5 # Exemplo: 5 vezes o comprimento médio da aresta
        curvatures = self._calculate_curvature(mesh, radius=curvature_radius)

        # 2. Iterar sobre os landmarks definidos e chamar a função de detecção apropriada
        for landmark_name in LANDMARK_NAMES:
            detection_func = self.detection_functions.get(landmark_name)
            if detection_func:
                try:
                    index, point = detection_func(mesh, kdtree, curvatures)
                    if point is not None:
                        landmarks_found[landmark_name] = point.tolist() # Armazenar como lista
                    else:
                        landmarks_found[landmark_name] = None
                        logging.warning(f"Falha ao detectar {landmark_name}.")
                except Exception as e:
                    landmarks_found[landmark_name] = None
                    logging.error(f"Erro ao detectar {landmark_name}: {e}", exc_info=True)
            else:
                landmarks_found[landmark_name] = None
                logging.warning(f"Função de detecção não implementada para: {landmark_name}")

        logging.info("Detecção geométrica de landmarks concluída.")
        return landmarks_found

# Exemplo de uso (requer mesh_processor e um arquivo STL dummy)
if __name__ == \'__main__\':
    from .mesh_processor import MeshProcessor # Import relativo

    # Criar diretórios e um arquivo STL dummy se não existirem para teste
    if not os.path.exists("data/skulls"): os.makedirs("data/skulls")
    dummy_stl_path = "data/skulls/dummy_skull.stl"
    if not os.path.exists(dummy_stl_path):
        logging.info(f"Criando arquivo STL dummy em {dummy_stl_path}")
        mesh_dummy = trimesh.primitives.Sphere(radius=50) # Usar esfera para ter curvatura
        mesh_dummy.vertices += [0, 0, 50] # Mover para cima
        mesh_dummy.export(dummy_stl_path)

    # Carregar e simplificar a malha
    processor = MeshProcessor(data_dir="./data/skulls", cache_dir="./data/cache")
    skull_mesh = processor.load_skull("dummy_skull.stl")

    if skull_mesh:
        # Simplificar um pouco para acelerar, mas manter detalhes suficientes
        simplified_skull = processor.simplify(skull_mesh, target_faces=2000, original_filename="dummy_skull.stl")

        if simplified_skull:
            logging.info("--- Iniciando Detecção Geométrica na Malha Dummy Simplificada ---")
            detector = GeometricDetector()
            detected_landmarks = detector.detect(simplified_skull)

            print("\n--- Landmarks Detectados (Geométrico) ---")
            if detected_landmarks:
                for name, coords in detected_landmarks.items():
                    if coords:
                        print(f"  {name}: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]")
                    else:
                        print(f"  {name}: Não detectado")
            else:
                print("Falha geral na detecção.")
        else:
            logging.error("Falha ao simplificar a malha dummy para detecção.")
    else:
        logging.error("Falha ao carregar a malha dummy.")

