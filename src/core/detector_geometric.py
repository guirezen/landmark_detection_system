# -*- coding: utf-8 -*-
"""Módulo para detecção de landmarks usando métodos geométricos simples."""

import numpy as np
import trimesh
from scipy.spatial import KDTree
import logging

from .landmarks import LANDMARK_NAMES

class GeometricDetector:
    """Detecta landmarks em uma malha 3D usando propriedades geométricas."""

    def __init__(self, landmark_definitions=None):
        """Inicializa o detector geométrico.

        Args:
            landmark_definitions (dict, optional): Dicionário com definições ou
                                                  heurísticas específicas para cada landmark.
        """
        self.landmark_definitions = landmark_definitions if landmark_definitions else {}
        
        # Mapeamento de nomes para funções de detecção específicas
        self.detection_functions = {
            "Glabela": self._find_glabela,
            "Nasion": self._find_nasion,
            "Bregma": self._find_bregma,
            "Opisthocranion": self._find_opisthocranion,
            "Euryon_Esquerdo": self._find_euryon_left,
            "Euryon_Direito": self._find_euryon_right,
            "Vertex": self._find_vertex,
            "Inion": self._find_inion,
        }
        logging.info("Detector Geométrico inicializado.")

    def _calculate_curvature(self, mesh, radius=None):
        """Calcula a curvatura Gaussiana para cada vértice.

        Args:
            mesh (trimesh.Trimesh): Malha de entrada.
            radius (float, optional): Raio para cálculo de curvatura local.

        Returns:
            np.ndarray: Array com a curvatura Gaussiana para cada vértice.
        """
        try:
            if radius is None:
                # Calcular raio baseado no tamanho médio das arestas
                avg_edge = np.mean(mesh.edges_unique_length)
                radius = avg_edge * 3.0  # Multiplicador conservador
            
            # Usar curvatura Gaussiana discreta
            curvatures = trimesh.curvature.discrete_gaussian_curvature_measure(
                mesh, mesh.vertices, radius=radius
            )
            
            # Tratar valores NaN ou infinitos
            curvatures = np.nan_to_num(curvatures, nan=0.0, posinf=1.0, neginf=-1.0)
            
            logging.info(f"Curvatura Gaussiana calculada para {len(mesh.vertices)} vértices "
                        f"(raio={radius:.3f})")
            return curvatures
            
        except Exception as e:
            logging.error(f"Erro ao calcular curvatura: {e}")
            # Retorna array de zeros se o cálculo falhar
            return np.zeros(len(mesh.vertices))

    def _find_extreme_vertex(self, mesh, direction_vector, region_mask=None):
        """Encontra o vértice mais extremo em uma determinada direção.
        
        Args:
            mesh: Malha 3D
            direction_vector: Vetor direção para busca
            region_mask: Máscara opcional para limitar região de busca
            
        Returns:
            tuple: (índice, coordenadas) do vértice extremo
        """
        vertices = mesh.vertices
        if region_mask is not None:
            vertices = vertices[region_mask]
            indices = np.where(region_mask)[0]
        else:
            indices = np.arange(len(vertices))
        
        dots = vertices @ np.array(direction_vector)
        extreme_idx_local = np.argmax(dots)
        extreme_idx_global = indices[extreme_idx_local]
        
        return extreme_idx_global, mesh.vertices[extreme_idx_global]

    def _get_midline_mask(self, mesh, tolerance_ratio=0.05):
        """Cria máscara para vértices próximos à linha média (X ≈ 0).
        
        Args:
            mesh: Malha 3D
            tolerance_ratio: Tolerância como fração da largura total
            
        Returns:
            np.ndarray: Máscara booleana para vértices na linha média
        """
        x_range = mesh.bounds[1, 0] - mesh.bounds[0, 0]
        tolerance = x_range * tolerance_ratio
        return np.abs(mesh.vertices[:, 0]) < tolerance

    def _find_glabela(self, mesh, kdtree, curvatures):
        """Encontra a Glabela (ponto mais proeminente frontal)."""
        try:
            # Região frontal superior
            center_z = mesh.centroid[2]
            bounds = mesh.bounds
            
            # Máscara para região frontal superior
            frontal_mask = (
                (mesh.vertices[:, 1] > bounds[0, 1] + (bounds[1, 1] - bounds[0, 1]) * 0.7) &  # Parte frontal
                (mesh.vertices[:, 2] > center_z) &  # Metade superior
                (np.abs(mesh.vertices[:, 0]) < (bounds[1, 0] - bounds[0, 0]) * 0.3)  # Próximo ao centro
            )
            
            if not np.any(frontal_mask):
                # Fallback: ponto mais frontal geral
                idx, point = self._find_extreme_vertex(mesh, [0, 1, 0])
                logging.warning("Usando ponto mais frontal como Glabela (fallback)")
                return idx, point
            
            # Entre os candidatos frontais, pegar o mais à frente
            idx, point = self._find_extreme_vertex(mesh, [0, 1, 0], frontal_mask)
            logging.info(f"Glabela detectada no índice: {idx}")
            return idx, point
            
        except Exception as e:
            logging.error(f"Erro ao detectar Glabela: {e}")
            return None, None

    def _find_nasion(self, mesh, kdtree, curvatures):
        """Encontra o Nasion (depressão nasal)."""
        try:
            bounds = mesh.bounds
            center_z = mesh.centroid[2]
            
            # Região de interesse: linha média frontal, altura média
            roi_mask = (
                self._get_midline_mask(mesh, 0.1) &  # Linha média
                (mesh.vertices[:, 1] > bounds[0, 1] + (bounds[1, 1] - bounds[0, 1]) * 0.6) &  # Frontal
                (mesh.vertices[:, 2] < center_z + (bounds[1, 2] - center_z) * 0.4) &  # Altura média
                (mesh.vertices[:, 2] > bounds[0, 2] + (bounds[1, 2] - bounds[0, 2]) * 0.3)   # Acima da base
            )
            
            if not np.any(roi_mask):
                logging.warning("ROI do Nasion vazia, usando heurística alternativa")
                # Alternativa: ponto de menor curvatura na região frontal
                frontal_mask = mesh.vertices[:, 1] > bounds[0, 1] + (bounds[1, 1] - bounds[0, 1]) * 0.7
                if np.any(frontal_mask):
                    roi_curvatures = curvatures[frontal_mask]
                    roi_indices = np.where(frontal_mask)[0]
                    min_idx = np.argmin(roi_curvatures)
                    nasion_idx = roi_indices[min_idx]
                    return nasion_idx, mesh.vertices[nasion_idx]
                else:
                    return None, None
            
            # Encontrar ponto de menor curvatura (mais côncavo) na ROI
            roi_curvatures = curvatures[roi_mask]
            roi_indices = np.where(roi_mask)[0]
            nasion_idx_local = np.argmin(roi_curvatures)
            nasion_idx_global = roi_indices[nasion_idx_local]
            
            logging.info(f"Nasion detectado no índice: {nasion_idx_global}")
            return nasion_idx_global, mesh.vertices[nasion_idx_global]
            
        except Exception as e:
            logging.error(f"Erro ao detectar Nasion: {e}")
            return None, None

    def _find_bregma(self, mesh, kdtree, curvatures):
        """Encontra o Bregma (junção das suturas no topo)."""
        try:
            bounds = mesh.bounds
            
            # Região superior na linha média
            top_midline_mask = (
                self._get_midline_mask(mesh, 0.1) &  # Linha média
                (mesh.vertices[:, 2] > bounds[0, 2] + (bounds[1, 2] - bounds[0, 2]) * 0.8)  # Topo
            )
            
            if not np.any(top_midline_mask):
                # Fallback: ponto mais alto geral
                idx, point = self._find_extreme_vertex(mesh, [0, 0, 1])
                logging.warning("Usando ponto mais alto como Bregma (fallback)")
                return idx, point
            
            # Ponto mais alto na linha média superior
            idx, point = self._find_extreme_vertex(mesh, [0, 0, 1], top_midline_mask)
            logging.info(f"Bregma detectado no índice: {idx}")
            return idx, point
            
        except Exception as e:
            logging.error(f"Erro ao detectar Bregma: {e}")
            return None, None

    def _find_vertex(self, mesh, kdtree, curvatures):
        """Encontra o Vertex (ponto mais superior)."""
        try:
            # Vertex é similar ao Bregma, mas pode estar ligeiramente deslocado
            idx, point = self._find_extreme_vertex(mesh, [0, 0, 1])
            logging.info(f"Vertex detectado no índice: {idx}")
            return idx, point
        except Exception as e:
            logging.error(f"Erro ao detectar Vertex: {e}")
            return None, None

    def _find_opisthocranion(self, mesh, kdtree, curvatures):
        """Encontra o Opisthocranion (ponto mais posterior)."""
        try:
            bounds = mesh.bounds
            
            # Região posterior na linha média
            rear_midline_mask = (
                self._get_midline_mask(mesh, 0.1) &  # Linha média
                (mesh.vertices[:, 1] < bounds[0, 1] + (bounds[1, 1] - bounds[0, 1]) * 0.3)  # Posterior
            )
            
            if not np.any(rear_midline_mask):
                # Fallback: ponto mais posterior geral
                idx, point = self._find_extreme_vertex(mesh, [0, -1, 0])
                logging.warning("Usando ponto mais posterior como Opisthocranion (fallback)")
                return idx, point
            
            # Ponto mais posterior na linha média
            idx, point = self._find_extreme_vertex(mesh, [0, -1, 0], rear_midline_mask)
            logging.info(f"Opisthocranion detectado no índice: {idx}")
            return idx, point
            
        except Exception as e:
            logging.error(f"Erro ao detectar Opisthocranion: {e}")
            return None, None

    def _find_euryon_left(self, mesh, kdtree, curvatures):
        """Encontra o Euryon Esquerdo (ponto mais lateral esquerdo)."""
        try:
            idx, point = self._find_extreme_vertex(mesh, [-1, 0, 0])
            logging.info(f"Euryon Esquerdo detectado no índice: {idx}")
            return idx, point
        except Exception as e:
            logging.error(f"Erro ao detectar Euryon Esquerdo: {e}")
            return None, None

    def _find_euryon_right(self, mesh, kdtree, curvatures):
        """Encontra o Euryon Direito (ponto mais lateral direito)."""
        try:
            idx, point = self._find_extreme_vertex(mesh, [1, 0, 0])
            logging.info(f"Euryon Direito detectado no índice: {idx}")
            return idx, point
        except Exception as e:
            logging.error(f"Erro ao detectar Euryon Direito: {e}")
            return None, None

    def _find_inion(self, mesh, kdtree, curvatures):
        """Encontra o Inion (protuberância occipital)."""
        try:
            bounds = mesh.bounds
            center_z = mesh.centroid[2]
            
            # Região occipital inferior (baseada no Opisthocranion mas mais baixa)
            roi_mask = (
                self._get_midline_mask(mesh, 0.15) &  # Linha média (um pouco mais tolerante)
                (mesh.vertices[:, 1] < bounds[0, 1] + (bounds[1, 1] - bounds[0, 1]) * 0.4) &  # Posterior
                (mesh.vertices[:, 2] < center_z) &  # Metade inferior
                (mesh.vertices[:, 2] > bounds[0, 2] + (bounds[1, 2] - bounds[0, 2]) * 0.2)   # Acima da base
            )
            
            if not np.any(roi_mask):
                logging.warning("ROI do Inion vazia, usando detecção alternativa")
                # Alternativa: procurar na região posterior inferior
                posterior_mask = (
                    (mesh.vertices[:, 1] < bounds[0, 1] + (bounds[1, 1] - bounds[0, 1]) * 0.5) &
                    (mesh.vertices[:, 2] < center_z)
                )
                if np.any(posterior_mask):
                    roi_mask = posterior_mask
                else:
                    return None, None
            
            # Encontrar ponto de maior curvatura convexa na ROI
            roi_curvatures = curvatures[roi_mask]
            roi_indices = np.where(roi_mask)[0]
            
            # Para Inion, procuramos curvatura convexa (positiva)
            inion_idx_local = np.argmax(roi_curvatures)
            inion_idx_global = roi_indices[inion_idx_local]
            
            logging.info(f"Inion detectado no índice: {inion_idx_global}")
            return inion_idx_global, mesh.vertices[inion_idx_global]
            
        except Exception as e:
            logging.error(f"Erro ao detectar Inion: {e}")
            return None, None

    def detect(self, mesh):
        """Detecta todos os landmarks definidos na malha fornecida.

        Args:
            mesh (trimesh.Trimesh): Malha pré-processada.

        Returns:
            dict: Dicionário mapeando nomes de landmarks para suas coordenadas [x, y, z],
                  ou None se a detecção falhar.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            logging.error("Input para detecção não é um objeto Trimesh válido.")
            return None

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            logging.error("Malha vazia fornecida para detecção.")
            return None

        logging.info(f"Iniciando detecção geométrica em malha com {len(mesh.vertices)} vértices.")
        landmarks_found = {}

        # Pré-computar informações úteis
        try:
            kdtree = KDTree(mesh.vertices)
            logging.debug("KDTree construída para consultas espaciais.")
        except Exception as e:
            logging.error(f"Erro ao construir KDTree: {e}")
            kdtree = None

        # Calcular curvatura
        curvatures = self._calculate_curvature(mesh)

        # Detectar cada landmark
        for landmark_name in LANDMARK_NAMES:
            detection_func = self.detection_functions.get(landmark_name)
            if detection_func:
                try:
                    index, point = detection_func(mesh, kdtree, curvatures)
                    if point is not None:
                        landmarks_found[landmark_name] = point.tolist()
                        logging.debug(f"{landmark_name} detectado: {point}")
                    else:
                        landmarks_found[landmark_name] = None
                        logging.warning(f"Falha ao detectar {landmark_name}")
                        
                except Exception as e:
                    landmarks_found[landmark_name] = None
                    logging.error(f"Erro ao detectar {landmark_name}: {e}")
            else:
                landmarks_found[landmark_name] = None
                logging.warning(f"Função de detecção não implementada para: {landmark_name}")

        # Estatísticas finais
        detected_count = sum(1 for coords in landmarks_found.values() if coords is not None)
        total_count = len(landmarks_found)
        logging.info(f"Detecção geométrica concluída: {detected_count}/{total_count} landmarks detectados")

        return landmarks_found

# Exemplo de uso e teste
if __name__ == '__main__':
    import os
    import sys
    
    # Adicionar path para imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from mesh_processor import MeshProcessor
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Criar diretórios de teste
    os.makedirs("data/skulls", exist_ok=True)
    
    dummy_stl_path = "data/skulls/dummy_skull_geom.stl"
    if not os.path.exists(dummy_stl_path):
        logging.info(f"Criando arquivo STL dummy em {dummy_stl_path}")
        # Usar esfera com subdivisões para ter geometria interessante
        mesh_dummy = trimesh.primitives.Sphere(radius=50, subdivisions=3)
        mesh_dummy.vertices += [0, 0, 50]  # Deslocar para cima
        mesh_dummy.export(dummy_stl_path)

    # Testar detecção
    processor = MeshProcessor(data_dir="./data/skulls", cache_dir="./data/cache")
    skull_mesh = processor.load_skull("dummy_skull_geom.stl")

    if skull_mesh:
        # Simplificar para acelerar
        simplified_skull = processor.simplify(skull_mesh, target_faces=1000, 
                                            original_filename="dummy_skull_geom.stl")

        if simplified_skull:
            logging.info("=== Iniciando Detecção Geométrica ===")
            detector = GeometricDetector()
            detected_landmarks = detector.detect(simplified_skull)

            print("\n=== Landmarks Detectados (Geométrico) ===")
            if detected_landmarks:
                for name, coords in detected_landmarks.items():
                    if coords:
                        print(f"  {name}: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]")
                    else:
                        print(f"  {name}: Não detectado")
            else:
                print("Falha geral na detecção.")
        else:
            logging.error("Falha ao simplificar a malha dummy.")
    else:
        logging.error("Falha ao carregar a malha dummy.")