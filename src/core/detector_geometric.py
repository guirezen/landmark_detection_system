# -*- coding: utf-8 -*-
"""Módulo para detecção de landmarks usando métodos geométricos simples."""

import numpy as np
import trimesh
from scipy.spatial import KDTree
import logging

from .landmarks import LANDMARK_NAMES

class GeometricDetector:
    """Detecta landmarks em uma malha 3D usando propriedades geométricas."""

    def __init__(self):
        # Mapear cada landmark a sua função específica de detecção
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
        logging.debug("GeometricDetector inicializado com funções de detecção para cada landmark.")

    def _calculate_curvature(self, mesh, radius=None):
        """Calcula a curvatura Gaussiana para cada vértice da malha."""
        try:
            # Se raio não fornecido, usar 3x o comprimento médio das arestas
            if radius is None:
                avg_edge = np.mean(mesh.edges_unique_length)
                radius = avg_edge * 3.0
            curvatures = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius=radius)
            # Substituir NaN/inf por valores finitos (0.0, 1.0, -1.0)
            curvatures = np.nan_to_num(curvatures, nan=0.0, posinf=1.0, neginf=-1.0)
            return curvatures
        except Exception as e:
            logging.error(f"Erro ao calcular curvatura: {e}")
            # Retorna zeros se falhar cálculo de curvatura
            return np.zeros(len(mesh.vertices))

    def _find_extreme_vertex(self, mesh, direction_vector, region_mask=None):
        """Encontra o índice e coordenada do vértice mais extremo em uma direção dada."""
        vertices = mesh.vertices
        if region_mask is not None:
            vertices = vertices[region_mask]
            indices = np.where(region_mask)[0]
        else:
            indices = np.arange(len(vertices))
        # Produto interno dos vetores posição com o vetor de direção para achar extremo
        dots = vertices @ np.array(direction_vector)
        extreme_idx_local = np.argmax(dots)
        extreme_idx_global = indices[extreme_idx_local]
        return extreme_idx_global, mesh.vertices[extreme_idx_global]

    def _get_midline_mask(self, mesh, tolerance_ratio=0.05):
        """Gera uma máscara booleana para vértices próximos ao plano mediano (x ~ 0)."""
        x_range = mesh.bounds[1, 0] - mesh.bounds[0, 0]
        tolerance = x_range * tolerance_ratio
        return np.abs(mesh.vertices[:, 0]) < tolerance

    def _find_glabela(self, mesh, kdtree, curvatures):
        """Encontra a Glabela: ponto frontal mais proeminente na parte superior do crânio."""
        try:
            center_z = mesh.centroid[2]
            bounds = mesh.bounds
            # Região frontal superior aproximada
            frontal_mask = (
                (mesh.vertices[:, 1] > bounds[0, 1] + 0.7 * (bounds[1, 1] - bounds[0, 1])) &  # parte frontal (Y alto)
                (mesh.vertices[:, 2] > center_z) &  # metade superior em Z
                (np.abs(mesh.vertices[:, 0]) < 0.3 * (bounds[1, 0] - bounds[0, 0]))  # próximo do plano mediano
            )
            if not np.any(frontal_mask):
                # Fallback: simplesmente o vértice mais à frente (maior Y)
                idx, point = self._find_extreme_vertex(mesh, [0, 1, 0])
                logging.warning("Glabela: usando vértice mais frontal como fallback")
                return idx, point
            # Dentro da região frontal, pega vértice mais projetado para frente (eixo Y)
            idx, point = self._find_extreme_vertex(mesh, [0, 1, 0], frontal_mask)
            return idx, point
        except Exception as e:
            logging.error(f"Erro ao detectar Glabela: {e}")
            return None, None

    def _find_nasion(self, mesh, kdtree, curvatures):
        """Encontra o Nasion: ponto de depressão nasal (entrequeda das sobrancelhas)."""
        try:
            bounds = mesh.bounds
            center_z = mesh.centroid[2]
            # ROI: região frontal mediana, altura média
            roi_mask = (
                self._get_midline_mask(mesh, 0.1) &
                (mesh.vertices[:, 1] > bounds[0, 1] + 0.6 * (bounds[1, 1] - bounds[0, 1])) &
                (mesh.vertices[:, 2] < center_z + 0.4 * (bounds[1, 2] - center_z)) &
                (mesh.vertices[:, 2] > bounds[0, 2] + 0.3 * (bounds[1, 2] - bounds[0, 2]))
            )
            if not np.any(roi_mask):
                logging.warning("Nasion: ROI vazia, usando fallback de curvatura mínima na face frontal")
                frontal_mask = mesh.vertices[:, 1] > bounds[0, 1] + 0.7 * (bounds[1, 1] - bounds[0, 1])
                if np.any(frontal_mask):
                    roi_curv = curvatures[frontal_mask]
                    roi_indices = np.where(frontal_mask)[0]
                    min_idx_local = np.argmin(roi_curv)
                    min_idx = roi_indices[min_idx_local]
                    return min_idx, mesh.vertices[min_idx]
                else:
                    return None, None
            # Dentro da ROI, escolhe vértice de menor coordenada Z (mais baixo na testa)
            roi_indices = np.where(roi_mask)[0]
            min_z_idx_local = np.argmin(mesh.vertices[roi_indices, 2])
            min_idx = roi_indices[min_z_idx_local]
            return min_idx, mesh.vertices[min_idx]
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
        """Detecta todos os landmarks definidos em uma malha pré-processada."""
        if not isinstance(mesh, trimesh.Trimesh):
            logging.error("Entrada inválida para detecção geométrica (esperado Trimesh).")
            return None
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            logging.error("Malha vazia fornecida ao detector geométrico.")
            return None
        logging.info(f"Iniciando detecção geométrica - malha com {len(mesh.vertices)} vértices.")
        landmarks_found = {}
        # Pré-calcular KDTree e curvatura para uso nos algoritmos
        kdtree = None
        try:
            kdtree = KDTree(mesh.vertices)
        except Exception as e:
            logging.warning(f"Não foi possível construir KDTree: {e}")
        curvatures = self._calculate_curvature(mesh)
        # Detectar cada landmark usando sua função respectiva
        for name in LANDMARK_NAMES:
            func = self.detection_functions.get(name)
            if func:
                try:
                    idx, point = func(mesh, kdtree, curvatures)
                    if point is not None:
                        landmarks_found[name] = point.tolist()
                        logging.debug(f"{name} detectado no vértice {idx}")
                    else:
                        landmarks_found[name] = None
                        logging.warning(f"{name} não pôde ser detectado.")
                except Exception as e:
                    landmarks_found[name] = None
                    logging.error(f"Erro ao detectar {name}: {e}")
            else:
                # Caso improvável: função não implementada
                landmarks_found[name] = None
                logging.error(f"Landmark '{name}' sem função de detecção implementada!")
        # Log de resumo
        detected = sum(1 for coords in landmarks_found.values() if coords is not None)
        logging.info(f"Detecção geométrica finalizada: {detected}/{len(LANDMARK_NAMES)} detectados.")
        return landmarks_found
    