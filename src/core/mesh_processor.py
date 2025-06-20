# -*- coding: utf-8 -*-
"""Módulo otimizado para carregar e simplificar malhas 3D - VERSÃO PERFORMANCE."""

import trimesh
import numpy as np
import os
import hashlib
import pickle
import logging
from scipy.spatial import ConvexHull

class MeshProcessor:
    """Processa malhas 3D com otimizações para arquivos grandes."""

    def __init__(self, data_dir="./data/skulls", cache_dir="./data/cache"):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logging.info(f"Processador otimizado inicializado. Cache em: {self.cache_dir}")

    def _get_cache_filename(self, original_filename, params):
        """Gera nome de cache único."""
        hasher = hashlib.md5()
        hasher.update(original_filename.encode('utf-8'))
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        hasher.update(param_str.encode('utf-8'))
        return os.path.join(self.cache_dir, f"{hasher.hexdigest()}.pkl")

    def _save_to_cache(self, mesh, cache_filepath):
        """Salva malha no cache."""
        try:
            with open(cache_filepath, 'wb') as f:
                mesh_data = {
                    'vertices': mesh.vertices.copy(),
                    'faces': mesh.faces.copy(),
                    'metadata': {
                        'vertex_count': len(mesh.vertices),
                        'face_count': len(mesh.faces)
                    }
                }
                pickle.dump(mesh_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Malha salva no cache: {cache_filepath}")
        except Exception as e:
            logging.error(f"Erro ao salvar cache {cache_filepath}: {e}")

    def _load_from_cache(self, cache_filepath):
        """Carrega malha do cache."""
        if not os.path.exists(cache_filepath):
            return None
            
        try:
            with open(cache_filepath, 'rb') as f:
                mesh_data = pickle.load(f)
            
            if not all(key in mesh_data for key in ['vertices', 'faces']):
                logging.warning(f"Cache corrompido: {cache_filepath}")
                return None
            
            mesh = trimesh.Trimesh(
                vertices=mesh_data['vertices'], 
                faces=mesh_data['faces'],
                validate=False,  # OTIMIZAÇÃO: pular validação no cache
                process=False    # OTIMIZAÇÃO: pular processamento
            )
            logging.info(f"Malha carregada do cache: {cache_filepath}")
            return mesh
            
        except Exception as e:
            logging.error(f"Erro ao carregar cache {cache_filepath}: {e}")
            try:
                os.remove(cache_filepath)
            except OSError:
                pass
            return None

    def load_skull(self, filename, use_cache=True, aggressive_load=True):
        """Carrega modelo STL com otimizações para arquivos grandes."""
        filepath = os.path.join(self.data_dir, filename)
        cache_params = {'operation': 'load', 'version': '2.0_optimized'}
        cache_filepath = self._get_cache_filename(filename, cache_params)

        # Cache primeiro
        if use_cache:
            mesh = self._load_from_cache(cache_filepath)
            if mesh is not None:
                return mesh

        if not os.path.exists(filepath):
            logging.error(f"Arquivo não encontrado: {filepath}")
            return None

        try:
            # OTIMIZAÇÃO: Carregar sem validação para arquivos grandes
            if aggressive_load:
                mesh = trimesh.load_mesh(filepath, force='mesh', validate=False, process=False)
            else:
                mesh = trimesh.load_mesh(filepath, force='mesh')
            
            logging.info(f"STL carregado: {filepath} "
                        f"(Vértices: {len(mesh.vertices)}, Faces: {len(mesh.faces)})")

            # Lidar com Scene
            if isinstance(mesh, trimesh.Scene):
                logging.warning(f"Arquivo {filename} é Scene, concatenando...")
                try:
                    mesh = mesh.dump(concatenate=True)
                except:
                    geometries = list(mesh.geometry.values())
                    if geometries:
                        mesh = geometries[0]
                    else:
                        logging.error(f"Falha ao extrair geometria de {filename}")
                        return None

            if not isinstance(mesh, trimesh.Trimesh):
                logging.error(f"Objeto não é Trimesh válido: {filename}")
                return None

            # Pré-processamento MÍNIMO para arquivos grandes
            if len(mesh.faces) > 50000:  # Arquivo grande
                logging.info("Arquivo grande detectado - pré-processamento mínimo")
                try:
                    # Apenas remover faces degeneradas críticas
                    mesh.remove_degenerate_faces()
                except:
                    logging.warning("Falha no pré-processamento mínimo")
            else:
                # Pré-processamento completo para arquivos pequenos
                try:
                    mesh.fix_normals()
                    if not mesh.is_watertight:
                        mesh.fill_holes()
                    mesh.remove_duplicate_faces()
                    mesh.remove_degenerate_faces()
                except Exception as e:
                    logging.warning(f"Erro no pré-processamento: {e}")

            # Salvar no cache
            if use_cache:
                self._save_to_cache(mesh, cache_filepath)

            return mesh

        except Exception as e:
            logging.error(f"Erro ao carregar {filepath}: {e}")
            return None

    def _ultra_fast_simplification(self, mesh, target_faces):
        """Simplificação ultra-rápida para casos extremos."""
        try:
            current_faces = len(mesh.faces)
            
            # Se já é pequeno, não simplificar
            if current_faces <= target_faces:
                return mesh
            
            # Para reduções muito drásticas, fazer em etapas
            if target_faces < current_faces * 0.1:  # Redução > 90%
                logging.info("Redução drástica detectada - simplificação em etapas")
                
                # Etapa 1: Reduzir para 20% do original
                intermediate_target = int(current_faces * 0.2)
                try:
                    intermediate = self._try_fast_simplification(mesh, intermediate_target)
                    if intermediate and len(intermediate.faces) < current_faces:
                        mesh = intermediate
                except:
                    pass
                
                # Etapa 2: Reduzir para o alvo final
                try:
                    final = self._try_fast_simplification(mesh, target_faces)
                    if final and len(final.faces) > 0:
                        return final
                except:
                    pass
            
            # Tentativa direta
            return self._try_fast_simplification(mesh, target_faces)
            
        except Exception as e:
            logging.error(f"Erro na simplificação ultra-rápida: {e}")
            return None

    def _try_fast_simplification(self, mesh, target_faces):
        """Tenta simplificação rápida com fast_simplification."""
        try:
            # Usar fast_simplification diretamente se disponível
            import fast_simplification
            
            # Preparar arrays
            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.uint32)
            
            # Calcular ratio
            ratio = target_faces / len(mesh.faces)
            ratio = max(0.01, min(0.99, ratio))  # Limitar entre 1% e 99%
            
            # Simplificar
            simplified_vertices, simplified_faces = fast_simplification.simplify(
                vertices, faces, target_count=target_faces
            )
            
            # Criar nova malha
            simplified_mesh = trimesh.Trimesh(
                vertices=simplified_vertices, 
                faces=simplified_faces,
                validate=False,
                process=False
            )
            
            logging.info(f"Fast simplification: {len(mesh.faces)} → {len(simplified_faces)} faces")
            return simplified_mesh
            
        except ImportError:
            # Fallback para método Trimesh
            try:
                ratio = 1.0 - (target_faces / len(mesh.faces))
                ratio = max(0.01, min(0.99, ratio))
                return mesh.simplify_quadric_decimation(ratio)
            except:
                return None
        except Exception as e:
            logging.warning(f"Erro na simplificação rápida: {e}")
            return None

    def _extreme_simplification_fallback(self, mesh, target_faces):
        """Fallback extremo usando decimação por clustering."""
        try:
            from sklearn.cluster import KMeans
            
            n_vertices = len(mesh.vertices)
            n_clusters = min(target_faces * 2, n_vertices // 2)
            
            if n_clusters < 10:
                return None
            
            # Clustering de vértices
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            clusters = kmeans.fit_predict(mesh.vertices)
            
            # Usar centroides como novos vértices
            new_vertices = kmeans.cluster_centers_
            
            # Triangulação 2D projetada
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            vertices_2d = pca.fit_transform(new_vertices)
            
            from scipy.spatial import Delaunay
            tri = Delaunay(vertices_2d)
            
            # Filtrar faces válidas
            valid_faces = []
            for face in tri.simplices:
                if len(np.unique(face)) == 3:
                    valid_faces.append(face)
            
            if len(valid_faces) == 0:
                return None
            
            simplified_mesh = trimesh.Trimesh(
                vertices=new_vertices, 
                faces=np.array(valid_faces),
                validate=False,
                process=False
            )
            
            logging.info(f"Clustering fallback: {len(mesh.faces)} → {len(simplified_mesh.faces)} faces")
            return simplified_mesh
            
        except Exception as e:
            logging.warning(f"Erro no fallback extremo: {e}")
            return None

    def simplify(self, mesh, target_faces, use_cache=True, original_filename="unknown"):
        """Simplifica malha com otimizações para performance."""
        if not isinstance(mesh, trimesh.Trimesh):
            logging.error("Input não é Trimesh válido")
            return None

        current_faces = len(mesh.faces)
        
        if current_faces <= target_faces:
            logging.info(f"Malha já tem {current_faces} faces (<= {target_faces})")
            return mesh

        # Cache
        cache_params = {
            'operation': 'simplify_optimized', 
            'target_faces': target_faces,
            'original_faces': current_faces,
            'version': '2.0'
        }
        cache_filepath = self._get_cache_filename(original_filename, cache_params)

        if use_cache:
            cached_mesh = self._load_from_cache(cache_filepath)
            if cached_mesh is not None:
                return cached_mesh

        logging.info(f"Simplificando {current_faces} → {target_faces} faces...")
        
        # Estratégias em ordem de preferência
        strategies = [
            ("Ultra Fast Simplification", self._ultra_fast_simplification),
            ("Extreme Fallback", self._extreme_simplification_fallback)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logging.info(f"Tentando: {strategy_name}")
                simplified_mesh = strategy_func(mesh, target_faces)
                
                if simplified_mesh and len(simplified_mesh.faces) > 0:
                    actual_faces = len(simplified_mesh.faces)
                    reduction = (current_faces - actual_faces) / current_faces * 100
                    logging.info(f"✅ {strategy_name}: {actual_faces} faces ({reduction:.1f}% redução)")
                    
                    # Salvar cache
                    if use_cache:
                        self._save_to_cache(simplified_mesh, cache_filepath)
                    
                    return simplified_mesh
                else:
                    logging.warning(f"❌ {strategy_name}: resultado inválido")
                    
            except Exception as e:
                logging.warning(f"❌ {strategy_name}: {e}")
                continue
        
        # Se tudo falhou, retornar original com aviso
        logging.warning("⚠️ Todas as simplificações falharam - usando malha original")
        return mesh

    def get_mesh_stats(self, mesh):
        """Retorna estatísticas da malha."""
        if not isinstance(mesh, trimesh.Trimesh):
            return None
            
        return {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'edges': len(mesh.edges) if hasattr(mesh, 'edges') else 0,
            'volume': mesh.volume if mesh.is_volume else None,
            'surface_area': mesh.area,
            'bounds': mesh.bounds.tolist(),
            'extents': mesh.extents.tolist(),
            'centroid': mesh.centroid.tolist()
        }
