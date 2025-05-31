# -*- coding: utf-8 -*-
"""Módulo para carregar, pré-processar e simplificar malhas 3D de crânios - VERSÃO ROBUSTA."""

import trimesh
import numpy as np
import os
import hashlib
import pickle
import logging
from scipy.spatial import ConvexHull

class MeshProcessor:
    """Processa malhas 3D, incluindo carregamento, simplificação e cache."""

    def __init__(self, data_dir="./data/skulls", cache_dir="./data/cache"):
        """Inicializa o processador de malhas.

        Args:
            data_dir (str): Diretório onde os arquivos STL originais estão localizados.
            cache_dir (str): Diretório para armazenar/recuperar malhas processadas.
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logging.info(f"Processador de Malha inicializado. Cache em: {self.cache_dir}")

    def _get_cache_filename(self, original_filename, params):
        """Gera um nome de arquivo de cache único baseado no nome original e parâmetros."""
        hasher = hashlib.md5()
        hasher.update(original_filename.encode('utf-8'))
        # Inclui parâmetros relevantes no hash para diferenciar caches
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        hasher.update(param_str.encode('utf-8'))
        return os.path.join(self.cache_dir, f"{hasher.hexdigest()}.pkl")

    def _save_to_cache(self, mesh, cache_filepath):
        """Salva a malha processada no cache."""
        try:
            with open(cache_filepath, 'wb') as f:
                # Salvar dados essenciais da malha (vértices, faces)
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
            logging.error(f"Erro ao salvar malha no cache {cache_filepath}: {e}")

    def _load_from_cache(self, cache_filepath):
        """Carrega a malha do cache, se existir."""
        if not os.path.exists(cache_filepath):
            return None
            
        try:
            with open(cache_filepath, 'rb') as f:
                mesh_data = pickle.load(f)
            
            # Verificar integridade dos dados
            if not all(key in mesh_data for key in ['vertices', 'faces']):
                logging.warning(f"Cache corrompido (chaves faltando): {cache_filepath}")
                return None
            
            # Recriar o objeto Trimesh
            mesh = trimesh.Trimesh(
                vertices=mesh_data['vertices'], 
                faces=mesh_data['faces'],
                validate=True
            )
            logging.info(f"Malha carregada do cache: {cache_filepath}")
            return mesh
            
        except Exception as e:
            logging.error(f"Erro ao carregar malha do cache {cache_filepath}: {e}")
            # Se o cache estiver corrompido, remove para evitar erros futuros
            try:
                os.remove(cache_filepath)
                logging.warning(f"Arquivo de cache corrompido removido: {cache_filepath}")
            except OSError:
                pass
            return None

    def load_skull(self, filename, use_cache=True):
        """Carrega um modelo de crânio STL.

        Args:
            filename (str): Nome do arquivo STL (sem o caminho base).
            use_cache (bool): Se deve usar o sistema de cache para o carregamento inicial.

        Returns:
            trimesh.Trimesh or None: Objeto da malha carregada ou None se falhar.
        """
        filepath = os.path.join(self.data_dir, filename)
        cache_params = {'operation': 'load', 'version': '1.0'}
        cache_filepath = self._get_cache_filename(filename, cache_params)

        # Tentar carregar do cache primeiro
        if use_cache:
            mesh = self._load_from_cache(cache_filepath)
            if mesh is not None:
                return mesh

        # Se não está no cache ou cache desabilitado, carregar do arquivo
        if not os.path.exists(filepath):
            logging.error(f"Arquivo não encontrado: {filepath}")
            return None

        try:
            # Carregar usando trimesh
            mesh = trimesh.load_mesh(filepath, force='mesh')
            logging.info(f"Arquivo STL carregado: {filepath} "
                        f"(Vértices: {len(mesh.vertices)}, Faces: {len(mesh.faces)})")

            # Verificar se foi carregado como Scene (múltiplos objetos)
            if isinstance(mesh, trimesh.Scene):
                logging.warning(f"Arquivo {filename} carregado como Cena, concatenando geometrias.")
                try:
                    mesh = mesh.dump(concatenate=True)
                except:
                    # Se falhar, tentar pegar a primeira geometria
                    geometries = list(mesh.geometry.values())
                    if geometries:
                        mesh = geometries[0]
                    else:
                        logging.error(f"Não foi possível extrair geometria de {filename}")
                        return None

            # Validar que é um Trimesh válido
            if not isinstance(mesh, trimesh.Trimesh):
                logging.error(f"Objeto carregado não é um Trimesh válido: {filename}")
                return None

            # Pré-processamento básico
            try:
                # Verificar e corrigir orientação das normais
                mesh.fix_normals()
                
                # Tentar preencher buracos se não for watertight
                if not mesh.is_watertight:
                    logging.warning(f"Malha {filename} não é watertight. Tentando preencher buracos.")
                    mesh.fill_holes()
                
                # Remover vértices duplicados e faces degeneradas
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
                
            except Exception as e:
                logging.warning(f"Erro durante pré-processamento de {filename}: {e}")
                # Continuar mesmo com erro no pré-processamento

            # Salvar no cache se habilitado
            if use_cache:
                self._save_to_cache(mesh, cache_filepath)

            return mesh

        except Exception as e:
            logging.error(f"Erro ao carregar arquivo {filepath}: {e}")
            return None

    def _install_fast_simplification(self):
        """Tenta instalar fast_simplification se não estiver disponível."""
        try:
            import fast_simplification
            return True
        except ImportError:
            try:
                logging.info("Tentando instalar fast_simplification...")
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "fast_simplification"])
                import fast_simplification
                logging.info("fast_simplification instalado com sucesso")
                return True
            except:
                logging.warning("Não foi possível instalar fast_simplification")
                return False

    def _simplify_with_pymeshlab(self, mesh, target_faces):
        """Simplifica usando pymeshlab com métodos corretos."""
        try:
            import pymeshlab
            
            # Criar MeshSet
            ms = pymeshlab.MeshSet()
            
            # Converter trimesh para pymeshlab
            vertices = mesh.vertices.astype(np.float64)
            faces = mesh.faces.astype(np.int32)
            
            # Criar mesh pymeshlab
            pymesh = pymeshlab.Mesh(vertices, faces)
            ms.add_mesh(pymesh)
            
            # Tentar diferentes métodos de simplificação do pymeshlab
            original_faces = len(mesh.faces)
            
            # Método 1: Simplificação por edge collapse (mais comum)
            try:
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
                logging.info("Usando pymeshlab: meshing_decimation_quadric_edge_collapse")
            except:
                # Método 2: Clustering vertices
                try:
                    target_perc = target_faces / original_faces
                    ms.meshing_decimation_clustering(threshold=pymeshlab.Percentage(target_perc * 100))
                    logging.info("Usando pymeshlab: meshing_decimation_clustering")
                except:
                    # Método 3: Simplificação por sampling
                    ms.generate_simplified_point_cloud(samplenum=min(target_faces * 3, len(vertices)))
                    logging.info("Usando pymeshlab: generate_simplified_point_cloud")
            
            # Extrair malha simplificada
            simplified_mesh_data = ms.current_mesh()
            simplified_vertices = simplified_mesh_data.vertex_matrix()
            simplified_faces = simplified_mesh_data.face_matrix()
            
            # Criar nova malha trimesh
            simplified_mesh = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces)
            
            logging.info(f"Pymeshlab simplificação: {original_faces} → {len(simplified_faces)} faces")
            return simplified_mesh
            
        except Exception as e:
            logging.warning(f"Erro na simplificação com pymeshlab: {e}")
            return None

    def _simplify_by_vertex_clustering(self, mesh, target_faces):
        """Simplifica agrupando vértices próximos."""
        try:
            from sklearn.cluster import KMeans
            
            # Calcular número de clusters baseado no target_faces
            n_clusters = min(target_faces, len(mesh.vertices) // 3)
            
            if n_clusters < 10:
                logging.warning("Target_faces muito baixo para clustering")
                return None
            
            # Agrupar vértices
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(mesh.vertices)
            
            # Criar novos vértices (centroides dos clusters)
            new_vertices = kmeans.cluster_centers_
            
            # Criar faces usando Delaunay triangulation
            from scipy.spatial import Delaunay
            
            # Projetar vértices para 2D para triangulação
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            vertices_2d = pca.fit_transform(new_vertices)
            
            # Triangulação
            tri = Delaunay(vertices_2d)
            new_faces = tri.simplices
            
            # Filtrar faces inválidas
            valid_faces = []
            for face in new_faces:
                if len(np.unique(face)) == 3:  # Face válida (3 vértices únicos)
                    valid_faces.append(face)
            
            if len(valid_faces) == 0:
                logging.warning("Nenhuma face válida gerada no clustering")
                return None
            
            simplified_mesh = trimesh.Trimesh(vertices=new_vertices, faces=np.array(valid_faces))
            
            logging.info(f"Clustering simplificação: {len(mesh.faces)} → {len(simplified_mesh.faces)} faces")
            return simplified_mesh
            
        except Exception as e:
            logging.warning(f"Erro na simplificação por clustering: {e}")
            return None

    def _simplify_by_convex_hull_sampling(self, mesh, target_faces):
        """Simplifica usando convex hull de uma amostra de vértices."""
        try:
            n_vertices = len(mesh.vertices)
            
            # Determinar quantos vértices amostrar
            sample_size = min(max(target_faces * 2, 50), n_vertices)
            
            if sample_size >= n_vertices:
                # Se a amostra é quase o tamanho original, usar convex hull direto
                hull = ConvexHull(mesh.vertices)
                simplified_mesh = trimesh.Trimesh(vertices=hull.points, faces=hull.simplices)
            else:
                # Amostragem estratificada para preservar forma
                # Combinar amostragem aleatória com pontos extremos
                
                # Pontos extremos (bounding box)
                bounds = mesh.bounds
                extreme_points = []
                for i in range(3):  # x, y, z
                    for j in range(2):  # min, max
                        point = bounds[j]
                        # Encontrar vértice mais próximo
                        distances = np.linalg.norm(mesh.vertices - point, axis=1)
                        closest_idx = np.argmin(distances)
                        extreme_points.append(mesh.vertices[closest_idx])
                
                extreme_points = np.array(extreme_points)
                
                # Amostragem aleatória do restante
                remaining_sample_size = sample_size - len(extreme_points)
                if remaining_sample_size > 0:
                    random_indices = np.random.choice(
                        n_vertices, 
                        size=min(remaining_sample_size, n_vertices), 
                        replace=False
                    )
                    random_points = mesh.vertices[random_indices]
                    
                    # Combinar pontos extremos e aleatórios
                    sampled_vertices = np.vstack([extreme_points, random_points])
                else:
                    sampled_vertices = extreme_points
                
                # Convex hull da amostra
                hull = ConvexHull(sampled_vertices)
                simplified_mesh = trimesh.Trimesh(vertices=hull.points, faces=hull.simplices)
            
            logging.info(f"Convex hull simplificação: {len(mesh.faces)} → {len(simplified_mesh.faces)} faces")
            return simplified_mesh
            
        except Exception as e:
            logging.warning(f"Erro na simplificação por convex hull: {e}")
            return None

    def simplify(self, mesh, target_faces, use_cache=True, original_filename="unknown"):
        """Simplifica a malha para um número alvo de faces usando métodos robustos.

        Args:
            mesh (trimesh.Trimesh): Objeto da malha a ser simplificada.
            target_faces (int): Número alvo de faces para a malha simplificada.
            use_cache (bool): Se deve usar o cache para a malha simplificada.
            original_filename (str): Nome do arquivo original para gerar chave de cache.

        Returns:
            trimesh.Trimesh or None: Malha simplificada ou malha original se falhar.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            logging.error("Input para simplificação não é um objeto Trimesh válido.")
            return None

        current_faces = len(mesh.faces)
        
        if current_faces <= target_faces:
            logging.info(f"Malha já possui {current_faces} faces (<= alvo {target_faces}). "
                        "Nenhuma simplificação necessária.")
            return mesh

        cache_params = {
            'operation': 'simplify', 
            'target_faces': target_faces,
            'original_faces': current_faces,
            'version': '3.0'  # Nova versão robusta
        }
        cache_filepath = self._get_cache_filename(original_filename, cache_params)

        # Tentar carregar do cache
        if use_cache:
            cached_mesh = self._load_from_cache(cache_filepath)
            if cached_mesh is not None:
                return cached_mesh

        logging.info(f"Simplificando malha de {current_faces} faces para "
                    f"aproximadamente {target_faces} faces...")
        
        # Lista de métodos de simplificação em ordem de preferência
        simplification_methods = [
            ("trimesh simplify_quadric_decimation", self._try_trimesh_quadric),
            ("pymeshlab", self._simplify_with_pymeshlab),
            ("vertex clustering", self._simplify_by_vertex_clustering),
            ("convex hull sampling", self._simplify_by_convex_hull_sampling)
        ]
        
        simplified_mesh = None
        
        for method_name, method_func in simplification_methods:
            try:
                logging.info(f"Tentando método: {method_name}")
                simplified_mesh = method_func(mesh, target_faces)
                
                if simplified_mesh is not None and len(simplified_mesh.faces) > 0:
                    actual_faces = len(simplified_mesh.faces)
                    reduction_pct = (current_faces - actual_faces) / current_faces * 100
                    logging.info(f"✅ Sucesso com {method_name}: {actual_faces} faces "
                               f"({reduction_pct:.1f}% redução)")
                    
                    # Salvar no cache se habilitado
                    if use_cache:
                        self._save_to_cache(simplified_mesh, cache_filepath)
                    
                    return simplified_mesh
                else:
                    logging.warning(f"❌ {method_name} resultou em malha vazia/inválida")
                    
            except Exception as e:
                logging.warning(f"❌ {method_name} falhou: {e}")
                continue
        
        # Se todos os métodos falharam, retornar malha original
        logging.warning("⚠️ Todas as tentativas de simplificação falharam. Retornando malha original.")
        return mesh

    def _try_trimesh_quadric(self, mesh, target_faces):
        """Tenta usar o método quadric do trimesh."""
        # Instalar fast_simplification se necessário
        if self._install_fast_simplification():
            try:
                # Calcular ratio de redução (valor entre 0 e 1)
                current_faces = len(mesh.faces)
                target_reduction = 1.0 - (target_faces / current_faces)
                target_reduction = max(0.0, min(0.99, target_reduction))  # Limitar entre 0 e 0.99
                
                if hasattr(mesh, 'simplify_quadric_decimation'):
                    return mesh.simplify_quadric_decimation(target_reduction)
                elif hasattr(mesh, 'simplify_quadratic_decimation'):
                    return mesh.simplify_quadratic_decimation(target_reduction)
            except Exception as e:
                logging.warning(f"Método quadric do trimesh falhou: {e}")
        
        return None

    def get_mesh_stats(self, mesh):
        """Retorna estatísticas básicas da malha.
        
        Args:
            mesh (trimesh.Trimesh): Malha para análise
            
        Returns:
            dict: Dicionário com estatísticas da malha
        """
        if not isinstance(mesh, trimesh.Trimesh):
            return None
            
        stats = {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'edges': len(mesh.edges),
            'volume': mesh.volume if mesh.is_volume else None,
            'surface_area': mesh.area,
            'is_watertight': mesh.is_watertight,
            'is_winding_consistent': mesh.is_winding_consistent,
            'bounds': mesh.bounds.tolist(),
            'extents': mesh.extents.tolist(),
            'centroid': mesh.centroid.tolist()
        }
        
        return stats