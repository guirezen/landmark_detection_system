# -*- coding: utf-8 -*-
"""Módulo para carregar, pré-processar e simplificar malhas 3D de crânios."""

import trimesh
import numpy as np
import os
import hashlib
import pickle
import logging

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

    def simplify(self, mesh, target_faces, use_cache=True, original_filename="unknown"):
        """Simplifica a malha para um número alvo de faces usando decimação quadrática.

        Args:
            mesh (trimesh.Trimesh): Objeto da malha a ser simplificada.
            target_faces (int): Número alvo de faces para a malha simplificada.
            use_cache (bool): Se deve usar o cache para a malha simplificada.
            original_filename (str): Nome do arquivo original para gerar chave de cache.

        Returns:
            trimesh.Trimesh or None: Malha simplificada ou None se falhar.
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
            'version': '1.0'
        }
        cache_filepath = self._get_cache_filename(original_filename, cache_params)

        # Tentar carregar do cache
        if use_cache:
            cached_mesh = self._load_from_cache(cache_filepath)
            if cached_mesh is not None:
                return cached_mesh

        logging.info(f"Simplificando malha de {current_faces} faces para "
                    f"aproximadamente {target_faces} faces...")
        
        try:
            # Usar decimação quadrática que preserva melhor as características
            simplified_mesh = mesh.simplify_quadratic_decimation(target_faces)
            
            if simplified_mesh is None or len(simplified_mesh.faces) == 0:
                logging.error("Simplificação resultou em malha vazia.")
                return None
                
            actual_faces = len(simplified_mesh.faces)
            logging.info(f"Malha simplificada para {actual_faces} faces "
                        f"({(current_faces - actual_faces) / current_faces * 100:.1f}% redução).")

            # Salvar no cache se habilitado
            if use_cache:
                self._save_to_cache(simplified_mesh, cache_filepath)

            return simplified_mesh
            
        except Exception as e:
            logging.error(f"Erro durante a simplificação da malha: {e}")
            
            # Tentar método alternativo com menor agressividade
            try:
                logging.info("Tentando simplificação alternativa...")
                # Reduzir menos agressivamente
                conservative_target = max(target_faces, current_faces // 2)
                simplified_mesh = mesh.simplify_quadratic_decimation(conservative_target)
                
                if simplified_mesh is not None and len(simplified_mesh.faces) > 0:
                    logging.info(f"Simplificação alternativa bem-sucedida: {len(simplified_mesh.faces)} faces")
                    return simplified_mesh
                    
            except:
                pass
            
            # Se tudo falhar, retornar malha original
            logging.warning("Todas as tentativas de simplificação falharam. Retornando malha original.")
            return mesh

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

# Exemplo de uso
if __name__ == '__main__':
    import time
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Criar diretórios de teste
    os.makedirs("data/skulls", exist_ok=True)
    
    # Criar arquivo STL dummy para teste
    dummy_stl_path = "data/skulls/dummy_skull_test.stl"
    if not os.path.exists(dummy_stl_path):
        logging.info(f"Criando arquivo STL dummy em {dummy_stl_path}")
        # Criar uma esfera como teste
        mesh_dummy = trimesh.primitives.Sphere(radius=50, subdivisions=3)
        mesh_dummy.export(dummy_stl_path)

    # Testar o processador
    processor = MeshProcessor(data_dir="./data/skulls", cache_dir="./data/cache")

    # Teste de carregamento
    logging.info("=== Teste de Carregamento ===")
    start_time = time.time()
    skull_mesh = processor.load_skull("dummy_skull_test.stl", use_cache=True)
    load_time = time.time() - start_time

    if skull_mesh:
        stats = processor.get_mesh_stats(skull_mesh)
        logging.info(f"Malha carregada em {load_time:.4f}s:")
        for key, value in stats.items():
            logging.info(f"  {key}: {value}")

        # Teste de simplificação
        logging.info("=== Teste de Simplificação ===")
        target_faces = 100
        start_time = time.time()
        simplified_skull = processor.simplify(skull_mesh, 
                                            target_faces=target_faces, 
                                            use_cache=True, 
                                            original_filename="dummy_skull_test.stl")
        simplify_time = time.time() - start_time

        if simplified_skull:
            simplified_stats = processor.get_mesh_stats(simplified_skull)
            logging.info(f"Malha simplificada em {simplify_time:.4f}s:")
            for key, value in simplified_stats.items():
                if key in ['vertices', 'faces']:
                    logging.info(f"  {key}: {value}")

        # Teste de cache (segunda chamada deve ser mais rápida)
        logging.info("=== Teste de Cache ===")
        start_time = time.time()
        skull_mesh_cached = processor.load_skull("dummy_skull_test.stl", use_cache=True)
        cache_time = time.time() - start_time
        logging.info(f"Carregamento do cache: {cache_time:.4f}s")

        start_time = time.time()
        simplified_cached = processor.simplify(skull_mesh, 
                                             target_faces=target_faces, 
                                             use_cache=True, 
                                             original_filename="dummy_skull_test.stl")
        cache_simplify_time = time.time() - start_time
        logging.info(f"Simplificação do cache: {cache_simplify_time:.4f}s")

    else:
        logging.error("Falha ao carregar a malha dummy.")