# -*- coding: utf-8 -*-
"""Módulo para carregar, pré-processar e simplificar malhas 3D de crânios."""

import trimesh
import numpy as np
import os
import hashlib
import pickle
import logging

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                # Trimesh objects podem ser complexos para pickle diretamente
                # É mais seguro salvar os componentes numpy
                mesh_data = {'vertices': mesh.vertices, 'faces': mesh.faces}
                pickle.dump(mesh_data, f)
            logging.info(f"Malha salva no cache: {cache_filepath}")
        except Exception as e:
            logging.error(f"Erro ao salvar malha no cache {cache_filepath}: {e}")

    def _load_from_cache(self, cache_filepath):
        """Carrega a malha do cache, se existir."""
        if os.path.exists(cache_filepath):
            try:
                with open(cache_filepath, 'rb') as f:
                    mesh_data = pickle.load(f)
                # Recriar o objeto Trimesh
                mesh = trimesh.Trimesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
                logging.info(f"Malha carregada do cache: {cache_filepath}")
                return mesh
            except Exception as e:
                logging.error(f"Erro ao carregar malha do cache {cache_filepath}: {e}")
                # Se o cache estiver corrompido, removemos para evitar erros futuros
                try:
                    os.remove(cache_filepath)
                    logging.warning(f"Arquivo de cache corrompido removido: {cache_filepath}")
                except OSError as remove_error:
                    logging.error(f"Erro ao remover arquivo de cache corrompido: {remove_error}")
                return None
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
        cache_params = {'operation': 'load'}
        cache_filepath = self._get_cache_filename(filename, cache_params)

        if use_cache:
            mesh = self._load_from_cache(cache_filepath)
            if mesh:
                return mesh

        if not os.path.exists(filepath):
            logging.error(f"Arquivo não encontrado: {filepath}")
            return None

        try:
            # force='mesh' tenta carregar como uma malha única
            mesh = trimesh.load_mesh(filepath, force='mesh')
            logging.info(f"Arquivo STL carregado: {filepath} (Vértices: {len(mesh.vertices)}, Faces: {len(mesh.faces)})")

            # Pré-processamento básico (opcional, mas recomendado)
            if isinstance(mesh, trimesh.Scene):
                 # Se o STL contiver múltiplos objetos, tenta combiná-los
                 # Ou pode-se escolher o maior componente
                 logging.warning(f"Arquivo {filename} carregado como Cena, tentando concatenar geometrias.")
                 mesh = mesh.dump(concatenate=True)
                 if not isinstance(mesh, trimesh.Trimesh):
                     logging.error(f"Não foi possível converter a cena em Trimesh: {filename}")
                     return None

            if not mesh.is_watertight:
                logging.warning(f"Malha {filename} não é watertight. Tentando preencher buracos.")
                mesh.fill_holes()

            mesh.fix_normals()

            if use_cache:
                self._save_to_cache(mesh, cache_filepath)

            return mesh

        except Exception as e:
            logging.error(f"Erro ao carregar ou processar o arquivo {filepath}: {e}")
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

        if len(mesh.faces) <= target_faces:
            logging.info(f"Malha já possui {len(mesh.faces)} faces (<= alvo {target_faces}). Nenhuma simplificação necessária.")
            return mesh

        cache_params = {'operation': 'simplify', 'target_faces': target_faces}
        cache_filepath = self._get_cache_filename(original_filename, cache_params)

        if use_cache:
            cached_mesh = self._load_from_cache(cache_filepath)
            if cached_mesh:
                return cached_mesh

        logging.info(f"Simplificando malha de {len(mesh.faces)} faces para aproximadamente {target_faces} faces...")
        try:
            # A decimação quadrática geralmente preserva melhor as características
            simplified_mesh = mesh.simplify_quadratic_decimation(target_faces)
            logging.info(f"Malha simplificada para {len(simplified_mesh.faces)} faces.")

            if use_cache:
                self._save_to_cache(simplified_mesh, cache_filepath)

            return simplified_mesh
        except Exception as e:
            # Trimesh pode ter problemas com algumas malhas durante a simplificação
            logging.error(f"Erro durante a simplificação da malha: {e}")
            # Tentar um método alternativo ou retornar None?
            # Por simplicidade, retornamos None
            return None

# Exemplo de uso (requer um arquivo STL em data/skulls para funcionar)
if __name__ == '__main__':
    # Criar diretórios e um arquivo STL dummy se não existirem para teste
    if not os.path.exists("data/skulls"): os.makedirs("data/skulls")
    dummy_stl_path = "data/skulls/dummy_skull.stl"
    if not os.path.exists(dummy_stl_path):
        # Cria um cubo como malha dummy
        logging.info(f"Criando arquivo STL dummy em {dummy_stl_path}")
        mesh_dummy = trimesh.primitives.Box()
        mesh_dummy.export(dummy_stl_path)

    processor = MeshProcessor(data_dir="./data/skulls", cache_dir="./data/cache")

    # Teste de carregamento
    logging.info("--- Teste de Carregamento ---")
    skull_mesh = processor.load_skull("dummy_skull.stl", use_cache=True)

    if skull_mesh:
        logging.info(f"Malha dummy carregada: Vértices={len(skull_mesh.vertices)}, Faces={len(skull_mesh.faces)}")

        # Teste de simplificação
        logging.info("--- Teste de Simplificação ---")
        target_faces_simplify = 5 # Reduzir o cubo (12 faces) para menos
        simplified_skull = processor.simplify(skull_mesh, target_faces=target_faces_simplify, use_cache=True, original_filename="dummy_skull.stl")

        if simplified_skull:
            logging.info(f"Malha dummy simplificada: Vértices={len(simplified_skull.vertices)}, Faces={len(simplified_skull.faces)}")
        else:
            logging.error("Falha ao simplificar a malha dummy.")

        # Teste de cache (segunda chamada deve ser mais rápida e usar cache)
        logging.info("--- Teste de Cache (Carregamento) ---")
        skull_mesh_cached = processor.load_skull("dummy_skull.stl", use_cache=True)
        logging.info("--- Teste de Cache (Simplificação) ---")
        simplified_skull_cached = processor.simplify(skull_mesh, target_faces=target_faces_simplify, use_cache=True, original_filename="dummy_skull.stl")

    else:
        logging.error("Falha ao carregar a malha dummy.")

