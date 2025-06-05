# -*- coding: utf-8 -*-
"""Funções auxiliares e utilitários gerais para o sistema de detecção."""

import os
import logging
import time
import numpy as np
import json
from functools import wraps

def setup_logging(log_level=logging.INFO):
    """Configura o formato e nível do logging global da aplicação."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Reduzir verbosidade de bibliotecas externas para evitar poluição do log
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("open3d").setLevel(logging.WARNING)

def timeit(func):
    """Decorator para medir o tempo de execução de uma função e logar o resultado."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        logging.info(f"Função '{func.__name__}' executada em {elapsed:.4f} segundos")
        return result
    return wrapper

def list_stl_files(data_dir):
    """Retorna uma lista ordenada de todos os arquivos .stl presentes em um diretório."""
    try:
        if not os.path.isdir(data_dir):
            logging.error(f"Diretório de dados não encontrado: {data_dir}")
            return []
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(".stl")]
        files.sort()
        logging.info(f"{len(files)} arquivos STL encontrados em {data_dir}")
        return files
    except Exception as e:
        logging.error(f"Erro ao listar arquivos STL em {data_dir}: {e}")
        return []

def save_landmarks_to_json(landmarks_dict, filepath):
    """Salva um dicionário de landmarks detectados em um arquivo JSON."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Converter arrays numpy para listas, e garantir None se não detectado
        serializable = {}
        for name, coords in landmarks_dict.items():
            if isinstance(coords, np.ndarray):
                serializable[name] = coords.tolist()
            else:
                serializable[name] = coords  # Pode ser lista padrão ou None
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=4, ensure_ascii=False)
        logging.info(f"Landmarks salvos em arquivo: {filepath}")
        return True
    except Exception as e:
        logging.error(f"Erro ao salvar JSON de landmarks ({filepath}): {e}")
        return False

def load_landmarks_from_json(filepath):
    """Carrega um dicionário de landmarks a partir de um arquivo JSON."""
    if not os.path.exists(filepath):
        logging.error(f"Arquivo de landmarks não encontrado: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Converter listas de volta para np.array para uso interno, se for o caso
        for name, coords in data.items():
            if isinstance(coords, list) and len(coords) == 3:
                data[name] = np.array(coords, dtype=float)
        logging.info(f"Landmarks carregados de {filepath}")
        return data
    except Exception as e:
        logging.error(f"Erro ao carregar landmarks do JSON ({filepath}): {e}")
        return None

def validate_mesh(mesh):
    """Verifica se uma malha (trimesh.Trimesh) é válida para processamento."""
    if mesh is None:
        return False
    # Uma malha válida deve ter atributos vertices e faces com conteúdo
    if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
        return False
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return False
    return True
