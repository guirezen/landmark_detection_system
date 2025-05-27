# -*- coding: utf-8 -*-
"""Funções auxiliares e utilitários gerais para o sistema."""

import os
import logging
import time
import numpy as np
import json

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

def setup_logging(log_level=logging.INFO):
    """Configura o logging para o sistema."""
    logging.basicConfig(level=log_level,
                        format=\'%(asctime)s - %(name)s - %(levelname)s - %(message)s\
                        , datefmt=\'%Y-%m-%d %H:%M:%S\")
    # Exemplo: Desativar logging de bibliotecas muito verbosas
    # logging.getLogger(\"trimesh\").setLevel(logging.WARNING)

def timeit(method):
    """Decorador para medir o tempo de execução de uma função."""
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logging.info(f"Função \'{method.__name__}\' executada em {te - ts:.4f} seg")
        # print(f"Função \'{method.__name__}\' executada em {te - ts:.4f} seg") # Alternativa
        return result
    return timed

def list_stl_files(data_dir):
    """Lista todos os arquivos .stl no diretório especificado."""
    try:
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(\".stl\")]
        logging.info(f"{len(files)} arquivos STL encontrados em {data_dir}")
        return files
    except FileNotFoundError:
        logging.error(f"Diretório de dados não encontrado: {data_dir}")
        return []
    except Exception as e:
        logging.error(f"Erro ao listar arquivos STL em {data_dir}: {e}")
        return []

def save_landmarks_to_json(landmarks_dict, filepath):
    """Salva o dicionário de landmarks detectados em um arquivo JSON.

    Args:
        landmarks_dict (dict): Dicionário com nomes de landmarks e coordenadas.
        filepath (str): Caminho completo para salvar o arquivo JSON.
    """
    try:
        # Converter coordenadas numpy para listas, se necessário
        serializable_landmarks = {}
        for name, coords in landmarks_dict.items():
            if isinstance(coords, np.ndarray):
                serializable_landmarks[name] = coords.tolist()
            else:
                serializable_landmarks[name] = coords # Assume que já é serializável (lista ou None)

        with open(filepath, \'w\
                 , encoding=\'utf-8\
                 ) as f:
            json.dump(serializable_landmarks, f, indent=4, ensure_ascii=False)
        logging.info(f"Landmarks salvos em: {filepath}")
        return True
    except Exception as e:
        logging.error(f"Erro ao salvar landmarks em JSON ({filepath}): {e}")
        return False

def load_landmarks_from_json(filepath):
    """Carrega um dicionário de landmarks de um arquivo JSON.

    Args:
        filepath (str): Caminho completo do arquivo JSON.

    Returns:
        dict or None: Dicionário de landmarks carregado ou None se falhar.
    """
    if not os.path.exists(filepath):
        logging.error(f"Arquivo de landmarks não encontrado: {filepath}")
        return None
    try:
        with open(filepath, \'r\
                 , encoding=\'utf-8\
                 ) as f:
            landmarks_dict = json.load(f)
        # Opcional: Converter listas de volta para arrays numpy
        for name, coords in landmarks_dict.items():
            if isinstance(coords, list):
                landmarks_dict[name] = np.array(coords)
        logging.info(f"Landmarks carregados de: {filepath}")
        return landmarks_dict
    except Exception as e:
        logging.error(f"Erro ao carregar landmarks do JSON ({filepath}): {e}")
        return None

# Exemplo de uso
if __name__ == \'__main__\
    :
    setup_logging()
    logging.info("--- Testando Funções Auxiliares ---")

    # Teste list_stl_files (cria dir dummy se não existir)
    dummy_data_dir = "./dummy_data_helpers"
    if not os.path.exists(dummy_data_dir): os.makedirs(dummy_data_dir)
    with open(os.path.join(dummy_data_dir, "test1.stl"), \'w\
             ) as f: f.write("dummy")
    with open(os.path.join(dummy_data_dir, "test2.STL"), \'w\
             ) as f: f.write("dummy")
    with open(os.path.join(dummy_data_dir, "ignore.txt"), \'w\
             ) as f: f.write("dummy")

    stl_files = list_stl_files(dummy_data_dir)
    print(f"Arquivos STL encontrados: {stl_files}")

    # Teste save/load landmarks
    dummy_landmarks = {
        "Glabela": np.array([1.0, 2.0, 3.0]),
        "Nasion": [4.0, 5.0, 6.0], # Já em lista
        "Bregma": None
    }
    dummy_json_path = os.path.join(dummy_data_dir, "landmarks.json")

    if save_landmarks_to_json(dummy_landmarks, dummy_json_path):
        loaded_landmarks = load_landmarks_from_json(dummy_json_path)
        if loaded_landmarks:
            print("\nLandmarks Salvos:")
            print(dummy_landmarks)
            print("\nLandmarks Carregados:")
            print(loaded_landmarks)
            # Verificar se numpy array foi restaurado
            print(f"Tipo Glabela carregado: {type(loaded_landmarks.get(\'Glabela\'))}")

    # Limpeza (opcional)
    # import shutil
    # shutil.rmtree(dummy_data_dir)

