# -*- coding: utf-8 -*-
"""Funções auxiliares e utilitários gerais para o sistema."""

import os
import logging
import time
import numpy as np
import json
from functools import wraps

def setup_logging(log_level=logging.INFO):
    """Configura o logging para o sistema."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Reduzir verbosidade de bibliotecas externas
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

def timeit(func):
    """Decorador para medir o tempo de execução de uma função."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Função '{func.__name__}' executada em {execution_time:.4f} seg")
        return result
    return wrapper

def list_stl_files(data_dir):
    """Lista todos os arquivos .stl no diretório especificado."""
    try:
        if not os.path.exists(data_dir):
            logging.error(f"Diretório de dados não encontrado: {data_dir}")
            return []
        
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(".stl")]
        logging.info(f"{len(files)} arquivos STL encontrados em {data_dir}")
        return sorted(files)  # Retornar ordenado para consistência
    except Exception as e:
        logging.error(f"Erro ao listar arquivos STL em {data_dir}: {e}")
        return []

def save_landmarks_to_json(landmarks_dict, filepath):
    """Salva o dicionário de landmarks detectados em um arquivo JSON.

    Args:
        landmarks_dict (dict): Dicionário com nomes de landmarks e coordenadas.
        filepath (str): Caminho completo para salvar o arquivo JSON.
    
    Returns:
        bool: True se salvou com sucesso, False caso contrário.
    """
    try:
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Converter coordenadas numpy para listas, se necessário
        serializable_landmarks = {}
        for name, coords in landmarks_dict.items():
            if isinstance(coords, np.ndarray):
                serializable_landmarks[name] = coords.tolist()
            else:
                serializable_landmarks[name] = coords  # Já é serializável ou None

        with open(filepath, 'w', encoding='utf-8') as f:
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
        with open(filepath, 'r', encoding='utf-8') as f:
            landmarks_dict = json.load(f)
        
        # Converter listas de volta para arrays numpy para consistência interna
        for name, coords in landmarks_dict.items():
            if isinstance(coords, list) and len(coords) == 3:
                landmarks_dict[name] = np.array(coords)
        
        logging.info(f"Landmarks carregados de: {filepath}")
        return landmarks_dict
    except Exception as e:
        logging.error(f"Erro ao carregar landmarks do JSON ({filepath}): {e}")
        return None

def validate_mesh(mesh):
    """Valida se uma malha é adequada para processamento.
    
    Args:
        mesh: Objeto trimesh.Trimesh
        
    Returns:
        bool: True se a malha é válida, False caso contrário
    """
    if mesh is None:
        return False
    
    if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
        return False
    
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return False
    
    return True

def create_directory_structure(base_dir):
    """Cria a estrutura de diretórios necessária para o projeto.
    
    Args:
        base_dir (str): Diretório base do projeto
    """
    directories = [
        'data/skulls',
        'data/cache',
        'data/ground_truth',
        'models',
        'results',
        'results/geometric',
        'results/ml'
    ]
    
    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        os.makedirs(full_path, exist_ok=True)
        logging.debug(f"Diretório criado/verificado: {full_path}")

# Exemplo de uso e testes
if __name__ == '__main__':
    setup_logging()
    logging.info("--- Testando Funções Auxiliares ---")

    # Teste list_stl_files (cria dir dummy se não existir)
    dummy_data_dir = "./dummy_data_helpers"
    os.makedirs(dummy_data_dir, exist_ok=True)
    
    # Criar alguns arquivos de teste
    with open(os.path.join(dummy_data_dir, "test1.stl"), 'w') as f: 
        f.write("dummy content")
    with open(os.path.join(dummy_data_dir, "test2.STL"), 'w') as f: 
        f.write("dummy content")
    with open(os.path.join(dummy_data_dir, "ignore.txt"), 'w') as f: 
        f.write("dummy content")

    stl_files = list_stl_files(dummy_data_dir)
    print(f"Arquivos STL encontrados: {stl_files}")

    # Teste save/load landmarks
    dummy_landmarks = {
        "Glabela": np.array([1.0, 2.0, 3.0]),
        "Nasion": [4.0, 5.0, 6.0],
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
            print(f"Tipo Glabela carregado: {type(loaded_landmarks.get('Glabela'))}")

    # Teste do decorador timeit
    @timeit
    def test_function():
        time.sleep(0.1)
        return "teste concluído"
    
    result = test_function()
    print(f"Resultado da função testada: {result}")

    # Limpeza
    import shutil
    try:
        shutil.rmtree(dummy_data_dir)
        print("Arquivos de teste removidos.")
    except:
        pass