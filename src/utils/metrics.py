# -*- coding: utf-8 -*-
"""Módulo para calcular métricas de avaliação da detecção de landmarks."""

import numpy as np
import logging
import pandas as pd
import time
import os

from ..core.landmarks import LANDMARK_NAMES
from .helpers import load_landmarks_from_json # Usar import relativo

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

def calculate_landmark_distance(predicted_coord, ground_truth_coord):
    """Calcula a distância Euclidiana entre duas coordenadas 3D.

    Args:
        predicted_coord (np.ndarray or list): Coordenadas previstas [x, y, z].
        ground_truth_coord (np.ndarray or list): Coordenadas ground truth [x, y, z].

    Returns:
        float or None: Distância Euclidiana ou None se alguma coordenada for inválida.
    """
    if predicted_coord is None or ground_truth_coord is None:
        return None
    try:
        pred = np.asarray(predicted_coord)
        gt = np.asarray(ground_truth_coord)
        if pred.shape != (3,) or gt.shape != (3,):
            logging.warning(f"Coordenadas inválidas para cálculo de distância: pred={pred}, gt={gt}")
            return None
        distance = np.linalg.norm(pred - gt)
        return distance
    except Exception as e:
        logging.error(f"Erro ao calcular distância Euclidiana: {e}")
        return None

def evaluate_detection(predicted_landmarks, ground_truth_landmarks):
    """Calcula a distância (erro) para cada landmark detectado.

    Args:
        predicted_landmarks (dict): Dicionário de landmarks previstos {"Nome": [x,y,z] or None, ...}.
        ground_truth_landmarks (dict): Dicionário de landmarks ground truth {"Nome": [x,y,z] or None, ...}.

    Returns:
        dict: Dicionário com a distância (erro em mm, assumindo unidades consistentes)
              para cada landmark. {"Nome": distancia or None, ...}.
        float or None: Erro médio de detecção (Mean Detection Error - MDE) sobre os landmarks
                       detectados com sucesso e presentes no ground truth.
    """
    landmark_errors = {}
    valid_distances = []

    if not predicted_landmarks or not ground_truth_landmarks:
        logging.warning("Dicionários de landmarks previstos ou ground truth estão vazios/inválidos.")
        return {}, None

    # Iterar sobre os landmarks definidos para garantir consistência
    for name in LANDMARK_NAMES:
        pred_coord = predicted_landmarks.get(name)
        gt_coord = ground_truth_landmarks.get(name)

        if gt_coord is None:
            # Se não há ground truth, não podemos calcular erro
            landmark_errors[name] = None
            logging.debug(f"Ground truth não disponível para {name}. Erro não calculado.")
            continue

        if pred_coord is None:
            # Se a predição falhou, mas GT existe, marcamos como erro infinito ou None?
            # Usaremos None para indicar falha na detecção.
            landmark_errors[name] = None
            logging.debug(f"Predição falhou para {name} (GT disponível). Erro não calculado.")
            continue

        # Calcular distância se ambos existirem
        distance = calculate_landmark_distance(pred_coord, gt_coord)
        landmark_errors[name] = distance
        if distance is not None:
            valid_distances.append(distance)
            logging.debug(f"Erro para {name}: {distance:.4f} mm")
        else:
             logging.warning(f"Cálculo de distância falhou para {name}.")

    # Calcular erro médio (MDE)
    mean_detection_error = np.mean(valid_distances) if valid_distances else None
    if mean_detection_error is not None:
        logging.info(f"Erro Médio de Detecção (MDE): {mean_detection_error:.4f} mm")
    else:
        logging.info("Não foi possível calcular o Erro Médio de Detecção (nenhuma distância válida).")

    return landmark_errors, mean_detection_error

def run_evaluation_on_dataset(results_dir, ground_truth_dir, method_name):
    """Executa a avaliação em um conjunto de resultados e ground truths.

    Args:
        results_dir (str): Diretório contendo os arquivos JSON com landmarks previstos.
                           (Ex: results/geometric/A0001_landmarks.json)
        ground_truth_dir (str): Diretório contendo os arquivos JSON com landmarks ground truth.
                                (Ex: data/ground_truth/A0001_landmarks_gt.json)
        method_name (str): Nome do método sendo avaliado (ex: "Geometric", "ML").

    Returns:
        pandas.DataFrame: DataFrame contendo os erros por landmark e o MDE para cada arquivo.
                        Colunas: [\"FileID\", "Landmark", "Error", "MDE"]
        pandas.DataFrame: DataFrame com estatísticas agregadas (média, std) por landmark.
                        Colunas: ["Landmark", "MeanError", "StdError", "DetectionRate"]
    """
    all_results = []
    aggregated_errors = {name: [] for name in LANDMARK_NAMES}
    detection_counts = {name: {"detected": 0, "total_gt": 0} for name in LANDMARK_NAMES}

    result_files = [f for f in os.listdir(results_dir) if f.endswith("_landmarks.json")]
    if not result_files:
        logging.error(f"Nenhum arquivo de resultado encontrado em: {results_dir}")
        return pd.DataFrame(), pd.DataFrame()

    logging.info(f"Iniciando avaliação para o método \"{method_name}\" em {len(result_files)} arquivos...")

    for result_filename in result_files:
        file_id = result_filename.replace("_landmarks.json", "")
        predicted_filepath = os.path.join(results_dir, result_filename)
        # Assumir um padrão para nome do arquivo GT
        gt_filename = f"{file_id}_landmarks_gt.json" # Ajustar se o padrão for diferente
        gt_filepath = os.path.join(ground_truth_dir, gt_filename)

        predicted_landmarks = load_landmarks_from_json(predicted_filepath)
        ground_truth_landmarks = load_landmarks_from_json(gt_filepath)

        if predicted_landmarks is None:
            logging.warning(f"Falha ao carregar predições para {file_id}. Pulando.")
            continue
        if ground_truth_landmarks is None:
            logging.warning(f"Falha ao carregar ground truth para {file_id}. Pulando.")
            continue

        landmark_errors, mde = evaluate_detection(predicted_landmarks, ground_truth_landmarks)

        # Registrar resultados individuais
        for name, error in landmark_errors.items():
            all_results.append({
                "FileID": file_id,
                "Method": method_name,
                "Landmark": name,
                "Error": error if error is not None else np.nan, # Usar NaN para facilitar agregação
                "MDE_File": mde if mde is not None else np.nan
            })
            # Atualizar contagens e erros agregados
            if ground_truth_landmarks.get(name) is not None:
                detection_counts[name]["total_gt"] += 1
                if predicted_landmarks.get(name) is not None and error is not None:
                    aggregated_errors[name].append(error)
                    detection_counts[name]["detected"] += 1

    results_df = pd.DataFrame(all_results)

    # Calcular estatísticas agregadas
    summary_stats = []
    for name in LANDMARK_NAMES:
        errors = aggregated_errors[name]
        total_gt = detection_counts[name]["total_gt"]
        detected = detection_counts[name]["detected"]
        detection_rate = (detected / total_gt) * 100 if total_gt > 0 else 0

        if errors:
            mean_err = np.mean(errors)
            std_err = np.std(errors)
        else:
            mean_err = np.nan
            std_err = np.nan

        summary_stats.append({
            "Landmark": name,
            "MeanError": mean_err,
            "StdError": std_err,
            "DetectionRate": detection_rate,
            "NumDetected": detected,
            "NumGT": total_gt
        })

    summary_df = pd.DataFrame(summary_stats)

    logging.info(f"Avaliação concluída para o método \"{method_name}\".")
    return results_df, summary_df

# Exemplo de uso (requer arquivos JSON dummy)
if __name__ == \"__main__\":
    from .helpers import save_landmarks_to_json # Import relativo

    logging.info("--- Testando Métricas de Avaliação ---")

    # Criar diretórios dummy
    dummy_results_dir = "./dummy_results_metrics"
    dummy_gt_dir = "./dummy_gt_metrics"
    os.makedirs(dummy_results_dir, exist_ok=True)
    os.makedirs(dummy_gt_dir, exist_ok=True)

    # Criar arquivos dummy
    pred1 = {"Glabela": [1, 2, 3], "Nasion": [4, 5, 6], "Bregma": None}
    gt1 = {"Glabela": [1.1, 2.1, 3.1], "Nasion": [4, 5, 7], "Bregma": [10, 11, 12]}
    save_landmarks_to_json(pred1, os.path.join(dummy_results_dir, "fileA_landmarks.json"))
    save_landmarks_to_json(gt1, os.path.join(dummy_gt_dir, "fileA_landmarks_gt.json"))

    pred2 = {"Glabela": [10, 20, 30], "Nasion": None, "Bregma": [10, 11, 12.5]}
    gt2 = {"Glabela": [10.5, 20.5, 30.5], "Nasion": [14, 15, 16], "Bregma": [10, 11, 12]}
    save_landmarks_to_json(pred2, os.path.join(dummy_results_dir, "fileB_landmarks.json"))
    save_landmarks_to_json(gt2, os.path.join(dummy_gt_dir, "fileB_landmarks_gt.json"))

    # Testar cálculo de distância individual
    dist = calculate_landmark_distance(pred1["Glabela"], gt1["Glabela"])
    print(f"Distância Glabela (Arquivo A): {dist:.4f}")
    dist_fail = calculate_landmark_distance(pred1["Bregma"], gt1["Bregma"])
    print(f"Distância Bregma (Arquivo A): {dist_fail}")

    # Testar avaliação de um arquivo
    errors, mde = evaluate_detection(pred1, gt1)
    print("\nErros Arquivo A:")
    print(errors)
    print(f"MDE Arquivo A: {mde:.4f}")

    # Testar avaliação do dataset
    print("\n--- Avaliação do Dataset Dummy ---")
    results_dataframe, summary_dataframe = run_evaluation_on_dataset(dummy_results_dir, dummy_gt_dir, "DummyMethod")

    print("\nResultados Individuais (DataFrame):")
    print(results_dataframe.to_string())

    print("\nEstatísticas Agregadas (DataFrame):")
    print(summary_dataframe.to_string())

    # Limpeza (opcional)
    # import shutil
    # shutil.rmtree(dummy_results_dir)
    # shutil.rmtree(dummy_gt_dir)

