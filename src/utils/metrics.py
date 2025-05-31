# -*- coding: utf-8 -*-
"""Módulo para calcular métricas de avaliação da detecção de landmarks."""

import numpy as np
import logging
import pandas as pd
import os
import sys

# Adicionar o diretório pai ao path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.core.landmarks import LANDMARK_NAMES
from src.utils.helpers import load_landmarks_from_json

def calculate_landmark_distance(predicted_coord, ground_truth_coord):
    """Calcula a distância Euclidiana entre duas coordenadas 3D.

    Args:
        predicted_coord: Coordenadas previstas [x, y, z].
        ground_truth_coord: Coordenadas ground truth [x, y, z].

    Returns:
        float or None: Distância Euclidiana ou None se inválida.
    """
    if predicted_coord is None or ground_truth_coord is None:
        return None
    
    try:
        pred = np.asarray(predicted_coord, dtype=float)
        gt = np.asarray(ground_truth_coord, dtype=float)
        
        if pred.shape != (3,) or gt.shape != (3,):
            logging.warning(f"Coordenadas inválidas: pred={pred.shape}, gt={gt.shape}")
            return None
        
        # Verificar se há valores NaN ou infinitos
        if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
            logging.warning("Coordenadas contêm NaN ou infinito")
            return None
            
        distance = np.linalg.norm(pred - gt)
        return float(distance)
        
    except Exception as e:
        logging.error(f"Erro ao calcular distância Euclidiana: {e}")
        return None

def evaluate_detection(predicted_landmarks, ground_truth_landmarks):
    """Calcula a distância (erro) para cada landmark detectado.

    Args:
        predicted_landmarks (dict): Landmarks previstos {"Nome": [x,y,z] or None, ...}.
        ground_truth_landmarks (dict): Landmarks ground truth {"Nome": [x,y,z] or None, ...}.

    Returns:
        tuple: (dict de erros por landmark, erro médio de detecção)
    """
    if not predicted_landmarks or not ground_truth_landmarks:
        logging.warning("Dicionários de landmarks vazios ou inválidos")
        return {}, None

    landmark_errors = {}
    valid_distances = []

    for name in LANDMARK_NAMES:
        pred_coord = predicted_landmarks.get(name)
        gt_coord = ground_truth_landmarks.get(name)

        if gt_coord is None:
            # Sem ground truth disponível
            landmark_errors[name] = None
            logging.debug(f"GT não disponível para {name}")
            continue

        if pred_coord is None:
            # Predição falhou
            landmark_errors[name] = None
            logging.debug(f"Predição falhou para {name}")
            continue

        # Calcular distância
        distance = calculate_landmark_distance(pred_coord, gt_coord)
        landmark_errors[name] = distance
        
        if distance is not None:
            valid_distances.append(distance)
            logging.debug(f"Erro para {name}: {distance:.4f} mm")
        else:
            logging.warning(f"Cálculo de distância falhou para {name}")

    # Calcular erro médio de detecção (MDE)
    mean_detection_error = np.mean(valid_distances) if valid_distances else None
    
    if mean_detection_error is not None:
        logging.info(f"Erro Médio de Detecção (MDE): {mean_detection_error:.4f} mm")
    else:
        logging.info("MDE não pôde ser calculado (nenhuma distância válida)")

    return landmark_errors, mean_detection_error

def calculate_detection_rate(predicted_landmarks, ground_truth_landmarks):
    """Calcula a taxa de detecção para cada landmark.
    
    Args:
        predicted_landmarks (dict): Landmarks previstos
        ground_truth_landmarks (dict): Landmarks ground truth
        
    Returns:
        dict: Taxa de detecção por landmark
    """
    detection_rates = {}
    
    for name in LANDMARK_NAMES:
        gt_coord = ground_truth_landmarks.get(name)
        pred_coord = predicted_landmarks.get(name)
        
        if gt_coord is not None:  # GT existe
            if pred_coord is not None:  # Predição existe
                detection_rates[name] = 1.0  # Detectado
            else:
                detection_rates[name] = 0.0  # Não detectado
        else:
            detection_rates[name] = None  # GT não disponível
    
    return detection_rates

def run_evaluation_on_dataset(results_dir, ground_truth_dir, method_name):
    """Executa avaliação em um conjunto de resultados e ground truths.

    Args:
        results_dir (str): Diretório com arquivos JSON de landmarks previstos.
        ground_truth_dir (str): Diretório com arquivos JSON de ground truth.
        method_name (str): Nome do método sendo avaliado.

    Returns:
        tuple: (DataFrame detalhado, DataFrame resumo)
    """
    if not os.path.exists(results_dir):
        logging.error(f"Diretório de resultados não encontrado: {results_dir}")
        return pd.DataFrame(), pd.DataFrame()
    
    if not os.path.exists(ground_truth_dir):
        logging.error(f"Diretório de ground truth não encontrado: {ground_truth_dir}")
        return pd.DataFrame(), pd.DataFrame()

    all_results = []
    aggregated_errors = {name: [] for name in LANDMARK_NAMES}
    detection_counts = {name: {"detected": 0, "total_gt": 0} for name in LANDMARK_NAMES}

    # Encontrar arquivos de resultado
    try:
        result_files = [f for f in os.listdir(results_dir) if f.endswith("_landmarks.json")]
    except Exception as e:
        logging.error(f"Erro ao listar arquivos em {results_dir}: {e}")
        return pd.DataFrame(), pd.DataFrame()

    if not result_files:
        logging.error(f"Nenhum arquivo de resultado encontrado em: {results_dir}")
        return pd.DataFrame(), pd.DataFrame()

    logging.info(f"Avaliando método '{method_name}' em {len(result_files)} arquivos...")

    for result_filename in result_files:
        try:
            # Extrair ID do arquivo
            file_id = result_filename.replace("_landmarks.json", "")
            
            # Caminhos dos arquivos
            predicted_filepath = os.path.join(results_dir, result_filename)
            gt_filename = f"{file_id}_landmarks_gt.json"
            gt_filepath = os.path.join(ground_truth_dir, gt_filename)

            # Carregar landmarks
            predicted_landmarks = load_landmarks_from_json(predicted_filepath)
            ground_truth_landmarks = load_landmarks_from_json(gt_filepath)

            if predicted_landmarks is None:
                logging.warning(f"Falha ao carregar predições para {file_id}")
                continue
                
            if ground_truth_landmarks is None:
                logging.warning(f"Falha ao carregar GT para {file_id}")
                continue

            # Avaliar detecção
            landmark_errors, mde = evaluate_detection(predicted_landmarks, ground_truth_landmarks)
            detection_rates = calculate_detection_rate(predicted_landmarks, ground_truth_landmarks)

            # Registrar resultados individuais
            for name in LANDMARK_NAMES:
                error = landmark_errors.get(name)
                detection_rate = detection_rates.get(name)
                
                all_results.append({
                    "FileID": file_id,
                    "Method": method_name,
                    "Landmark": name,
                    "Error": error if error is not None else np.nan,
                    "MDE_File": mde if mde is not None else np.nan,
                    "Detected": 1 if detection_rate == 1.0 else 0 if detection_rate == 0.0 else np.nan
                })

                # Atualizar contagens agregadas
                if ground_truth_landmarks.get(name) is not None:
                    detection_counts[name]["total_gt"] += 1
                    if predicted_landmarks.get(name) is not None and error is not None:
                        aggregated_errors[name].append(error)
                        detection_counts[name]["detected"] += 1

        except Exception as e:
            logging.error(f"Erro ao processar arquivo {result_filename}: {e}")
            continue

    # Criar DataFrame de resultados
    results_df = pd.DataFrame(all_results)

    # Calcular estatísticas agregadas
    summary_stats = []
    for name in LANDMARK_NAMES:
        errors = aggregated_errors[name]
        total_gt = detection_counts[name]["total_gt"]
        detected = detection_counts[name]["detected"]
        
        # Taxa de detecção
        detection_rate = (detected / total_gt) * 100 if total_gt > 0 else 0

        # Estatísticas de erro
        if errors:
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            median_err = np.median(errors)
            min_err = np.min(errors)
            max_err = np.max(errors)
        else:
            mean_err = std_err = median_err = min_err = max_err = np.nan

        summary_stats.append({
            "Landmark": name,
            "MeanError": mean_err,
            "StdError": std_err,
            "MedianError": median_err,
            "MinError": min_err,
            "MaxError": max_err,
            "DetectionRate": detection_rate,
            "NumDetected": detected,
            "NumGT": total_gt
        })

    summary_df = pd.DataFrame(summary_stats)

    # Log estatísticas gerais
    if not results_df.empty:
        overall_detection_rate = results_df["Detected"].mean() * 100
        overall_mean_error = results_df["Error"].mean()
        logging.info(f"Avaliação concluída para '{method_name}':")
        logging.info(f"  Taxa de detecção geral: {overall_detection_rate:.1f}%")
        logging.info(f"  Erro médio geral: {overall_mean_error:.4f} mm")

    return results_df, summary_df

def compare_methods(results_dict):
    """Compara múltiplos métodos de detecção.
    
    Args:
        results_dict (dict): {"method_name": (results_df, summary_df), ...}
        
    Returns:
        pd.DataFrame: Comparação resumida entre métodos
    """
    comparison_data = []
    
    for method_name, (results_df, summary_df) in results_dict.items():
        if results_df.empty:
            continue
            
        overall_stats = {
            "Method": method_name,
            "OverallDetectionRate": results_df["Detected"].mean() * 100,
            "OverallMeanError": results_df["Error"].mean(),
            "OverallStdError": results_df["Error"].std(),
            "MedianError": results_df["Error"].median(),
            "NumLandmarks": len(LANDMARK_NAMES),
            "NumFiles": results_df["FileID"].nunique()
        }
        comparison_data.append(overall_stats)
    
    return pd.DataFrame(comparison_data)

# Exemplo de uso e teste
if __name__ == "__main__":
    import json
    import tempfile
    import shutil
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("=== Testando Métricas de Avaliação ===")

    # Criar diretórios temporários
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy_results_dir = os.path.join(temp_dir, "results")
        dummy_gt_dir = os.path.join(temp_dir, "ground_truth")
        os.makedirs(dummy_results_dir)
        os.makedirs(dummy_gt_dir)

        # Criar dados de teste
        test_cases = [
            {
                "file_id": "testA",
                "pred": {"Glabela": [1, 2, 3], "Nasion": [4, 5, 6], "Bregma": None},
                "gt": {"Glabela": [1.1, 2.1, 3.1], "Nasion": [4, 5, 7], "Bregma": [10, 11, 12]}
            },
            {
                "file_id": "testB", 
                "pred": {"Glabela": [10, 20, 30], "Nasion": None, "Bregma": [10, 11, 12.5]},
                "gt": {"Glabela": [10.5, 20.5, 30.5], "Nasion": [14, 15, 16], "Bregma": [10, 11, 12]}
            }
        ]

        # Salvar arquivos de teste
        for case in test_cases:
            pred_path = os.path.join(dummy_results_dir, f"{case['file_id']}_landmarks.json")
            gt_path = os.path.join(dummy_gt_dir, f"{case['file_id']}_landmarks_gt.json")
            
            with open(pred_path, 'w') as f:
                json.dump(case['pred'], f)
            with open(gt_path, 'w') as f:
                json.dump(case['gt'], f)

        # Testar cálculo de distância individual
        dist = calculate_landmark_distance([1, 2, 3], [1.1, 2.1, 3.1])
        print(f"Distância teste: {dist:.4f}")

        # Testar avaliação de arquivo individual
        pred1 = test_cases[0]['pred']
        gt1 = test_cases[0]['gt']
        errors, mde = evaluate_detection(pred1, gt1)
        print(f"\nErros arquivo A: {errors}")
        print(f"MDE arquivo A: {mde:.4f}" if mde else "MDE arquivo A: N/A")

        # Testar avaliação do dataset
        print("\n=== Avaliação do Dataset Teste ===")
        results_df, summary_df = run_evaluation_on_dataset(
            dummy_results_dir, dummy_gt_dir, "TestMethod"
        )

        if not results_df.empty:
            print("\nResultados Detalhados:")
            print(results_df.round(4))
            
            print("\nResumo por Landmark:")
            print(summary_df.round(4))
        else:
            print("Nenhum resultado gerado")