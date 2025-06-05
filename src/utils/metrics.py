# -*- coding: utf-8 -*-
"""Métricas de avaliação para detecção de landmarks (cálculo de erros e estatísticas)."""
import numpy as np
import logging
import pandas as pd
import os
from src.core.landmarks import LANDMARK_NAMES
from src.utils.helpers import load_landmarks_from_json

def calculate_landmark_distance(predicted_coord, ground_truth_coord):
    """Calcula a distância Euclidiana (erro) entre duas coordenadas 3D."""
    if predicted_coord is None or ground_truth_coord is None:
        return None
    try:
        pred = np.asarray(predicted_coord, dtype=float)
        gt = np.asarray(ground_truth_coord, dtype=float)
        if pred.shape != (3,) or gt.shape != (3,):
            logging.warning(f"Coordenadas inválidas para cálculo de distância: pred={pred.shape}, gt={gt.shape}")
            return None
        # Verificar valores válidos
        if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
            logging.warning("Coordenadas contêm valores não finitos (NaN/Inf)")
            return None
        dist = np.linalg.norm(pred - gt)
        return float(dist)
    except Exception as e:
        logging.error(f"Erro ao calcular distância Euclidiana: {e}")
        return None

def evaluate_detection(predicted_landmarks, ground_truth_landmarks):
    """Computa o erro por landmark e o erro médio (MDE) dado um resultado predito e o ground truth correspondente."""
    if not predicted_landmarks or not ground_truth_landmarks:
        logging.warning("Dicionários de landmarks vazios ou inválidos fornecidos para avaliação.")
        return {}, None
    landmark_errors = {}
    valid_distances = []
    for name in LANDMARK_NAMES:
        pred_coord = predicted_landmarks.get(name)
        gt_coord = ground_truth_landmarks.get(name)
        if gt_coord is None:
            # Ground truth ausente para este ponto
            landmark_errors[name] = None
            logging.debug(f"Sem GT para {name}, ignorando na métrica.")
            continue
        if pred_coord is None:
            # Falha na predição deste ponto
            landmark_errors[name] = None
            logging.debug(f"{name} não detectado (None).")
            continue
        # Calcular distância e registrar
        dist = calculate_landmark_distance(pred_coord, gt_coord)
        landmark_errors[name] = dist
        if dist is not None:
            valid_distances.append(dist)
            logging.debug(f"Erro {name}: {dist:.4f} mm")
        else:
            logging.warning(f"Não foi possível calcular erro para {name} (coords inválidas).")
    # Erro médio de detecção (MDE)
    mde = np.mean(valid_distances) if valid_distances else None
    return landmark_errors, mde

def run_evaluation_on_dataset(results_dir, gt_dir, method_label="Metodo"):
    """Varre diretório de resultados e ground truth, calculando métricas para todos os arquivos correspondentes."""
    aggregated_errors = {name: [] for name in LANDMARK_NAMES}
    detection_counts = {name: {"detected": 0, "total_gt": 0} for name in LANDMARK_NAMES}
    results_records = []  # para DataFrame detalhado
    # Encontrar todos os arquivos de resultado no diretório
    for filename in os.listdir(results_dir):
        if not filename.endswith("_landmarks.json"):
            continue
        file_id = filename.replace("_landmarks.json", "")
        gt_file = os.path.join(gt_dir, f"{file_id}_landmarks_gt.json")
        result_file = os.path.join(results_dir, filename)
        if not os.path.exists(gt_file):
            logging.warning(f"GT não encontrado para {file_id}, pulando.")
            continue
        pred = load_landmarks_from_json(result_file)
        gt = load_landmarks_from_json(gt_file)
        if pred is None or gt is None:
            logging.error(f"Falha ao carregar JSON de resultado ou GT para {file_id}.")
            continue
        errors, mde = evaluate_detection(pred, gt)
        # Registro detalhado para este arquivo
        record = {"FileID": file_id, "Method": method_label}
        for name, err in errors.items():
            record[name] = err
            # Atualizar agregados
            if gt.get(name) is not None:
                detection_counts[name]["total_gt"] += 1
                if err is not None:
                    detection_counts[name]["detected"] += 1
            if err is not None:
                aggregated_errors[name].append(err)
        record["MDE"] = mde
        results_records.append(record)
    # Criar DataFrame de resultados detalhados
    results_df = pd.DataFrame(results_records)
    # Criar DataFrame de resumo (média por landmark, taxa de detecção por landmark)
    summary_data = []
    for name in LANDMARK_NAMES:
        errors = aggregated_errors[name]
        total_gt = detection_counts[name]["total_gt"]
        detected = detection_counts[name]["detected"]
        avg_error = float(np.mean(errors)) if errors else None
        detect_rate = (detected / total_gt * 100) if total_gt > 0 else None
        summary_data.append({
            "Landmark": name,
            "AvgError": avg_error,
            "DetectionRate(%)": detect_rate,
            "Count_GT": total_gt,
            "Count_Detected": detected
        })
    summary_df = pd.DataFrame(summary_data)
    return results_df, summary_df
