# -*- coding: utf-8 -*-
"""Script principal para executar o sistema de detecção de landmarks."""

import argparse
import os
import logging
import time

# Configurar imports relativos corretos
# Adicionar o diretório pai (landmark_detection_system) ao sys.path
# Isso é comum em scripts que rodam módulos de um pacote
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.core.mesh_processor import MeshProcessor
from src.core.detector_geometric import GeometricDetector
from src.core.detector_ml import MLDetector
from src.utils.helpers import list_stl_files, save_landmarks_to_json, timeit, setup_logging
from src.utils.visualization import plot_landmarks
from src.utils.metrics import evaluate_detection, load_landmarks_from_json as load_gt_landmarks # Renomear para clareza

# Configuração do logging (pode ser ajustado por argumentos)
setup_logging()

@timeit
def process_single_file(args):
    """Processa um único arquivo STL."""
    logging.info(f"--- Processando arquivo único: {args.input_file} ---")
    filename = os.path.basename(args.input_file)
    data_dir = os.path.dirname(args.input_file)
    if not data_dir: # Se apenas o nome do arquivo foi passado, assume o diretório padrão
        data_dir = args.data_dir
        filepath = os.path.join(data_dir, filename)
    else:
        filepath = args.input_file # Caminho completo foi fornecido

    if not os.path.exists(filepath):
        logging.error(f"Arquivo de entrada não encontrado: {filepath}")
        return

    # 1. Carregar e Pré-processar a Malha
    processor = MeshProcessor(data_dir=data_dir, cache_dir=args.cache_dir)
    mesh = processor.load_skull(filename, use_cache=not args.no_cache)
    if not mesh:
        logging.error(f"Falha ao carregar a malha: {filename}")
        return

    simplified_mesh = mesh # Usar malha original por padrão
    if args.simplify_faces > 0:
        logging.info(f"Simplificando malha para {args.simplify_faces} faces...")
        simplified_mesh = processor.simplify(mesh, target_faces=args.simplify_faces,
                                           use_cache=not args.no_cache, original_filename=filename)
        if not simplified_mesh:
            logging.warning("Falha ao simplificar a malha. Usando a original.")
            simplified_mesh = mesh

    # 2. Selecionar e Executar o Método de Detecção
    landmarks_detected = None
    if args.method == "geometric":
        logging.info("Usando método de detecção Geométrico.")
        detector = GeometricDetector()
        landmarks_detected = detector.detect(simplified_mesh)
    elif args.method == "ml":
        logging.info("Usando método de detecção por Machine Learning.")
        # Verificar se os modelos ML existem (treinamento não é feito aqui)
        ml_detector = MLDetector(model_dir=args.model_dir)
        # Tentar carregar modelos necessários (a função detect faz isso internamente)
        landmarks_detected = ml_detector.detect(simplified_mesh)
    else:
        logging.error(f"Método de detecção desconhecido: {args.method}")
        return

    if not landmarks_detected:
        logging.error(f"Falha na detecção de landmarks para {filename}.")
        return

    logging.info(f"Landmarks detectados para {filename}:")
    for name, coords in landmarks_detected.items():
        coord_str = f"[{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]" if coords else "Não detectado"
        logging.info(f"  - {name}: {coord_str}")

    # 3. Salvar Resultados
    output_filename = f"{os.path.splitext(filename)[0]}_{args.method}_landmarks.json"
    output_path = os.path.join(args.output_dir, output_filename)
    os.makedirs(args.output_dir, exist_ok=True)
    if not save_landmarks_to_json(landmarks_detected, output_path):
        logging.error(f"Falha ao salvar resultados em {output_path}")

    # 4. Avaliar (se ground truth for fornecido)
    if args.gt_file:
        logging.info(f"Avaliando resultado contra ground truth: {args.gt_file}")
        gt_landmarks = load_gt_landmarks(args.gt_file)
        if gt_landmarks:
            errors, mde = evaluate_detection(landmarks_detected, gt_landmarks)
            logging.info("Erros de detecção (distância em mm):")
            for name, error in errors.items():
                error_str = f"{error:.4f}" if error is not None else "N/A"
                logging.info(f"  - {name}: {error_str}")
            mde_str = f"{mde:.4f}" if mde is not None else "N/A"
            logging.info(f"Erro Médio de Detecção (MDE): {mde_str} mm")
        else:
            logging.warning(f"Não foi possível carregar o arquivo ground truth: {args.gt_file}")

    # 5. Visualizar (opcional)
    if args.visualize:
        logging.info("Gerando visualização...")
        vis_title = f"Landmarks ({args.method.capitalize()}) - {filename}"
        vis_save_path = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_{args.method}_visualization.png")
        # Usar malha simplificada para visualização se foi usada na detecção
        plot_landmarks(simplified_mesh, landmarks_detected, title=vis_title,
                       use_3d=not args.force_2d_vis, save_path_2d=vis_save_path)
        logging.info(f"Visualização (potencialmente) salva em: {vis_save_path} (se 2D foi usada)")

@timeit
def process_batch(args):
    """Processa múltiplos arquivos STL em um diretório."""
    logging.info(f"--- Processando Lote de Arquivos no Diretório: {args.input_dir} ---")
    stl_files = list_stl_files(args.input_dir)
    if not stl_files:
        logging.error(f"Nenhum arquivo STL encontrado em {args.input_dir}")
        return

    num_files = len(stl_files)
    logging.info(f"Encontrados {num_files} arquivos STL para processar.")

    # Criar subdiretório de saída para o método
    method_output_dir = os.path.join(args.output_dir, args.method)
    os.makedirs(method_output_dir, exist_ok=True)

    # Preparar processador e detector fora do loop para eficiência
    processor = MeshProcessor(data_dir=args.input_dir, cache_dir=args.cache_dir)
    if args.method == "geometric":
        detector = GeometricDetector()
    elif args.method == "ml":
        detector = MLDetector(model_dir=args.model_dir)
        # Pré-carregar modelos pode ser útil aqui se a memória permitir
        # for name in LANDMARK_NAMES: detector.load_model(name)
    else:
        logging.error(f"Método de detecção desconhecido: {args.method}")
        return

    processed_count = 0
    failed_count = 0
    start_time_batch = time.time()

    for i, filename in enumerate(stl_files):
        logging.info(f"Processando arquivo {i+1}/{num_files}: {filename}")
        file_start_time = time.time()
        try:
            mesh = processor.load_skull(filename, use_cache=not args.no_cache)
            if not mesh:
                logging.warning(f"Falha ao carregar {filename}. Pulando.")
                failed_count += 1
                continue

            simplified_mesh = mesh
            if args.simplify_faces > 0:
                simplified_mesh = processor.simplify(mesh, target_faces=args.simplify_faces,
                                                   use_cache=not args.no_cache, original_filename=filename)
                if not simplified_mesh:
                    logging.warning(f"Falha ao simplificar {filename}. Usando original.")
                    simplified_mesh = mesh

            landmarks_detected = detector.detect(simplified_mesh)

            if not landmarks_detected:
                logging.warning(f"Falha na detecção para {filename}. Pulando.")
                failed_count += 1
                continue

            # Salvar resultados
            output_filename = f"{os.path.splitext(filename)[0]}_landmarks.json" # Nome genérico dentro da pasta do método
            output_path = os.path.join(method_output_dir, output_filename)
            save_landmarks_to_json(landmarks_detected, output_path)

            # Visualizar (opcional, pode gerar muitas janelas/arquivos)
            if args.visualize:
                vis_title = f"Landmarks ({args.method.capitalize()}) - {filename}"
                vis_save_path = os.path.join(method_output_dir, f"{os.path.splitext(filename)[0]}_visualization.png")
                plot_landmarks(simplified_mesh, landmarks_detected, title=vis_title,
                               use_3d=not args.force_2d_vis, save_path_2d=vis_save_path)

            processed_count += 1
            file_end_time = time.time()
            logging.info(f"Arquivo {filename} processado em {file_end_time - file_start_time:.2f} seg.")

        except Exception as e:
            logging.error(f"Erro inesperado ao processar {filename}: {e}", exc_info=True)
            failed_count += 1

    end_time_batch = time.time()
    logging.info(f"--- Processamento em Lote Concluído ---")
    logging.info(f"Total de arquivos: {num_files}")
    logging.info(f"Processados com sucesso: {processed_count}")
    logging.info(f"Falhas: {failed_count}")
    logging.info(f"Tempo total do lote: {end_time_batch - start_time_batch:.2f} seg.")

    # Avaliação do Lote (se diretório GT for fornecido)
    if args.gt_dir:
        logging.info(f"--- Avaliando Resultados do Lote ({args.method}) --- ")
        from src.utils.metrics import run_evaluation_on_dataset # Importar aqui para evitar dependência circular no top-level
        results_df, summary_df = run_evaluation_on_dataset(method_output_dir, args.gt_dir, args.method.capitalize())

        if not results_df.empty:
            results_csv_path = os.path.join(args.output_dir, f"evaluation_{args.method}_detailed.csv")
            summary_csv_path = os.path.join(args.output_dir, f"evaluation_{args.method}_summary.csv")
            try:
                results_df.to_csv(results_csv_path, index=False)
                summary_df.to_csv(summary_csv_path, index=False)
                logging.info(f"Resultados detalhados da avaliação salvos em: {results_csv_path}")
                logging.info(f"Sumário da avaliação salvo em: {summary_csv_path}")
            except Exception as e:
                logging.error(f"Erro ao salvar resultados da avaliação em CSV: {e}")
        else:
            logging.warning("Avaliação do lote não gerou resultados.")

def main():
    parser = argparse.ArgumentParser(description="Sistema de Detecção de Landmarks em Crânios 3D")

    # --- Argumentos Comuns ---
    parser.add_argument("--method", type=str, required=True, choices=["geometric", "ml"], help="Método de detecção a ser utilizado.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Diretório para salvar os resultados (JSON, visualizações, CSVs de avaliação).")
    parser.add_argument("--cache_dir", type=str, default="./data/cache", help="Diretório para cache de malhas processadas.")
    parser.add_argument("--no_cache", action="store_true", help="Desativa o uso de cache para carregamento e simplificação.")
    parser.add_argument("--simplify_faces", type=int, default=5000, help="Número alvo de faces para simplificação da malha (0 para não simplificar).")
    parser.add_argument("--visualize", action="store_true", help="Gera visualizações dos landmarks na malha.")
    parser.add_argument("--force_2d_vis", action="store_true", help="Força o uso de visualização 2D (Matplotlib) mesmo que Open3D esteja disponível.")
    parser.add_argument("--model_dir", type=str, default="./models", help="Diretório contendo os modelos ML treinados (usado apenas se method='ml').")
    parser.add_argument("-v", "--verbose", action="store_true", help="Ativa logging mais detalhado (DEBUG).")

    # --- Subparsers para modos Single e Batch ---
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Modo de operação: processar um único arquivo ou um lote.")

    # --- Modo Single File ---
    parser_single = subparsers.add_parser("single", help="Processa um único arquivo STL.")
    parser_single.add_argument("-i", "--input_file", type=str, required=True, help="Caminho para o arquivo STL de entrada.")
    parser_single.add_argument("--data_dir", type=str, default="./data/skulls", help="Diretório base de dados (usado se input_file for apenas nome).")
    parser_single.add_argument("--gt_file", type=str, help="(Opcional) Caminho para o arquivo JSON de ground truth correspondente para avaliação.")
    parser_single.set_defaults(func=process_single_file)

    # --- Modo Batch ---
    parser_batch = subparsers.add_parser("batch", help="Processa todos os arquivos STL em um diretório.")
    parser_batch.add_argument("-i", "--input_dir", type=str, required=True, help="Diretório contendo os arquivos STL de entrada.")
    parser_batch.add_argument("--gt_dir", type=str, help="(Opcional) Diretório contendo os arquivos JSON de ground truth para avaliação em lote.")
    parser_batch.set_defaults(func=process_batch)

    args = parser.parse_args()

    # Ajustar nível de logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Logging de DEBUG ativado.")
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Executar a função correspondente ao modo selecionado
    args.func(args)

if __name__ == "__main__":
    main()

