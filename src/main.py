# -*- coding: utf-8 -*-
"""Script principal para executar o sistema de detecção de landmarks - CORREÇÃO DE CAMINHO."""

import argparse
import os
import logging
import time
import sys

# Configurar imports - adicionar diretório raiz ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.core.mesh_processor import MeshProcessor
from src.core.detector_geometric import GeometricDetector
from src.core.detector_ml import MLDetector
from src.utils.helpers import list_stl_files, save_landmarks_to_json, timeit, setup_logging
from src.utils.visualization import plot_landmarks
from src.utils.metrics import evaluate_detection, load_landmarks_from_json

@timeit
def process_single_file(args):
    """Processa um único arquivo STL."""
    logging.info(f"=== Processando arquivo único: {args.input_file} ===")
    
    # CORREÇÃO: Determinar caminho completo do arquivo corretamente
    input_file = args.input_file
    
    # Se o caminho é absoluto, usar como está
    if os.path.isabs(input_file):
        filepath = input_file
        data_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
    # Se contém separadores de caminho, tratar como relativo ao diretório atual
    elif os.path.sep in input_file or '/' in input_file:
        filepath = os.path.abspath(input_file)
        data_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
    # Se é apenas nome do arquivo, usar data_dir padrão
    else:
        filename = input_file
        data_dir = args.data_dir
        filepath = os.path.join(data_dir, filename)

    logging.info(f"Caminho resolvido: {filepath}")
    logging.info(f"Data dir: {data_dir}")
    logging.info(f"Filename: {filename}")

    if not os.path.exists(filepath):
        logging.error(f"Arquivo de entrada não encontrado: {filepath}")
        return False

    try:
        # 1. Carregar e pré-processar a malha
        processor = MeshProcessor(data_dir=data_dir, cache_dir=args.cache_dir)
        
        logging.info(f"Carregando malha: {filename}")
        mesh = processor.load_skull(filename, use_cache=not args.no_cache)
        if not mesh:
            logging.error(f"Falha ao carregar a malha: {filename}")
            return False

        logging.info(f"Malha carregada: {len(mesh.vertices)} vértices, {len(mesh.faces)} faces")

        # Simplificar se solicitado
        simplified_mesh = mesh
        if args.simplify_faces > 0:
            logging.info(f"Simplificando malha para {args.simplify_faces} faces...")
            simplified_mesh = processor.simplify(
                mesh, 
                target_faces=args.simplify_faces,
                use_cache=not args.no_cache, 
                original_filename=filename
            )
            if not simplified_mesh:
                logging.warning("Falha na simplificação. Usando malha original.")
                simplified_mesh = mesh
            else:
                logging.info(f"Malha simplificada: {len(simplified_mesh.vertices)} vértices, "
                           f"{len(simplified_mesh.faces)} faces")

        # 2. Executar detecção
        landmarks_detected = None
        if args.method == "geometric":
            logging.info("Executando detecção geométrica...")
            detector = GeometricDetector()
            landmarks_detected = detector.detect(simplified_mesh)
            
        elif args.method == "ml":
            logging.info("Executando detecção por Machine Learning...")
            ml_detector = MLDetector(model_dir=args.model_dir)
            landmarks_detected = ml_detector.detect(simplified_mesh)
            
        else:
            logging.error(f"Método de detecção desconhecido: {args.method}")
            return False

        if not landmarks_detected:
            logging.error(f"Falha na detecção de landmarks para {filename}")
            return False

        # Log dos resultados
        detected_count = sum(1 for coords in landmarks_detected.values() if coords is not None)
        total_count = len(landmarks_detected)
        logging.info(f"Detecção concluída: {detected_count}/{total_count} landmarks detectados")
        
        if args.verbose:
            for name, coords in landmarks_detected.items():
                if coords:
                    coord_str = f"[{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]"
                else:
                    coord_str = "Não detectado"
                logging.info(f"  {name}: {coord_str}")

        # 3. Salvar resultados
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = f"{os.path.splitext(filename)[0]}_{args.method}_landmarks.json"
        output_path = os.path.join(args.output_dir, output_filename)
        
        if save_landmarks_to_json(landmarks_detected, output_path):
            logging.info(f"Resultados salvos em: {output_path}")
        else:
            logging.error(f"Falha ao salvar resultados em: {output_path}")

        # 4. Avaliar contra ground truth (se fornecido)
        if hasattr(args, 'gt_file') and args.gt_file:
            logging.info(f"Avaliando contra ground truth: {args.gt_file}")
            gt_landmarks = load_landmarks_from_json(args.gt_file)
            if gt_landmarks:
                errors, mde = evaluate_detection(landmarks_detected, gt_landmarks)
                
                logging.info("=== Resultados da Avaliação ===")
                for name, error in errors.items():
                    if error is not None:
                        logging.info(f"  {name}: {error:.4f} mm")
                    else:
                        logging.info(f"  {name}: N/A")
                
                if mde is not None:
                    logging.info(f"Erro Médio de Detecção (MDE): {mde:.4f} mm")
                else:
                    logging.info("MDE: N/A")
            else:
                logging.warning(f"Falha ao carregar ground truth: {args.gt_file}")

        # 5. Gerar visualização (se solicitado)
        if args.visualize:
            logging.info("Gerando visualização...")
            vis_title = f"Landmarks ({args.method.capitalize()}) - {filename}"
            vis_save_path = os.path.join(args.output_dir, 
                                       f"{os.path.splitext(filename)[0]}_{args.method}_visualization.png")
            
            success = plot_landmarks(
                simplified_mesh, 
                landmarks_detected, 
                title=vis_title,
                use_3d=not args.force_2d_vis, 
                save_path_2d=vis_save_path
            )
            
            if success:
                logging.info(f"Visualização salva em: {vis_save_path}")
            else:
                logging.warning("Falha na geração da visualização")

        logging.info(f"Processamento de {filename} concluído com sucesso")
        return True

    except Exception as e:
        logging.error(f"Erro durante processamento de {filename}: {e}", exc_info=True)
        return False

@timeit
def process_batch(args):
    """Processa múltiplos arquivos STL em um diretório."""
    logging.info(f"=== Processamento em Lote: {args.input_dir} ===")
    
    if not os.path.exists(args.input_dir):
        logging.error(f"Diretório de entrada não encontrado: {args.input_dir}")
        return False

    stl_files = list_stl_files(args.input_dir)
    if not stl_files:
        logging.error(f"Nenhum arquivo STL encontrado em {args.input_dir}")
        return False

    num_files = len(stl_files)
    logging.info(f"Encontrados {num_files} arquivos STL para processar")

    # Criar estrutura de diretórios de saída
    method_output_dir = os.path.join(args.output_dir, args.method)
    os.makedirs(method_output_dir, exist_ok=True)

    # Preparar componentes de processamento
    processor = MeshProcessor(data_dir=args.input_dir, cache_dir=args.cache_dir)
    
    if args.method == "geometric":
        detector = GeometricDetector()
    elif args.method == "ml":
        detector = MLDetector(model_dir=args.model_dir)
    else:
        logging.error(f"Método de detecção desconhecido: {args.method}")
        return False

    # Estatísticas de processamento
    processed_count = 0
    failed_count = 0
    total_landmarks_detected = 0
    batch_start_time = time.time()

    # Processar cada arquivo
    for i, filename in enumerate(stl_files):
        file_start_time = time.time()
        logging.info(f"Processando arquivo {i+1}/{num_files}: {filename}")
        
        try:
            # Carregar malha
            mesh = processor.load_skull(filename, use_cache=not args.no_cache)
            if not mesh:
                logging.warning(f"Falha ao carregar {filename}. Pulando.")
                failed_count += 1
                continue

            # Simplificar se necessário
            simplified_mesh = mesh
            if args.simplify_faces > 0:
                simplified_mesh = processor.simplify(
                    mesh, 
                    target_faces=args.simplify_faces,
                    use_cache=not args.no_cache, 
                    original_filename=filename
                )
                if not simplified_mesh:
                    logging.warning(f"Falha na simplificação de {filename}. Usando original.")
                    simplified_mesh = mesh

            # Detectar landmarks
            landmarks_detected = detector.detect(simplified_mesh)
            if not landmarks_detected:
                logging.warning(f"Falha na detecção para {filename}. Pulando.")
                failed_count += 1
                continue

            # Contar landmarks detectados
            file_landmarks_count = sum(1 for coords in landmarks_detected.values() 
                                     if coords is not None)
            total_landmarks_detected += file_landmarks_count

            # Salvar resultados
            output_filename = f"{os.path.splitext(filename)[0]}_landmarks.json"
            output_path = os.path.join(method_output_dir, output_filename)
            save_landmarks_to_json(landmarks_detected, output_path)

            # Gerar visualização (se solicitado)
            if args.visualize:
                vis_title = f"Landmarks ({args.method.capitalize()}) - {filename}"
                vis_save_path = os.path.join(method_output_dir, 
                                           f"{os.path.splitext(filename)[0]}_visualization.png")
                plot_landmarks(
                    simplified_mesh, 
                    landmarks_detected, 
                    title=vis_title,
                    use_3d=not args.force_2d_vis, 
                    save_path_2d=vis_save_path
                )

            processed_count += 1
            file_time = time.time() - file_start_time
            logging.info(f"Arquivo {filename} processado em {file_time:.2f}s "
                        f"({file_landmarks_count} landmarks detectados)")

        except Exception as e:
            logging.error(f"Erro inesperado ao processar {filename}: {e}")
            failed_count += 1
            continue

    # Estatísticas finais
    batch_time = time.time() - batch_start_time
    logging.info("=== Processamento em Lote Concluído ===")
    logging.info(f"Total de arquivos: {num_files}")
    logging.info(f"Processados com sucesso: {processed_count}")
    logging.info(f"Falhas: {failed_count}")
    logging.info(f"Total de landmarks detectados: {total_landmarks_detected}")
    logging.info(f"Tempo total: {batch_time:.2f}s")
    logging.info(f"Tempo médio por arquivo: {batch_time/max(processed_count, 1):.2f}s")

    # Executar avaliação em lote (se ground truth fornecido)
    if hasattr(args, 'gt_dir') and args.gt_dir and processed_count > 0:
        logging.info("=== Executando Avaliação em Lote ===")
        try:
            from src.utils.metrics import run_evaluation_on_dataset
            
            results_df, summary_df = run_evaluation_on_dataset(
                method_output_dir, args.gt_dir, args.method.capitalize()
            )

            if not results_df.empty:
                # Salvar resultados da avaliação
                results_csv = os.path.join(args.output_dir, f"evaluation_{args.method}_detailed.csv")
                summary_csv = os.path.join(args.output_dir, f"evaluation_{args.method}_summary.csv")
                
                results_df.to_csv(results_csv, index=False)
                summary_df.to_csv(summary_csv, index=False)
                
                logging.info(f"Avaliação detalhada salva em: {results_csv}")
                logging.info(f"Resumo da avaliação salvo em: {summary_csv}")
                
                # Log de estatísticas principais
                overall_detection_rate = results_df["Detected"].mean() * 100
                overall_mean_error = results_df["Error"].mean()
                logging.info(f"Taxa de detecção geral: {overall_detection_rate:.1f}%")
                logging.info(f"Erro médio geral: {overall_mean_error:.4f} mm")
            else:
                logging.warning("Avaliação em lote não gerou resultados")
                
        except Exception as e:
            logging.error(f"Erro durante avaliação em lote: {e}")

    return processed_count > 0

def setup_argument_parser():
    """Configura e retorna o parser de argumentos."""
    parser = argparse.ArgumentParser(
        description="Sistema de Detecção de Landmarks em Crânios 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Processar um arquivo com caminho relativo
  python src/main.py single --method geometric -i cranio.stl --visualize

  # Processar um arquivo com caminho completo  
  python src/main.py single --method geometric -i data/skulls/cranio.stl --visualize

  # Processar lote com ML e avaliação
  python src/main.py batch --method ml -i data/skulls/ --gt_dir data/ground_truth/ --output_dir results/

  # Processar com simplificação customizada
  python src/main.py single --method geometric -i cranio.stl --simplify_faces 2000 --verbose
        """
    )

    # Subparsers para modos
    subparsers = parser.add_subparsers(dest="mode", required=True, 
                                      help="Modo de operação")

    # Modo single file
    parser_single = subparsers.add_parser("single", 
                                         help="Processa um único arquivo STL")
    
    # Argumentos do single
    parser_single.add_argument("--method", type=str, required=True, 
                              choices=["geometric", "ml"], 
                              help="Método de detecção a ser utilizado")
    parser_single.add_argument("-i", "--input_file", type=str, required=True, 
                              help="Caminho para o arquivo STL (nome do arquivo ou caminho completo)")
    parser_single.add_argument("--data_dir", type=str, default="./data/skulls", 
                              help="Diretório base (usado quando input_file é apenas nome)")
    parser_single.add_argument("--gt_file", type=str, 
                              help="Arquivo JSON de ground truth para avaliação")
    parser_single.add_argument("--output_dir", type=str, default="./results", 
                              help="Diretório para salvar os resultados")
    parser_single.add_argument("--cache_dir", type=str, default="./data/cache", 
                              help="Diretório para cache de malhas processadas")
    parser_single.add_argument("--no_cache", action="store_true", 
                              help="Desativa o uso de cache")
    parser_single.add_argument("--simplify_faces", type=int, default=5000, 
                              help="Número alvo de faces para simplificação (0 para não simplificar)")
    parser_single.add_argument("--visualize", action="store_true", 
                              help="Gera visualizações dos landmarks")
    parser_single.add_argument("--force_2d_vis", action="store_true", 
                              help="Força visualização 2D mesmo que Open3D esteja disponível")
    parser_single.add_argument("--model_dir", type=str, default="./models", 
                              help="Diretório dos modelos ML (apenas para method='ml')")
    parser_single.add_argument("-v", "--verbose", action="store_true", 
                              help="Ativa logging detalhado")
    parser_single.set_defaults(func=process_single_file)

    # Modo batch
    parser_batch = subparsers.add_parser("batch", 
                                        help="Processa todos os STL em um diretório")
    
    # Argumentos do batch
    parser_batch.add_argument("--method", type=str, required=True, 
                             choices=["geometric", "ml"], 
                             help="Método de detecção a ser utilizado")
    parser_batch.add_argument("-i", "--input_dir", type=str, required=True, 
                             help="Diretório com arquivos STL")
    parser_batch.add_argument("--gt_dir", type=str, 
                             help="Diretório com arquivos JSON de ground truth")
    parser_batch.add_argument("--output_dir", type=str, default="./results", 
                             help="Diretório para salvar os resultados")
    parser_batch.add_argument("--cache_dir", type=str, default="./data/cache", 
                             help="Diretório para cache de malhas processadas")
    parser_batch.add_argument("--no_cache", action="store_true", 
                             help="Desativa o uso de cache")
    parser_batch.add_argument("--simplify_faces", type=int, default=5000, 
                             help="Número alvo de faces para simplificação (0 para não simplificar)")
    parser_batch.add_argument("--visualize", action="store_true", 
                             help="Gera visualizações dos landmarks")
    parser_batch.add_argument("--force_2d_vis", action="store_true", 
                             help="Força visualização 2D mesmo que Open3D esteja disponível")
    parser_batch.add_argument("--model_dir", type=str, default="./models", 
                             help="Diretório dos modelos ML (apenas para method='ml')")
    parser_batch.add_argument("-v", "--verbose", action="store_true", 
                             help="Ativa logging detalhado")
    parser_batch.set_defaults(func=process_batch)

    return parser

def main():
    """Função principal do script."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Configurar logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    # Validar argumentos
    if args.method == "ml" and not os.path.exists(args.model_dir):
        logging.warning(f"Diretório de modelos ML não encontrado: {args.model_dir}")
        logging.warning("Certifique-se de treinar os modelos antes de usar o método ML")

    if args.simplify_faces < 0:
        logging.error("--simplify_faces deve ser >= 0")
        return 1

    # Log da configuração
    logging.info("=== Configuração do Sistema ===")
    logging.info(f"Modo: {args.mode}")
    logging.info(f"Método: {args.method}")
    logging.info(f"Diretório de saída: {args.output_dir}")
    logging.info(f"Cache: {'Desabilitado' if args.no_cache else 'Habilitado'}")
    logging.info(f"Simplificação: {'Desabilitada' if args.simplify_faces == 0 else f'{args.simplify_faces} faces'}")
    logging.info(f"Visualização: {'Habilitada' if args.visualize else 'Desabilitada'}")

    # Executar função correspondente
    try:
        success = args.func(args)
        if success:
            logging.info("=== Execução Concluída com Sucesso ===")
            return 0
        else:
            logging.error("=== Execução Finalizada com Erros ===")
            return 1
            
    except KeyboardInterrupt:
        logging.info("Execução interrompida pelo usuário")
        return 1
    except Exception as e:
        logging.error(f"Erro inesperado durante execução: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())