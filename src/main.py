# -*- coding: utf-8 -*-
"""Script principal otimizado para detecção de landmarks (suporta single e batch)."""

import argparse
import os
import logging
import time
import sys
import signal
import psutil
from contextlib import contextmanager

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

class TimeoutError(Exception):
    """Exceção para timeout personalizado."""
    pass

@contextmanager
def timeout_context(seconds):
    """Context manager para limitar tempo de execução de um bloco de código."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operação excedeu {seconds} segundos")
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
    try:
        yield
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

def check_system_resources():
    """Verifica recursos do sistema e sugere parâmetros apropriados."""
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        cpu_count = psutil.cpu_count()
        logging.info(f"Recursos do sistema: {available_gb:.1f} GB RAM, {cpu_count} CPUs")
        if available_gb < 4:
            return {
                'max_faces_load': 200_000,
                'default_simplify': 500,
                'timeout_load': 60,
                'timeout_simplify': 120,
                'timeout_detect': 30,
                'aggressive_mode': True
            }
        elif available_gb < 8:
            return {
                'max_faces_load': 500_000,
                'default_simplify': 1000,
                'timeout_load': 120,
                'timeout_simplify': 180,
                'timeout_detect': 60,
                'aggressive_mode': True
            }
        else:
            return {
                'max_faces_load': 1_000_000,
                'default_simplify': 2000,
                'timeout_load': 180,
                'timeout_simplify': 300,
                'timeout_detect': 120,
                'aggressive_mode': False
            }
    except Exception as e:
        logging.error(f"Falha ao verificar recursos: {e}")
        return {
            'max_faces_load': 300_000,
            'default_simplify': 800,
            'timeout_load': 90,
            'timeout_simplify': 150,
            'timeout_detect': 45,
            'aggressive_mode': True
        }

def get_file_recommendations(filepath):
    """Analisa um arquivo STL e retorna recomendações de simplificação e tempo estimado."""
    try:
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if file_size_mb > 200:
            return {
                'recommended_faces': 300,
                'estimated_time': "2-5 minutos",
                'warning': "Arquivo muito grande - processamento pode ser lento",
                'aggressive_load': True
            }
        elif file_size_mb > 100:
            return {
                'recommended_faces': 500,
                'estimated_time': "1-3 minutos",
                'warning': "Arquivo grande - usando configurações otimizadas",
                'aggressive_load': True
            }
        elif file_size_mb > 50:
            return {
                'recommended_faces': 1000,
                'estimated_time': "30-90 segundos",
                'warning': None,
                'aggressive_load': True
            }
        else:
            return {
                'recommended_faces': 2000,
                'estimated_time': "10-30 segundos",
                'warning': None,
                'aggressive_load': False
            }
    except Exception as e:
        logging.error(f"Não foi possível obter informações do arquivo: {e}")
        return {
            'recommended_faces': 1000,
            'estimated_time': "tempo variável",
            'warning': "Não foi possível analisar o arquivo",
            'aggressive_load': True
        }

@timeit
def process_single_file(args):
    """Processa um único arquivo STL com detecção de landmarks, incluindo carga, simplificação e detecção."""
    logging.info(f"=== Processando arquivo: {args.input_file} (método {args.method}) ===")
    config = check_system_resources()
    if os.path.isabs(args.input_file):
        filepath = args.input_file
        data_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
    else:
        data_dir = args.data_dir
        filename = args.input_file
        filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        logging.error(f"Arquivo não encontrado: {filepath}")
        return False
    # Analisar arquivo e recomendar parâmetros
    rec = get_file_recommendations(filepath)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logging.info(f"Arquivo {filename} - tamanho: {file_size_mb:.1f} MB")
    if rec['warning']:
        logging.warning(rec['warning'])
    # Se o usuário não especificou --simplify_faces (valor padrão), usar recomendação
    if args.simplify_faces == 5000:
        args.simplify_faces = rec['recommended_faces']
        logging.info(f"Simplificação ajustada para {args.simplify_faces} faces (recomendado)")
    logging.info(f"Tempo estimado de processamento: {rec['estimated_time']}")
    try:
        # 1. Carregamento da malha com timeout
        processor = MeshProcessor(data_dir=data_dir, cache_dir=args.cache_dir)
        logging.info(f"Carregando malha com timeout de {config['timeout_load']}s...")
        mesh = None
        try:
            with timeout_context(config['timeout_load']):
                mesh = processor.load_skull(filename, use_cache=not args.no_cache)
        except TimeoutError:
            logging.error(f"TIMEOUT: carregamento excedeu {config['timeout_load']}s")
            return False
        except Exception as e:
            logging.error(f"Erro ao carregar malha: {e}")
            return False
        if mesh is None:
            logging.error("Falha ao carregar a malha (objeto vazio).")
            return False
        initial_faces = len(mesh.faces)
        logging.info(f"Malha carregada com {len(mesh.vertices):,} vértices e {initial_faces:,} faces.")
        # Verificar se malha muito grande para carga
        if initial_faces > config['max_faces_load']:
            logging.warning(f"Malha muito grande ({initial_faces:,} faces). "
                            f"Forçando simplificação agressiva para {args.simplify_faces} faces.")
        # 2. Simplificação da malha com timeout (se aplicável)
        simplified_mesh = mesh
        if args.simplify_faces > 0 and initial_faces > args.simplify_faces:
            logging.info(f"Simplificando para {args.simplify_faces} faces (timeout {config['timeout_simplify']}s)...")
            try:
                with timeout_context(config['timeout_simplify']):
                    simplified_mesh = processor.simplify(
                        mesh, target_faces=args.simplify_faces,
                        use_cache=not args.no_cache, original_filename=filename
                    )
            except TimeoutError:
                logging.error(f"TIMEOUT: simplificação excedeu {config['timeout_simplify']}s")
                logging.warning("Usando malha original sem simplificação.")
                simplified_mesh = mesh
            except Exception as e:
                logging.error(f"Erro durante simplificação: {e}")
                logging.warning("Usando malha original (simplificação falhou).")
                simplified_mesh = mesh
            # Log de redução
            if simplified_mesh is not None:
                final_faces = len(simplified_mesh.faces)
                if final_faces < initial_faces:
                    reduction = (1 - final_faces/initial_faces) * 100
                    logging.info(f"Simplificação bem-sucedida: {final_faces:,} faces ({reduction:.1f}% redução).")
                else:
                    logging.warning("Simplificação não reduziu o número de faces.")
            else:
                logging.warning("Simplificação retornou None - usando malha original.")
                simplified_mesh = mesh
        # 3. Detecção de landmarks com timeout
        logging.info(f"Executando detecção '{args.method}' (timeout {config['timeout_detect']}s)...")
        landmarks_detected = None
        try:
            with timeout_context(config['timeout_detect']):
                if args.method == "geometric":
                    detector = GeometricDetector()
                    landmarks_detected = detector.detect(simplified_mesh)
                elif args.method == "ml":
                    detector = MLDetector(model_dir=args.model_dir)
                    landmarks_detected = detector.detect(simplified_mesh)
                else:
                    logging.error(f"Método desconhecido: {args.method}")
                    return False
        except TimeoutError:
            logging.error(f"TIMEOUT: detecção excedeu {config['timeout_detect']}s")
            return False
        except Exception as e:
            logging.error(f"Erro na detecção: {e}")
            return False
        if not landmarks_detected:
            logging.error(f"Detecção retornou resultado vazio para {filename}.")
            return False
        # Log de resumo dos resultados
        detected_count = sum(1 for coords in landmarks_detected.values() if coords is not None)
        total_count = len(landmarks_detected)
        success_rate = (detected_count / total_count) * 100
        logging.info(f"Detecção concluída: {detected_count}/{total_count} landmarks detectados ({success_rate:.1f}%).")
        if args.verbose:
            for name, coords in landmarks_detected.items():
                coord_str = (f"[{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]" 
                             if coords else "Não detectado")
                logging.info(f"  {name}: {coord_str}")
        # 4. Salvar resultados em arquivo JSON
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = f"{os.path.splitext(filename)[0]}_{args.method}_landmarks.json"
        output_path = os.path.join(args.output_dir, output_filename)
        if save_landmarks_to_json(landmarks_detected, output_path):
            logging.info(f"Resultados salvos em {output_path}")
        else:
            logging.error(f"Falha ao salvar resultados em {output_path}")
        # 5. Avaliação opcional se ground truth fornecido
        if getattr(args, 'gt_file', None):
            logging.info(f"Avaliando resultado usando ground truth: {args.gt_file}")
            gt_landmarks = load_landmarks_from_json(args.gt_file)
            if gt_landmarks:
                errors, mde = evaluate_detection(landmarks_detected, gt_landmarks)
                logging.info("=== Erros por landmark ===")
                for lname, err in errors.items():
                    if err is not None:
                        logging.info(f"  {lname}: {err:.4f} mm")
                    else:
                        logging.info(f"  {lname}: N/A")
                if mde is not None:
                    logging.info(f"**Erro Médio de Detecção (MDE)**: {mde:.4f} mm")
        # 6. Visualização opcional
        if args.visualize:
            logging.info("Gerando visualização dos resultados...")
            vis_title = f"Landmarks ({args.method.capitalize()}) - {filename}"
            vis_path_2d = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}_{args.method}_visualization.png")
            try:
                success = plot_landmarks(simplified_mesh, landmarks_detected,
                                         title=vis_title, use_3d=not args.force_2d_vis,
                                         save_path_2d=vis_path_2d)
                if success:
                    logging.info(f"Visualização 2D salva em: {vis_path_2d}")
                else:
                    logging.warning("Falha ao gerar visualização (resultado vazio).")
            except Exception as e:
                logging.warning(f"Erro ao gerar visualização: {e}")
        # Estatísticas finais de processamento
        final_faces = len(simplified_mesh.faces)
        reduction_info = ""
        if final_faces < initial_faces:
            reduction = (1 - final_faces/initial_faces) * 100
            reduction_info = f" (redução de {reduction:.1f}%)"
        logging.info("✅ Processamento finalizado.")
        logging.info(f"   Faces processadas: {final_faces:,}{reduction_info}")
        logging.info(f"   Landmarks detectados: {detected_count}/{total_count}")
        logging.info(f"   Taxa de sucesso: {success_rate:.1f}%")
        return True
    except Exception as e:
        logging.error(f"Erro inesperado no processamento: {e}", exc_info=True)
        return False

def process_batch(args):
    """Processa todos os arquivos STL em um diretório."""
    logging.info(f"=== Processando múltiplos arquivos em: {args.input_dir} ===")
    if not os.path.isdir(args.input_dir):
        logging.error(f"O caminho especificado não é um diretório: {args.input_dir}")
        return False
    files = list_stl_files(args.input_dir)
    if not files:
        logging.error(f"Nenhum arquivo STL encontrado em {args.input_dir}")
        return False
    success_all = True
    # Processar cada arquivo individualmente
    for filename in files:
        file_args = argparse.Namespace(**vars(args))  # copia todos os argumentos atuais
        file_args.input_file = filename
        file_args.data_dir = args.input_dir  # garantir que data_dir aponte para o dir batch
        file_args.mode = "single"
        logging.info(f"\n[Batch] Iniciando arquivo: {filename}")
        result = process_single_file(file_args)
        if not result:
            success_all = False
            logging.error(f"[Batch] Falha ao processar {filename}")
    logging.info("=== Processamento em lote concluído ===")
    return success_all

def setup_argument_parser():
    """Configura e retorna o argparse.ArgumentParser com subcomandos e opções."""
    parser = argparse.ArgumentParser(
        description="Sistema de Detecção de Landmarks 3D - Versão Otimizada",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python src/main.py single --method geometric -i A0001.stl --output_dir results --visualize
  python src/main.py batch --method ml -i data/skulls/ --output_dir results --simplify_faces 1000
        """
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Modo de operação (single ou batch)")
    # Subcomando: processar um único arquivo
    parser_single = subparsers.add_parser("single", help="Processar um único arquivo STL")
    parser_single.add_argument("--method", type=str, required=True, choices=["geometric", "ml"],
                               help="Método de detecção a usar")
    parser_single.add_argument("-i", "--input_file", type=str, required=True,
                               help="Nome do arquivo STL de entrada (ou caminho relativo)")
    parser_single.add_argument("--data_dir", type=str, default="./data/skulls",
                               help="Diretório base dos arquivos de entrada (usado se -i for nome simples)")
    parser_single.add_argument("--gt_file", type=str, help="Arquivo JSON de ground truth para avaliação")
    parser_single.add_argument("--output_dir", type=str, default="./results",
                               help="Diretório para salvar os resultados")
    parser_single.add_argument("--cache_dir", type=str, default="./data/cache",
                               help="Diretório de cache para malhas processadas")
    parser_single.add_argument("--no_cache", action="store_true", help="Desativar uso de cache de malhas")
    parser_single.add_argument("--simplify_faces", type=int, default=5000,
                               help="Número alvo de faces para simplificação (0 para não simplificar)")
    parser_single.add_argument("--auto", action="store_true", help="Ajustar parâmetros automaticamente ao arquivo")  # flag informativa
    parser_single.add_argument("--fast", action="store_true", help="Modo ultra-rápido (simplificação agressiva fixa)")
    parser_single.add_argument("--visualize", action="store_true", help="Gerar visualização dos resultados")
    parser_single.add_argument("--force_2d_vis", action="store_true", help="Forçar visualização 2D (mesmo com open3D disponível)")
    parser_single.add_argument("--model_dir", type=str, default="./models", help="Diretório dos modelos ML treinados")
    parser_single.add_argument("-v", "--verbose", action="store_true", help="Exibir logs detalhados (debug)")
    parser_single.set_defaults(func=process_single_file)
    # Subcomando: processar todos os arquivos de um diretório (batch)
    parser_batch = subparsers.add_parser("batch", help="Processar todos os arquivos STL em um diretório")
    parser_batch.add_argument("--method", type=str, required=True, choices=["geometric", "ml"],
                              help="Método de detecção a usar para cada arquivo")
    parser_batch.add_argument("-i", "--input_dir", type=str, required=True,
                              help="Diretório contendo arquivos .stl de entrada")
    parser_batch.add_argument("--output_dir", type=str, default="./results",
                              help="Diretório para salvar os resultados")
    parser_batch.add_argument("--cache_dir", type=str, default="./data/cache",
                              help="Diretório de cache para malhas processadas")
    parser_batch.add_argument("--no_cache", action="store_true", help="Desativar cache de malhas")
    parser_batch.add_argument("--simplify_faces", type=int, default=5000,
                              help="Número alvo de faces para simplificação (0 para não simplificar)")
    parser_batch.add_argument("--visualize", action="store_true", help="Gerar visualização para cada arquivo")
    parser_batch.add_argument("--force_2d_vis", action="store_true", help="Forçar visualização 2D (ignorar 3D)")
    parser_batch.add_argument("--model_dir", type=str, default="./models", help="Diretório dos modelos ML")
    parser_batch.add_argument("-v", "--verbose", action="store_true", help="Exibir logs detalhados")
    parser_batch.set_defaults(func=process_batch)
    return parser

def main():
    """Função principal que interpreta argumentos e invoca o modo apropriado."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    # Configurar logging básico (INFO por padrão, DEBUG se verbose)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    # Aplicar modos especiais
    if getattr(args, 'auto', False):
        logging.info("🔍 Modo automático ativado - parâmetros serão ajustados com base no arquivo.")
    if getattr(args, 'fast', False):
        logging.info("⚡ Modo ultra-rápido ativado - simplificação agressiva fixa em 300 faces.")
        args.simplify_faces = 300  # Forçar simplificação máxima se fast mode
    # Avisos/validações finais
    if args.method == "ml" and not os.path.exists(args.model_dir):
        logging.warning(f"Diretório de modelos ML não encontrado: {args.model_dir}")
    if hasattr(args, 'input_dir') and args.input_dir:
        # Se for batch, remover eventual barra final para consistência
        args.input_dir = args.input_dir.rstrip("/\\")
    if hasattr(args, 'input_file') and args.input_file and os.path.sep in args.input_file:
        # Se usuário passou caminho em --input_file no modo single, ajustar data_dir automaticamente
        args.data_dir = os.path.dirname(os.path.abspath(args.input_file))
        args.input_file = os.path.basename(args.input_file)
    if args.simplify_faces < 0:
        logging.error("--simplify_faces deve ser >= 0")
        return 1
    # Registro da configuração escolhida
    logging.info("=== Iniciando Sistema de Detecção de Landmarks ===")
    logging.info(f"Modo de operação: {args.mode}")
    logging.info(f"Método de detecção: {args.method}")
    if args.mode == "batch":
        logging.info(f"Diretório de entrada: {args.input_dir}")
        logging.info(f"{len(list_stl_files(args.input_dir))} arquivos STL serão processados.")
    else:
        logging.info(f"Arquivo de entrada: {args.input_file}")
    # Verificar recursos e logar se modo agressivo foi ativado automaticamente
    sys_config = check_system_resources()
    if sys_config.get('aggressive_mode'):
        logging.info("🔧 Modo agressivo ativado automaticamente (recursos limitados detectados).")
    # Executar função correspondente (single ou batch)
    try:
        success = args.func(args)
        if success:
            logging.info("=== ✅ PROCESSAMENTO BEM-SUCEDIDO ===")
            return 0
        else:
            logging.error("=== ❌ FALHAS NO PROCESSAMENTO ===")
            return 1
    except KeyboardInterrupt:
        logging.error("Execução interrompida pelo usuário.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
