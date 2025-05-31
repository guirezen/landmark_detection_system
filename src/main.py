# -*- coding: utf-8 -*-
"""Script principal OTIMIZADO para detecção de landmarks - Resolve timeouts com arquivos grandes."""

import argparse
import os
import logging
import time
import sys
import signal
import psutil
from contextlib import contextmanager

# Configurar imports
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
    """Context manager para timeout de operações."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operação excedeu {seconds} segundos")
    
    # Configurar handler apenas em sistemas Unix
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
    """Verifica recursos do sistema e retorna configurações recomendadas."""
    try:
        # Memória disponível
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # CPU
        cpu_count = psutil.cpu_count()
        
        logging.info(f"Recursos do sistema: {available_gb:.1f}GB RAM, {cpu_count} CPUs")
        
        # Determinar configurações baseadas nos recursos
        if available_gb < 4:
            return {
                'max_faces_load': 200000,  # Máximo para carregamento
                'default_simplify': 500,   # Simplificação muito agressiva
                'timeout_load': 60,        # 1 minuto para carregamento
                'timeout_simplify': 120,   # 2 minutos para simplificação
                'timeout_detect': 30,      # 30 segundos para detecção
                'aggressive_mode': True
            }
        elif available_gb < 8:
            return {
                'max_faces_load': 500000,
                'default_simplify': 1000,
                'timeout_load': 120,
                'timeout_simplify': 180,
                'timeout_detect': 60,
                'aggressive_mode': True
            }
        else:
            return {
                'max_faces_load': 1000000,
                'default_simplify': 2000,
                'timeout_load': 180,
                'timeout_simplify': 300,
                'timeout_detect': 120,
                'aggressive_mode': False
            }
    except:
        # Configuração conservadora se falhar
        return {
            'max_faces_load': 300000,
            'default_simplify': 800,
            'timeout_load': 90,
            'timeout_simplify': 150,
            'timeout_detect': 45,
            'aggressive_mode': True
        }

def get_file_recommendations(filepath):
    """Analisa arquivo e retorna recomendações de processamento."""
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
    except:
        return {
            'recommended_faces': 1000,
            'estimated_time': "tempo variável",
            'warning': "Não foi possível analisar arquivo",
            'aggressive_load': True
        }

@timeit
def process_single_file_optimized(args):
    """Processa arquivo único com otimizações e timeouts."""
    logging.info(f"=== Processamento Otimizado: {args.input_file} ===")
    
    # Verificar recursos do sistema
    system_config = check_system_resources()
    
    # Determinar caminho do arquivo
    input_file = args.input_file
    
    if os.path.isabs(input_file):
        filepath = input_file
        data_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
    elif os.path.sep in input_file or '/' in input_file:
        filepath = os.path.abspath(input_file)
        data_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
    else:
        filename = input_file
        data_dir = args.data_dir
        filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        logging.error(f"Arquivo não encontrado: {filepath}")
        return False

    # Analisar arquivo e dar recomendações
    file_rec = get_file_recommendations(filepath)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    logging.info(f"Arquivo: {filename} ({file_size_mb:.1f} MB)")
    if file_rec['warning']:
        logging.warning(file_rec['warning'])
    
    # Ajustar simplificação se não especificada
    if args.simplify_faces == 5000:  # Valor padrão
        recommended_faces = file_rec['recommended_faces']
        logging.info(f"Usando simplificação recomendada: {recommended_faces} faces")
        args.simplify_faces = recommended_faces
    
    logging.info(f"Tempo estimado: {file_rec['estimated_time']}")

    try:
        # 1. Carregamento com timeout
        processor = MeshProcessor(data_dir=data_dir, cache_dir=args.cache_dir)
        
        logging.info(f"Carregando malha com timeout de {system_config['timeout_load']}s...")
        
        mesh = None
        try:
            with timeout_context(system_config['timeout_load']):
                mesh = processor.load_skull(
                    filename, 
                    use_cache=not args.no_cache
                )
        except TimeoutError:
            logging.error(f"TIMEOUT no carregamento após {system_config['timeout_load']}s")
            return False
        except Exception as e:
            logging.error(f"Erro no carregamento: {e}")
            return False

        if not mesh:
            logging.error(f"Falha ao carregar: {filename}")
            return False

        initial_faces = len(mesh.faces)
        logging.info(f"Malha carregada: {len(mesh.vertices):,} vértices, {initial_faces:,} faces")

        # Verificar se a malha é muito grande
        if initial_faces > system_config['max_faces_load']:
            logging.warning(f"Malha muito grande ({initial_faces:,} faces). "
                          f"Forçando simplificação agressiva para {args.simplify_faces} faces.")

        # 2. Simplificação com timeout
        simplified_mesh = mesh
        if args.simplify_faces > 0 and initial_faces > args.simplify_faces:
            logging.info(f"Simplificando para {args.simplify_faces} faces com timeout de {system_config['timeout_simplify']}s...")
            
            try:
                with timeout_context(system_config['timeout_simplify']):
                    simplified_mesh = processor.simplify(
                        mesh,
                        target_faces=args.simplify_faces,
                        use_cache=not args.no_cache,
                        original_filename=filename
                    )
            except TimeoutError:
                logging.error(f"TIMEOUT na simplificação após {system_config['timeout_simplify']}s")
                logging.warning("Usando malha original sem simplificação")
                simplified_mesh = mesh
            except Exception as e:
                logging.error(f"Erro na simplificação: {e}")
                logging.warning("Usando malha original")
                simplified_mesh = mesh

            if simplified_mesh:
                final_faces = len(simplified_mesh.faces)
                if final_faces < initial_faces:
                    reduction = (1 - final_faces / initial_faces) * 100
                    logging.info(f"Simplificação bem-sucedida: {final_faces:,} faces ({reduction:.1f}% redução)")
                else:
                    logging.warning("Simplificação não reduziu faces - usando malha original")
            else:
                logging.warning("Simplificação retornou None - usando malha original")
                simplified_mesh = mesh

        # 3. Detecção com timeout
        landmarks_detected = None
        
        logging.info(f"Executando detecção {args.method} com timeout de {system_config['timeout_detect']}s...")
        
        try:
            with timeout_context(system_config['timeout_detect']):
                if args.method == "geometric":
                    detector = GeometricDetector()
                    landmarks_detected = detector.detect(simplified_mesh)
                elif args.method == "ml":
                    ml_detector = MLDetector(model_dir=args.model_dir)
                    landmarks_detected = ml_detector.detect(simplified_mesh)
                else:
                    logging.error(f"Método desconhecido: {args.method}")
                    return False
        except TimeoutError:
            logging.error(f"TIMEOUT na detecção após {system_config['timeout_detect']}s")
            return False
        except Exception as e:
            logging.error(f"Erro na detecção: {e}")
            return False

        if not landmarks_detected:
            logging.error(f"Falha na detecção para {filename}")
            return False

        # Log dos resultados
        detected_count = sum(1 for coords in landmarks_detected.values() if coords is not None)
        total_count = len(landmarks_detected)
        success_rate = (detected_count / total_count) * 100
        
        logging.info(f"Detecção concluída: {detected_count}/{total_count} landmarks ({success_rate:.1f}%)")
        
        if args.verbose:
            for name, coords in landmarks_detected.items():
                if coords:
                    coord_str = f"[{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]"
                else:
                    coord_str = "Não detectado"
                logging.info(f"  {name}: {coord_str}")

        # 4. Salvar resultados
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = f"{os.path.splitext(filename)[0]}_{args.method}_landmarks.json"
        output_path = os.path.join(args.output_dir, output_filename)
        
        if save_landmarks_to_json(landmarks_detected, output_path):
            logging.info(f"Resultados salvos: {output_path}")
        else:
            logging.error(f"Falha ao salvar: {output_path}")

        # 5. Avaliação contra ground truth
        if hasattr(args, 'gt_file') and args.gt_file:
            logging.info(f"Avaliando contra GT: {args.gt_file}")
            gt_landmarks = load_landmarks_from_json(args.gt_file)
            if gt_landmarks:
                errors, mde = evaluate_detection(landmarks_detected, gt_landmarks)
                
                logging.info("=== Avaliação ===")
                for name, error in errors.items():
                    if error is not None:
                        logging.info(f"  {name}: {error:.4f} mm")
                    else:
                        logging.info(f"  {name}: N/A")
                
                if mde is not None:
                    logging.info(f"MDE: {mde:.4f} mm")

        # 6. Visualização
        if args.visualize:
            logging.info("Gerando visualização...")
            vis_title = f"Landmarks ({args.method.capitalize()}) - {filename}"
            vis_save_path = os.path.join(args.output_dir, 
                                       f"{os.path.splitext(filename)[0]}_{args.method}_visualization.png")
            
            try:
                success = plot_landmarks(
                    simplified_mesh, 
                    landmarks_detected, 
                    title=vis_title,
                    use_3d=not args.force_2d_vis, 
                    save_path_2d=vis_save_path
                )
                
                if success:
                    logging.info(f"Visualização salva: {vis_save_path}")
                else:
                    logging.warning("Falha na visualização")
            except Exception as e:
                logging.warning(f"Erro na visualização: {e}")

        # Estatísticas finais
        final_faces = len(simplified_mesh.faces)
        reduction_info = ""
        if final_faces < initial_faces:
            reduction = (1 - final_faces / initial_faces) * 100
            reduction_info = f" (redução: {reduction:.1f}%)"
        
        logging.info(f"✅ Processamento concluído:")
        logging.info(f"   Faces processadas: {final_faces:,}{reduction_info}")
        logging.info(f"   Landmarks detectados: {detected_count}/{total_count}")
        logging.info(f"   Taxa de sucesso: {success_rate:.1f}%")
        
        return True

    except Exception as e:
        logging.error(f"Erro inesperado: {e}", exc_info=True)
        return False

def setup_argument_parser():
    """Configura parser com opções otimizadas."""
    parser = argparse.ArgumentParser(
        description="Sistema de Detecção de Landmarks - VERSÃO OTIMIZADA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos para arquivos grandes:

  # Ultra rápido (arquivos >200MB)
  python src/main_optimized.py single --method geometric -i A0001_clear.stl --simplify_faces 300 --fast

  # Rápido (arquivos >100MB)  
  python src/main_optimized.py single --method geometric -i A0001_clear.stl --simplify_faces 500

  # Equilibrado (arquivos 50-100MB)
  python src/main_optimized.py single --method geometric -i arquivo.stl --simplify_faces 1000

  # Com análise automática
  python src/main_optimized.py single --method geometric -i arquivo.stl --auto
        """
    )

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Modo de operação")

    # Modo single file otimizado
    parser_single = subparsers.add_parser("single", help="Processa arquivo único (otimizado)")
    
    parser_single.add_argument("--method", type=str, required=True, 
                              choices=["geometric", "ml"], 
                              help="Método de detecção")
    parser_single.add_argument("-i", "--input_file", type=str, required=True, 
                              help="Arquivo STL (nome ou caminho)")
    parser_single.add_argument("--data_dir", type=str, default="./data/skulls", 
                              help="Diretório base")
    parser_single.add_argument("--gt_file", type=str, help="Ground truth JSON")
    parser_single.add_argument("--output_dir", type=str, default="./results", 
                              help="Diretório de saída")
    parser_single.add_argument("--cache_dir", type=str, default="./data/cache", 
                              help="Diretório de cache")
    parser_single.add_argument("--no_cache", action="store_true", help="Desativar cache")
    parser_single.add_argument("--simplify_faces", type=int, default=5000, 
                              help="Faces alvo (0=sem simplificação, auto=análise automática)")
    parser_single.add_argument("--auto", action="store_true", 
                              help="Análise automática e configuração otimizada")
    parser_single.add_argument("--fast", action="store_true", 
                              help="Modo ultra-rápido (máxima simplificação)")
    parser_single.add_argument("--visualize", action="store_true", help="Gerar visualizações")
    parser_single.add_argument("--force_2d_vis", action="store_true", 
                              help="Forçar visualização 2D")
    parser_single.add_argument("--model_dir", type=str, default="./models", 
                              help="Diretório dos modelos ML")
    parser_single.add_argument("-v", "--verbose", action="store_true", 
                              help="Logging detalhado")
    parser_single.set_defaults(func=process_single_file_optimized)

    return parser

def main():
    """Função principal otimizada."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Configurar logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    # Aplicar configurações especiais
    if hasattr(args, 'auto') and args.auto:
        logging.info("🔍 Modo automático ativado - analisando arquivo...")
    
    if hasattr(args, 'fast') and args.fast:
        logging.info("⚡ Modo ultra-rápido ativado")
        args.simplify_faces = 300  # Forçar simplificação agressiva

    # Validações
    if args.method == "ml" and not os.path.exists(args.model_dir):
        logging.warning(f"Diretório ML não encontrado: {args.model_dir}")

    if args.simplify_faces < 0:
        logging.error("--simplify_faces deve ser >= 0")
        return 1

    # Log da configuração
    logging.info("=== Sistema de Detecção Otimizado ===")
    logging.info(f"Modo: {args.mode}")
    logging.info(f"Método: {args.method}")
    
    # Verificar recursos do sistema
    system_config = check_system_resources()
    if system_config['aggressive_mode']:
        logging.info("🔧 Modo agressivo ativado (recursos limitados)")

    # Executar
    try:
        success = args.func(args)
        if success:
            logging.info("=== ✅ SUCESSO ===")
            return 0
        else:
            logging.error("=== ❌ FALHA ===")
            return 1
            
    except KeyboardInterrupt:
        logging.info("Interrompido pelo usuário")
        return 1
    except Exception as e:
        logging.error(f"Erro inesperado: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())