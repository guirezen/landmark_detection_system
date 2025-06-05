#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Otimizações de performance para o sistema de detecção de landmarks."""

import os
import sys
import time
import logging
import trimesh
import numpy as np
from pathlib import Path

def apply_aggressive_simplification_fix():
    """Aplica correções para melhorar performance com arquivos grandes."""
    
    print("🚀 APLICANDO OTIMIZAÇÕES DE PERFORMANCE")
    print("="*60)
    
    # 1. Corrigir mesh_processor.py para ser mais agressivo
    mesh_processor_content = '''# -*- coding: utf-8 -*-
"""Módulo otimizado para carregar e simplificar malhas 3D - VERSÃO PERFORMANCE."""

import trimesh
import numpy as np
import os
import hashlib
import pickle
import logging
from scipy.spatial import ConvexHull

class MeshProcessor:
    """Processa malhas 3D com otimizações para arquivos grandes."""

    def __init__(self, data_dir="./data/skulls", cache_dir="./data/cache"):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logging.info(f"Processador otimizado inicializado. Cache em: {self.cache_dir}")

    def _get_cache_filename(self, original_filename, params):
        """Gera nome de cache único."""
        hasher = hashlib.md5()
        hasher.update(original_filename.encode('utf-8'))
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        hasher.update(param_str.encode('utf-8'))
        return os.path.join(self.cache_dir, f"{hasher.hexdigest()}.pkl")

    def _save_to_cache(self, mesh, cache_filepath):
        """Salva malha no cache."""
        try:
            with open(cache_filepath, 'wb') as f:
                mesh_data = {
                    'vertices': mesh.vertices.copy(),
                    'faces': mesh.faces.copy(),
                    'metadata': {
                        'vertex_count': len(mesh.vertices),
                        'face_count': len(mesh.faces)
                    }
                }
                pickle.dump(mesh_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Malha salva no cache: {cache_filepath}")
        except Exception as e:
            logging.error(f"Erro ao salvar cache {cache_filepath}: {e}")

    def _load_from_cache(self, cache_filepath):
        """Carrega malha do cache."""
        if not os.path.exists(cache_filepath):
            return None
            
        try:
            with open(cache_filepath, 'rb') as f:
                mesh_data = pickle.load(f)
            
            if not all(key in mesh_data for key in ['vertices', 'faces']):
                logging.warning(f"Cache corrompido: {cache_filepath}")
                return None
            
            mesh = trimesh.Trimesh(
                vertices=mesh_data['vertices'], 
                faces=mesh_data['faces'],
                validate=False,  # OTIMIZAÇÃO: pular validação no cache
                process=False    # OTIMIZAÇÃO: pular processamento
            )
            logging.info(f"Malha carregada do cache: {cache_filepath}")
            return mesh
            
        except Exception as e:
            logging.error(f"Erro ao carregar cache {cache_filepath}: {e}")
            try:
                os.remove(cache_filepath)
            except OSError:
                pass
            return None

    def load_skull(self, filename, use_cache=True, aggressive_load=True):
        """Carrega modelo STL com otimizações para arquivos grandes."""
        filepath = os.path.join(self.data_dir, filename)
        cache_params = {'operation': 'load', 'version': '2.0_optimized'}
        cache_filepath = self._get_cache_filename(filename, cache_params)

        # Cache primeiro
        if use_cache:
            mesh = self._load_from_cache(cache_filepath)
            if mesh is not None:
                return mesh

        if not os.path.exists(filepath):
            logging.error(f"Arquivo não encontrado: {filepath}")
            return None

        try:
            # OTIMIZAÇÃO: Carregar sem validação para arquivos grandes
            if aggressive_load:
                mesh = trimesh.load_mesh(filepath, force='mesh', validate=False, process=False)
            else:
                mesh = trimesh.load_mesh(filepath, force='mesh')
            
            logging.info(f"STL carregado: {filepath} "
                        f"(Vértices: {len(mesh.vertices)}, Faces: {len(mesh.faces)})")

            # Lidar com Scene
            if isinstance(mesh, trimesh.Scene):
                logging.warning(f"Arquivo {filename} é Scene, concatenando...")
                try:
                    mesh = mesh.dump(concatenate=True)
                except:
                    geometries = list(mesh.geometry.values())
                    if geometries:
                        mesh = geometries[0]
                    else:
                        logging.error(f"Falha ao extrair geometria de {filename}")
                        return None

            if not isinstance(mesh, trimesh.Trimesh):
                logging.error(f"Objeto não é Trimesh válido: {filename}")
                return None

            # Pré-processamento MÍNIMO para arquivos grandes
            if len(mesh.faces) > 50000:  # Arquivo grande
                logging.info("Arquivo grande detectado - pré-processamento mínimo")
                try:
                    # Apenas remover faces degeneradas críticas
                    mesh.remove_degenerate_faces()
                except:
                    logging.warning("Falha no pré-processamento mínimo")
            else:
                # Pré-processamento completo para arquivos pequenos
                try:
                    mesh.fix_normals()
                    if not mesh.is_watertight:
                        mesh.fill_holes()
                    mesh.remove_duplicate_faces()
                    mesh.remove_degenerate_faces()
                except Exception as e:
                    logging.warning(f"Erro no pré-processamento: {e}")

            # Salvar no cache
            if use_cache:
                self._save_to_cache(mesh, cache_filepath)

            return mesh

        except Exception as e:
            logging.error(f"Erro ao carregar {filepath}: {e}")
            return None

    def _ultra_fast_simplification(self, mesh, target_faces):
        """Simplificação ultra-rápida para casos extremos."""
        try:
            current_faces = len(mesh.faces)
            
            # Se já é pequeno, não simplificar
            if current_faces <= target_faces:
                return mesh
            
            # Para reduções muito drásticas, fazer em etapas
            if target_faces < current_faces * 0.1:  # Redução > 90%
                logging.info("Redução drástica detectada - simplificação em etapas")
                
                # Etapa 1: Reduzir para 20% do original
                intermediate_target = int(current_faces * 0.2)
                try:
                    intermediate = self._try_fast_simplification(mesh, intermediate_target)
                    if intermediate and len(intermediate.faces) < current_faces:
                        mesh = intermediate
                except:
                    pass
                
                # Etapa 2: Reduzir para o alvo final
                try:
                    final = self._try_fast_simplification(mesh, target_faces)
                    if final and len(final.faces) > 0:
                        return final
                except:
                    pass
            
            # Tentativa direta
            return self._try_fast_simplification(mesh, target_faces)
            
        except Exception as e:
            logging.error(f"Erro na simplificação ultra-rápida: {e}")
            return None

    def _try_fast_simplification(self, mesh, target_faces):
        """Tenta simplificação rápida com fast_simplification."""
        try:
            # Usar fast_simplification diretamente se disponível
            import fast_simplification
            
            # Preparar arrays
            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.uint32)
            
            # Calcular ratio
            ratio = target_faces / len(mesh.faces)
            ratio = max(0.01, min(0.99, ratio))  # Limitar entre 1% e 99%
            
            # Simplificar
            simplified_vertices, simplified_faces = fast_simplification.simplify(
                vertices, faces, target_count=target_faces
            )
            
            # Criar nova malha
            simplified_mesh = trimesh.Trimesh(
                vertices=simplified_vertices, 
                faces=simplified_faces,
                validate=False,
                process=False
            )
            
            logging.info(f"Fast simplification: {len(mesh.faces)} → {len(simplified_faces)} faces")
            return simplified_mesh
            
        except ImportError:
            # Fallback para método Trimesh
            try:
                ratio = 1.0 - (target_faces / len(mesh.faces))
                ratio = max(0.01, min(0.99, ratio))
                return mesh.simplify_quadric_decimation(ratio)
            except:
                return None
        except Exception as e:
            logging.warning(f"Erro na simplificação rápida: {e}")
            return None

    def _extreme_simplification_fallback(self, mesh, target_faces):
        """Fallback extremo usando decimação por clustering."""
        try:
            from sklearn.cluster import KMeans
            
            n_vertices = len(mesh.vertices)
            n_clusters = min(target_faces * 2, n_vertices // 2)
            
            if n_clusters < 10:
                return None
            
            # Clustering de vértices
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            clusters = kmeans.fit_predict(mesh.vertices)
            
            # Usar centroides como novos vértices
            new_vertices = kmeans.cluster_centers_
            
            # Triangulação 2D projetada
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            vertices_2d = pca.fit_transform(new_vertices)
            
            from scipy.spatial import Delaunay
            tri = Delaunay(vertices_2d)
            
            # Filtrar faces válidas
            valid_faces = []
            for face in tri.simplices:
                if len(np.unique(face)) == 3:
                    valid_faces.append(face)
            
            if len(valid_faces) == 0:
                return None
            
            simplified_mesh = trimesh.Trimesh(
                vertices=new_vertices, 
                faces=np.array(valid_faces),
                validate=False,
                process=False
            )
            
            logging.info(f"Clustering fallback: {len(mesh.faces)} → {len(simplified_mesh.faces)} faces")
            return simplified_mesh
            
        except Exception as e:
            logging.warning(f"Erro no fallback extremo: {e}")
            return None

    def simplify(self, mesh, target_faces, use_cache=True, original_filename="unknown"):
        """Simplifica malha com otimizações para performance."""
        if not isinstance(mesh, trimesh.Trimesh):
            logging.error("Input não é Trimesh válido")
            return None

        current_faces = len(mesh.faces)
        
        if current_faces <= target_faces:
            logging.info(f"Malha já tem {current_faces} faces (<= {target_faces})")
            return mesh

        # Cache
        cache_params = {
            'operation': 'simplify_optimized', 
            'target_faces': target_faces,
            'original_faces': current_faces,
            'version': '2.0'
        }
        cache_filepath = self._get_cache_filename(original_filename, cache_params)

        if use_cache:
            cached_mesh = self._load_from_cache(cache_filepath)
            if cached_mesh is not None:
                return cached_mesh

        logging.info(f"Simplificando {current_faces} → {target_faces} faces...")
        
        # Estratégias em ordem de preferência
        strategies = [
            ("Ultra Fast Simplification", self._ultra_fast_simplification),
            ("Extreme Fallback", self._extreme_simplification_fallback)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logging.info(f"Tentando: {strategy_name}")
                simplified_mesh = strategy_func(mesh, target_faces)
                
                if simplified_mesh and len(simplified_mesh.faces) > 0:
                    actual_faces = len(simplified_mesh.faces)
                    reduction = (current_faces - actual_faces) / current_faces * 100
                    logging.info(f"✅ {strategy_name}: {actual_faces} faces ({reduction:.1f}% redução)")
                    
                    # Salvar cache
                    if use_cache:
                        self._save_to_cache(simplified_mesh, cache_filepath)
                    
                    return simplified_mesh
                else:
                    logging.warning(f"❌ {strategy_name}: resultado inválido")
                    
            except Exception as e:
                logging.warning(f"❌ {strategy_name}: {e}")
                continue
        
        # Se tudo falhou, retornar original com aviso
        logging.warning("⚠️ Todas as simplificações falharam - usando malha original")
        return mesh

    def get_mesh_stats(self, mesh):
        """Retorna estatísticas da malha."""
        if not isinstance(mesh, trimesh.Trimesh):
            return None
            
        return {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'edges': len(mesh.edges) if hasattr(mesh, 'edges') else 0,
            'volume': mesh.volume if mesh.is_volume else None,
            'surface_area': mesh.area,
            'bounds': mesh.bounds.tolist(),
            'extents': mesh.extents.tolist(),
            'centroid': mesh.centroid.tolist()
        }
'''

    # Salvar o arquivo otimizado
    mesh_processor_path = "src/core/mesh_processor_optimized.py"
    try:
        os.makedirs(os.path.dirname(mesh_processor_path), exist_ok=True)
        with open(mesh_processor_path, 'w', encoding='utf-8') as f:
            f.write(mesh_processor_content)
        print(f"✅ Processador otimizado salvo em: {mesh_processor_path}")
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar processador otimizado: {e}")
        return False

def create_performance_test_script():
    """Cria script de teste de performance específico."""
    
    test_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teste de performance otimizado para arquivos grandes."""

import os
import sys
import time
import logging

# Configurar logging para ser menos verboso
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

sys.path.append('.')

def test_large_file_performance():
    """Testa performance com arquivo grande real."""
    print("🧪 TESTE DE PERFORMANCE OTIMIZADO")
    print("="*50)
    
    from src.core.mesh_processor_optimized import MeshProcessor
    from src.core.detector_geometric import GeometricDetector
    
    # Configuração
    DATA_DIR = "./data/skulls"
    CACHE_DIR = "./data/cache"
    
    # Encontrar arquivo de teste
    stl_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.stl')]
    
    if not stl_files:
        print("❌ Nenhum arquivo STL encontrado")
        return False
    
    # Usar o maior arquivo disponível
    test_file = None
    max_size = 0
    
    for file in stl_files:
        file_path = os.path.join(DATA_DIR, file)
        size = os.path.getsize(file_path)
        if size > max_size:
            max_size = size
            test_file = file
    
    print(f"🎯 Testando com: {test_file} ({max_size/(1024*1024):.1f} MB)")
    
    # Teste com diferentes configurações
    test_configs = [
        ("Ultra Agressivo", {"target_faces": 500, "aggressive_load": True}),
        ("Agressivo", {"target_faces": 1000, "aggressive_load": True}),
        ("Moderado", {"target_faces": 2000, "aggressive_load": True}),
        ("Conservador", {"target_faces": 5000, "aggressive_load": False})
    ]
    
    processor = MeshProcessor(data_dir=DATA_DIR, cache_dir=CACHE_DIR)
    detector = GeometricDetector()
    
    results = []
    
    for config_name, config in test_configs:
        print(f"\\n🔧 Testando configuração: {config_name}")
        print(f"   Target faces: {config['target_faces']}")
        
        try:
            # Carregamento
            start_time = time.time()
            mesh = processor.load_skull(test_file, aggressive_load=config['aggressive_load'])
            load_time = time.time() - start_time
            
            if not mesh:
                print(f"   ❌ Falha no carregamento")
                continue
            
            print(f"   ✅ Carregamento: {load_time:.2f}s ({len(mesh.vertices):,} vértices)")
            
            # Simplificação
            start_time = time.time()
            simplified = processor.simplify(
                mesh, 
                target_faces=config['target_faces'],
                original_filename=test_file
            )
            simplify_time = time.time() - start_time
            
            if not simplified:
                print(f"   ❌ Falha na simplificação")
                continue
            
            reduction = (1 - len(simplified.faces) / len(mesh.faces)) * 100
            print(f"   ✅ Simplificação: {simplify_time:.2f}s ({len(simplified.faces):,} faces, {reduction:.1f}% redução)")
            
            # Detecção
            start_time = time.time()
            landmarks = detector.detect(simplified)
            detect_time = time.time() - start_time
            
            if landmarks:
                detected_count = sum(1 for coords in landmarks.values() if coords is not None)
                print(f"   ✅ Detecção: {detect_time:.2f}s ({detected_count}/8 landmarks)")
                
                total_time = load_time + simplify_time + detect_time
                results.append({
                    'config': config_name,
                    'total_time': total_time,
                    'load_time': load_time,
                    'simplify_time': simplify_time,
                    'detect_time': detect_time,
                    'final_faces': len(simplified.faces),
                    'landmarks_detected': detected_count,
                    'success': True
                })
                
                print(f"   🏁 Tempo total: {total_time:.2f}s")
                
            else:
                print(f"   ❌ Falha na detecção")
                
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            continue
    
    # Resumo dos resultados
    if results:
        print(f"\\n📊 RESUMO DOS RESULTADOS")
        print("="*50)
        
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            fastest = min(successful_results, key=lambda x: x['total_time'])
            
            print(f"🏆 Configuração mais rápida: {fastest['config']}")
            print(f"   Tempo total: {fastest['total_time']:.2f}s")
            print(f"   Faces finais: {fastest['final_faces']:,}")
            print(f"   Landmarks: {fastest['landmarks_detected']}/8")
            
            print(f"\\n📋 Todas as configurações:")
            for result in successful_results:
                print(f"   {result['config']}: {result['total_time']:.2f}s "
                      f"({result['final_faces']:,} faces, {result['landmarks_detected']}/8 landmarks)")
            
            # Recomendar configuração
            if fastest['total_time'] < 30:  # Menos de 30 segundos
                print(f"\\n✅ SISTEMA OTIMIZADO COM SUCESSO!")
                print(f"💡 Configuração recomendada: {fastest['config']}")
                print(f"📋 Comando sugerido:")
                print(f"   python src/main.py single --method geometric -i {test_file} \\\\")
                print(f"          --simplify_faces {config['target_faces']} --output_dir results/optimized")
                return True
            else:
                print(f"\\n⚠️  Sistema funcional mas ainda lento ({fastest['total_time']:.2f}s)")
                print(f"💡 Considere usar configuração: {fastest['config']}")
                return True
        else:
            print("❌ Nenhuma configuração teve sucesso completo")
            return False
    else:
        print("❌ Nenhum teste foi bem-sucedido")
        return False

if __name__ == "__main__":
    success = test_large_file_performance()
    exit(0 if success else 1)
'''
    
    test_script_path = "test_performance_optimized.py"
    try:
        with open(test_script_path, 'w', encoding='utf-8') as f:
            f.write(test_script)
        print(f"✅ Script de teste criado: {test_script_path}")
        return True
    except Exception as e:
        print(f"❌ Erro ao criar script de teste: {e}")
        return False

def main():
    """Aplica todas as otimizações de performance."""
    print("🔧 APLICANDO OTIMIZAÇÕES DE PERFORMANCE PARA ARQUIVOS GRANDES")
    print("="*70)
    
    steps = [
        ("Criar MeshProcessor Otimizado", apply_aggressive_simplification_fix),
        ("Criar Script de Teste de Performance", create_performance_test_script)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        print(f"\\n{'='*70}")
        print(f"📋 {step_name}")
        print('='*70)
        
        if step_func():
            success_count += 1
            print(f"✅ {step_name} - CONCLUÍDO")
        else:
            print(f"❌ {step_name} - FALHOU")
    
    print(f"\\n{'='*70}")
    print("📊 RESULTADO DAS OTIMIZAÇÕES")
    print('='*70)
    
    print(f"Etapas concluídas: {success_count}/2")
    
    if success_count >= 1:
        print("\\n🚀 OTIMIZAÇÕES APLICADAS!")
        print("\\n📋 PRÓXIMOS PASSOS:")
        print("1. Execute: python test_performance_optimized.py")
        print("2. Se bem-sucedido, substitua mesh_processor.py pelo otimizado")
        print("3. Teste com: python src/main.py single --method geometric -i A0001_clear.stl --simplify_faces 500")
        
        print("\\n💡 CONFIGURAÇÕES RECOMENDADAS PARA ARQUIVOS GRANDES:")
        print("   --simplify_faces 500   (ultra rápido)")
        print("   --simplify_faces 1000  (rápido)")
        print("   --simplify_faces 2000  (equilibrado)")
        
        return 0
    else:
        print("\\n❌ Falha nas otimizações")
        return 1

if __name__ == "__main__":
    exit(main())