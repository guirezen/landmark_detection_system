#!/usr/bin/env python3
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
        print(f"\n🔧 Testando configuração: {config_name}")
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
        print(f"\n📊 RESUMO DOS RESULTADOS")
        print("="*50)
        
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            fastest = min(successful_results, key=lambda x: x['total_time'])
            
            print(f"🏆 Configuração mais rápida: {fastest['config']}")
            print(f"   Tempo total: {fastest['total_time']:.2f}s")
            print(f"   Faces finais: {fastest['final_faces']:,}")
            print(f"   Landmarks: {fastest['landmarks_detected']}/8")
            
            print(f"\n📋 Todas as configurações:")
            for result in successful_results:
                print(f"   {result['config']}: {result['total_time']:.2f}s "
                      f"({result['final_faces']:,} faces, {result['landmarks_detected']}/8 landmarks)")
            
            # Recomendar configuração
            if fastest['total_time'] < 30:  # Menos de 30 segundos
                print(f"\n✅ SISTEMA OTIMIZADO COM SUCESSO!")
                print(f"💡 Configuração recomendada: {fastest['config']}")
                print(f"📋 Comando sugerido:")
                print(f"   python src/main.py single --method geometric -i {test_file} \\")
                print(f"          --simplify_faces {config['target_faces']} --output_dir results/optimized")
                return True
            else:
                print(f"\n⚠️  Sistema funcional mas ainda lento ({fastest['total_time']:.2f}s)")
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
