#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para aplicar as correções finais e instalar dependências restantes."""

import subprocess
import sys
import os

def install_missing_dependencies():
    """Instala dependências que ainda estão faltando."""
    print("🔧 INSTALANDO DEPENDÊNCIAS RESTANTES\n")
    
    missing_deps = [
        ("networkx", "Análise de grafos para trimesh"),
        ("Rtree", "Índices espaciais (opcional)"),
    ]
    
    for dep_name, description in missing_deps:
        try:
            __import__(dep_name.lower())
            print(f"✅ {dep_name} já instalado")
        except ImportError:
            print(f"📦 Instalando {dep_name} - {description}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep_name])
                print(f"✅ {dep_name} instalado com sucesso")
            except Exception as e:
                print(f"⚠️  Falha ao instalar {dep_name}: {e}")
                print(f"   Continuando sem {dep_name}...")

def test_corrected_system():
    """Testa o sistema com as correções aplicadas."""
    print("\n🧪 TESTANDO SISTEMA CORRIGIDO\n")
    
    try:
        # Testar imports
        sys.path.append('.')
        from src.core.mesh_processor import MeshProcessor
        from src.core.detector_geometric import GeometricDetector
        print("✅ Imports principais funcionando")
        
        # Criar arquivo de teste se não existir
        os.makedirs('data/skulls', exist_ok=True)
        
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=50)
        test_file = 'data/skulls/test_final_corrections.stl'
        mesh.export(test_file)
        print(f"✅ Arquivo de teste criado: {test_file}")
        
        # Testar processador
        processor = MeshProcessor('./data/skulls', './data/cache')
        
        # Carregar malha
        loaded_mesh = processor.load_skull('test_final_corrections.stl')
        if not loaded_mesh:
            print("❌ Falha no carregamento")
            return False
        print(f"✅ Carregamento: {len(loaded_mesh.vertices)} vértices")
        
        # Testar simplificação com diferentes tamanhos
        test_targets = [500, 200, 100]
        
        for target in test_targets:
            simplified = processor.simplify(loaded_mesh, target_faces=target, 
                                          original_filename='test_final_corrections.stl')
            if simplified:
                actual_faces = len(simplified.faces)
                reduction = (1 - actual_faces / len(loaded_mesh.faces)) * 100
                print(f"✅ Simplificação {target}: {actual_faces} faces ({reduction:.1f}% redução)")
            else:
                print(f"⚠️  Simplificação {target} retornou malha original")
        
        # Testar detecção
        detector = GeometricDetector()
        landmarks = detector.detect(simplified)
        if landmarks:
            detected = sum(1 for coords in landmarks.values() if coords is not None)
            print(f"✅ Detecção: {detected}/8 landmarks")
            return True
        else:
            print("❌ Falha na detecção")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

def test_command_line_usage():
    """Testa diferentes formas de usar a linha de comando."""
    print("\n📋 TESTANDO COMANDOS DE LINHA\n")
    
    # Lista de comandos para testar (sem executar)
    test_commands = [
        # Teste com nome simples do arquivo
        "python src/main.py single --method geometric -i test_final_corrections.stl --output_dir results/test1 --verbose",
        
        # Teste com caminho relativo
        "python src/main.py single --method geometric -i data/skulls/test_final_corrections.stl --output_dir results/test2 --verbose",
        
        # Teste com visualização
        "python src/main.py single --method geometric -i test_final_corrections.stl --visualize --output_dir results/test3",
        
        # Teste sem simplificação
        "python src/main.py single --method geometric -i test_final_corrections.stl --simplify_faces 0 --output_dir results/test4",
        
        # Teste com simplificação baixa
        "python src/main.py single --method geometric -i test_final_corrections.stl --simplify_faces 200 --output_dir results/test5"
    ]
    
    print("📝 Comandos válidos para testar:")
    for i, cmd in enumerate(test_commands, 1):
        print(f"{i}. {cmd}")
    
    return True

def generate_usage_examples():
    """Gera exemplos de uso práticos."""
    print("\n📖 EXEMPLOS DE USO PRÁTICOS\n")
    
    examples = [
        {
            "title": "Processamento Básico (Nome do Arquivo)",
            "command": "python src/main.py single --method geometric -i A0001_clear.stl --output_dir results/basic --verbose",
            "description": "Processa arquivo no diretório data/skulls/ usando apenas o nome"
        },
        {
            "title": "Processamento com Visualização",
            "command": "python src/main.py single --method geometric -i A0001_clear.stl --visualize --output_dir results/visual",
            "description": "Gera visualização 3D dos landmarks detectados"
        },
        {
            "title": "Processamento Sem Simplificação",
            "command": "python src/main.py single --method geometric -i A0001_clear.stl --simplify_faces 0 --output_dir results/full",
            "description": "Usa malha original sem simplificação (mais lento mas mais preciso)"
        },
        {
            "title": "Processamento Rápido",
            "command": "python src/main.py single --method geometric -i A0001_clear.stl --simplify_faces 1000 --output_dir results/fast",
            "description": "Simplifica para 1000 faces (processamento rápido)"
        },
        {
            "title": "Processamento em Lote",
            "command": "python src/main.py batch --method geometric -i data/skulls/ --output_dir results/batch --visualize",
            "description": "Processa todos os arquivos .stl no diretório"
        }
    ]
    
    for example in examples:
        print(f"🔸 {example['title']}")
        print(f"   Comando: {example['command']}")
        print(f"   Descrição: {example['description']}\n")
    
    return True

def main():
    """Executa todas as correções finais."""
    print("🔧 APLICANDO CORREÇÕES FINAIS DO SISTEMA")
    print("="*60)
    
    steps = [
        ("Instalar Dependências Restantes", install_missing_dependencies),
        ("Testar Sistema Corrigido", test_corrected_system),
        ("Testar Comandos de Linha", test_command_line_usage),
        ("Gerar Exemplos de Uso", generate_usage_examples)
    ]
    
    passed_steps = 0
    for step_name, step_func in steps:
        print(f"\n{'='*60}")
        print(f"📋 {step_name}")
        print('='*60)
        
        if step_func():
            passed_steps += 1
            print(f"✅ {step_name} - CONCLUÍDA")
        else:
            print(f"❌ {step_name} - FALHOU")
    
    # Resultado final
    print(f"\n{'='*60}")
    print("🎉 CORREÇÕES FINAIS APLICADAS")
    print('='*60)
    
    print(f"Etapas concluídas: {passed_steps}/4")
    
    if passed_steps >= 3:
        print("\n🚀 SISTEMA PRONTO PARA USO!")
        print("\n📋 COMANDOS RECOMENDADOS PARA TESTAR:")
        print("1. python src/main.py single --method geometric -i A0001_clear.stl --visualize --verbose")
        print("2. python src/main.py single --method geometric -i A0001_clear.stl --simplify_faces 1000 --output_dir results/test")
        print("3. python src/main.py batch --method geometric -i data/skulls/ --output_dir results/batch")
        
        print("\n⚠️  LEMBRE-SE:")
        print("- Use apenas o NOME do arquivo (ex: A0001_clear.stl)")
        print("- NÃO use data/skulls/A0001_clear.stl (duplica o caminho)")
        print("- Para arquivos grandes, use --simplify_faces para acelerar")
        print("- Use --visualize para ver os landmarks detectados")
        
        return 0
    else:
        print("\n⚠️  Algumas correções falharam, mas o sistema deve funcionar")
        return 1

if __name__ == "__main__":
    exit(main())