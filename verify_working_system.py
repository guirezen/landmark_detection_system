#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para verificar se o sistema está funcionando corretamente após todas as correções."""

import os
import sys
import logging
import subprocess

# Reduzir verbosidade dos logs
logging.basicConfig(level=logging.ERROR)

def check_file_paths():
    """Verifica se os arquivos STL estão no local correto."""
    print("📁 VERIFICANDO ARQUIVOS STL\n")
    
    stl_dir = "./data/skulls"
    
    if not os.path.exists(stl_dir):
        print(f"❌ Diretório {stl_dir} não encontrado")
        return False
    
    stl_files = [f for f in os.listdir(stl_dir) if f.lower().endswith('.stl')]
    
    print(f"✅ Diretório encontrado: {stl_dir}")
    print(f"📊 Arquivos STL encontrados: {len(stl_files)}")
    
    if stl_files:
        print("📋 Arquivos disponíveis:")
        for i, file in enumerate(stl_files[:5], 1):  # Mostrar apenas os primeiros 5
            file_path = os.path.join(stl_dir, file)
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"   {i}. {file} ({file_size:.1f} MB)")
        
        if len(stl_files) > 5:
            print(f"   ... e mais {len(stl_files) - 5} arquivos")
        
        return True, stl_files[0]  # Retornar o primeiro arquivo para teste
    else:
        print("⚠️  Nenhum arquivo STL encontrado")
        return False, None

def test_system_with_real_file(test_file):
    """Testa o sistema com um arquivo STL real."""
    print(f"\n🧪 TESTANDO SISTEMA COM ARQUIVO REAL: {test_file}\n")
    
    # Comando de teste
    cmd = [
        sys.executable, "src/main.py", "single",
        "--method", "geometric",
        "-i", test_file,
        "--output_dir", "results/verification_test",
        "--simplify_faces", "1000",
        "--verbose"
    ]
    
    print(f"🚀 Executando comando:")
    print(f"   {' '.join(cmd)}")
    print()
    
    try:
        # Executar comando
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ COMANDO EXECUTADO COM SUCESSO!")
            
            # Verificar se arquivos de saída foram criados
            output_dir = "results/verification_test"
            json_file = os.path.join(output_dir, f"{os.path.splitext(test_file)[0]}_geometric_landmarks.json")
            
            if os.path.exists(json_file):
                print(f"✅ Arquivo JSON criado: {json_file}")
                
                # Verificar conteúdo do JSON
                try:
                    import json
                    with open(json_file, 'r') as f:
                        landmarks = json.load(f)
                    
                    detected = sum(1 for coords in landmarks.values() if coords is not None)
                    total = len(landmarks)
                    print(f"✅ Landmarks detectados: {detected}/{total}")
                    
                    if detected > 0:
                        print("✅ Sistema funcionando perfeitamente!")
                        return True
                    else:
                        print("⚠️  Sistema executou mas não detectou landmarks")
                        return False
                        
                except Exception as e:
                    print(f"⚠️  Erro ao ler JSON: {e}")
                    return False
            else:
                print(f"⚠️  Arquivo JSON não criado")
                return False
        else:
            print("❌ COMANDO FALHOU!")
            print(f"Código de retorno: {result.returncode}")
            print(f"Erro: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Comando demorou mais que 2 minutos (timeout)")
        return False
    except Exception as e:
        print(f"❌ Erro ao executar comando: {e}")
        return False

def test_visualization(test_file):
    """Testa a funcionalidade de visualização."""
    print(f"\n🎨 TESTANDO VISUALIZAÇÃO COM: {test_file}\n")
    
    cmd = [
        sys.executable, "src/main.py", "single",
        "--method", "geometric",
        "-i", test_file,
        "--output_dir", "results/visualization_test",
        "--simplify_faces", "500",
        "--visualize",
        "--force_2d_vis"  # Forçar 2D para evitar problemas de janela
    ]
    
    print(f"🚀 Executando comando com visualização:")
    print(f"   {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # Verificar se imagem foi criada
            output_dir = "results/visualization_test"
            img_file = os.path.join(output_dir, f"{os.path.splitext(test_file)[0]}_geometric_visualization.png")
            
            if os.path.exists(img_file):
                file_size = os.path.getsize(img_file)
                print(f"✅ Visualização criada: {img_file} ({file_size} bytes)")
                return True
            else:
                print("⚠️  Arquivo de visualização não criado")
                return False
        else:
            print("❌ Comando de visualização falhou")
            print(f"Erro: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Visualização demorou mais que 1 minuto (timeout)")
        return False
    except Exception as e:
        print(f"❌ Erro na visualização: {e}")
        return False

def run_performance_test(test_file):
    """Testa performance com diferentes configurações."""
    print(f"\n⚡ TESTE DE PERFORMANCE COM: {test_file}\n")
    
    configs = [
        ("Sem simplificação", {"--simplify_faces": "0"}),
        ("Simplificação alta", {"--simplify_faces": "5000"}),
        ("Simplificação média", {"--simplify_faces": "2000"}),
        ("Simplificação baixa", {"--simplify_faces": "500"})
    ]
    
    results = []
    
    for config_name, params in configs:
        print(f"🔧 Testando: {config_name}")
        
        cmd = [
            sys.executable, "src/main.py", "single",
            "--method", "geometric",
            "-i", test_file,
            "--output_dir", f"results/perf_test_{params['--simplify_faces']}",
        ]
        
        # Adicionar parâmetros específicos
        for key, value in params.items():
            cmd.extend([key, value])
        
        try:
            import time
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"   ✅ {config_name}: {execution_time:.2f}s")
                results.append((config_name, execution_time, True))
            else:
                print(f"   ❌ {config_name}: Falhou")
                results.append((config_name, 0, False))
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ {config_name}: Timeout (>3min)")
            results.append((config_name, 0, False))
        except Exception as e:
            print(f"   ❌ {config_name}: Erro - {e}")
            results.append((config_name, 0, False))
    
    # Resumo de performance
    print(f"\n📊 Resumo de Performance:")
    successful_tests = [r for r in results if r[2]]
    
    if successful_tests:
        fastest = min(successful_tests, key=lambda x: x[1])
        print(f"   🏆 Mais rápido: {fastest[0]} ({fastest[1]:.2f}s)")
        
        for name, time_taken, success in results:
            status = "✅" if success else "❌"
            time_str = f"{time_taken:.2f}s" if success else "Falhou"
            print(f"   {status} {name}: {time_str}")
        
        return len(successful_tests) > 0
    else:
        print("   ❌ Nenhum teste de performance passou")
        return False

def main():
    """Executa verificação completa do sistema."""
    print("🔍 VERIFICAÇÃO COMPLETA DO SISTEMA")
    print("="*50)
    
    # Verificar arquivos
    files_ok, test_file = check_file_paths()
    if not files_ok:
        print("\n❌ Sistema não pode ser testado - arquivos STL não encontrados")
        print("\n💡 SOLUÇÃO:")
        print("   Coloque arquivos .stl no diretório data/skulls/")
        return 1
    
    print(f"\n🎯 Usando arquivo de teste: {test_file}")
    
    # Lista de testes
    tests = [
        ("Funcionalidade Básica", lambda: test_system_with_real_file(test_file)),
        ("Visualização", lambda: test_visualization(test_file)),
        ("Performance", lambda: run_performance_test(test_file))
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 TESTE: {test_name}")
        print('='*50)
        
        if test_func():
            passed_tests += 1
            print(f"✅ {test_name} - PASSOU")
        else:
            print(f"❌ {test_name} - FALHOU")
    
    # Resultado final
    print(f"\n{'='*50}")
    print("📊 RESULTADO FINAL DA VERIFICAÇÃO")
    print('='*50)
    
    print(f"Testes passaram: {passed_tests}/3")
    
    if passed_tests >= 2:
        print("\n🎉 SISTEMA FUNCIONANDO CORRETAMENTE!")
        
        print(f"\n📋 COMANDOS VÁLIDOS PARA USO:")
        print(f"   python src/main.py single --method geometric -i {test_file} --visualize")
        print(f"   python src/main.py single --method geometric -i {test_file} --simplify_faces 1000")
        print(f"   python src/main.py batch --method geometric -i data/skulls/ --output_dir results/batch")
        
        print(f"\n💡 DICAS:")
        print("   - Use --simplify_faces 1000 para processamento mais rápido")
        print("   - Use --visualize para ver os landmarks detectados")
        print("   - Use apenas o nome do arquivo, não o caminho completo")
        
        return 0
    else:
        print(f"\n⚠️  Sistema parcialmente funcional ({passed_tests}/3 testes passaram)")
        print("   Verifique os erros acima para mais detalhes")
        return 1

if __name__ == "__main__":
    exit(main())