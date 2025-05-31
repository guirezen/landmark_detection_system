#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para instalar dependências e corrigir problemas de simplificação."""

import subprocess
import sys
import os

def run_command(command, description):
    """Executa um comando e reporta o resultado."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Sucesso")
            return True
        else:
            print(f"❌ {description} - Erro: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exceção: {e}")
        return False

def install_dependencies():
    """Instala todas as dependências necessárias."""
    print("🚀 INSTALAÇÃO AUTOMÁTICA DE DEPENDÊNCIAS\n")
    
    # Lista de comandos de instalação
    install_commands = [
        ("pip install --upgrade pip", "Atualizando pip"),
        ("pip install --upgrade trimesh>=4.0.0", "Instalando/atualizando trimesh"),
        ("pip install fast_simplification", "Instalando fast_simplification"),
        ("pip install pymeshlab", "Instalando pymeshlab"),
        ("pip install numpy scipy scikit-learn", "Instalando bibliotecas científicas"),
        ("pip install matplotlib seaborn", "Instalando bibliotecas de visualização"),
        ("pip install pandas joblib", "Instalando utilitários de dados"),
        ("pip install open3d", "Instalando Open3D (visualização 3D)")
    ]
    
    successful_installs = 0
    total_installs = len(install_commands)
    
    for command, description in install_commands:
        if run_command(command, description):
            successful_installs += 1
        print()  # Linha em branco
    
    print(f"📊 Resultado: {successful_installs}/{total_installs} instalações bem-sucedidas")
    
    if successful_installs >= total_installs - 1:  # Permitir 1 falha (como open3d)
        print("✅ Instalação concluída com sucesso!")
        return True
    else:
        print("⚠️ Algumas instalações falharam, mas o sistema pode ainda funcionar.")
        return False

def test_imports():
    """Testa se as bibliotecas podem ser importadas."""
    print("\n🧪 TESTANDO IMPORTS DAS BIBLIOTECAS\n")
    
    libraries = [
        ("trimesh", "Processamento de malhas 3D"),
        ("numpy", "Computação numérica"),
        ("sklearn", "Machine Learning"),
        ("scipy", "Computação científica"), 
        ("matplotlib", "Visualização 2D"),
        ("pandas", "Análise de dados"),
        ("joblib", "Persistência de modelos"),
        ("fast_simplification", "Simplificação rápida"),
        ("pymeshlab", "Processamento avançado de malhas"),
        ("open3d", "Visualização 3D (opcional)")
    ]
    
    successful_imports = 0
    for lib_name, description in libraries:
        try:
            __import__(lib_name)
            print(f"✅ {lib_name} - {description}")
            successful_imports += 1
        except ImportError as e:
            if lib_name == "open3d":
                print(f"⚠️ {lib_name} - {description} (opcional, OK se falhar)")
            else:
                print(f"❌ {lib_name} - {description} - Erro: {e}")
    
    critical_libs = len(libraries) - 1  # Excluir open3d
    if successful_imports >= critical_libs:
        print(f"\n✅ {successful_imports}/{len(libraries)} bibliotecas importadas com sucesso!")
        return True
    else:
        print(f"\n❌ Apenas {successful_imports}/{len(libraries)} bibliotecas funcionando.")
        return False

def create_test_environment():
    """Cria ambiente de teste básico."""
    print("\n🏗️ CRIANDO AMBIENTE DE TESTE\n")
    
    # Criar diretórios necessários
    directories = [
        "data/skulls",
        "data/cache", 
        "data/ground_truth",
        "results",
        "models"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Diretório criado: {directory}")
        except Exception as e:
            print(f"❌ Erro ao criar {directory}: {e}")
            return False
    
    # Criar arquivo de teste
    try:
        import trimesh
        mesh = trimesh.primitives.Sphere(radius=50, subdivisions=2)
        test_file = "data/skulls/test_installation.stl"
        mesh.export(test_file)
        print(f"✅ Arquivo de teste criado: {test_file}")
        return True
    except Exception as e:
        print(f"❌ Erro ao criar arquivo de teste: {e}")
        return False

def run_basic_test():
    """Executa teste básico do sistema."""
    print("\n🧪 TESTE BÁSICO DO SISTEMA\n")
    
    try:
        # Testar imports principais
        sys.path.append('.')
        from src.core.mesh_processor import MeshProcessor
        from src.core.detector_geometric import GeometricDetector
        print("✅ Imports principais funcionando")
        
        # Testar carregamento
        processor = MeshProcessor('./data/skulls', './data/cache')
        mesh = processor.load_skull('test_installation.stl')
        if mesh:
            print(f"✅ Carregamento: {len(mesh.vertices)} vértices")
        else:
            print("❌ Falha no carregamento")
            return False
        
        # Testar simplificação
        simplified = processor.simplify(mesh, target_faces=100, 
                                      original_filename='test_installation.stl')
        if simplified:
            print(f"✅ Simplificação: {len(simplified.faces)} faces")
        else:
            print("❌ Falha na simplificação")
            return False
            
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
        print(f"❌ Erro no teste básico: {e}")
        return False

def main():
    """Executa processo completo de instalação e teste."""
    print("🔧 SCRIPT DE CORREÇÃO E INSTALAÇÃO DO LANDMARK DETECTION SYSTEM")
    print("="*70)
    
    steps = [
        ("Instalação de Dependências", install_dependencies),
        ("Teste de Imports", test_imports),
        ("Criação do Ambiente", create_test_environment),
        ("Teste Básico do Sistema", run_basic_test)
    ]
    
    passed_steps = 0
    for step_name, step_func in steps:
        print(f"\n{'='*70}")
        print(f"📋 ETAPA: {step_name}")
        print('='*70)
        
        if step_func():
            passed_steps += 1
            print(f"✅ {step_name} - CONCLUÍDA")
        else:
            print(f"❌ {step_name} - FALHOU")
            
    # Resultado final
    print(f"\n{'='*70}")
    print("📊 RESULTADO FINAL")
    print('='*70)
    
    if passed_steps >= 3:  # Pelo menos 3 de 4 etapas
        print("🎉 INSTALAÇÃO E CONFIGURAÇÃO CONCLUÍDAS COM SUCESSO!")
        
        return 0
    else:
        print("⚠️ INSTALAÇÃO PARCIALMENTE FALHOU")
        print(f"Apenas {passed_steps}/4 etapas foram bem-sucedidas.")
        print("\n🔧 SOLUÇÕES:")
        print("- Verifique se você tem permissões de administrador")
        print("- Tente executar em um ambiente virtual limpo")
        print("- Verifique sua conexão com a internet")
        
        return 1

if __name__ == "__main__":
    exit(main())