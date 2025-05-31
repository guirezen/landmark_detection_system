# -*- coding: utf-8 -*-
"""Módulo para visualização de malhas 3D e landmarks detectados."""

import numpy as np
import matplotlib.pyplot as plt
import logging
import trimesh
import os

# Tentar importar open3d, mas não falhar se não estiver disponível
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    logging.info("Open3D disponível - visualização 3D interativa habilitada")
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.info("Open3D não disponível - usando apenas visualização 2D")

def plot_landmarks_2d(mesh, landmarks_dict, title="Landmarks Detectados (2D)", save_path=None):
    """Plota a malha e landmarks usando projeções 2D (XY, XZ, YZ)."""
    if not isinstance(mesh, trimesh.Trimesh):
        logging.error("Entrada para plot_landmarks_2d não é um Trimesh válido")
        return False

    try:
        vertices = mesh.vertices
        landmark_coords = []
        landmark_names = []
        
        # Extrair coordenadas e nomes dos landmarks válidos
        if landmarks_dict:
            for name, coords in landmarks_dict.items():
                if coords is not None:
                    try:
                        coord_array = np.asarray(coords)
                        if coord_array.shape == (3,) and np.isfinite(coord_array).all():
                            landmark_coords.append(coord_array)
                            landmark_names.append(name)
                    except:
                        logging.warning(f"Coordenada inválida para landmark {name}: {coords}")
                        continue
        
        landmark_coords = np.array(landmark_coords) if landmark_coords else np.empty((0, 3))

        # Criar subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=16)

        # Projeção XY (Vista Superior)
        axs[0].scatter(vertices[:, 0], vertices[:, 1], s=0.5, alpha=0.3, 
                      c='lightblue', label="Vértices")
        if len(landmark_coords) > 0:
            axs[0].scatter(landmark_coords[:, 0], landmark_coords[:, 1], 
                          c='red', s=80, marker='o', edgecolors='black', 
                          linewidth=1, label="Landmarks", zorder=5)
            # Adicionar labels dos landmarks
            for i, name in enumerate(landmark_names):
                axs[0].annotate(name, (landmark_coords[i, 0], landmark_coords[i, 1]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color='darkred',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        axs[0].set_xlabel("X (mm)")
        axs[0].set_ylabel("Y (mm)")
        axs[0].set_title("Vista Superior (XY)")
        axs[0].set_aspect('equal', adjustable='box')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()

        # Projeção XZ (Vista Frontal)
        axs[1].scatter(vertices[:, 0], vertices[:, 2], s=0.5, alpha=0.3, 
                      c='lightgreen', label="Vértices")
        if len(landmark_coords) > 0:
            axs[1].scatter(landmark_coords[:, 0], landmark_coords[:, 2], 
                          c='red', s=80, marker='o', edgecolors='black', 
                          linewidth=1, label="Landmarks", zorder=5)
            for i, name in enumerate(landmark_names):
                axs[1].annotate(name, (landmark_coords[i, 0], landmark_coords[i, 2]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color='darkred',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        axs[1].set_xlabel("X (mm)")
        axs[1].set_ylabel("Z (mm)")
        axs[1].set_title("Vista Frontal (XZ)")
        axs[1].set_aspect('equal', adjustable='box')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()

        # Projeção YZ (Vista Lateral)
        axs[2].scatter(vertices[:, 1], vertices[:, 2], s=0.5, alpha=0.3, 
                      c='lightcoral', label="Vértices")
        if len(landmark_coords) > 0:
            axs[2].scatter(landmark_coords[:, 1], landmark_coords[:, 2], 
                          c='red', s=80, marker='o', edgecolors='black', 
                          linewidth=1, label="Landmarks", zorder=5)
            for i, name in enumerate(landmark_names):
                axs[2].annotate(name, (landmark_coords[i, 1], landmark_coords[i, 2]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color='darkred',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        axs[2].set_xlabel("Y (mm)")
        axs[2].set_ylabel("Z (mm)")
        axs[2].set_title("Vista Lateral (YZ)")
        axs[2].set_aspect('equal', adjustable='box')
        axs[2].grid(True, alpha=0.3)
        axs[2].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Salvar se caminho fornecido
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Visualização 2D salva em: {save_path}")
            except Exception as e:
                logging.error(f"Erro ao salvar visualização 2D: {e}")
        
        # Fechar figura para liberar memória
        plt.close(fig)
        return True

    except Exception as e:
        logging.error(f"Erro durante plotagem 2D: {e}")
        return False

def plot_landmarks_3d_o3d(mesh, landmarks_dict, title="Landmarks 3D", show_window=True):
    """Visualiza malha e landmarks em 3D usando Open3D."""
    if not OPEN3D_AVAILABLE:
        logging.warning("Open3D não disponível - visualização 3D não possível")
        return False

    if not isinstance(mesh, trimesh.Trimesh):
        logging.error("Entrada para plot_landmarks_3d_o3d não é um Trimesh válido")
        return False

    try:
        # Converter Trimesh para Open3D
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh.vertices),
            triangles=o3d.utility.Vector3iVector(mesh.faces)
        )
        
        # Computar normais e pintar a malha
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Cinza claro

        geometries = [o3d_mesh]

        # Adicionar landmarks como esferas
        if landmarks_dict:
            # Calcular tamanho apropriado para landmarks baseado na malha
            mesh_scale = np.max(mesh.extents)
            sphere_radius = mesh_scale * 0.02  # 2% do tamanho da malha
            
            colors = [
                [1.0, 0.0, 0.0],  # Vermelho
                [0.0, 1.0, 0.0],  # Verde
                [0.0, 0.0, 1.0],  # Azul
                [1.0, 1.0, 0.0],  # Amarelo
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Ciano
                [1.0, 0.5, 0.0],  # Laranja
                [0.5, 0.0, 1.0],  # Roxo
            ]
            
            color_idx = 0
            for name, coords in landmarks_dict.items():
                if coords is not None:
                    try:
                        coord_array = np.asarray(coords)
                        if coord_array.shape == (3,) and np.isfinite(coord_array).all():
                            # Criar esfera para o landmark
                            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                            sphere.translate(coord_array)
                            sphere.paint_uniform_color(colors[color_idx % len(colors)])
                            geometries.append(sphere)
                            color_idx += 1
                    except Exception as e:
                        logging.warning(f"Erro ao criar esfera para {name}: {e}")

        # Visualizar
        if show_window:
            logging.info("Abrindo janela de visualização 3D...")
            try:
                # Configurar visualizador
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name=title, width=1024, height=768)
                
                for geom in geometries:
                    vis.add_geometry(geom)
                
                # Configurar vista
                vis.get_render_option().point_size = 1.0
                vis.get_render_option().line_width = 1.0
                
                # Executar visualizador
                vis.run()
                vis.destroy_window()
                
                logging.info("Janela de visualização 3D fechada")
                return True
                
            except Exception as e:
                logging.error(f"Erro na visualização 3D: {e}")
                return False
        else:
            # Modo sem janela (para captura de tela)
            logging.info("Visualização 3D em modo headless")
            return True

    except Exception as e:
        logging.error(f"Erro durante criação da visualização 3D: {e}")
        return False

def save_screenshot_3d(mesh, landmarks_dict, save_path, image_size=(1024, 768)):
    """Salva screenshot da visualização 3D sem mostrar janela."""
    if not OPEN3D_AVAILABLE:
        logging.warning("Open3D não disponível - screenshot 3D não possível")
        return False

    try:
        # Converter para Open3D
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh.vertices),
            triangles=o3d.utility.Vector3iVector(mesh.faces)
        )
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])

        geometries = [o3d_mesh]

        # Adicionar landmarks
        if landmarks_dict:
            mesh_scale = np.max(mesh.extents)
            sphere_radius = mesh_scale * 0.02
            
            for name, coords in landmarks_dict.items():
                if coords is not None:
                    try:
                        coord_array = np.asarray(coords)
                        if coord_array.shape == (3,) and np.isfinite(coord_array).all():
                            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                            sphere.translate(coord_array)
                            sphere.paint_uniform_color([1.0, 0.0, 0.0])
                            geometries.append(sphere)
                    except:
                        continue

        # Criar visualizador off-screen
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=image_size[0], height=image_size[1])
        
        for geom in geometries:
            vis.add_geometry(geom)
        
        # Renderizar e salvar
        vis.poll_events()
        vis.update_renderer()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vis.capture_screen_image(save_path, do_render=True)
        vis.destroy_window()
        
        logging.info(f"Screenshot 3D salvo em: {save_path}")
        return True

    except Exception as e:
        logging.error(f"Erro ao salvar screenshot 3D: {e}")
        return False

def plot_landmarks(mesh, landmarks_dict, title="Landmarks Detectados", 
                  use_3d=True, save_path_2d=None, save_path_3d=None):
    """Função principal para plotar landmarks.
    
    Args:
        mesh: Malha 3D
        landmarks_dict: Dicionário de landmarks
        title: Título da visualização
        use_3d: Se deve tentar visualização 3D
        save_path_2d: Caminho para salvar imagem 2D
        save_path_3d: Caminho para salvar screenshot 3D
    """
    if not isinstance(mesh, trimesh.Trimesh):
        logging.error("Malha fornecida não é um Trimesh válido")
        return False

    visualization_success = False

    # Tentar visualização 3D se solicitado e disponível
    if use_3d and OPEN3D_AVAILABLE:
        logging.info("Tentando visualização 3D interativa...")
        success_3d = plot_landmarks_3d_o3d(mesh, landmarks_dict, title=f"{title} (3D)")
        if success_3d:
            visualization_success = True
        
        # Salvar screenshot 3D se caminho fornecido
        if save_path_3d:
            save_screenshot_3d(mesh, landmarks_dict, save_path_3d)

    # Sempre gerar visualização 2D (como fallback ou complemento)
    if not use_3d or not OPEN3D_AVAILABLE:
        if not OPEN3D_AVAILABLE:
            logging.info("Open3D não disponível - usando visualização 2D")
        else:
            logging.info("Gerando visualização 2D")

    success_2d = plot_landmarks_2d(mesh, landmarks_dict, 
                                  title=f"{title} (2D)", 
                                  save_path=save_path_2d)
    if success_2d:
        visualization_success = True

    if not visualization_success:
        logging.error("Falha em todos os métodos de visualização")
        return False

    return True

def plot_error_distribution(errors_dict, title="Distribuição de Erros", save_path=None):
    """Plota distribuição de erros de detecção para cada landmark.
    
    Args:
        errors_dict: Dicionário {landmark_name: [list of errors]}
        title: Título do gráfico
        save_path: Caminho para salvar
    """
    try:
        # Filtrar landmarks com dados válidos
        valid_data = {name: errors for name, errors in errors_dict.items() 
                     if errors and len(errors) > 0}
        
        if not valid_data:
            logging.warning("Nenhum dado de erro válido para plotar")
            return False

        # Criar boxplot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        landmarks = list(valid_data.keys())
        errors_lists = [valid_data[landmark] for landmark in landmarks]
        
        bp = ax.boxplot(errors_lists, labels=landmarks, patch_artist=True)
        
        # Colorir boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(landmarks)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Erro de Detecção (mm)")
        ax.set_xlabel("Landmarks")
        ax.grid(True, alpha=0.3)
        
        # Rotacionar labels se necessário
        if len(landmarks) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Gráfico de distribuição salvo em: {save_path}")
        
        plt.close()
        return True
        
    except Exception as e:
        logging.error(f"Erro ao plotar distribuição de erros: {e}")
        return False

# Exemplo de uso e teste
if __name__ == "__main__":
    import sys
    import os
    
    # Adicionar path para imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("=== Testando Visualização ===")

    # Criar malha de teste
    test_mesh = trimesh.primitives.Sphere(radius=50, subdivisions=3)
    test_mesh.vertices += [0, 0, 50]

    # Landmarks de teste
    test_landmarks = {
        "Glabela": [0, 50, 50],
        "Bregma": [0, 0, 100],
        "Euryon_Direito": [50, 0, 50],
        "Landmark_Faltando": None
    }

    print(f"Malha de teste: {len(test_mesh.vertices)} vértices, {len(test_mesh.faces)} faces")
    print(f"Landmarks de teste: {len([x for x in test_landmarks.values() if x is not None])} válidos")

    # Testar visualização 2D
    print("\n--- Teste Visualização 2D ---")
    success_2d = plot_landmarks_2d(test_mesh, test_landmarks, 
                                  title="Teste Landmarks 2D",
                                  save_path="./test_visualization_2d.png")
    print(f"Visualização 2D: {'Sucesso' if success_2d else 'Falha'}")

    # Testar visualização principal
    print("\n--- Teste Visualização Principal ---")
    success_main = plot_landmarks(test_mesh, test_landmarks,
                                 title="Teste Visualização Completa",
                                 save_path_2d="./test_complete_2d.png")
    print(f"Visualização principal: {'Sucesso' if success_main else 'Falha'}")

    # Testar gráfico de distribuição de erros
    print("\n--- Teste Distribuição de Erros ---")
    test_errors = {
        "Glabela": [1.2, 2.1, 0.8, 1.5, 2.0],
        "Nasion": [0.5, 1.0, 1.2, 0.8],
        "Bregma": [2.5, 3.0, 1.8, 2.2, 1.9, 2.1]
    }
    
    success_error = plot_error_distribution(test_errors,
                                           title="Teste Distribuição de Erros",
                                           save_path="./test_error_distribution.png")
    print(f"Gráfico de erros: {'Sucesso' if success_error else 'Falha'}")

    print(f"\nOpen3D disponível: {OPEN3D_AVAILABLE}")
    if OPEN3D_AVAILABLE:
        print("Visualização 3D interativa disponível")
    else:
        print("Apenas visualização 2D disponível")