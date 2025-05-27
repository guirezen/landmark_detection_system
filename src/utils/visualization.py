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
    logging.info("Biblioteca Open3D encontrada. Visualização 3D interativa estará disponível.")
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.warning("Biblioteca Open3D não encontrada. Visualização 3D interativa desabilitada. Usando Matplotlib para projeções 2D.")

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')

def plot_landmarks_2d(mesh, landmarks_dict, title="Landmarks Detectados (Projeção 2D)", save_path=None):
    """Plota a malha (como nuvem de pontos projetada) e os landmarks usando Matplotlib (projeções XY, XZ, YZ)."""
    if not isinstance(mesh, trimesh.Trimesh):
        logging.error("Entrada para plot_landmarks_2d não é um objeto Trimesh válido.")
        return

    vertices = mesh.vertices
    landmark_coords = []
    landmark_names = []
    if landmarks_dict:
        for name, coords in landmarks_dict.items():
            if coords is not None:
                landmark_coords.append(coords)
                landmark_names.append(name)
        landmark_coords = np.array(landmark_coords)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    # Projeção XY (Vista Superior/Inferior)
    axs[0].scatter(vertices[:, 0], vertices[:, 1], s=1, alpha=0.1, label=\"Vértices da Malha\")
    if len(landmark_coords) > 0:
        axs[0].scatter(landmark_coords[:, 0], landmark_coords[:, 1], c=\"red\", s=50, label=\"Landmarks\")
        for i, name in enumerate(landmark_names):
            axs[0].text(landmark_coords[i, 0], landmark_coords[i, 1], name, fontsize=9, color=\"red\")
    axs[0].set_xlabel(\"X\")
    axs[0].set_ylabel(\"Y\")
    axs[0].set_title(\"Projeção XY (Vista Superior)\")
    axs[0].set_aspect(\"equal\", adjustable=\"box\")
    axs[0].grid(True)

    # Projeção XZ (Vista Frontal/Traseira)
    axs[1].scatter(vertices[:, 0], vertices[:, 2], s=1, alpha=0.1, label=\"Vértices da Malha\")
    if len(landmark_coords) > 0:
        axs[1].scatter(landmark_coords[:, 0], landmark_coords[:, 2], c=\"red\", s=50, label=\"Landmarks\")
        for i, name in enumerate(landmark_names):
            axs[1].text(landmark_coords[i, 0], landmark_coords[i, 2], name, fontsize=9, color=\"red\")
    axs[1].set_xlabel(\"X\")
    axs[1].set_ylabel(\"Z\")
    axs[1].set_title(\"Projeção XZ (Vista Frontal)\")
    axs[1].set_aspect(\"equal\", adjustable=\"box\")
    axs[1].grid(True)

    # Projeção YZ (Vista Lateral)
    axs[2].scatter(vertices[:, 1], vertices[:, 2], s=1, alpha=0.1, label=\"Vértices da Malha\")
    if len(landmark_coords) > 0:
        axs[2].scatter(landmark_coords[:, 1], landmark_coords[:, 2], c=\"red\", s=50, label=\"Landmarks\")
        for i, name in enumerate(landmark_names):
            axs[2].text(landmark_coords[i, 1], landmark_coords[i, 2], name, fontsize=9, color=\"red\")
    axs[2].set_xlabel(\"Y\")
    axs[2].set_ylabel(\"Z\")
    axs[2].set_title(\"Projeção YZ (Vista Lateral)\")
    axs[2].set_aspect(\"equal\", adjustable=\"box\")
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar layout para o título principal

    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            logging.info(f"Visualização 2D salva em: {save_path}")
        except Exception as e:
            logging.error(f"Erro ao salvar visualização 2D em {save_path}: {e}")
    else:
        # Em ambientes sem GUI, plt.show() pode não funcionar ou bloquear.
        # É mais seguro salvar a imagem ou usá-la em notebooks.
        logging.info("Visualização 2D gerada. Use save_path para salvar ou exiba em um notebook.")
        # plt.show() # Descomentar se em ambiente com GUI

    plt.close(fig) # Fechar a figura para liberar memória

def plot_landmarks_3d_o3d(mesh, landmarks_dict, title="Landmarks Detectados (3D Interativo)"):
    """Visualiza a malha e os landmarks em 3D usando Open3D (se disponível)."""
    if not OPEN3D_AVAILABLE:
        logging.warning("Open3D não está disponível. Não é possível gerar visualização 3D interativa.")
        return None

    if not isinstance(mesh, trimesh.Trimesh):
        logging.error("Entrada para plot_landmarks_3d_o3d não é um objeto Trimesh válido.")
        return None

    # Converter malha Trimesh para Open3D TriangleMesh
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    o3d_mesh.compute_vertex_normals()
    # Pintar a malha de cinza claro para destacar os landmarks
    o3d_mesh.paint_uniform_color([0.8, 0.8, 0.8])

    geometries = [o3d_mesh]

    # Criar esferas para representar os landmarks
    if landmarks_dict:
        for name, coords in landmarks_dict.items():
            if coords is not None:
                landmark_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.0) # Ajustar raio conforme escala
                landmark_sphere.translate(np.array(coords))
                landmark_sphere.paint_uniform_color([1.0, 0.0, 0.0]) # Vermelho
                geometries.append(landmark_sphere)
                # Adicionar texto (pode poluir a visualização, usar com cuidado)
                # label = o3d.geometry.create_text_geometry(name, font_size=10, depth=1)
                # label.translate(np.array(coords) + [0, 0, 5]) # Deslocar um pouco
                # label.paint_uniform_color([0, 0, 0])
                # geometries.append(label)

    # Visualizar
    logging.info("Abrindo janela de visualização 3D interativa Open3D...")
    try:
        # A visualização interativa pode não funcionar em todos os ambientes remotos/notebooks.
        o3d.visualization.draw_geometries(geometries, window_name=title)
        logging.info("Janela de visualização 3D fechada.")
        # Nota: draw_geometries é bloqueante. O código continua após fechar a janela.
        # Para salvar uma imagem da visualização sem abrir janela, pode-se usar:
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=False) # Executar fora da tela
        # for geom in geometries: vis.add_geometry(geom)
        # vis.update_geometry()
        # vis.poll_events()
        # vis.update_renderer()
        # vis.capture_screen_image("landmark_visualization.png", do_render=True)
        # vis.destroy_window()
        return True # Indica que a visualização foi tentada
    except Exception as e:
        logging.error(f"Erro ao tentar exibir visualização 3D com Open3D: {e}")
        return False # Indica falha na visualização

def plot_landmarks(mesh, landmarks_dict, title="Landmarks Detectados", use_3d=True, save_path_2d=None):
    """Função principal para plotar landmarks. Tenta 3D com Open3D, senão usa projeções 2D com Matplotlib.

    Args:
        mesh (trimesh.Trimesh): Malha a ser visualizada.
        landmarks_dict (dict): Dicionário de landmarks {"Nome": [x,y,z] or None, ...}.
        title (str): Título para a visualização.
        use_3d (bool): Se deve tentar a visualização 3D interativa primeiro.
        save_path_2d (str, optional): Caminho para salvar a imagem da visualização 2D.
                                      Se None, a imagem não é salva automaticamente.
    """
    visualization_done = False
    if use_3d and OPEN3D_AVAILABLE:
        logging.info("Tentando visualização 3D interativa com Open3D...")
        success_3d = plot_landmarks_3d_o3d(mesh, landmarks_dict, title=f"{title} (3D)")
        if success_3d:
            visualization_done = True

    if not visualization_done:
        if use_3d and not OPEN3D_AVAILABLE:
            logging.info("Open3D não disponível. Recorrendo à visualização 2D com Matplotlib.")
        else:
            logging.info("Visualização 3D não foi usada ou falhou. Gerando visualização 2D com Matplotlib.")

        plot_landmarks_2d(mesh, landmarks_dict, title=f"{title} (Projeções 2D)", save_path=save_path_2d)
        visualization_done = True # Mesmo que salve ou não, a tentativa 2D foi feita

    if not visualization_done:
         logging.error("Nenhum método de visualização pôde ser executado com sucesso.")

# Exemplo de uso (requer mesh_processor e um arquivo STL dummy)
if __name__ == \"__main__\":
    from ..core.mesh_processor import MeshProcessor # Import relativo
    from ..core.detector_geometric import GeometricDetector # Para obter landmarks dummy

    logging.info("--- Testando Funções de Visualização ---")

    # Criar diretórios e um arquivo STL dummy se não existirem para teste
    if not os.path.exists("data/skulls"): os.makedirs("data/skulls")
    dummy_stl_path = "data/skulls/dummy_vis_skull.stl"
    if not os.path.exists(dummy_stl_path):
        logging.info(f"Criando arquivo STL dummy em {dummy_stl_path}")
        mesh_dummy = trimesh.primitives.Sphere(radius=50, subdivisions=4)
        mesh_dummy.vertices += [0, 0, 50]
        mesh_dummy.export(dummy_stl_path)

    # Carregar a malha
    processor = MeshProcessor(data_dir="./data/skulls", cache_dir="./data/cache")
    skull_mesh = processor.load_skull("dummy_vis_skull.stl")

    if skull_mesh:
        # Obter alguns landmarks dummy (usando o detector geométrico como exemplo)
        detector = GeometricDetector()
        # Simplificar um pouco para o detector
        simplified_mesh = processor.simplify(skull_mesh, target_faces=1000, original_filename="dummy_vis_skull.stl")
        if simplified_mesh:
            detected_landmarks = detector.detect(simplified_mesh)
        else:
            logging.warning("Falha ao simplificar, usando malha original para landmarks dummy.")
            # Criar landmarks dummy manualmente se a detecção falhar
            detected_landmarks = {
                "Glabela": skull_mesh.vertices[np.argmax(skull_mesh.vertices[:,1])].tolist(),
                "Bregma": skull_mesh.vertices[np.argmax(skull_mesh.vertices[:,2])].tolist(),
                "Euryon_Direito": skull_mesh.vertices[np.argmax(skull_mesh.vertices[:,0])].tolist(),
                "Ponto_Faltando": None
            }

        if detected_landmarks:
            print("\nLandmarks Dummy para Visualização:")
            print(detected_landmarks)

            # Testar visualização principal (tentará 3D primeiro)
            print("\n--- Testando plot_landmarks (tentativa 3D primeiro) ---")
            # Definir um caminho para salvar a imagem 2D caso 3D falhe ou não seja usado
            save_2d_path = "./dummy_visualization_2d.png"
            plot_landmarks(skull_mesh, detected_landmarks, title="Teste Visualização Esfera", save_path_2d=save_2d_path)

            # Forçar visualização 2D
            print("\n--- Testando plot_landmarks (forçando 2D) ---")
            plot_landmarks(skull_mesh, detected_landmarks, title="Teste Visualização Esfera (Forçado 2D)", use_3d=False, save_path_2d=save_2d_path)

            # Verificar se o arquivo 2D foi criado (se a visualização 2D foi chamada)
            if os.path.exists(save_2d_path):
                print(f"\nArquivo de visualização 2D salvo em: {save_2d_path}")
                # Poderia usar image_view aqui para mostrar a imagem salva
                # image_view(save_2d_path)
            else:
                # Isso pode acontecer se a visualização 3D funcionou e foi fechada
                print("\nArquivo de visualização 2D não foi criado (provavelmente a visualização 3D funcionou). ")

        else:
            logging.error("Falha ao obter landmarks dummy para visualização.")
    else:
        logging.error("Falha ao carregar a malha dummy para visualização.")

