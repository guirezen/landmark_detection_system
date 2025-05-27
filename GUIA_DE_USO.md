# Guia de Uso - Sistema Simplificado de Detecção de Landmarks

Este guia fornece instruções sobre como instalar, configurar e utilizar o Sistema Simplificado de Detecção de Landmarks em Crânios 3D.

## 1. Instalação Rápida

Siga estes passos para configurar o ambiente e instalar as dependências necessárias.

### Pré-requisitos

*   **Python:** Versão 3.8 ou superior recomendada.
*   **pip:** Gerenciador de pacotes Python.
*   **Git:** (Opcional) Para clonar o repositório, se aplicável.

### Passos de Instalação

1.  **Obtenha o Código:**
    *   Se você recebeu o projeto como um arquivo `.zip`, descompacte-o em um local de sua preferência.
    *   Se estiver usando Git: `git clone <url_do_repositorio>`
    *   Navegue até o diretório raiz do projeto (`landmark_detection_system`) no seu terminal.

2.  **Crie um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv venv
    ```
    *   Ative o ambiente virtual:
        *   **Windows:** `.\venv\Scripts\activate`
        *   **macOS/Linux:** `source venv/bin/activate`

3.  **Instale as Dependências:**
    Execute o seguinte comando no diretório raiz do projeto (onde `requirements.txt` está localizado):
    ```bash
    pip install -r requirements.txt
    ```
    Isso instalará `trimesh`, `numpy`, `scikit-learn`, `matplotlib`, `open3d` (opcional), `jupyterlab`, `pandas`, `joblib` e outras dependências necessárias.

4.  **Verificação da Instalação (Opcional):**
    Você pode tentar importar as bibliotecas principais em um interpretador Python dentro do ambiente ativado para verificar se a instalação ocorreu sem erros:
    ```python
    import trimesh
    import sklearn
    import open3d # Se instalado
    print("Dependências principais carregadas com sucesso!")
    ```

## 2. Estrutura de Diretórios Esperada

Certifique-se de que os diretórios de dados estejam configurados corretamente. A estrutura padrão é:

```
landmark_detection_system/
├── data/
│   ├── skulls/          # COLOQUE SEUS ARQUIVOS .STL AQUI!
│   ├── cache/           # Será criado automaticamente
│   └── ground_truth/    # (Opcional) Coloque seus arquivos GT .json aqui
├── models/              # Modelos ML treinados (serão criados/usados)
├── notebooks/           # Notebooks para exploração e análise
├── results/             # Resultados das detecções (serão criados)
├── src/                 # Código fonte
├── venv/                # Ambiente virtual (se criado)
├── requirements.txt
├── README.md
└── GUIA_DE_USO.md
```

*   **`data/skulls/`:** É **essencial** que você coloque seus arquivos `.stl` dos modelos de crânio neste diretório para que o sistema possa encontrá-los.
*   **`data/ground_truth/`:** Se você possui dados ground truth (coordenadas corretas dos landmarks) para avaliação, coloque-os como arquivos `.json` neste diretório. O nome do arquivo GT deve corresponder ao arquivo STL (ex: `A0001.stl` -> `A0001_landmarks_gt.json`).

## 3. Uso Básico (Linha de Comando)

O script principal `src/main.py` oferece uma interface de linha de comando para executar as detecções.

**Certifique-se de que seu ambiente virtual esteja ativado antes de executar os comandos.**

### 3.1 Processando um Único Arquivo

Use o modo `single` para processar um arquivo STL específico.

**Exemplo (Método Geométrico):**
```bash
python src/main.py single --method geometric -i data/skulls/SEU_ARQUIVO.stl --visualize
```

**Exemplo (Método ML):**
```bash
python src/main.py single --method ml -i data/skulls/SEU_ARQUIVO.stl --visualize
```

**Argumentos Comuns:**

*   `--method {geometric, ml}`: (Obrigatório) Escolhe o método de detecção.
*   `-i / --input_file`: (Obrigatório) Caminho para o arquivo `.stl` de entrada.
*   `--output_dir`: Diretório para salvar os resultados (padrão: `./results`).
*   `--cache_dir`: Diretório de cache (padrão: `./data/cache`).
*   `--no_cache`: Desativa o uso do cache.
*   `--simplify_faces N`: Simplifica a malha para `N` faces antes da detecção (padrão: 5000). Use 0 para não simplificar.
*   `--visualize`: Gera e tenta exibir/salvar uma visualização dos landmarks detectados.
*   `--force_2d_vis`: Força a visualização 2D (matplotlib) mesmo se Open3D estiver disponível.
*   `--model_dir`: Diretório dos modelos ML (padrão: `./models`, relevante para `--method ml`).
*   `--gt_file`: (Opcional) Caminho para o arquivo JSON de ground truth para avaliação deste arquivo.
*   `-v / --verbose`: Ativa logging mais detalhado.

### 3.2 Processando Múltiplos Arquivos (Lote)

Use o modo `batch` para processar todos os arquivos `.stl` dentro de um diretório.

**Exemplo (Método Geométrico):**
```bash
python src/main.py batch --method geometric -i data/skulls/ --output_dir results/geometric_batch
```

**Exemplo (Método ML com Avaliação):**
```bash
python src/main.py batch --method ml -i data/skulls/ --gt_dir data/ground_truth/ --output_dir results/ml_batch --visualize
```

**Argumentos Específicos do Batch:**

*   `-i / --input_dir`: (Obrigatório) Diretório contendo os arquivos `.stl` a serem processados.
*   `--gt_dir`: (Opcional) Diretório contendo os arquivos JSON de ground truth para avaliação em lote.
*   `--output_dir`: Diretório base onde os resultados serão salvos (será criado um subdiretório para o método, ex: `results/ml_batch/ml/`).

Os resultados (arquivos JSON de landmarks, visualizações PNG se `--visualize`, e CSVs de avaliação se `--gt_dir`) serão salvos no diretório de saída especificado.

## 4. Uso com Notebooks Jupyter

Os notebooks na pasta `notebooks/` fornecem uma maneira interativa de explorar os dados, demonstrar os métodos e analisar os resultados.

1.  **Inicie o Jupyter Lab:**
    No diretório raiz do projeto (com o ambiente virtual ativado):
    ```bash
    jupyter lab
    ```
    Isso abrirá o Jupyter Lab no seu navegador.

2.  **Navegue e Execute:**
    *   Abra a pasta `notebooks` na interface do Jupyter.
    *   Execute os notebooks na ordem:
        *   `01_exploracao_dados.ipynb`: Carrega, visualiza e simplifica malhas.
        *   `02_demonstracao_metodos.ipynb`: Executa ambos os métodos de detecção em um exemplo.
        *   `03_analise_resultados.ipynb`: Avalia e compara os resultados (requer dados GT e resultados gerados previamente).
    *   Siga as instruções dentro de cada notebook e execute as células de código.

**Importante:** Os notebooks contêm código para criar arquivos e dados *dummy* caso os dados reais (STL, GT, modelos ML) não estejam presentes. Adapte os caminhos e nomes de arquivos conforme necessário se estiver usando seus próprios dados.

## 5. Entendendo os Resultados

### 5.1 Arquivos de Landmarks (`.json`)

Para cada arquivo STL processado, o sistema gera um arquivo JSON (ex: `SEU_ARQUIVO_geometric_landmarks.json`) no diretório de resultados. Este arquivo contém um dicionário onde as chaves são os nomes dos landmarks (definidos em `src/core/landmarks.py`) e os valores são as coordenadas [x, y, z] detectadas ou `null` (`None` em Python) se a detecção falhou para aquele landmark.

**Exemplo de Conteúdo JSON:**
```json
{
    "Glabela": [10.5, 80.2, 120.1],
    "Nasion": [10.2, 75.0, 110.5],
    "Bregma": null, // Detecção falhou ou não implementada
    "Opisthocranion": [11.0, -90.5, 95.3],
    ...
}
```

### 5.2 Arquivos de Avaliação (`.csv`)

Se a avaliação foi executada (usando `--gt_file` ou `--gt_dir`), dois arquivos CSV são gerados no diretório de saída principal:

*   `evaluation_<method>_detailed.csv`: Contém o erro de detecção (distância em mm) para cada landmark em cada arquivo processado.
*   `evaluation_<method>_summary.csv`: Contém estatísticas agregadas por landmark (erro médio, desvio padrão, taxa de detecção) para o método avaliado.

### 5.3 Visualizações (`.png`)

Se a opção `--visualize` foi usada, imagens `.png` podem ser geradas:

*   **Visualização 2D:** Se Open3D não estiver disponível ou `--force_2d_vis` for usado, uma imagem com projeções 2D (XY, XZ, YZ) da malha e dos landmarks será salva.
*   **Visualização 3D:** Se Open3D estiver disponível, uma janela interativa será aberta. A funcionalidade de salvar a imagem 3D diretamente não está habilitada por padrão no script `main.py` ou nos notebooks, mas pode ser implementada modificando `src/utils/visualization.py`.

## 6. Aspectos Técnicos (Foco TCC)

Consulte o `README.md` para uma descrição detalhada dos aspectos técnicos, algoritmos, estruturas de dados e decisões de implementação relevantes para o TCC.

## 7. Treinamento do Modelo de Machine Learning

O script `main.py` **não** realiza o treinamento dos modelos de Machine Learning. Ele assume que os modelos já existem no diretório especificado por `--model_dir` (padrão: `./models`).

Para treinar os modelos:

1.  **Prepare os Dados:** Você precisará de um conjunto de malhas STL (`data/skulls/`) e os arquivos JSON de ground truth correspondentes (`data/ground_truth/`).
2.  **Use o Código de Treinamento:** O módulo `src/core/detector_ml.py` contém a função `train()`. Você pode:
    *   Adaptar o código de exemplo no final de `detector_ml.py` para carregar suas malhas e GTs e chamar `train()` para cada landmark.
    *   Criar um script de treinamento separado que importe `MLDetector` e `MeshProcessor`, carregue os dados e execute o loop de treinamento, salvando os modelos no diretório `models/`.

**Exemplo Conceitual (em um script `train_ml.py`):**
```python
import os
import sys
module_path = os.path.abspath(os.path.join(".")) # Assumindo que roda da raiz
if module_path not in sys.path: sys.path.append(module_path)

from src.core.mesh_processor import MeshProcessor
from src.core.detector_ml import MLDetector
from src.utils.helpers import list_stl_files, load_landmarks_from_json
from src.core.landmarks import LANDMARK_NAMES
import logging

logging.basicConfig(level=logging.INFO)

DATA_DIR = "./data/skulls"
GT_DIR = "./data/ground_truth"
MODEL_DIR = "./models"
CACHE_DIR = "./data/cache"
SIMPLIFY_FACES = 5000

processor = MeshProcessor(data_dir=DATA_DIR, cache_dir=CACHE_DIR)
ml_detector = MLDetector(model_dir=MODEL_DIR)

all_meshes = []
all_gts = []
file_ids = [os.path.splitext(f)[0] for f in list_stl_files(DATA_DIR)]

logging.info(f"Carregando {len(file_ids)} malhas e GTs para treinamento...")
for file_id in file_ids:
    mesh = processor.load_skull(f"{file_id}.stl")
    if not mesh: continue
    simplified_mesh = processor.simplify(mesh, SIMPLIFY_FACES, original_filename=f"{file_id}.stl")
    if not simplified_mesh: simplified_mesh = mesh # Usar original se falhar

    gt_path = os.path.join(GT_DIR, f"{file_id}_landmarks_gt.json")
    gt_landmarks = load_landmarks_from_json(gt_path)

    if simplified_mesh and gt_landmarks:
        all_meshes.append(simplified_mesh)
        all_gts.append(gt_landmarks)
    else:
        logging.warning(f"Pulando {file_id} por falta de malha ou GT.")

if not all_meshes:
    logging.error("Nenhum dado válido carregado para treinamento.")
else:
    logging.info(f"Iniciando treinamento para {len(LANDMARK_NAMES)} landmarks...")
    for landmark_name in LANDMARK_NAMES:
        logging.info(f"--- Treinando para: {landmark_name} ---")
        ml_detector.train(all_meshes, all_gts, landmark_name)
    logging.info("Treinamento ML concluído.")
```

## 8. Problemas Comuns

*   **Erro `FileNotFoundError`:** Verifique se os caminhos para os arquivos STL, GT ou modelos estão corretos e se os arquivos existem.
*   **Memória Insuficiente:** Se encontrar erros de memória, especialmente com malhas grandes:
    *   Aumente o nível de simplificação (reduza `--simplify_faces`).
    *   Processe arquivos em lote menores ou individualmente.
    *   Feche outras aplicações que consomem muita memória.
*   **Falha na Detecção (Landmark == `null`):**
    *   **Geométrico:** A heurística pode não ter encontrado um candidato válido. Verifique a orientação da malha ou ajuste as heurísticas em `detector_geometric.py`.
    *   **ML:** O modelo pode não ter sido treinado para esse landmark, ou a confiança da predição foi muito baixa. Verifique se o modelo `.joblib` existe e se o treinamento foi adequado.
*   **Erro na Visualização 3D (Open3D):** Pode ocorrer em ambientes sem suporte gráfico adequado (servidores remotos, alguns containers Docker). Use `--force_2d_vis` para usar a visualização Matplotlib.
*   **Dependências Ausentes:** Certifique-se de que todas as bibliotecas em `requirements.txt` foram instaladas corretamente no ambiente virtual ativo.

Se encontrar outros problemas, verifique os logs de erro detalhados (use a opção `-v` ou `--verbose` na linha de comando) para obter mais informações.

