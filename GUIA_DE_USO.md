# Guia de Uso - Sistema de Detecção de Landmarks em Crânios 3D

Este guia fornece instruções completas sobre como instalar, configurar e utilizar o Sistema de Detecção de Landmarks em Crânios 3D.

## 📋 Índice

1. [Instalação Rápida](#instalação-rápida)
2. [Estrutura de Diretórios](#estrutura-de-diretórios)
3. [Uso via Linha de Comando](#uso-via-linha-de-comando)
4. [Uso com Notebooks Jupyter](#uso-com-notebooks-jupyter)
5. [Entendendo os Resultados](#entendendo-os-resultados)
6. [Treinamento de Modelos ML](#treinamento-de-modelos-ml)
7. [Problemas Comuns](#problemas-comuns)
8. [Exemplos Práticos](#exemplos-práticos)

## 🚀 Instalação Rápida

### Pré-requisitos

- **Python:** Versão 3.8 ou superior
- **pip:** Gerenciador de pacotes Python
- **Git:** (Opcional) Para clonar o repositório

### Passos de Instalação

1. **Obter o Código:**
   ```bash
   # Se você recebeu como arquivo .zip
   unzip landmark_detection_system.zip
   cd landmark_detection_system
   
   # Ou se estiver usando Git
   git clone <url_do_repositorio>
   cd landmark_detection_system
   ```

2. **Criar Ambiente Virtual (Recomendado):**
   ```bash
   python -m venv venv
   ```
   
   **Ativar o ambiente virtual:**
   - **Windows:** `.\venv\Scripts\activate`
   - **macOS/Linux:** `source venv/bin/activate`

3. **Instalar Dependências:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Isso instalará todas as dependências necessárias:
   - `trimesh` - Processamento de malhas 3D
   - `numpy` - Computação numérica
   - `scikit-learn` - Machine Learning
   - `matplotlib` - Visualização 2D
   - `open3d` - Visualização 3D (opcional)
   - `pandas` - Análise de dados
   - `scipy` - Computação científica
   - `seaborn` - Visualização estatística

4. **Verificar Instalação:**
   ```bash
   python -c "import trimesh, sklearn, numpy; print('✅ Dependências principais instaladas!')"
   ```

## 📁 Estrutura de Diretórios

Certifique-se de que os diretórios estejam organizados corretamente:

```
landmark_detection_system/
├── data/
│   ├── skulls/          # 📂 COLOQUE SEUS ARQUIVOS .STL AQUI!
│   ├── cache/           # 🔄 Cache automático (criado automaticamente)
│   └── ground_truth/    # 📊 Arquivos GT .json (opcional)
├── models/              # 🤖 Modelos ML treinados
├── notebooks/           # 📓 Notebooks Jupyter para exploração
├── results/             # 📈 Resultados das detecções
├── src/                 # 💾 Código fonte
│   ├── core/           # Módulos principais
│   └── utils/          # Utilitários
├── venv/               # 🐍 Ambiente virtual (se criado)
├── requirements.txt
├── README.md
└── GUIA_DE_USO.md
```

**⚠️ IMPORTANTE:** Coloque seus arquivos `.stl` no diretório `data/skulls/` para que o sistema possa encontrá-los.

## 💻 Uso via Linha de Comando

O script principal `src/main.py` oferece uma interface completa de linha de comando.

**Certifique-se de que seu ambiente virtual esteja ativado antes de executar os comandos.**

### Processando um Único Arquivo

```bash
# Método geométrico básico
python src/main.py single --method geometric -i data/skulls/seu_arquivo.stl --visualize

# Método ML com ground truth
python src/main.py single --method ml -i data/skulls/seu_arquivo.stl --gt_file data/ground_truth/seu_arquivo_landmarks_gt.json --visualize

# Com simplificação customizada
python src/main.py single --method geometric -i seu_arquivo.stl --simplify_faces 3000 --verbose
```

### Processando Múltiplos Arquivos (Lote)

```bash
# Processar todos os STL com método geométrico
python src/main.py batch --method geometric -i data/skulls/ --output_dir results/geometric_batch --visualize

# Processar com ML e avaliação automática
python src/main.py batch --method ml -i data/skulls/ --gt_dir data/ground_truth/ --output_dir results/ml_batch

# Processamento silencioso (sem visualizações)
python src/main.py batch --method geometric -i data/skulls/ --simplify_faces 2000
```

### Argumentos Principais

| Argumento | Descrição | Exemplo |
|-----------|-----------|---------|
| `--method` | Método de detecção (`geometric` ou `ml`) | `--method geometric` |
| `-i` | Arquivo ou diretório de entrada | `-i data/skulls/cranio.stl` |
| `--output_dir` | Diretório de saída | `--output_dir results/teste` |
| `--simplify_faces` | Número de faces alvo (0 = não simplificar) | `--simplify_faces 2000` |
| `--visualize` | Gerar visualizações | `--visualize` |
| `--gt_file` / `--gt_dir` | Ground truth para avaliação | `--gt_dir data/ground_truth/` |
| `--no_cache` | Desativar cache | `--no_cache` |
| `--verbose` | Logging detalhado | `--verbose` |

## 📔 Uso com Notebooks Jupyter

Os notebooks oferecem uma experiência interativa para exploração e análise.

### Iniciando o Jupyter Lab

```bash
# No diretório raiz do projeto (com ambiente virtual ativado)
jupyter lab
```

Isso abrirá o Jupyter Lab no seu navegador.

### Notebooks Disponíveis

1. **`01_exploracao_dados.ipynb`**
   - Carregamento e visualização de malhas
   - Demonstração do pré-processamento
   - Análise de propriedades geométricas
   - Verificação do sistema de cache

2. **`02_demonstracao_metodos.ipynb`**
   - Execução dos métodos de detecção
   - Comparação visual dos resultados
   - Análise de performance básica

3. **`03_analise_resultados.ipynb`**
   - Avaliação quantitativa detalhada
   - Métricas estatísticas
   - Visualizações comparativas
   - Relatórios de performance

### Executando os Notebooks

1. Navegue até a pasta `notebooks/` na interface do Jupyter
2. Abra os notebooks na ordem (01 → 02 → 03)
3. Execute as células sequencialmente (`Shift + Enter`)
4. Adapte os caminhos de arquivos conforme necessário

## 📊 Entendendo os Resultados

### Arquivos de Landmarks (`.json`)

Para cada arquivo STL processado, o sistema gera um arquivo JSON:

```json
{
    "Glabela": [10.5, 80.2, 120.1],
    "Nasion": [10.2, 75.0, 110.5],
    "Bregma": null,
    "Opisthocranion": [11.0, -90.5, 95.3],
    "Euryon_Esquerdo": [-45.2, 0.1, 85.7],
    "Euryon_Direito": [45.8, -0.3, 85.9],
    "Vertex": [0.2, 1.5, 125.3],
    "Inion": [0.5, -85.2, 75.1]
}
```

- **Coordenadas válidas:** Array `[x, y, z]` em milímetros
- **`null`:** Landmark não detectado ou detecção falhou

### Arquivos de Avaliação (`.csv`)

Se a avaliação foi executada, dois arquivos CSV são gerados:

1. **`evaluation_[método]_detailed.csv`**
   - Erro de detecção para cada landmark em cada arquivo
   - Colunas: `FileID`, `Method`, `Landmark`, `Error`, `MDE_File`

2. **`evaluation_[método]_summary.csv`**
   - Estatísticas agregadas por landmark
   - Colunas: `Landmark`, `MeanError`, `StdError`, `DetectionRate`, `NumDetected`

### Visualizações (`.png`)

- **Visualização 2D:** Projeções XY, XZ, YZ da malha com landmarks
- **Visualização 3D:** Janela interativa (se Open3D disponível)

## 🤖 Treinamento de Modelos ML

O script `main.py` **não** realiza treinamento - apenas usa modelos existentes.

### Para Treinar Modelos

1. **Prepare os Dados:**
   - Malhas STL em `data/skulls/`
   - Arquivos JSON GT correspondentes em `data/ground_truth/`

2. **Execute o Treinamento:**

```python
# Exemplo de script de treinamento
import sys
sys.path.append('.')

from src.core.mesh_processor import MeshProcessor
from src.core.detector_ml import MLDetector
from src.utils.helpers import list_stl_files, load_landmarks_from_json
from src.core.landmarks import LANDMARK_NAMES

# Configuração
DATA_DIR = "./data/skulls"
GT_DIR = "./data/ground_truth"
MODEL_DIR = "./models"
CACHE_DIR = "./data/cache"

# Carregar dados
processor = MeshProcessor(data_dir=DATA_DIR, cache_dir=CACHE_DIR)
ml_detector = MLDetector(model_dir=MODEL_DIR)

all_meshes = []
all_gts = []

stl_files = list_stl_files(DATA_DIR)
for filename in stl_files:
    file_id = filename.replace('.stl', '')
    
    # Carregar malha
    mesh = processor.load_skull(filename)
    if mesh:
        simplified_mesh = processor.simplify(mesh, target_faces=5000, original_filename=filename)
        
        # Carregar GT
        gt_path = f"{GT_DIR}/{file_id}_landmarks_gt.json"
        gt_landmarks = load_landmarks_from_json(gt_path)
        
        if simplified_mesh and gt_landmarks:
            all_meshes.append(simplified_mesh)
            all_gts.append(gt_landmarks)

# Treinar para cada landmark
for landmark_name in LANDMARK_NAMES:
    print(f"Treinando {landmark_name}...")
    success = ml_detector.train(all_meshes, all_gts, landmark_name)
    if success:
        print(f"✅ Modelo {landmark_name} treinado")
    else:
        print(f"❌ Falha no treinamento de {landmark_name}")
```

## 🔧 Problemas Comuns

### Erro `FileNotFoundError`
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/skulls/arquivo.stl'
```
**Solução:** Verifique se os caminhos estão corretos e os arquivos existem.

### Erro de Memória
```
MemoryError: Unable to allocate array
```
**Soluções:**
- Reduza `--simplify_faces` (ex: `--simplify_faces 1000`)
- Processe arquivos individualmente
- Feche outras aplicações que consomem memória

### Falha na Detecção (Landmark == `null`)

**Método Geométrico:**
- Verifique orientação da malha (deve estar em posição anatômica)
- Malha pode ter qualidade baixa ou geometria atípica

**Método ML:**
- Modelo pode não existir para esse landmark
- Confiança da predição muito baixa
- Modelo precisa ser retreinado com mais dados

### Erro na Visualização 3D
```
RuntimeError: GLFW Error: [65544] WGL: The driver does not appear to support OpenGL
```
**Soluções:**
- Use `--force_2d_vis` para forçar visualização 2D
- Instale drivers gráficos atualizados
- Em servidores remotos, use sempre visualização 2D

### Dependências Ausentes
```
ModuleNotFoundError: No module named 'trimesh'
```
**Soluções:**
- Certifique-se de que o ambiente virtual está ativado
- Execute `pip install -r requirements.txt` novamente
- Verifique se está no diretório correto do projeto

## 💡 Exemplos Práticos

### Exemplo 1: Processamento Básico
```bash
# Processar um arquivo com método geométrico
python src/main.py single --method geometric -i data/skulls/cranio001.stl --visualize --verbose

# Output esperado:
# ✅ Landmarks detectados: 6/8 (75.0%)
# 💾 Resultados salvos em: results/cranio001_geometric_landmarks.json
# 🎨 Visualização salva em: results/cranio001_geometric_visualization.png
```

### Exemplo 2: Avaliação com Ground Truth
```bash
# Avaliar precisão com dados GT
python src/main.py single --method ml -i cranio001.stl --gt_file data/ground_truth/cranio001_landmarks_gt.json --verbose

# Output esperado:
# 📊 Erro Médio de Detecção (MDE): 2.345 mm
# 📋 Glabela: 1.23 mm
# 📋 Nasion: 2.45 mm
```

### Exemplo 3: Processamento em Lote com Avaliação
```bash
# Processar dataset completo e gerar relatório
python src/main.py batch --method geometric -i data/skulls/ --gt_dir data/ground_truth/ --output_dir results/evaluation_complete --visualize

# Output esperado:
# 📊 100 arquivos processados
# 📈 Taxa de detecção geral: 78.5%
# 📉 Erro médio geral: 3.124 mm
# 💾 Relatório detalhado: results/evaluation_geometric_summary.csv
```

### Exemplo 4: Comparação de Métodos
```bash
# Processar com ambos os métodos
python src/main.py batch --method geometric -i data/skulls/ --output_dir results/comparison
python src/main.py batch --method ml -i data/skulls/ --output_dir results/comparison

# Analisar resultados nos notebooks
jupyter lab notebooks/03_analise_resultados.ipynb
```

## 📞 Suporte

Para problemas não cobertos neste guia:

1. **Verifique os Logs:** Use `--verbose` para diagnóstico detalhado
2. **Teste com Dados Simples:** Use os notebooks com dados dummy primeiro
3. **Verifique Dependências:** Confirme que todas as bibliotecas estão instaladas
4. **Documentação Técnica:** Consulte o `README.md` para detalhes técnicos

---

**🎉 Sistema configurado e pronto para uso!**

Este guia deve cobrir a maioria dos casos de uso. Para funcionalidades avançadas ou customizações, consulte o código fonte e a documentação técnica no `README.md`.