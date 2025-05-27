# Sistema Simplificado de Detecção de Landmarks em Crânios 3D (TCC)

## Visão Geral

Este projeto implementa um sistema simplificado para a detecção automática de landmarks anatômicos em modelos 3D de crânios humanos, especificamente arquivos no formato STL. O sistema foi desenvolvido como parte de um Trabalho de Conclusão de Curso (TCC) em Ciência da Computação, com foco nos aspectos computacionais, algoritmos de processamento de malhas 3D e comparação de abordagens (geométrica vs. machine learning básico).

O principal objetivo é fornecer uma solução funcional, eficiente e academicamente sólida, otimizada para execução em hardware com recursos limitados (CPU-bound, sem GPU dedicada) e que demonstre competências em programação, estruturas de dados e algoritmos relacionados ao processamento 3D.

## Foco Técnico e Computacional

O desenvolvimento priorizou os seguintes aspectos técnicos:

1.  **Processamento Eficiente de Malhas 3D:**
    *   Utilização da biblioteca `trimesh` para carregamento, manipulação e análise de malhas STL.
    *   Implementação de **simplificação de malha** (decimação quadrática) para reduzir a complexidade computacional, permitindo o processamento de modelos grandes em hardware limitado. O número alvo de faces é configurável.
    *   Sistema de **cache** para malhas carregadas e simplificadas, evitando reprocessamento redundante e acelerando execuções subsequentes. O cache utiliza hashing do nome do arquivo e parâmetros de processamento para garantir a consistência.

2.  **Algoritmos de Detecção:**
    *   **Método Geométrico:** Abordagem baseada puramente em propriedades geométricas da malha:
        *   Cálculo de **curvatura** (Gaussiana como proxy) para identificar regiões de interesse (picos, vales).
        *   Identificação de **vértices extremos** ao longo dos eixos principais (X, Y, Z).
        *   Uso de **KD-Tree** (via `scipy.spatial`) para consultas eficientes de vizinhança, embora as heurísticas atuais sejam mais focadas em extremos e regiões.
        *   Implementação de **heurísticas específicas** para cada landmark (Glabela, Nasion, Bregma, etc.), combinando informações de posição, linha média e curvatura.
    *   **Método de Machine Learning (ML):** Abordagem supervisionada utilizando classificação:
        *   **Extração de Features Locais:** Para cada vértice, são extraídas características como coordenadas normalizadas, normal do vértice, curvatura local e distância ao centroide.
        *   **Classificador:** Utilização de `RandomForestClassifier` da biblioteca `scikit-learn`, um modelo robusto e adequado para dados tabulares, com bom desempenho em CPU.
        *   **Treinamento Individual:** Um modelo separado é treinado para cada landmark, tratando o problema como classificação binária (vértice é o landmark alvo vs. não é).
        *   **Tratamento de Desbalanceamento:** Implementação de subamostragem simples da classe majoritária (não-landmark) durante o treinamento para mitigar o desbalanceamento extremo.
        *   **Escalonamento de Features:** Uso de `StandardScaler` para normalizar as features antes do treinamento e predição.
        *   **Persistência de Modelos:** Modelos treinados e scalers são salvos usando `joblib` para reutilização.

3.  **Estruturas de Dados e Otimização:**
    *   Uso extensivo de `numpy` para operações vetorizadas eficientes em arrays de vértices e faces.
    *   KD-Trees para buscas espaciais rápidas no método geométrico (embora possa ser mais explorado).
    *   Cache baseado em arquivos `pickle` para objetos `trimesh` processados (armazenando vértices e faces).
    *   Logging configurável para monitoramento e depuração.
    *   Script principal (`main.py`) com interface de linha de comando (`argparse`) para facilitar a execução em modo single-file ou batch.

4.  **Avaliação e Comparação:**
    *   Módulo `metrics.py` dedicado ao cálculo de métricas de avaliação:
        *   **Erro de Detecção:** Distância Euclidiana (em mm) entre o landmark previsto e o ground truth.
        *   **Erro Médio de Detecção (MDE):** Média dos erros para os landmarks detectados com sucesso em um modelo.
        *   **Taxa de Detecção:** Percentual de vezes que um landmark foi detectado com sucesso (predição não nula) quando um ground truth estava disponível.
    *   Funções para avaliação em lote, gerando DataFrames (`pandas`) com resultados detalhados e sumários por método.
    *   Notebooks Jupyter (`03_analise_resultados.ipynb`) para visualização comparativa das métricas (boxplots de erro, gráficos de barras de taxa de detecção).

## Estrutura do Projeto

```
landmark_detection_system/
│
├── data/                 # Dados de entrada e processados
│   ├── skulls/           # Modelos .stl originais (ex: MUG500+)
│   ├── cache/            # Malhas processadas (carregadas, simplificadas)
│   └── ground_truth/     # (Opcional) Landmarks ground truth em formato JSON
│
├── models/               # Modelos de Machine Learning treinados (.joblib)
│
├── notebooks/            # Jupyter notebooks para exploração, demonstração e análise
│   ├── 01_exploracao_dados.ipynb
│   ├── 02_demonstracao_metodos.ipynb
│   └── 03_analise_resultados.ipynb
│
├── results/              # Resultados gerados pelo sistema
│   ├── geometric/        # Resultados do método geométrico (JSON, visualizações)
│   ├── ml/               # Resultados do método ML (JSON, visualizações)
│   └── evaluation_*.csv  # Arquivos CSV com métricas de avaliação
│   └── *.png             # Gráficos comparativos e outras visualizações
│
├── src/                  # Código fonte do sistema
│   ├── core/             # Módulos principais
│   │   ├── __init__.py
│   │   ├── mesh_processor.py    # Carregar, simplificar, cache
│   │   ├── detector_geometric.py # Detecção geométrica
│   │   ├── detector_ml.py       # Detecção ML (treinamento e predição)
│   │   └── landmarks.py         # Definições dos landmarks
│   │
│   ├── utils/            # Módulos utilitários
│   │   ├── __init__.py
│   │   ├── visualization.py     # Funções de plot (2D e 3D com Open3D)
│   │   ├── metrics.py          # Cálculo de métricas de avaliação
│   │   └── helpers.py          # Funções auxiliares (logging, I/O, etc.)
│   │
│   └── main.py                  # Script principal (interface de linha de comando)
│
├── requirements.txt      # Dependências Python
├── README.md             # Este arquivo
├── GUIA_DE_USO.md        # Guia prático para instalação e uso
└── todo.md               # Checklist de desenvolvimento (interno)
```

## Limitações e Próximos Passos

*   **Dependência de Dados:** A performance do método ML depende crucialmente da qualidade e quantidade dos dados de treinamento (malhas e landmarks ground truth), que não foram fornecidos neste escopo.
*   **Heurísticas Geométricas:** As heurísticas do método geométrico são simplificadas e podem não generalizar bem para todas as variações de crânios. Refinamentos podem ser necessários.
*   **Features ML:** O conjunto de features para o método ML é básico. Features mais sofisticadas (descritores de forma, HOG 3D, etc.) poderiam melhorar a performance.
*   **Validação:** A validação apresentada nos notebooks utiliza dados dummy. Uma validação rigorosa com dados reais e métricas estatísticas apropriadas é necessária.
*   **Visualização 3D:** A visualização 3D interativa depende da biblioteca `open3d`, que pode não estar disponível em todos os ambientes.

Possíveis trabalhos futuros incluem: refinar as heurísticas geométricas, explorar features e modelos ML mais avançados, implementar um pipeline de treinamento mais robusto, e realizar uma validação extensiva em um dataset real como o MUG500+.

## Como Usar

Consulte o arquivo `GUIA_DE_USO.md` para instruções detalhadas sobre instalação, configuração e execução do sistema.

