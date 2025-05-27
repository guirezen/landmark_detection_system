# -*- coding: utf-8 -*-
"""Define os nomes e descrições dos landmarks a serem detectados."""

# Lista dos landmarks principais a serem detectados
# Baseado nos requisitos simplificados do TCC
LANDMARK_NAMES = [
    "Glabela",        # Ponto mais proeminente na testa, entre as sobrancelhas
    "Nasion",         # Ponto mais profundo na raiz do nariz (depressão nasal)
    "Bregma",         # Ponto de junção das suturas coronal e sagital (topo do crânio)
    "Opisthocranion", # Ponto mais posterior do crânio na linha média sagital
    "Euryon_Esquerdo",# Ponto mais lateral esquerdo do crânio
    "Euryon_Direito", # Ponto mais lateral direito do crânio
    "Vertex",         # Ponto mais superior do crânio na linha média sagital (pode coincidir ou ser próximo ao Bregma)
    "Inion",          # Ponto na base do crânio, na protuberância occipital externa (alternativa ao Opisthocranion se mais fácil)
    # Adicionar mais 1-2 pontos de alta curvatura se necessário/viável
    # "Ponto_Curvatura_Max_1",
    # "Ponto_Curvatura_Max_2",
]

# Dicionário para mapear nomes a índices (opcional, mas pode ser útil)
LANDMARK_MAP = {name: i for i, name in enumerate(LANDMARK_NAMES)}

# Você pode adicionar mais detalhes ou estruturas aqui se necessário,
# como coordenadas esperadas aproximadas (se conhecidas a priori para um template)
# ou características geométricas típicas de cada landmark.

if __name__ == '__main__':
    print("Landmarks definidos:")
    for i, name in enumerate(LANDMARK_NAMES):
        print(f"{i}: {name}")
    print("\nMapeamento Nome -> Índice:")
    print(LANDMARK_MAP)

