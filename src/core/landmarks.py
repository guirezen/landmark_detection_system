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
    "Vertex",         # Ponto mais superior do crânio na linha média sagital
    "Inion",          # Ponto na base do crânio, na protuberância occipital externa
]

# Dicionário para mapear nomes a índices
LANDMARK_MAP = {name: i for i, name in enumerate(LANDMARK_NAMES)}

# Descrições detalhadas dos landmarks
LANDMARK_DESCRIPTIONS = {
    "Glabela": "Ponto mais proeminente na testa, entre as sobrancelhas",
    "Nasion": "Ponto mais profundo na raiz do nariz (depressão nasal)",
    "Bregma": "Ponto de junção das suturas coronal e sagital (topo do crânio)",
    "Opisthocranion": "Ponto mais posterior do crânio na linha média sagital",
    "Euryon_Esquerdo": "Ponto mais lateral esquerdo do crânio",
    "Euryon_Direito": "Ponto mais lateral direito do crânio",
    "Vertex": "Ponto mais superior do crânio na linha média sagital",
    "Inion": "Ponto na base do crânio, na protuberância occipital externa"
}

def get_landmark_info():
    """Retorna informações sobre todos os landmarks definidos."""
    return {
        'names': LANDMARK_NAMES,
        'map': LANDMARK_MAP,
        'descriptions': LANDMARK_DESCRIPTIONS,
        'count': len(LANDMARK_NAMES)
    }

if __name__ == '__main__':
    print("Landmarks definidos:")
    for i, name in enumerate(LANDMARK_NAMES):
        print(f"{i}: {name}")
    
    print("\nMapeamento Nome -> Índice:")
    print(LANDMARK_MAP)
    
    print("\nDescrições:")
    for name, desc in LANDMARK_DESCRIPTIONS.items():
        print(f"{name}: {desc}")