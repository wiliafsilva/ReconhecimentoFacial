import os
import json
import numpy as np
from .utils import load_json


def decide_from_vector(vec, automaton_map):
    """Decisão simples para a descrição dada:
    - Se o vetor conter apenas 1s -> face feliz
    - Se o vetor conter apenas 0s -> face triste
    - Se tiver mistura, usar regra de maioria: se maioria 1 -> feliz, se maioria 0 -> triste
    - Se encontrar qualquer valor fora de {0,1} -> rejeição
    Também retornamos a decisão lendo apenas o primeiro símbolo (simulação MT simples).
    """
    arr = np.array(vec)
    # verificar valores inválidos
    unique_vals = np.unique(arr)
    if any((v not in (0,1) for v in unique_vals)):
        return {'decision':'reject','reason':'invalid_symbols','by_first':None,'by_majority':None}

    # by first symbol (simulação de fita lendo primeiro bit)
    first = int(arr[0]) if arr.size>0 else None
    by_first = 'happy' if first==1 else ('sad' if first==0 else 'reject')

    # by majority
    ones = int((arr==1).sum())
    zeros = int((arr==0).sum())
    if ones > zeros:
        by_majority = 'happy'
    elif zeros > ones:
        by_majority = 'sad'
    else:
        # empate -> rejeição/indeterminado
        by_majority = 'reject'

    # consultar automaton_map (if present) for mapping
    # automaton_map is expected like {'neutral->happy':1,'neutral->sad':0}
    # here we only return final result as by_majority, but include both.
    final = by_majority
    return {'decision': final, 'by_first': by_first, 'by_majority': by_majority}


def main(digraphs_dir):
    auto_path = os.path.join(digraphs_dir, 'automaton.json')
    sad_meta = os.path.join(digraphs_dir, 'diff_neutral_sad_meta.json')
    target_meta = os.path.join(digraphs_dir, 'diff_neutral_happy_meta.json')
    if not os.path.exists(auto_path):
        raise FileNotFoundError('automaton.json não encontrado em ' + digraphs_dir)
    automaton_map = load_json(auto_path)

    smeta = load_json(sad_meta)
    tmeta = load_json(target_meta)

    svec = smeta['binary']
    tvec = tmeta['binary']

    sres = decide_from_vector(svec, automaton_map)
    tres = decide_from_vector(tvec, automaton_map)

    print('Automaton map:', automaton_map)
    print('\nNEUTRAL -> SAD decision:')
    print('  by_first_symbol:', sres['by_first'])
    print('  by_majority:', sres['by_majority'])
    print('  final_decision:', sres['decision'])

    print('\nNEUTRAL -> HAPPY decision:')
    print('  by_first_symbol:', tres['by_first'])
    print('  by_majority:', tres['by_majority'])
    print('  final_decision:', tres['decision'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='diretório com os digraphs')
    args = parser.parse_args()
    main(args.dir)
