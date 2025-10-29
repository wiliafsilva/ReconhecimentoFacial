import os
import json
import numpy as np
import argparse
from .landmark_extractor import LandmarkExtractor
from .digraph import build_face_digraph, digraph_from_difference
from .utils import save_json, ensure_dir
from .visualize import plot_diff_graph


def graph_to_dict(G):
    nodes = []
    for n, attrs in G.nodes(data=True):
        node = {'id': int(n)}
        for k, v in attrs.items():
            # convert numpy types
            try:
                json.dumps(v)
                node[k] = v
            except Exception:
                if hasattr(v, 'tolist'):
                    node[k] = v.tolist()
                else:
                    node[k] = str(v)
        nodes.append(node)
    edges = []
    for a, b, attrs in G.edges(data=True):
        e = {'source': int(a), 'target': int(b)}
        e.update({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in attrs.items()})
        edges.append(e)
    return {'nodes': nodes, 'edges': edges}


def save_graph(path, G):
    d = graph_to_dict(G)
    save_json(path, d)


def main(neutral, sad, happy, out_dir, threshold=0.05):
    ensure_dir(out_dir)
    ext = LandmarkExtractor()

    import cv2
    nb = cv2.imread(neutral)
    sb = cv2.imread(sad)
    tb = cv2.imread(happy)
    if nb is None or sb is None or tb is None:
        raise FileNotFoundError('Uma das imagens não pôde ser lida. Verifique os caminhos')

    n_lm = ext.from_bgr(nb)
    s_lm = ext.from_bgr(sb)
    t_lm = ext.from_bgr(tb)
    if n_lm is None:
        raise RuntimeError('Não foi possível extrair landmarks da imagem neutra')
    if s_lm is None:
        raise RuntimeError('Não foi possível extrair landmarks da imagem sad')
    if t_lm is None:
        raise RuntimeError('Não foi possível extrair landmarks da imagem happy')

    # gerar grafos de face individuais
    G_neutral = build_face_digraph(n_lm)
    G_sad = build_face_digraph(s_lm)
    G_target = build_face_digraph(t_lm)

    save_graph(os.path.join(out_dir, 'face_neutral.json'), G_neutral)
    save_graph(os.path.join(out_dir, 'face_sad.json'), G_sad)
    save_graph(os.path.join(out_dir, 'face_happy.json'), G_target)

    # gerar grafos de diferença usando neutro como base
    diff_ns, binary_ns, difs_ns = digraph_from_difference(n_lm, s_lm, threshold=threshold, normalize=True)
    diff_nt, binary_nt, difs_nt = digraph_from_difference(n_lm, t_lm, threshold=threshold, normalize=True)

    # salvar diffs (grafo + vetores)
    save_graph(os.path.join(out_dir, 'diff_neutral_sad_graph.json'), diff_ns)
    save_json(os.path.join(out_dir, 'diff_neutral_sad_meta.json'), {
        'binary': binary_ns.tolist() if hasattr(binary_ns, 'tolist') else list(binary_ns),
        'difs': difs_ns.tolist() if hasattr(difs_ns, 'tolist') else list(difs_ns)
    })

    # salvar visualização PNG dos diffs usando a imagem neutra como pano de fundo
    try:
        plot_diff_graph(nb, n_lm, diff_ns, out_path=os.path.join(out_dir, 'diff_neutral_sad.png'))
    except Exception as e:
        print('Falha ao gerar visualização diff_neutral_sad.png:', e)

    save_graph(os.path.join(out_dir, 'diff_neutral_happy_graph.json'), diff_nt)
    save_json(os.path.join(out_dir, 'diff_neutral_happy_meta.json'), {
        'binary': binary_nt.tolist() if hasattr(binary_nt, 'tolist') else list(binary_nt),
        'difs': difs_nt.tolist() if hasattr(difs_nt, 'tolist') else list(difs_nt)
    })

    try:
        plot_diff_graph(nb, n_lm, diff_nt, out_path=os.path.join(out_dir, 'diff_neutral_happy.png'))
    except Exception as e:
        print('Falha ao gerar visualização diff_neutral_happy.png:', e)

    # Calcular decisões reais baseadas nos vetores binários
    def decide(binary_vec):
        """Decide baseado na maioria de 1s vs 0s no vetor binário"""
        ones = int((binary_vec == 1).sum() if hasattr(binary_vec, 'sum') else sum(1 for v in binary_vec if v == 1))
        zeros = int((binary_vec == 0).sum() if hasattr(binary_vec, 'sum') else sum(1 for v in binary_vec if v == 0))
        if ones > zeros:
            return 'happy'
        elif zeros > ones:
            return 'sad'
        else:
            return 'neutral'
    
    dec_sad = decide(binary_ns)
    dec_happy = decide(binary_nt)
    
    # Mapear decisões para valores numéricos: happy=1, sad=0, neutral=0.5, reject=-1
    label_map = {'happy': 1, 'sad': 0, 'neutral': 0.5, 'reject': -1}
    
    # construir autômato com valores reais calculados
    automaton = {
        'neutral->sad': label_map.get(dec_sad, -1),
        'neutral->happy': label_map.get(dec_happy, -1),
        '_metadata': {
            'neutral->sad_label': dec_sad,
            'neutral->happy_label': dec_happy,
            'neutral->sad_ones': int((binary_ns == 1).sum() if hasattr(binary_ns, 'sum') else sum(1 for v in binary_ns if v == 1)),
            'neutral->sad_total': int(len(binary_ns)),
            'neutral->happy_ones': int((binary_nt == 1).sum() if hasattr(binary_nt, 'sum') else sum(1 for v in binary_nt if v == 1)),
            'neutral->happy_total': int(len(binary_nt)),
            'threshold': threshold
        }
    }
    save_json(os.path.join(out_dir, 'automaton.json'), automaton)

    # Gerar Máquina de Turing com valores reais baseados na análise
    # Arquitetura simplificada: a TM lê símbolos 0 (tentativa de sad) ou 1 (tentativa de happy)
    # e decide se aceita ou rejeita baseado nas decisões reais da análise
    
    # Criar estados específicos para cada decisão
    turing_machine = {
        'Q': ['q_start', 'q_check_sad', 'q_check_happy', 'q_accept', 'q_reject'],
        'Sigma': ['0', '1'],  # 0=testar sad, 1=testar happy
        'Gamma': ['0', '1', '_'],
        'blank': '_',
        'q0': 'q_start',
        'accept': ['q_accept'],
        'reject': ['q_reject'],
        'delta': {
            # Estado inicial: ler símbolo e ir para estado de checagem correspondente
            'q_start,0': {
                'next': 'q_check_sad',
                'write': '0',
                'dir': 'R'
            },
            'q_start,1': {
                'next': 'q_check_happy',
                'write': '1',
                'dir': 'R'
            },
            'q_start,_': {
                'next': 'q_reject',
                'write': '_',
                'dir': 'N'
            },
            
            # Estado q_check_sad: aceita se a análise decidiu 'sad', rejeita caso contrário
            'q_check_sad,_': {
                'next': 'q_accept' if dec_sad == 'sad' else 'q_reject',
                'write': '_',
                'dir': 'N'
            },
            
            # Estado q_check_happy: aceita se a análise decidiu 'happy', rejeita caso contrário
            'q_check_happy,_': {
                'next': 'q_accept' if dec_happy == 'happy' else 'q_reject',
                'write': '_',
                'dir': 'N'
            }
        },
        '_metadata': {
            'description': 'Máquina de Turing gerada com base na análise real das expressões faciais',
            'neutral->sad_decision': dec_sad,
            'neutral->happy_decision': dec_happy,
            'threshold': threshold,
            'symbol_mapping': {
                '0': 'Testa se neutral->sad foi classificado como sad',
                '1': 'Testa se neutral->happy foi classificado como happy',
                '_': 'blank (fim da fita)'
            },
            'logic': {
                'entrada_0': f"Lê 0 (sad) → vai para q_check_sad → {'ACEITA' if dec_sad == 'sad' else 'REJEITA'} (análise decidiu: {dec_sad})",
                'entrada_1': f"Lê 1 (happy) → vai para q_check_happy → {'ACEITA' if dec_happy == 'happy' else 'REJEITA'} (análise decidiu: {dec_happy})"
            }
        }
    }
    save_json(os.path.join(out_dir, 'turing_machine.json'), turing_machine)

    summary = {
        'face_graphs': ['face_neutral.json', 'face_sad.json', 'face_happy.json'],
        'diff_graphs': ['diff_neutral_sad_graph.json', 'diff_neutral_happy_graph.json'],
        'meta': ['diff_neutral_sad_meta.json', 'diff_neutral_happy_meta.json'],
        'automaton': 'automaton.json',
        'turing_machine': 'turing_machine.json'
    }
    save_json(os.path.join(out_dir, 'summary.json'), summary)
    print('Arquivos salvos em', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera dígrafos de faces e diferenças usando neutro como base')
    parser.add_argument('--neutral', required=True)
    parser.add_argument('--sad', required=True)
    parser.add_argument('--happy', required=True, help='imagem com a expressão alvo (happy)')
    parser.add_argument('--out', default='out_digraphs')
    parser.add_argument('--threshold', type=float, default=0.05, help='limiar normalizado (fração da diagonal)')
    args = parser.parse_args()
    main(args.neutral, args.sad, args.happy, args.out, threshold=args.threshold)
