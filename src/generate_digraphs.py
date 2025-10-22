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

    # construir autômato simples: neutro->target (happy) = 1, neutro->sad = 0
    automaton = {
        'neutral->happy': 1,
        'neutral->sad': 0
    }
    save_json(os.path.join(out_dir, 'automaton.json'), automaton)

    summary = {
        'face_graphs': ['face_neutral.json', 'face_sad.json', 'face_happy.json'],
        'diff_graphs': ['diff_neutral_sad_graph.json', 'diff_neutral_happy_graph.json'],
        'meta': ['diff_neutral_sad_meta.json', 'diff_neutral_happy_meta.json'],
        'automaton': 'automaton.json'
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
