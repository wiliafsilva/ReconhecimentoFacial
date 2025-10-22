import numpy as np
from .landmark_extractor import LandmarkExtractor
from .digraph import digraph_from_difference, build_face_digraph
from .dfa import SimpleEmotionDFA
from .utils import map_landmarks_to_regions, bbox_from_landmarks


class FacialStatePipeline:
    def __init__(self, extractor=None, dfa=None, threshold=2.0):
        self.extractor = extractor or LandmarkExtractor()
        self.threshold = threshold
        self.dfa = dfa or SimpleEmotionDFA()

    def analyze_pair(self, neutral_bgr, happy_bgr):
        n_lm = self.extractor.from_bgr(neutral_bgr)
        t_lm = self.extractor.from_bgr(happy_bgr)
        if n_lm is None or t_lm is None:
            return {'label':'reject','reason':'no_face'}
        if len(n_lm) != len(t_lm):
            return {'label':'reject','reason':'landmark_count_mismatch'}

        # construir grafos (opcional)
        G_neutral = build_face_digraph(n_lm)
        G_target = build_face_digraph(t_lm)

        # mapear regiões a partir do neutro
        bbox = bbox_from_landmarks(n_lm)
        regions = map_landmarks_to_regions(n_lm, bbox=bbox)

        # gerar diff normalizado por escala da face
        diff_graph, binary, difs = digraph_from_difference(n_lm, t_lm, threshold=self.threshold, normalize=True)

        # instanciar DFA com regiões encontradas
        dfa = SimpleEmotionDFA(regions=regions)
        # passar tanto o vetor binário quanto as magnitudes para o DFA (se suportado)
        dfa_input = {
            'counts': {r: int(binary[idxs].sum()) if idxs else 0 for r, idxs in regions.items()},
            'sizes': {r: len(idxs) for r, idxs in regions.items()},
            'difs': difs,
            'binary': binary
        }
        label = dfa.predict(dfa_input)

        # estatísticas por região
        counts = {r: int(binary[idxs].sum()) if idxs else 0 for r, idxs in regions.items()}
        sizes = {r: len(idxs) for r, idxs in regions.items()}

        return {
            'label': label,
            'binary': binary.tolist() if isinstance(binary, np.ndarray) else binary,
            'diff_nodes': list(diff_graph.nodes()),
            'diff_graph': diff_graph,
            'counts': counts,
            'sizes': sizes,
            'difs': difs.tolist() if isinstance(difs, np.ndarray) else difs
        }

    def analyze_images(self, neutral_path, happy_path):
        import cv2
        nb = cv2.imread(neutral_path)
        tb = cv2.imread(happy_path)
        return self.analyze_pair(nb, tb)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--neutral', required=True)
    parser.add_argument('--happy', required=True, help='imagem com expressão alvo (happy)')
    parser.add_argument('--threshold', type=float, default=2.0, help='threshold em pixels (>1) ou fração (<=1)')
    parser.add_argument('--visualize', action='store_true', help='gera imagens de visualização em --out')
    parser.add_argument('--out', default=None, help='diretório de saída para visualizações')
    args = parser.parse_args()
    p = FacialStatePipeline(threshold=args.threshold)
    out = p.analyze_images(args.neutral, args.happy)
    import json, os
    # diff_graph não é serializável; substituir por lista de arestas resumida
    out_printable = dict(out)
    if 'diff_graph' in out_printable:
        dg = out_printable.pop('diff_graph')
        out_printable['diff_edges'] = [(int(a), int(b), float(dg[a][b].get('weight', 0.0))) for a,b in dg.edges()]
    print(json.dumps(out_printable, indent=2))
    if args.visualize and args.out:
        os.makedirs(args.out, exist_ok=True)
        from .visualize import plot_diff_graph
        import cv2
        nb = cv2.imread(args.neutral)
        # res já contém 'diff_graph' e 'difs'
        res = out
        n_lm = p.extractor.from_bgr(nb)
        # salvar visualização do diff
        plot_diff_graph(nb, n_lm, res['diff_graph'], out_path=os.path.join(args.out, 'diff.png'))
