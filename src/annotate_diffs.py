import os
import numpy as np
from .utils import load_json, ensure_dir
from .landmark_extractor import LandmarkExtractor
from .visualize import plot_landmarks


def annotate_topk(digraphs_dir, neutral_image_path, k=8):
    # paths
    sad_meta = os.path.join(digraphs_dir, 'diff_neutral_sad_meta.json')
    target_meta = os.path.join(digraphs_dir, 'diff_neutral_target_meta.json')
    ensure_dir(digraphs_dir)

    sm = load_json(sad_meta)
    tm = load_json(target_meta)

    s_difs = np.array(sm['difs'], dtype=float)
    t_difs = np.array(tm['difs'], dtype=float)

    s_idx = list(np.argsort(-s_difs)[:k])
    t_idx = list(np.argsort(-t_difs)[:k])

    # extrair landmarks da imagem neutra
    import cv2
    nb = cv2.imread(neutral_image_path)
    ext = LandmarkExtractor()
    n_lm = ext.from_bgr(nb)
    if n_lm is None:
        raise RuntimeError('Não foi possível extrair landmarks da imagem neutra')

    # gerar imagens anotadas: vamos usar plot_landmarks com marcação manual para destacar top-K
    # Cria cópia das imagens com os pontos destacados: para simplicidade, desenharemos círculos grandes nos pontos top-K
    # Usamos matplotlib drawing via plot_landmarks (acrescentando highlights via axes se necessário)

    # helper para desenhar com destaque
    def _plot_with_highlights(image, landmarks, highlight_idxs, out_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,8))
        plot_landmarks(image, landmarks, ax=ax, show=False, out_path=None, draw_cube=True)
        lm = np.array(landmarks)
        if len(highlight_idxs)>0:
            pts = lm[highlight_idxs]
            ax.scatter(pts[:,0], pts[:,1], c='yellow', s=120, edgecolors='black', linewidths=1.2, zorder=10)
            for idx,(x,y) in zip(highlight_idxs, pts):
                ax.text(x+2, y+2, str(int(idx)), color='black', fontsize=10, weight='bold')
        ax.axis('off')
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

    out_s = os.path.join(digraphs_dir, 'annotated_neutral_sad.png')
    out_t = os.path.join(digraphs_dir, 'annotated_neutral_target.png')
    _plot_with_highlights(nb, n_lm, s_idx, out_s)
    _plot_with_highlights(nb, n_lm, t_idx, out_t)

    return out_s, out_t


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--neutral', required=True)
    parser.add_argument('--k', type=int, default=8)
    args = parser.parse_args()
    a,b = annotate_topk(args.dir, args.neutral, k=args.k)
    print('Anotações salvas em:')
    print(' ', a)
    print(' ', b)
