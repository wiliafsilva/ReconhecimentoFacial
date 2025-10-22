import os
import json
import numpy as np
from .utils import load_json, bbox_from_landmarks, map_landmarks_to_regions
from .landmark_extractor import LandmarkExtractor


def summarize_diff(meta_path, landmarks):
    meta = load_json(meta_path)
    binary = np.array(meta['binary'], dtype=int)
    difs = np.array(meta['difs'], dtype=float)
    changed_idx = np.where(binary == 1)[0].tolist()
    total_changed = int(binary.sum())
    # top changes by magnitude
    top_k = 8
    idx_sorted = np.argsort(-difs)
    top = [(int(i), float(difs[i])) for i in idx_sorted[:top_k] if difs[i] > 0]

    # regions
    regions = map_landmarks_to_regions(landmarks)
    region_counts = {r: int(binary[idxs].sum()) if idxs else 0 for r, idxs in regions.items()}

    return {
        'total_changed': total_changed,
        'changed_indices': changed_idx,
        'top_changes': top,
        'region_counts': region_counts
    }


def main(digraphs_dir, neutral_img_path=None):
    # carregar landmarks da face neutra para mapear regiões
    ext = LandmarkExtractor()
    import cv2
    if neutral_img_path is None:
        # tentar encontrar imagens e extrair landmarks da pasta pai
        raise ValueError('Forneça o caminho da imagem neutra para extrair landmarks')
    nb = cv2.imread(neutral_img_path)
    if nb is None:
        raise FileNotFoundError(neutral_img_path)
    n_lm = ext.from_bgr(nb)
    if n_lm is None:
        raise RuntimeError('Não foi possível extrair landmarks da imagem neutra')

    sad_meta = os.path.join(digraphs_dir, 'diff_neutral_sad_meta.json')
    target_meta = os.path.join(digraphs_dir, 'diff_neutral_happy_meta.json')
    if not os.path.exists(sad_meta) or not os.path.exists(target_meta):
        raise FileNotFoundError('Arquivos de meta diffs não encontrados em ' + digraphs_dir)

    ssum = summarize_diff(sad_meta, n_lm)
    tsum = summarize_diff(target_meta, n_lm)

    print('Resumo da comparação: NEUTRAL -> SAD')
    print('  Landmarks alterados (total):', ssum['total_changed'])
    print('  Contagem por região:', ssum['region_counts'])
    print('  Top mudanças (idx, magnitude):')
    for i, v in ssum['top_changes']:
        print(f'    {i}: {v:.4f}')

    print('\nResumo da comparação: NEUTRAL -> HAPPY')
    print('  Landmarks alterados (total):', tsum['total_changed'])
    print('  Contagem por região:', tsum['region_counts'])
    print('  Top mudanças (idx, magnitude):')
    for i, v in tsum['top_changes']:
        print(f'    {i}: {v:.4f}')

    # indicar onde estão os PNGs
    print('\nVisualizações geradas:')
    print('  ', os.path.join(digraphs_dir, 'diff_neutral_sad.png'))
    print('  ', os.path.join(digraphs_dir, 'diff_neutral_happy.png'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='diretório com os digraphs (summary.json)')
    parser.add_argument('--neutral', required=True, help='caminho para a imagem neutra (para extrair landmarks)')
    args = parser.parse_args()
    main(args.dir, args.neutral)
