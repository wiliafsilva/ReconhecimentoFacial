from typing import List, Dict

class SimpleEmotionDFA:
    """Máquina de estados que mapeia padrões binários para estados emocionais simples.
    - Pode receber vetor binário (N,) e um mapeamento de regiões (dict com listas de índices)
    - Usa thresholds percentuais por região para decidir emoção
    Suporta parâmetros legados mouth_indices, eye_indices, brow_indices para compatibilidade.
    """
    def __init__(self, regions=None, mouth_thresh=0.18, eyes_thresh=0.20, brows_thresh=0.12,
                 mouth_indices=None, eye_indices=None, brow_indices=None):
        # regions: {'mouth': [...], 'eyes': [...], 'brows': [...]} com índices
        self.regions = regions or {}
        # compatibilidade: se passado mouth_indices etc., converte para regions
        if mouth_indices:
            self.regions.setdefault('mouth', list(mouth_indices))
        if eye_indices:
            self.regions.setdefault('eyes', list(eye_indices))
        if brow_indices:
            self.regions.setdefault('brows', list(brow_indices))
        self.mouth_thresh = mouth_thresh
        self.eyes_thresh = eyes_thresh
        self.brows_thresh = brows_thresh

    def predict(self, binary_vec):
        # se fornecido dicionário com contagens
        if isinstance(binary_vec, dict) and 'counts' in binary_vec:
            counts = binary_vec['counts']
            sizes = binary_vec.get('sizes', {})
        else:
            v = binary_vec
            counts = {}
            sizes = {}
            for r in ('mouth','eyes','brows'):
                idxs = self.regions.get(r, [])
                sizes[r] = len(idxs)
                counts[r] = int(v[idxs].sum()) if idxs else 0

        props = {}
        for r in ('mouth','eyes','brows'):
            s = sizes.get(r, 0) or 1
            props[r] = counts.get(r, 0) / s

        # regras heurísticas em proporção
        # Detecção 'happy' também verifica se os top-K deslocamentos estão majoritariamente na boca
        if props['mouth'] >= self.mouth_thresh and props['eyes'] <= self.eyes_thresh:
            return 'happy'

        # Checar top-k variações (se v for vetor de 0/1, não temos magnitudes; o chamador pode passar dict com 'difs')
        # Se for passado vetor binário puro, essa verificação será ignorada
        if isinstance(binary_vec, dict) and 'difs' in binary_vec:
            difs = binary_vec['difs']
            # top-k indices
            k = min(10, len(difs))
            topk = sorted(range(len(difs)), key=lambda i: difs[i], reverse=True)[:k]
            mouth_idxs = set(self.regions.get('mouth', []))
            mouth_topk = sum(1 for i in topk if i in mouth_idxs)
            if k > 0 and (mouth_topk / k) >= 0.6 and props['mouth'] >= 0.08:
                return 'happy'
        if props['brows'] >= self.brows_thresh and props['mouth'] <= 0.1:
            return 'sad'
        total_changed = sum(counts.get(r,0) for r in ('mouth','eyes','brows'))
        if total_changed == 0:
            return 'neutral'
        # se muitos nós fora das 3 regiões principais -> reject
        # calcular total_ones e tamanho total (len_total)
        if isinstance(binary_vec, dict) and 'binary' in binary_vec:
            try:
                total_ones = int(binary_vec['binary'].sum())
            except Exception:
                total_ones = int(sum(binary_vec['binary']))
            len_total = len(binary_vec['binary'])
        elif not isinstance(binary_vec, dict):
            try:
                total_ones = int(binary_vec.sum())
            except Exception:
                total_ones = int(sum(binary_vec))
            len_total = len(binary_vec)
        else:
            total_ones = total_changed
            len_total = sum(sizes.values())
        if total_ones > max(50, 0.5 * len_total):
            return 'reject'
        return 'neutral'
