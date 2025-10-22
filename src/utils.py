import os
import json
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(path, obj):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def landmarks_to_np(landmarks):
    """Converte lista de (x,y) ou (x,y,z) para numpy array shape (N,2) ou (N,3)"""
    return np.array(landmarks, dtype=float)


def bbox_from_landmarks(landmarks):
    """Retorna bbox (x_min, y_min, x_max, y_max) para array (N,2)"""
    lm = np.array(landmarks)
    x_min = float(lm[:,0].min())
    y_min = float(lm[:,1].min())
    x_max = float(lm[:,0].max())
    y_max = float(lm[:,1].max())
    return (x_min, y_min, x_max, y_max)


def face_scale_from_bbox(bbox):
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min
    # usar diagonal como escala
    return float((w**2 + h**2) ** 0.5)


def map_landmarks_to_regions(landmarks):
    """Mapeia landmarks em regiões.

    - Se detectar 68 pontos (formato OpenFace / dlib 68), usa mapeamento fixo para mouth/eyes/brows.
    - Caso contrário, cai na heurística por bandas verticais.
    """
    lm = np.array(landmarks)
    N = len(lm)
    if N == 68:
        return map_landmarks_to_regions_openface68()
    ys = lm[:,1]
    # dividir em três bandas verticais simples
    y_min, y_max = ys.min(), ys.max()
    h = y_max - y_min + 1e-9
    top_thr = y_min + 0.33 * h
    mid_thr = y_min + 0.66 * h
    mouth = [i for i in range(N) if lm[i,1] >= mid_thr]
    eyes = [i for i in range(N) if lm[i,1] >= top_thr and lm[i,1] < mid_thr]
    brows = [i for i in range(N) if lm[i,1] < top_thr]
    return {'mouth': mouth, 'eyes': eyes, 'brows': brows}


def map_landmarks_to_regions_openface68():
    """Retorna mapeamento de índices para o esquema de 68 landmarks (dlib/OpenFace).

    Índices baseados no padrão dlib 68:
    - olhos: 36-47 (inclusive)
    - sobrancelhas: 17-26 (inclusive)
    - boca: 48-67 (inclusive)
    Retorna dict com listas de índices (0-based).
    """
    mouth = list(range(48, 68))
    eyes = list(range(36, 48))
    brows = list(range(17, 27))
    return {'mouth': mouth, 'eyes': eyes, 'brows': brows}


def bbox_from_landmarks(landmarks):
    """Retorna bbox (x_min,y_min,x_max,y_max) a partir de landmarks (N,2)"""
    lm = np.array(landmarks)
    x_min = float(np.min(lm[:, 0]))
    x_max = float(np.max(lm[:, 0]))
    y_min = float(np.min(lm[:, 1]))
    y_max = float(np.max(lm[:, 1]))
    return x_min, y_min, x_max, y_max


def face_scale_from_bbox(bbox):
    """Retorna uma medida de escala da face a partir do bbox: diagonal (pixels)
    Usado para normalizar deslocamentos absolutos.
    """
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min
    return float(np.sqrt(w * w + h * h))


def map_landmarks_to_regions(landmarks, bbox=None):
    """Mapeia índices de landmarks para regiões aproximadas (mouth, eyes, brows)
    Usa partição vertical relativa do bbox: brows (top 0-33%), eyes (33-60%), mouth (60-100%).
    Retorna dict com chaves 'mouth','eyes','brows' e valores lista de índices.
    """
    lm = np.array(landmarks)
    if bbox is None:
        bbox = bbox_from_landmarks(lm)
    x_min, y_min, x_max, y_max = bbox
    h = y_max - y_min
    # limites relativos
    brow_th = y_min + 0.33 * h
    eye_th = y_min + 0.60 * h
    mouth_th = y_min + 0.60 * h

    mouth_idx = []
    eyes_idx = []
    brows_idx = []
    for i, (x, y) in enumerate(lm):
        if y >= mouth_th:
            mouth_idx.append(i)
        elif y >= brow_th and y < eye_th:
            eyes_idx.append(i)
        else:
            brows_idx.append(i)

    return {'mouth': mouth_idx, 'eyes': eyes_idx, 'brows': brows_idx}
