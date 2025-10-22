import cv2
import numpy as np
from .utils import landmarks_to_np

try:
    import mediapipe as mp
except Exception:
    mp = None


class LandmarkExtractor:
    """Extrator de landmarks. Prefere MediaPipe, pode usar CSV do OpenFace com adapter."""
    def __init__(self, use_mediapipe=True):
        self.use_mediapipe = use_mediapipe and (mp is not None)
        if self.use_mediapipe:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

    def from_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        return self.from_bgr(img)

    def from_bgr(self, bgr_image):
        if self.use_mediapipe:
            img_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return None
            lm = results.multi_face_landmarks[0]
            h, w, _ = bgr_image.shape
            pts = [(p.x * w, p.y * h) for p in lm.landmark]
            return landmarks_to_np(pts)
        else:
            raise RuntimeError('MediaPipe não disponível. Use adapter OpenFace.')


def parse_openface_csv(csv_path):
    """Leitor simples do CSV de FeatureExtraction de OpenFace para retornar landmarks x,y.
    Procura colunas como x_0, y_0, ... ou landmark_0_x etc. Implementação simplificada.
    """
    import pandas as pd
    import re
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()
    # busca padrões comuns: 'x_0','y_0' ou 'X0','Y0' ou 'landmark_0_x'
    pairs = {}
    rx1 = re.compile(r'(?i)^(?:x[_-]?|X)(\d+)$')
    rx2 = re.compile(r'(?i)^(?:y[_-]?|Y)(\d+)$')
    rx3 = re.compile(r'(?i)^(?:.*?_)?(\d+)[_\-]?(?:x)$')
    rx4 = re.compile(r'(?i)^(?:.*?_)?(\d+)[_\-]?(?:y)$')
    for c in cols:
        m = rx1.match(c)
        if m:
            pairs.setdefault(int(m.group(1)), {})['x'] = c
            continue
        m = rx2.match(c)
        if m:
            pairs.setdefault(int(m.group(1)), {})['y'] = c
            continue
        m = rx3.match(c)
        if m:
            pairs.setdefault(int(m.group(1)), {})['x'] = c
            continue
        m = rx4.match(c)
        if m:
            pairs.setdefault(int(m.group(1)), {})['y'] = c
            continue

    # sort by index
    idxs = sorted([i for i, d in pairs.items() if 'x' in d and 'y' in d])
    if not idxs:
        # tenta colunas com pattern x_0 y_0 anywhere
        xcols = [c for c in cols if re.search(r'(?i)\bx[_-]?\d+\b', c)]
        ycols = [c for c in cols if re.search(r'(?i)\by[_-]?\d+\b', c)]
        if len(xcols) == len(ycols) and len(xcols) > 0:
            xcols.sort(); ycols.sort()
            pts = [(df[xcols[i]].iat[0], df[ycols[i]].iat[0]) for i in range(len(xcols))]
            return landmarks_to_np(pts)
        raise ValueError('Formato CSV OpenFace não reconhecido; verifique as colunas')

    pts = [(df[pairs[i]['x']].iat[0], df[pairs[i]['y']].iat[0]) for i in idxs]
    return landmarks_to_np(pts)
