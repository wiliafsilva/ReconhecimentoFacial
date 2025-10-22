import networkx as nx
import numpy as np
from .utils import bbox_from_landmarks, face_scale_from_bbox


def build_face_digraph(landmarks):
    """Cria um dígrafo onde cada landmark é um nó. A aresta direção pode representar distância/ângulo relativo.
    Implementação simples: conexão k-nearest neighbors direcionada por vetor de diferença.
    landmarks: np.array (N,2)
    """
    N = len(landmarks)
    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, xy=tuple(landmarks[i].tolist()))

    # k-NN (k pequeno)
    from sklearn.neighbors import NearestNeighbors
    k = min(6, N-1)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(landmarks)
    distances, indices = nbrs.kneighbors(landmarks)
    for i in range(N):
        for j_idx in range(1, k+1):
            j = indices[i, j_idx]
            vec = landmarks[j] - landmarks[i]
            weight = np.linalg.norm(vec)
            G.add_edge(i, j, weight=float(weight))
    return G


def digraph_from_difference(neutral_landmarks, target_landmarks, threshold=0.05, normalize=True):
    """Gera um digrafo de diferença.

    - Se normalize=True: calcula deslocamento por nó normalizado pela escala da face (diagonal do bbox).
      threshold nesse caso é uma fração (ex.: 0.05 = 5% da diagonal).
    - Retorna (G_changed, binary_vector)
    """
    N = len(neutral_landmarks)
    if normalize:
        bbox = bbox_from_landmarks(neutral_landmarks)
        scale = face_scale_from_bbox(bbox)
        # if threshold > 1 assume user passed pixels; convert to normalized fraction
        if threshold > 1.0:
            thr = float(threshold) / (scale + 1e-9)
        else:
            thr = float(threshold)
        dif = np.linalg.norm(target_landmarks - neutral_landmarks, axis=1) / (scale + 1e-9)
    else:
        thr = float(threshold)
        dif = np.linalg.norm(target_landmarks - neutral_landmarks, axis=1)

    changed = (dif >= thr).astype(int)
    G = nx.DiGraph()
    for i in range(N):
        if changed[i]:
            G.add_node(i, change=float(dif[i]))

    # conectar nós que mudaram entre si se estiverem a menos de 0.15 * scale (se normalize) ou 50 px
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            dist = np.linalg.norm(target_landmarks[a] - target_landmarks[b])
            limit = 0.15 * scale if normalize else 50.0
            if dist < limit:
                G.add_edge(a, b, weight=float(dist))
                G.add_edge(b, a, weight=float(dist))
    return G, changed, dif
