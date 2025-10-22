import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def _draw_stylized_cube(ax, bbox, color='blue', alpha=0.8, linewidth=2):
    # bbox: (x_min, y_min, x_max, y_max)
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    # 2D projection of a rotated cube for style (simple offset)
    offset_x = 0.15 * w
    offset_y = -0.12 * h
    # front rectangle
    front = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    # back rectangle (shifted)
    back = front + np.array([offset_x, offset_y])
    # draw filled polygons (light) and edges
    ax.plot(front[:,0], front[:,1], color=color, linewidth=linewidth)
    ax.plot(back[:,0], back[:,1], color=color, linewidth=linewidth)
    for i in range(4):
        ax.plot([front[i,0], back[i,0]], [front[i,1], back[i,1]], color=color, linewidth=linewidth)


def plot_landmarks(image, landmarks, ax=None, show=True, out_path=None, title=None, draw_cube=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(image[...,::-1])
    lm = np.array(landmarks)
    # landmarks: red dots
    ax.scatter(lm[:,0], lm[:,1], c='#cc0000', s=18, zorder=3)
    # minimal index labels for debugging (optional small alpha)
    #for i,(x,y) in enumerate(lm):
    #    ax.text(x,y,str(i),color='yellow',fontsize=6, alpha=0.6)
    if draw_cube:
        # compute bbox from landmarks (tight)
        x_min, y_min = lm[:,0].min(), lm[:,1].min()
        x_max, y_max = lm[:,0].max(), lm[:,1].max()
        _draw_stylized_cube(ax, (x_min, y_min, x_max, y_max), color='#1f4fa6', linewidth=2)
    # highlight approximate eye regions if indices available (heuristic)
    n = len(lm)
    if n >= 68:
        # OpenFace/Dlib 68 -> eyes indices
        left_eye = lm[36:42]
        right_eye = lm[42:48]
    else:
        # fallback: assume eyes roughly upper third
        ys = lm[:,1]
        top_mask = ys < ys.mean()
        top_idx = np.where(top_mask)[0]
        left_eye = lm[top_idx[:6]] if len(top_idx) >= 6 else lm[:6]
        right_eye = lm[top_idx[6:12]] if len(top_idx) >= 12 else lm[6:12]
    def _draw_eye_ring(pts):
        if len(pts) == 0:
            return
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
        rx = max(4.0, np.ptp(pts[:,0]) * 0.6)
        ry = max(3.0, np.ptp(pts[:,1]) * 0.6)
        circ = plt.Circle((cx, cy), max(rx, ry), color='#33cc33', fill=False, linewidth=2, alpha=0.9)
        ax.add_patch(circ)
    _draw_eye_ring(np.array(left_eye))
    _draw_eye_ring(np.array(right_eye))

    if title:
        ax.set_title(title)
    ax.axis('off')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    if show:
        plt.show()


def plot_diff_graph(image, landmarks, diff_graph, out_path=None):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(image[...,::-1])
    lm = np.array(landmarks)
    # draw cube in background
    try:
        x_min, y_min = lm[:,0].min(), lm[:,1].min()
        x_max, y_max = lm[:,0].max(), lm[:,1].max()
            # cubo removido conforme solicitado (mantemos a função caso queira reativar)
            # _draw_stylized_cube(ax, (x_min, y_min, x_max, y_max), color='#33cc33', linewidth=2.6)
    except Exception:
        pass

    # Draw yellow landmarks across the face with reduced intensity
    try:
        ax.scatter(lm[:,0], lm[:,1], c='#FF0000', s=2, alpha=1, zorder=2)
    except Exception:
        pass

    # Helper: extract numeric weight for an edge data dict
    def _edge_weight(edata, a, b):
        if not isinstance(edata, dict):
            return 1.0
        for k in ('weight', 'w', 'dist', 'distance'):
            if k in edata:
                try:
                    return float(edata[k])
                except Exception:
                    pass
        # fallback: geometric distance between points if available
        try:
            return float(np.linalg.norm(lm[a] - lm[b]))
        except Exception:
            return 1.0

    # collect edges with weights
    raw_edges = list(diff_graph.edges(data=True))
    if len(raw_edges) == 0:
        # nothing to draw
        ax.axis('off')
        if out_path:
            plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        return

    edges_with_w = []
    for a,b,edata in raw_edges:
        w = _edge_weight(edata, a, b)
        edges_with_w.append((a,b,w))

    weights = np.array([w for _,_,w in edges_with_w])
    # choose a threshold dynamically (keep very top edges) to avoid grid-like noise
    try:
        pct = 98
        thresh = np.percentile(weights, pct)
    except Exception:
        thresh = weights.mean()

    selected = [(a,b,w) for (a,b,w) in edges_with_w if w >= thresh]
    # If percentile left us with none (very skewed), fallback to top-K
    if len(selected) == 0:
        # fallback to a very small number of strongest edges
        k = min(8, max(2, int(len(edges_with_w) * 0.06)))
        edges_sorted = sorted(edges_with_w, key=lambda x: x[2], reverse=True)
        selected = edges_sorted[:k]

    # limit selected edges to top-K by weight to avoid many small grid-lines
    if len(selected) > 8:
        selected = sorted(selected, key=lambda x: x[2], reverse=True)[:8]

    # Build set of nodes involved in selected edges
    sel_nodes = set()
    for a,b,_ in selected:
        sel_nodes.add(a); sel_nodes.add(b)

    # For the visual style requested, we won't draw the diff edges as lines
    # — instead we emphasize the nodes involved.

    # emphasize selected nodes (moderate red dots)
    if len(sel_nodes) > 0:
        pts = lm[list(sel_nodes)]
    # red highlighted nodes with white border (reduced size)
    ax.scatter(pts[:,0], pts[:,1], c='#ff3333', s=24, edgecolors='white', linewidths=0.6, zorder=6)


    ax.axis('off')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    plt.close()
