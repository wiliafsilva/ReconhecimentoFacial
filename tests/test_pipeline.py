import numpy as np
from src.digraph import digraph_from_difference
from src.dfa import SimpleEmotionDFA


def test_digraph_and_binary():
    # cria landmarks neutros (pontos em linha)
    neutral = np.array([[i*10.0, 50.0] for i in range(20)])
    # target com movimento na boca (últimos 5 pontos)
    target = neutral.copy()
    target[15:] += np.array([0.0, 30.0])
    G, binary, difs = digraph_from_difference(neutral, target, threshold=5.0)
    assert sum(binary) >= 5


def test_simple_dfa():
    # cria vetor binário com mudanças na boca indices 15-19
    v = np.zeros(20, dtype=int)
    v[15:20] = 1
    dfa = SimpleEmotionDFA(mouth_indices=list(range(15,20)), eye_indices=[0,1], brow_indices=[10,11])
    label = dfa.predict(v)
    assert label in ('happy','neutral')
