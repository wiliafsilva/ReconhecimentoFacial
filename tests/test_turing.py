import pytest
from src.turing import TuringMachine


def test_tm_to_dict_and_graphviz():
    tm = TuringMachine.sample_majority_tm()
    d = tm.to_dict()
    assert 'Q' in d and isinstance(d['Q'], list)
    assert 'delta' in d and isinstance(d['delta'], dict)
    gv = tm.to_graphviz()
    assert 'digraph' in gv


def test_tm_transitions_serializable():
    tm = TuringMachine.sample_majority_tm()
    # garantir que todas as chaves e valores são serializáveis em strings simples
    for k, v in tm.to_dict()['delta'].items():
        assert isinstance(k, str)
        assert 'next' in v and 'write' in v and 'dir' in v
