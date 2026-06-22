import numpy as np
from domain_kv.compressor import evict_by_section, compress_kv_cache


def test_evict_by_section():
    kv = {
        'a': [(f'a{i}', np.array([i])) for i in range(5)],
        'b': [(f'b{i}', np.array([i])) for i in range(3)],
    }
    budgets = {'a': 2, 'b': 1}
    out = evict_by_section(kv, budgets)
    assert len(out['a']) == 2
    assert out['a'][0][0] == 'a3'
    assert len(out['b']) == 1


def test_quantize_and_compress():
    kv = {'sec': [(f'k{i}', np.linspace(0,1,8).astype(np.float32)) for i in range(4)]}
    budgets = {'sec': 2}
    comp = compress_kv_cache(kv, budgets, quantize='uint8')
    assert 'sec' in comp
    assert len(comp['sec']) == 2
    item = comp['sec'][0][1]
    assert isinstance(item, dict) and 'q' in item
