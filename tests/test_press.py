import torch
import pytest

from kvpress.presses.knorm_press import KnormPress
from domain_kv.press import DomainAwarePress


class FakeModule:
    """Stands in for an nn.Module attention layer; DomainAwarePress.compress
    only reads `.head_dim` from it."""

    head_dim = 4


def make_keys_values(seq_len: int, head_dim: int = 4, num_kv_heads: int = 2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(1, num_kv_heads, seq_len, head_dim, generator=g)
    values = torch.randn(1, num_kv_heads, seq_len, head_dim, generator=g)
    return keys, values


def test_compress_respects_per_section_budgets():
    seq_len = 10
    # sections: [0,0,0,0,0, 1,1,1,1,1]
    section_ids = torch.tensor([0] * 5 + [1] * 5)
    budgets = {0: 1, 1: 4}

    keys, values = make_keys_values(seq_len)
    hidden_states = torch.zeros(1, seq_len, 8)

    press = DomainAwarePress(base_press=KnormPress())
    press.set_document(section_ids, budgets)

    new_keys, new_values = press.compress(FakeModule(), hidden_states, keys, values, None, {})

    assert new_keys.shape[2] == 5  # 1 + 4
    assert new_values.shape[2] == 5


def test_compress_keeps_lowest_norm_tokens_within_a_section():
    # Single section; budget=2 out of 5. KnormPress scores -norm (per
    # https://arxiv.org/pdf/2406.11430, low key-norm correlates with high
    # attention), so the two *smallest*-norm keys must be the ones kept.
    seq_len = 5
    section_ids = torch.zeros(seq_len, dtype=torch.long)
    budgets = {0: 2}

    keys = torch.zeros(1, 1, seq_len, 4)
    for i in range(seq_len):
        keys[0, 0, i, :] = float(i + 1)  # norms strictly increasing with index
    values = torch.zeros(1, 1, seq_len, 4)
    hidden_states = torch.zeros(1, seq_len, 4)

    press = DomainAwarePress(base_press=KnormPress())
    press.set_document(section_ids, budgets)
    new_keys, _ = press.compress(FakeModule(), hidden_states, keys, values, None, {})

    kept_norms = new_keys[0, 0, :, 0].tolist()
    assert sorted(kept_norms) == [1.0, 2.0]


def test_compress_raises_without_set_document():
    press = DomainAwarePress(base_press=KnormPress())
    keys, values = make_keys_values(6)
    hidden_states = torch.zeros(1, 6, 8)
    with pytest.raises(RuntimeError):
        press.compress(FakeModule(), hidden_states, keys, values, None, {})


def test_compress_rejects_mismatched_section_ids_length():
    press = DomainAwarePress(base_press=KnormPress())
    press.set_document(torch.zeros(4, dtype=torch.long), {0: 2})
    keys, values = make_keys_values(6)  # length mismatch: 6 != 4
    hidden_states = torch.zeros(1, 6, 8)
    with pytest.raises(ValueError):
        press.compress(FakeModule(), hidden_states, keys, values, None, {})
