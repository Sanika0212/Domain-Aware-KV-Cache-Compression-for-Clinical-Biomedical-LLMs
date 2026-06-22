"""Real metrics computed from a HuggingFace `transformers` KV cache.

Operates on `transformers.cache_utils.DynamicCache` (or any object exposing a
`.layers` list with `.keys`/`.values` tensors), not on a mock dict.
"""
from typing import Dict


def cache_num_tokens(cache, layer_idx: int = 0) -> int:
    """Number of cached tokens (sequence length) at a given layer."""
    return int(cache.layers[layer_idx].keys.shape[2])


def cache_bytes(cache) -> int:
    """Total bytes used by keys+values across all layers of a KV cache."""
    total = 0
    for layer in cache.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
    return total


def retention_ratio(compressed_cache, original_num_tokens: int, layer_idx: int = 0) -> float:
    """Fraction of original tokens retained after compression."""
    if original_num_tokens == 0:
        return 0.0
    return cache_num_tokens(compressed_cache, layer_idx) / original_num_tokens


def per_section_retention(section_ids, kept_mask) -> Dict[int, float]:
    """Given the original per-token section ids and a boolean kept-mask of the
    same length, return {section_id: fraction_kept}.
    """
    import numpy as np

    section_ids = np.asarray(section_ids)
    kept_mask = np.asarray(kept_mask, dtype=bool)
    out: Dict[int, float] = {}
    for sec_id in np.unique(section_ids):
        sec_mask = section_ids == sec_id
        total = int(sec_mask.sum())
        kept = int((sec_mask & kept_mask).sum())
        out[int(sec_id)] = kept / total if total else 0.0
    return out
