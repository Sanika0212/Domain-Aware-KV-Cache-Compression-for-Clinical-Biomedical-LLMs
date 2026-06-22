"""Real metrics computed from a HuggingFace `transformers` KV cache.

Operates on `transformers.cache_utils.DynamicCache` (or any object exposing a
`.layers` list with `.keys`/`.values` tensors), not on a mock dict.
"""


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
