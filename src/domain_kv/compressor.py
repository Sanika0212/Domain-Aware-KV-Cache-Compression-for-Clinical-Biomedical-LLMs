"""Compression primitives for KV caches.

This module defines simple, explainable compression strategies:
- section-aware eviction to meet a per-section budget (LRU-style)
- dtype downcasting (float32 -> float16)
- naive 8-bit quantization via linear scaling
"""
from typing import Dict, Any, Tuple, List
import numpy as np


def float16_downcast(vector: np.ndarray) -> np.ndarray:
    return vector.astype(np.float16)


def linear_uint8_quantize(vector: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Quantize to uint8 with min/max scale. Returns (uint8_array, min, max)."""
    mn = float(vector.min())
    mx = float(vector.max())
    if mx - mn < 1e-8:
        return (np.zeros_like(vector, dtype=np.uint8), mn, mx)
    scaled = (vector - mn) / (mx - mn)
    q = np.clip((scaled * 255.0).round(), 0, 255).astype(np.uint8)
    return q, mn, mx


def evict_by_section(kv_cache: Dict[str, List[Tuple[str, np.ndarray]]], budgets: Dict[str, int]) -> Dict[str, List[Tuple[str, Any]]]:
    """Evict oldest items per section until per-section budgets met.

    kv_cache: mapping section -> list of (key, vector) ordered oldest->newest
    budgets: mapping section -> max allowed items
    Returns new kv_cache with evictions applied.
    """
    new_cache = {}
    for sec, items in kv_cache.items():
        allowed = budgets.get(sec, 0)
        if allowed <= 0:
            new_cache[sec] = []
            continue
        # keep newest allowed items
        new_cache[sec] = items[-allowed:]
    return new_cache


def compress_kv_cache(kv_cache: Dict[str, List[Tuple[str, np.ndarray]]], budgets: Dict[str, int], quantize: str = None) -> Dict[str, List[Tuple[str, Any]]]:
    """Apply eviction and optional quantization/downcast.

    quantize: None | 'float16' | 'uint8'
    """
    compressed = evict_by_section(kv_cache, budgets)
    out = {}
    for sec, items in compressed.items():
        new_items = []
        for k, v in items:
            if quantize == 'float16':
                new_items.append((k, float16_downcast(v)))
            elif quantize == 'uint8':
                q, mn, mx = linear_uint8_quantize(v)
                new_items.append((k, {'q': q, 'min': mn, 'max': mx}))
            else:
                new_items.append((k, v))
        out[sec] = new_items
    return out
