"""Adapter hooks for integrating with KVPress/vLLM style KV caches.

This module provides a lightweight class that demonstrates how to connect a
running KV store to the compressor. For real integration, adapt to the target
engine's callback API (KVPress or vLLM)."""
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class KVPressAdapter:
    """Minimal adapter modelling a section-tagged KV cache with simple LRU timestamps.

    Entries are stored as mapping section -> list of tuples `(key, vector, ts)` where
    `ts` is a timestamp used for eviction ordering. This class is intended as a
    lightweight shim for experiments and should be replaced by engine-specific
    adapters for production integration.
    """
    kv_cache: Dict[str, List[Tuple[str, Any, float]]] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, List[Tuple[str, Any, float]]]:
        """Return a shallow-copy snapshot of the KV cache organized by section."""
        return {k: list(v) for k, v in self.kv_cache.items()}

    def apply_compressed(self, compressed: Dict[str, List[Tuple[str, Any]]]):
        """Apply a compressed snapshot back into the running kv cache.

        Accepts mapping section -> list of (key, payload) and sets current timestamps.
        """
        self.kv_cache.clear()
        now = time.time()
        for k, v in compressed.items():
            self.kv_cache[k] = [(item[0], item[1], now + i * 1e-6) for i, item in enumerate(v)]

    def insert(self, section: str, key: str, vector: Any):
        """Insert a KV entry into a named section, tagging with a timestamp."""
        now = time.time()
        self.kv_cache.setdefault(section, []).append((key, vector, now))

    def size_by_section(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.kv_cache.items()}

    def evict_lru(self, section: str, keep: int):
        """Evict oldest entries in `section` to keep at most `keep` items."""
        items = self.kv_cache.get(section, [])
        if len(items) <= keep:
            return
        # items are oldest->newest if timestamps are increasing; sort to be safe
        items_sorted = sorted(items, key=lambda x: x[2])
        self.kv_cache[section] = items_sorted[-keep:]
