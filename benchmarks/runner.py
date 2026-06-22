"""Experiment runner that wires parsing, allocation, compression, and evaluation."""
from typing import Dict, Any
from domain_kv.section_parser import extract_sections
from domain_kv.allocator import allocate_budgets
from domain_kv.compressor import compress_kv_cache
from domain_kv.kvpress_adapter import KVPressAdapter
from benchmarks.loader import load_sample_notes
import time


class ExperimentRunner:
    def __init__(self, total_budget: int = 128, quantize: str = 'float16'):
        self.total_budget = total_budget
        self.quantize = quantize

    def build_kv_from_note(self, note_text: str) -> Dict[str, list]:
        secs = extract_sections(note_text)
        # naive per-section KV entries: one vector per sentence token
        kv = {}
        for s, t in secs.items():
            # represent each sentence by a small random vector placeholder
            lines = [ln for ln in t.splitlines() if ln.strip()]
            kv[s] = []
            for i, ln in enumerate(lines):
                vec = self._fake_vector(ln, i)
                kv[s].append((f"{s}_sent_{i}", vec))
        return kv

    def _fake_vector(self, text: str, seed: int):
        import numpy as np
        rng = np.random.RandomState(abs(hash(text)) % (2**31) + seed)
        return rng.randn(256).astype('float32')

    def run_once(self, note_id: str, note_text: str) -> Dict[str, Any]:
        start = time.time()
        kv = self.build_kv_from_note(note_text)
        secs = extract_sections(note_text)
        budgets = allocate_budgets(secs, total_budget=self.total_budget)
        compressed = compress_kv_cache(kv, budgets, quantize=self.quantize)
        adapter = KVPressAdapter()
        adapter.apply_compressed(compressed)
        size = adapter.size_by_section()
        elapsed = time.time() - start
        # mock metric: retained fraction of original items
        orig_counts = {k: len(v) for k, v in kv.items()}
        retained_frac = {k: size.get(k, 0) / max(1, orig_counts.get(k, 0)) for k in orig_counts}
        return {
            'note_id': note_id,
            'orig_counts': orig_counts,
            'retained': size,
            'retained_frac': retained_frac,
            'elapsed': elapsed,
            'budgets': budgets,
        }

    def run_dataset(self):
        results = []
        for nid, note in load_sample_notes():
            res = self.run_once(nid, note)
            results.append(res)
        return results
