"""Small demo showing section parsing, budget allocation, and compression."""
from domain_kv.section_parser import extract_sections
from domain_kv.allocator import allocate_budgets
from domain_kv.compressor import compress_kv_cache
from domain_kv.kvpress_adapter import KVPressAdapter
import numpy as np


def build_fake_kv(sections):
    kv = {}
    for s, txt in sections.items():
        tokens = max(1, len(txt.split()))
        # create a few fake vectors per section
        items = []
        for i in range(min(10, tokens)):
            vec = np.random.randn(768).astype(np.float32)
            items.append((f"{s}_tok_{i}", vec))
        kv[s] = items
    return kv


def main():
    sample = """
Chief complaint:
Patient reports chest pain.

History of present illness:
Symptoms ongoing for 2 days. No prior MI.

Medications:
Aspirin 81mg daily.

Assessment and plan:
Observe and obtain troponin.
"""

    secs = extract_sections(sample)
    print("Sections:", list(secs.keys()))
    kv = build_fake_kv(secs)
    adapter = KVPressAdapter(kv_cache=kv)
    snap = adapter.snapshot()
    budgets = allocate_budgets(secs, total_budget=10)
    print("Budgets:", budgets)
    compressed = compress_kv_cache(snap, budgets, quantize='float16')
    adapter.apply_compressed(compressed)
    print("Compressed cache sizes:", {k: len(v) for k, v in adapter.kv_cache.items()})


if __name__ == '__main__':
    main()
