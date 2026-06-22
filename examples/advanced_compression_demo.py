"""Advanced example: end-to-end section-aware compression with metrics."""
import sys
import json
from domain_kv.section_parser import extract_sections
from domain_kv.allocator import allocate_budgets
from domain_kv.compressor import compress_kv_cache
from domain_kv.kvpress_adapter import KVPressAdapter
from domain_kv.engine_integration import MockEngineAdapter
from benchmarks.runner import ExperimentRunner
import numpy as np


def simulate_inference_with_compression(note_text: str, total_budget: int = 256):
    """Full pipeline: parse, allocate, compress, measure impact."""
    runner = ExperimentRunner(total_budget=total_budget, quantize='float16')
    
    # Extract sections
    secs = extract_sections(note_text)
    print(f"\n📋 Extracted {len(secs)} sections: {list(secs.keys())}")
    
    # Build synthetic KV cache
    kv = runner.build_kv_from_note(note_text)
    orig_sizes = {s: len(items) for s, items in kv.items()}
    print(f"📦 Original KV cache sizes: {orig_sizes}")
    print(f"   Total KV pairs: {sum(orig_sizes.values())}")
    
    # Allocate per-section budgets
    budgets = allocate_budgets(secs, total_budget=total_budget)
    print(f"\n💾 Allocated budgets: {budgets}")
    print(f"   Total budget used: {sum(budgets.values())}")
    
    # Compress
    compressed = compress_kv_cache(kv, budgets, quantize='float16')
    comp_sizes = {s: len(items) for s, items in compressed.items()}
    print(f"\n✅ Compressed KV cache sizes: {comp_sizes}")
    print(f"   Total KV pairs: {sum(comp_sizes.values())}")
    
    # Compute compression stats
    retained_ratio = sum(comp_sizes.values()) / max(1, sum(orig_sizes.values()))
    reduction_ratio = 1.0 - retained_ratio
    print(f"\n📊 Compression statistics:")
    print(f"   Retention ratio: {retained_ratio:.2%}")
    print(f"   Reduction ratio: {reduction_ratio:.2%}")
    
    # Per-section stats
    for sec in secs:
        orig = orig_sizes.get(sec, 0)
        comp = comp_sizes.get(sec, 0)
        ratio = comp / max(1, orig) if orig > 0 else 0
        print(f"   {sec:30s}: {orig:3d} → {comp:3d} ({ratio:5.1%})")
    
    # Memory footprint estimate (float16 = 2 bytes per float)
    vec_dim = 256  # placeholder from runner
    orig_mb = sum(orig_sizes.values()) * vec_dim * 4 / (1024**2)  # float32
    comp_mb = sum(comp_sizes.values()) * vec_dim * 2 / (1024**2)  # float16
    print(f"\n🧠 Estimated memory (float32 → float16):")
    print(f"   Before: {orig_mb:.2f} MB")
    print(f"   After:  {comp_mb:.2f} MB")
    print(f"   Saved:  {orig_mb - comp_mb:.2f} MB ({(orig_mb - comp_mb) / orig_mb * 100:.1f}%)")
    
    return {
        'sections': secs,
        'original_sizes': orig_sizes,
        'compressed_sizes': comp_sizes,
        'budgets': budgets,
        'retention_ratio': retained_ratio,
        'reduction_ratio': reduction_ratio,
        'memory_before_mb': orig_mb,
        'memory_after_mb': comp_mb,
        'memory_saved_mb': orig_mb - comp_mb,
    }


def main():
    clinical_note = """
Chief complaint:
Chest pain for 2 days.

History of present illness:
Patient is a 65-year-old male presenting with acute chest pain radiating to left arm.
Symptoms started at rest this morning. Associated with dyspnea but no nausea.
Denies recent trauma or falls. Reports history of hypertension.

Past medical history:
Hypertension x 10 years, controlled on lisinopril.
Diabetes type 2, on metformin.
Hyperlipidemia.
Prior MI in 2015, treated with PCI.

Medications:
Lisinopril 10 mg daily
Metformin 500 mg BID
Atorvastatin 80 mg nightly
Aspirin 81 mg daily

Allergies:
NKDA

Physical examination:
Vital signs: BP 155/95, HR 102, RR 18, Temp 98.6°F
General: Alert, anxious-appearing male
Cardiac: Regular rate and rhythm, no murmurs
Lungs: Clear to auscultation bilaterally
Extremities: No edema

Assessment and plan:
Acute coronary syndrome vs. musculoskeletal pain. Will admit for cardiac workup.
Obtain EKG, troponin, CBC, CMP. Start heparin drip. Cardiology consult pending.
NPO except meds. Monitor telemetry.
"""
    
    print("=" * 70)
    print("Domain-Aware KV-Cache Compression for Clinical Notes")
    print("=" * 70)
    
    results = simulate_inference_with_compression(clinical_note, total_budget=32)
    
    print("\n" + "=" * 70)
    print("Summary JSON:")
    print(json.dumps({k: v for k, v in results.items() if k != 'sections'}, indent=2))
    

if __name__ == '__main__':
    main()
