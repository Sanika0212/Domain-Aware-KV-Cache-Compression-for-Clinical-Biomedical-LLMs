# Evaluation Plan: Domain-Aware KV-Cache Compression for Clinical LLMs

## Executive Summary

This document outlines the evaluation strategy for the domain-aware KV-cache compression system tailored to clinical/biomedical long-context inference. The goal is to demonstrate that **per-section KV budgets derived from EHR/clinical document structure** improve compression quality and inference efficiency compared to structure-agnostic baselines.

---

## 1. Research Questions

1. **RQ1:** Does allocating KV budgets per clinical section (vs. global compression) preserve downstream task performance better?
2. **RQ2:** What is the latency and memory reduction achieved at different compression ratios?
3. **RQ3:** How do quantization strategies (uint8, float16, mixed) affect clinical QA accuracy?
4. **RQ4:** Which sections (e.g., labs, vitals, HPI) are most important for maintaining clinical reasoning accuracy?

---

## 2. Datasets & Benchmarks

### 2.1 Public Clinical Datasets

| Dataset | Size | Task | Notes |
|---------|------|------|-------|
| **Medical QA (arXiv:2411.09834)** | ~8K Q&A pairs | Long-context QA on discharge summaries | Benchmark for clinical comprehension; uses 7B–13B models |
| **MIMIC-III (via PhysioNet)** | 40K+ notes | Readmission prediction, note extraction | Requires license agreement; representative of real EHR text |
| **PubMed & Clinical Notes (open)** | ~100K notes | Synthetic long-context tasks | For quick prototyping |

### 2.2 Synthetic Benchmark

- Generate long clinical notes (2K–4K tokens) by combining real-world templates (chief complaint, labs, meds, plan).
- Inject redundancy patterns observed in real EHRs (copy-forwarded text, templated sections).
- Query: "Extract key findings" / "Predict readmission risk" / "Summarize care plan".

---

## 3. Baselines

1. **No Compression (Oracle):** Full KV cache, no eviction. Baseline for accuracy.
2. **Uniform Compression:** LRU eviction across all tokens (e.g., SnapKV, H2O).
3. **Learned Compression:** Attention-head importance weighting (e.g., PyramidKV, StructKV).
4. **Domain-Aware (Ours):** Per-section budgets with learned or heuristic importance weights.

---

## 4. Metrics

### 4.1 Task Performance
- **Accuracy:** Exact-match, F1, or BLEU on downstream task (QA, prediction, extraction).
- **Degradation:** Relative accuracy loss vs. oracle (e.g., 0.5% degradation acceptable).

### 4.2 Efficiency
- **KV Cache Size:** Total tokens retained; reported as % of original.
- **Memory Footprint:** MB (after quantization/dtype conversion).
- **Latency:** End-to-end token generation time (ms).
- **Memory–Accuracy Trade-off:** Pareto curve across compression ratios (0.1–0.9).

### 4.3 Section Importance
- **Per-Section Retention Ratio:** For each clinical section, report % of KV retained.
- **Ablation:** Measure accuracy when one section is fully evicted (e.g., drop "Allergies" vs. drop "Labs").

---

## 5. Experimental Protocol

### 5.1 Setup
- **Model:** Llama-2-7B or Mistral-7B (vLLM-compatible).
- **Hardware:** Single A100 (80GB) or V100 (32GB).
- **Context Length:** 2K–4K tokens (typical clinical note length).
- **Repetitions:** 5 random seeds; report mean ± std dev.

### 5.2 Procedure

#### Experiment 1: Static Compression on QA Benchmark
```
for each model in [Llama-7B, Mistral-7B]:
  for each compression_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for each baseline in [oracle, uniform, learned, domain_aware]:
      - Load clinical QA prompt + context
      - Compress KV cache (per baseline strategy)
      - Generate answer; measure latency & accuracy
      - Report: accuracy, latency, memory saved
```

#### Experiment 2: Adaptive Compression During Generation
```
for each model in [Llama-7B, Mistral-7B]:
  - Run generation with streaming KV compression
  - Trigger compression every N tokens (e.g., N=512)
  - Measure: cumulative latency, final accuracy, peak memory
```

#### Experiment 3: Section Ablation
```
for each clinical section in [chief_complaint, labs, vitals, meds, assessment]:
  - Compress all sections uniformly EXCEPT target
  - Measure accuracy drop when section is evicted
  - Rank sections by importance
```

### 5.3 Reproducibility
- Fix random seeds (numpy, torch).
- Report model version (e.g., "llama-2-7b-hf@huggingface").
- Save KV cache snapshots for offline analysis.
- Release code & datasets on GitHub.

---

## 6. Ablation Studies

| Ablation | Hypothesis | Metric |
|----------|-----------|--------|
| **No quantization** | Float32 vs. float16 | Memory footprint, accuracy |
| **Uniform allocation** | Per-section vs. global budget | Accuracy, memory–latency trade-off |
| **LRU only** | Eviction only (no quantization) | Baseline vs. combined |
| **Simple vs. learned weights** | Hand-crafted section importance | Section retention ratio, accuracy |
| **Single section drop** | Which sections are critical? | Per-section importance ranking |

---

## 7. Expected Outcomes

- **RQ1:** Section-aware allocation preserves **≥2% better accuracy** on clinical QA vs. uniform.
- **RQ2:** **50–70% memory reduction** at ≤1% accuracy drop.
- **RQ3:** Float16 + uint8 quantization yields **40–50% total memory savings** with minimal accuracy impact.
- **RQ4:** Labs, vitals, HPI ranked as high-importance; allergies, demographics as low-importance.

---

## 8. Implementation Roadmap

### Phase 1: Baseline Setup (Week 1–2)
- [ ] Download MIMIC / Public Medical QA datasets.
- [ ] Build dataset loaders & evaluation harness.
- [ ] Implement uniform compression baseline.
- [ ] Measure oracle (full cache) accuracy.

### Phase 2: Domain-Aware Implementation (Week 2–3)
- [ ] Implement learned section importance weighting (attention-based).
- [ ] Integrate with vLLM via adapter.
- [ ] Run Experiment 1 (static compression).

### Phase 3: Analysis & Ablations (Week 3–4)
- [ ] Run ablation studies.
- [ ] Visualize memory–accuracy curves.
- [ ] Compute per-section importance ranking.
- [ ] Statistical significance testing.

### Phase 4: Writeup & Release (Week 4+)
- [ ] Draft paper or technical report.
- [ ] Create reproducible benchmark scripts.
- [ ] Push code & results to GitHub.

---

## 9. Evaluation Code Skeleton

The repository includes:

- `benchmarks/runner.py` — Full experiment runner.
- `examples/advanced_compression_demo.py` — End-to-end pipeline demo.
- `INTEGRATION.md` — How to use adapters with vLLM.
- `tests/test_*.py` — Unit tests for core logic.

**Quick start:**

```bash
PYTHONPATH=src python examples/advanced_compression_demo.py
PYTHONPATH=src pytest -q
```

---

## 10. Paper Positioning

**Title:** *Domain-Aware KV-Cache Compression for Clinical Long-Context LLM Inference*

**Key Claims:**
1. First work to apply per-section KV budgeting to clinical EHR documents.
2. Demonstrates that clinical structure (labs, vitals, HPI) can guide compression better than learned attention alone.
3. Achieves 50–70% memory reduction with <1% accuracy degradation on clinical QA.
4. Lightweight, engine-agnostic method compatible with vLLM, KVPress, and other frameworks.

**Positioning:** Instantiation of CodeComp/StructKV-style structure-aware compression + first medical KV-compression benchmark.

---

## 11. Open Questions & Future Work

- **Learned Section Weights:** Train a shallow model to predict section importance from task queries.
- **Cross-Domain Transfer:** Does clinical section weighting transfer to other domains (legal contracts, scientific papers)?
- **Multi-GPU Distributed:** How does distributed inference interact with per-section compression?
- **Streaming Notes:** Compress KV cache as note grows (e.g., real-time EHR charting).

---

## Contact & Reproduction

- **Repository:** https://github.com/Sanika0212/Domain-Aware-KV-Cache-Compression-for-Clinical-Biomedical-LLMs
- **Author:** Sanika0212 (sanikanajan@gmail.com)
- **License:** MIT
