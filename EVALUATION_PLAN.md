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

This project deliberately uses only public, no-credentialing-required data
(MIMIC-III/IV requires a PhysioNet license agreement we have not pursued, to
keep the benchmark reproducible by anyone who clones the repo).

### 2.1 Real public biomedical QA: PubMedQA

`benchmarks/loader.py::load_pubmedqa` loads `qiaojin/PubMedQA` (`pqa_labeled`
split) from the Hugging Face Hub — 1,000 expert-labeled biomedical QA
examples, each with a structured abstract (BACKGROUND/OBJECTIVE/METHODS/
RESULTS/CONCLUSIONS) and a yes/no/maybe ground-truth answer. The structured
labels become real sections via `domain_kv.section_parser.from_labeled_paragraphs`
— no regex header-guessing needed since the boundaries are already known.
This is biomedical *literature* QA, not EHR notes; see §2.2 for the
EHR-document-structure benchmark.

### 2.2 Synthetic EHR-structured benchmark

`benchmarks/loader.py::load_synthetic_notes` generates clinical notes (chief
complaint, HPI, PMH, medications, allergies, vitals, labs, physical exam,
assessment & plan) from randomized templates, each with a deterministic
medication-dose QA target for token-F1 scoring. Critically, it injects
**copy-forwarded redundancy** — the same HPI sentence verbatim in both the
HPI and assessment-and-plan sections — mirroring the repetition pattern
documented in Wornow et al. (arXiv:2412.16178) and the "Addressing Note
Bloat" EHR study cited in the project brief (46% of real EHR note text is
copy-forwarded). This is what makes per-section budgeting interesting: a
structure-agnostic compressor sees the duplicated sentence as independently
"important" wherever it appears, while a section-budget approach evicts
according to where it sits in the clinical structure.

---

## 3. Baselines

All implemented as real `kvpress` presses in `benchmarks/runner.py::build_press`,
run against the real model's KV cache (no simulation):

1. **`oracle`:** No compression (`press=None`). Upper bound for accuracy.
2. **`random`:** `kvpress.RandomPress` — structure-agnostic random eviction.
3. **`knorm`:** `kvpress.KnormPress` — structure-agnostic, key-L2-norm-based eviction.
4. **`snapkv`:** `kvpress.SnapKVPress` — structure-agnostic, attention-based eviction
   (this is the real SnapKV from arXiv:2404.14469, not a re-implementation).
5. **`domain_aware`:** `domain_kv.press.DomainAwarePress` (ours) — per-section
   budgets (`domain_kv.allocator.allocate_token_budgets`) with `KnormPress`-based
   intra-section ranking.

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
- **Default model:** `Qwen/Qwen2.5-0.5B-Instruct` (Qwen2 architecture — one of
  `kvpress.SUPPORTED_MODELS` — small enough to run real prefill+generation on
  a CPU-only machine in well under a second per example). `benchmarks/runner.py
  --model` accepts any `kvpress`-supported architecture, so the identical code
  scales to a 7-8B biomedical model (BioMistral-7B, Meditron-7B) given a GPU,
  without any code changes.
- **Hardware actually used for the numbers in this repo:** Apple M4, CPU-only
  (no CUDA). Numbers are reported as exactly what they are — a small-model,
  CPU-feasibility benchmark, not a claim about 7-8B-scale behavior.
- **Context length:** ~250-450 tokens (synthetic notes) / ~250-600 tokens
  (PubMedQA structured abstracts) — see `results/runs.csv` for actuals.

### 5.2 Procedure

#### Experiment 1: Static compression across ratios (implemented)
```
python benchmarks/runner.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --n-synthetic 20 --n-pubmedqa 15 \
  --ratios 0.3 0.5 0.7 \
  --presses oracle random knorm snapkv domain_aware \
  --out results/runs.csv
python benchmarks/plot_results.py --csv results/runs.csv --out-dir results/
```
For each (example, press, ratio): compress the real KV cache during prefill,
generate a real answer, score it, and record real latency + real KV memory
bytes. See `results/summary.csv` and `results/*.png` for the actual output of
this run (committed to the repo, regenerable by anyone).

#### Experiment 2: Section ablation (not yet run — future work)
For each clinical section, set its budget to 0 (full eviction) while keeping
others at their normal allocation; measure the resulting score drop to rank
sections by importance. The mechanism (`allocate_token_budgets` accepts an
explicit per-section override) already supports this; only the sweep script
is unwritten.

### 5.3 Reproducibility
- `benchmarks/loader.load_synthetic_notes(seed=...)` is deterministic.
- Model name and `kvpress` version are recorded in `results/summary.csv`'s
  companion run command; pin exact versions via `pyproject.toml`.
- All data sources are public, no-license-agreement datasets — anyone can
  `git clone` and reproduce `results/runs.csv` exactly.

---

## 6. Ablation Studies

| Ablation | Hypothesis | Status |
|----------|-----------|--------|
| **Uniform vs. per-section allocation** | Per-section budgets preserve accuracy better than a structure-agnostic compressor at the same ratio | Implemented — `oracle/random/knorm/snapkv` vs. `domain_aware` in `results/runs.csv` |
| **Base scorer choice** | `DomainAwarePress(base_press=...)` ranking-within-section sensitivity (Knorm vs. SnapKV) | Mechanism implemented (`base_press` param); sweep not yet run |
| **Single section drop** | Which clinical sections are most critical? | Not yet run — see Experiment 2 in §5.2 |

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
