# Domain-Aware KV-Cache Compression for Clinical/Biomedical LLMs

[![GitHub](https://img.shields.io/badge/GitHub-Sanika0212-blue?logo=github)](https://github.com/Sanika0212/Domain-Aware-KV-Cache-Compression-for-Clinical-Biomedical-LLMs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Sanika0212/Domain-Aware-KV-Cache-Compression-for-Clinical-Biomedical-LLMs/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

A real, working implementation of **per-section KV-cache budgeting for clinical/EHR documents** — built on the real [`kvpress`](https://github.com/NVIDIA/kvpress) library, a real HuggingFace `transformers` model, and a real (no-mock) attention KV cache. No fabricated numbers: every figure below comes from `results/runs.csv`, produced by actually running `benchmarks/runner.py`.

## What this is

`DomainAwarePress` (`src/domain_kv/press.py`) is a `kvpress.ScorerPress` that
enforces a **hard per-section token budget** during KV-cache compression,
where sections come from real clinical document structure (chief complaint,
HPI, labs, vitals, medications, assessment & plan, ...) instead of being
structure-agnostic like SnapKV/H2O/Knorm/Random. Per-section budgets are
allocated by `src/domain_kv/allocator.py` from a clinical importance profile
(HPI/labs/vitals weighted up, demographics/allergies weighted down), and
within each section, tokens are ranked by a delegated real importance scorer
(any `kvpress.ScorerPress`, e.g. key-norm or SnapKV).

To the best of our research (see `EVALUATION_PLAN.md` for related-work
positioning), this is the first KV-compression method to allocate budgets
from clinical document structure, and the first public benchmark of
KV-compression methods (Random/Knorm/SnapKV/ours) on a clinical/biomedical
long-context task.

---

## Quick Start

`kvpress`'s `fire` dependency imports the stdlib `pipes` module, removed in
Python 3.13 — use **Python 3.10, 3.11, or 3.12**.

```bash
git clone https://github.com/Sanika0212/Domain-Aware-KV-Cache-Compression-for-Clinical-Biomedical-LLMs.git
cd Domain-Aware-KV-Cache-Compression-for-Clinical-Biomedical-LLMs

python3.11 -m venv .venv   # create the venv OUTSIDE this repo if your local
source .venv/bin/activate  # path contains a ':' (breaks venv path resolution)
pip install -e ".[dev]"
```

### Run the interactive demo (real model, real compression, ~10s on CPU)

```bash
PYTHONPATH=src:. python run_demo.py
PYTHONPATH=src:. python run_demo.py --ratio 0.7 --model Qwen/Qwen2.5-0.5B-Instruct
```

### Run the tests

```bash
PYTHONPATH=src:. pytest               # fast unit tests (no model download)
PYTHONPATH=src:. pytest -m slow       # + real end-to-end model integration test
```

### Run the full benchmark and regenerate `results/`

```bash
PYTHONPATH=src:. python benchmarks/runner.py \
    --n-synthetic 20 --n-pubmedqa 15 --ratios 0.3 0.5 0.7 \
    --presses oracle random knorm snapkv domain_aware \
    --out results/runs.csv
PYTHONPATH=src:. python benchmarks/plot_results.py --csv results/runs.csv --out-dir results/
```

---

## Repository Structure

```
├── src/domain_kv/
│   ├── section_parser.py   # regex header segmentation + tokenizer-offset section tagging
│   ├── allocator.py        # per-section token budgets from a clinical importance profile
│   ├── press.py            # DomainAwarePress(ScorerPress) — the core contribution
│   └── metrics.py          # real KV-cache byte-size / retention metrics
├── benchmarks/
│   ├── loader.py            # PubMedQA (real, public) + seeded synthetic EHR notes
│   ├── runner.py            # real model, real compression, real generation, real scoring
│   └── plot_results.py      # turns results/runs.csv into results/summary.csv + plots
├── run_demo.py               # interactive single-note demo
├── tests/                    # unit tests + opt-in real-model integration test
├── INTEGRATION.md            # how DomainAwarePress hooks into real kvpress/transformers
├── EVALUATION_PLAN.md        # research questions, datasets, protocol, status
└── results/                  # committed output of the benchmark run below
```

---

## Architecture

```
clinical note (text)
   │
   ▼
SectionedNote                                  section_parser.py
   regex header segmentation (chief complaint, HPI, labs, meds, ...)
   OR from_labeled_paragraphs() for pre-structured sources (PubMedQA)
   │
   ▼
tag_token_sections(note, tokenizer)            section_parser.py
   real tokenizer offset_mapping -> per-token section_id array
   │
   ▼
allocate_token_budgets(section_ids, ...)       allocator.py
   per-section token budget, weighted by CLINICAL_IMPORTANCE_PROFILE
   │
   ▼
DomainAwarePress(base_press=KnormPress())      press.py  ← the contribution
   kvpress.ScorerPress subclass: within each section, keep the
   top-budget[section] tokens by the delegated scorer; gather kept K/V.
   │
   ▼
with press(model):
    model.model(input_ids=context_ids, past_key_values=cache)   # real prefill,
                                                                  # real compression
   │
   ▼
generate_answer(...)  →  real greedy decoding against the compressed cache
```

---

## Real Results

Run on `Qwen/Qwen2.5-0.5B-Instruct`, **Apple M4 CPU** (no CUDA on this
machine — see `EVALUATION_PLAN.md` §5.1 for why), 20 synthetic EHR-structured
notes + 15 real PubMedQA examples, compression ratios {0.3, 0.5, 0.7}, scored
by token-F1 (synthetic, open-generation medication QA) or exact-match
(PubMedQA, yes/no/maybe). Full data: `results/runs.csv` / `results/summary.csv`.
Regenerate with the command in Quick Start.

Mean task score by press and compression ratio (455 real runs, `results/summary.csv`):

| Press | ratio=0.30 | ratio=0.50 | ratio=0.70 |
|---|---|---|---|
| **`oracle`** (no compression) | 0.587 *(at ratio=0.0)* | — | — |
| `random` | 0.384 | 0.299 | 0.046 |
| `knorm` | 0.432 | 0.397 | 0.358 |
| `snapkv` | 0.425 | 0.262 | 0.224 |
| **`domain_aware` (ours)** | **0.471** | **0.501** | **0.405** |

`domain_aware` scores highest among all compressed conditions at every tested
ratio, and degrades far more gracefully at aggressive compression: at
ratio=0.70 (keeping only 30% of KV tokens), `domain_aware` retains 0.405
mean score versus `random`'s 0.046 (>8x), and still clearly ahead of
`knorm` (0.358) and `snapkv` (0.224). See `results/accuracy_vs_ratio.png` and
`results/memory_vs_ratio.png` for the full Pareto curves, and `results/runs.csv`
for every individual run (model answer, score, latency, KV memory bytes).

Two things to read honestly from this table (small model, small sample —
treat as a feasibility demonstration, not a publication-scale claim):
- **Sections matter for *which* tokens survive eviction**, not just how many.
  At matched compression ratios, `domain_aware` and the structure-agnostic
  baselines keep the *same number* of tokens but different *tokens*.
- A 0.5B model is a weak silver-bullet detector — at this scale, oracle
  accuracy itself is mediocre on the harder examples, which compresses the
  dynamic range available to show a compression-method gap. The mechanism
  (per-section budgeting on a real KV cache) is the contribution; the
  absolute numbers should be re-measured on a 7-8B biomedical model (the code
  already supports this via `--model`) before drawing task-performance
  conclusions.

---

## Research Positioning

**Novelty:**
- First KV-cache compression method to allocate budgets from **clinical
  document structure** (sections), as opposed to attention/layer/head
  topology (cf. StructKV) or program structure (cf. CodeComp, the closest
  conceptual precedent — per-chunk KV budgets from code structure).
- First benchmark of any KV-compression method (SnapKV/H2O/PyramidKV-style)
  on a clinical/biomedical long-context task.

**Related work:**
- SnapKV, H2O, PyramidKV, Knorm — general, structure-agnostic KV compression. Real implementations of Random/Knorm/SnapKV are used as direct baselines here via `kvpress`.
- TRACE (arXiv:2604.16364) — clinical note de-duplication at the *input-text* level, not the KV cache.
- Context Clues (Wornow et al., arXiv:2412.16178) — documents the copy-forwarded repetition in EHR notes that motivates this project's synthetic redundancy injection.
- GenCache, Prompt Cache, SemShareKV — prompt-reuse caching, orthogonal to compression.

See `EVALUATION_PLAN.md` for the full research-questions/protocol writeup and current implementation status.

---

## Contributing

Contributions welcome — open an issue or PR. Before committing, confirm your
git identity is set for this repo:

```bash
git config user.name "Sanika0212"
git config user.email "sanikanajan@gmail.com"
```

## License

[MIT](LICENSE)

## Contact

Sanika0212 — sanikanajan@gmail.com — https://github.com/Sanika0212
