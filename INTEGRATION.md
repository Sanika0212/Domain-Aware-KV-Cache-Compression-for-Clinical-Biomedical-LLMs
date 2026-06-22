# Integration Guide: Real `kvpress` Usage

This project compresses the **actual KV cache** of a real HuggingFace
`transformers` model using the real [`kvpress`](https://github.com/NVIDIA/kvpress)
library — not a mocked dict, and not vLLM (the brief explicitly avoids vLLM
engine surgery; `kvpress` patches `transformers` attention layers directly via
forward hooks, which is engine-agnostic and works on CPU, MPS, or CUDA).

## Supported models

`kvpress` (and therefore `DomainAwarePress`) only supports a fixed set of HF
architectures (checked against the installed package, see
`kvpress.SUPPORTED_MODELS`): **Llama, Mistral, Phi3, Qwen2, Qwen3, Gemma3**.
This project defaults to `Qwen/Qwen2.5-0.5B-Instruct` (small enough to run on
a CPU-only machine in seconds) but the exact same code works unmodified with
a 7-8B biomedical model on the same architecture family, e.g. `BioMistral-7B`
or `epfl-llm/meditron-7b` (both Llama/Mistral-family), given a GPU.

## Core API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kvpress import KnormPress

from domain_kv.section_parser import extract_sections, tag_token_sections
from domain_kv.allocator import allocate_token_budgets
from domain_kv.press import DomainAwarePress

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

note = extract_sections(clinical_note_text)
section_ids, names = tag_token_sections(note, tokenizer)

context_ids = tokenizer(note.text, return_tensors="pt").input_ids
total_budget = int(0.5 * context_ids.shape[1])  # keep 50% of tokens
budgets = allocate_token_budgets(section_ids, names, total_budget)

press = DomainAwarePress(base_press=KnormPress())  # any ScorerPress works as the
                                                     # intra-section ranking function
press.set_document(section_ids, budgets)

cache = DynamicCache()
with press(model):                      # real kvpress context manager:
    model.model(input_ids=context_ids,  # registers forward hooks on every
                past_key_values=cache)  # attention layer, compresses during
                                         # this prefill call, then un-hooks.
```

`DomainAwarePress.compress()` is called once per attention layer during
prefill. It ranks tokens *within each section* using a delegated base scorer
(any `kvpress.ScorerPress`, e.g. `KnormPress` or `SnapKVPress`), keeps the
top-`budget[section]` tokens per section, and gathers the corresponding
key/value tensors — enforcing the literal per-section KV budget claim, while
still using a real attention/key-norm-based importance signal for *which*
tokens within a section survive.

## Continuing generation against a compressed cache

`kvpress` doesn't restore chronological token order after pruning (confirmed
by reading `SnapKVPress`/`KnormPress` source — `topk` + `gather` keeps score
order). This is safe: compression only runs during prefill, and every
subsequent decode step attends to the *entire* past cache without a causal
mask over it, so permutation has no effect. `benchmarks/runner.py::generate_answer`
mirrors `kvpress.KVPressTextGenerationPipeline.generate_answer` for continuing
generation with the right `position_ids` (continuing from the *original*,
pre-compression context length, since RoPE rotation for kept tokens was baked
in at their true absolute position before compression).

## Adding a new baseline

Any class from `kvpress` (`RandomPress`, `SnapKVPress`, `ExpectedAttentionPress`,
`PyramidKVPress`, ...) can be dropped into `benchmarks/runner.py::build_press`
as a structure-agnostic baseline to compare against `DomainAwarePress` at a
matched overall compression ratio.

## Extending section parsing to a new document type

`domain_kv.section_parser` has two entry points:
- `extract_sections(text)` — regex/header-based segmentation for free text
  (clinical notes with headers like "Chief Complaint:", "Medications:", ...).
- `from_labeled_paragraphs(labels, paragraphs)` — direct construction from
  already-segmented sources (used for PubMedQA's structured-abstract labels:
  BACKGROUND/OBJECTIVE/METHODS/RESULTS/CONCLUSIONS).

Both return a `SectionedNote`, the common type consumed by
`tag_token_sections` and `allocate_token_budgets`. Add a new entry point
returning a `SectionedNote` to support another document type without
touching the allocator or press.
