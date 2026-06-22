# Engine Integration Guide

This guide explains how to integrate domain-aware KV-cache compression into vLLM or KVPress runtimes.

## Overview

The module `src/domain_kv/engine_integration.py` provides adapter skeletons for:
- **vLLM** (official LLM inference engine)
- **KVPress** (modular KV-cache caching layer)
- **MockEngineAdapter** (for local testing)

The adapter pattern is simple:
1. Instantiate adapter with your engine: `adapter = VLLMAdapter(engine)` or `KVPressAdapter(engine)`
2. Call `snapshot()` to capture the current KV cache state (organized by clinical section)
3. Apply `compress_kv_cache()` to compress per-section budgets
4. Call `apply_compressed()` to atomically update the engine's KV store

## vLLM Integration

### Prerequisites

```bash
pip install vllm>=0.6.0
```

### Minimal Example

```python
from vllm import LLM, SamplingParams
from domain_kv.engine_integration import VLLMAdapter
from domain_kv.section_parser import extract_sections
from domain_kv.allocator import allocate_budgets
from domain_kv.compressor import compress_kv_cache

# Create engine
llm = LLM(model="meta-llama/Llama-2-7b-hf")
adapter = VLLMAdapter(llm)

# Run inference on a clinical note
prompt = "Chief complaint: Chest pain. History of present illness: ..."
outputs = llm.generate(prompt, SamplingParams(max_tokens=100))

# Capture KV cache snapshot
kv_snapshot = adapter.snapshot()

# Compress with section-aware budgets
sections = extract_sections(prompt)
budgets = allocate_budgets(sections, total_budget=256)
compressed = compress_kv_cache(kv_snapshot, budgets, quantize='float16')

# Apply compressed cache back into engine
adapter.apply_compressed(compressed)

# Continue inference with reduced memory footprint
outputs = llm.generate("Continue with assessment...", SamplingParams(max_tokens=100))
```

### Customizing Section Mapping

By default, `VLLMAdapter.snapshot()` maps entries to sections based on metadata tags.
If your prompts lack explicit section headers, override the adapter:

```python
class CustomVLLMAdapter(VLLMAdapter):
    def snapshot(self):
        result = super().snapshot()
        # add custom section parsing logic if needed
        return result
```

## KVPress Integration

### Prerequisites

```bash
pip install kvpress
```

### Minimal Example

```python
from kvpress import KVPressEngine
from domain_kv.engine_integration import KVPressAdapter
# ... same import pattern as vLLM above

engine = KVPressEngine(...)
adapter = KVPressAdapter(engine)

# snapshot, compress, apply as above
```

## Safe Update Semantics

**Important:** Both vLLM and KVPress are concurrent systems. When calling `apply_compressed()`:

1. **No outstanding requests** should be reading the KV cache.
2. **Atomicity:** The adapter must ensure that updates are atomic (all-or-nothing) from the engine's perspective.
3. **Locks:** Consider using engine-specific locks or synchronization primitives.

For production use, implement engine-specific atomic updates. The current skeletons are for **experiments only**.

## MockEngineAdapter (Testing)

For unit tests or local experiments, use the `MockEngineAdapter`:

```python
from domain_kv.engine_integration import MockEngineAdapter

kv = {'section': [('k1', vec1), ('k2', vec2)]}
adapter = MockEngineAdapter(internal_store=kv)

snap = adapter.snapshot()
# ... compress ...
adapter.apply_compressed(compressed)
```

## Troubleshooting

### "Failed to snapshot vLLM KV"
- Verify the vLLM version matches the expected API (check `vllm.__version__`).
- The vLLM KV cache layout may differ; inspect `llm.engine.kv` structure and adapt.

### "Failed to apply compressed KV"
- Ensure no active inference threads are accessing the cache.
- Use engine-provided APIs instead of direct mutation (e.g., `llm.engine.set_kv(...)` if available).

### vLLM/KVPress modules not found
- Install them: `pip install vllm kvpress`
- The adapters gracefully fall back to `None` if imports fail.

## Next Steps

1. Run the example benchmark with your engine: `PYTHONPATH=src python examples/benchmark_demo.py`
2. Implement engine-specific tests in `tests/test_vllm_integration.py` or `tests/test_kvpress_integration.py`.
3. Measure latency and accuracy impact of compression on real clinical queries.
