"""Real evaluation harness: runs an actual HF causal LM, actually compresses
its KV cache during prefill via real `kvpress` presses (including our
`DomainAwarePress`), and actually generates an answer, then scores it.

No mocked tensors, no fabricated metrics. Memory and latency numbers come
from the real `DynamicCache` and `time.time()` around real forward passes.
"""
import contextlib
import time
from collections import Counter
from typing import List

import torch
from transformers import DynamicCache

from kvpress import RandomPress, KnormPress, SnapKVPress

from domain_kv.section_parser import tag_token_sections
from domain_kv.allocator import allocate_token_budgets
from domain_kv.press import DomainAwarePress
from domain_kv import metrics as dkv_metrics

from benchmarks.loader import BenchmarkExample


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def choice_score(pred: str, gold: str, choices: List[str]) -> float:
    pred_l = pred.lower()
    for c in choices:
        if c in pred_l:
            return float(c == gold.lower())
    return 0.0


def score_example(example: BenchmarkExample, generated: str) -> float:
    if example.answer_type == "f1":
        return f1_score(generated, example.expected_answer)
    return choice_score(generated, example.expected_answer, example.choices or [])


@torch.no_grad()
def generate_answer(model, tokenizer, question_ids: torch.Tensor, cache, context_length: int, max_new_tokens: int) -> str:
    """Greedy decoding continuation from a (possibly compressed) cache.

    Mirrors kvpress.KVPressTextGenerationPipeline.generate_answer: position
    ids continue from the *original* (pre-compression) context length, since
    RoPE rotation for kept tokens was already baked in at their true position.
    """
    position_ids = torch.arange(context_length, context_length + question_ids.shape[1], device=model.device).unsqueeze(0)
    outputs = model(input_ids=question_ids, past_key_values=cache, position_ids=position_ids)
    position_ids = position_ids[:, -1:] + 1
    generated_ids = [outputs.logits[0, -1].argmax()]

    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    for i in range(max_new_tokens - 1):
        outputs = model(input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0), past_key_values=cache, position_ids=position_ids + i)
        new_id = outputs.logits[0, -1].argmax()
        generated_ids.append(new_id)
        if new_id.item() in eos_ids:
            break
    return tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)


def build_press(name: str, ratio: float, example: BenchmarkExample, tokenizer, n_tokens: int):
    """Returns (press_or_None, extra_setup_callable_or_None)."""
    if name == "oracle":
        return None
    if name == "random":
        return RandomPress(compression_ratio=ratio)
    if name == "knorm":
        return KnormPress(compression_ratio=ratio)
    if name == "snapkv":
        return SnapKVPress(compression_ratio=ratio, window_size=min(32, max(1, n_tokens // 4)))
    if name == "domain_aware":
        section_ids, names = tag_token_sections(example.note, tokenizer)
        total_budget = max(1, int(round(n_tokens * (1 - ratio))))
        budgets = allocate_token_budgets(section_ids, names, total_budget)
        press = DomainAwarePress(base_press=KnormPress())
        press.set_document(section_ids, budgets)
        return press
    raise ValueError(f"Unknown press {name!r}")


def run_example(model, tokenizer, example: BenchmarkExample, press_name: str, ratio: float, max_new_tokens: int) -> dict:
    context_ids = tokenizer(example.note.text, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    question_ids = tokenizer(example.question + "\nAnswer:", return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    n_tokens = context_ids.shape[1]

    press = build_press(press_name, ratio, example, tokenizer, n_tokens)
    cache = DynamicCache()

    t0 = time.time()
    cm = press(model) if press is not None else contextlib.nullcontext()
    with cm:
        with torch.no_grad():
            model.model(input_ids=context_ids, past_key_values=cache)
    answer = generate_answer(model, tokenizer, question_ids, cache, n_tokens, max_new_tokens)
    elapsed = time.time() - t0

    return {
        "example_id": example.example_id,
        "press": press_name,
        "ratio": ratio,
        "answer": answer,
        "score": score_example(example, answer),
        "elapsed_s": elapsed,
        "orig_tokens": n_tokens,
        "kept_tokens": dkv_metrics.cache_num_tokens(cache),
        "mem_bytes": dkv_metrics.cache_bytes(cache),
    }
