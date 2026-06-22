"""Real evaluation harness: runs an actual HF causal LM, actually compresses
its KV cache during prefill via real `kvpress` presses (including our
`DomainAwarePress`), and actually generates an answer, then scores it.

No mocked tensors, no fabricated metrics. Memory and latency numbers come
from the real `DynamicCache` and `time.time()` around real forward passes.
"""
from collections import Counter
from typing import List

import torch

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
