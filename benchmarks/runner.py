"""Real evaluation harness: runs an actual HF causal LM, actually compresses
its KV cache during prefill via real `kvpress` presses (including our
`DomainAwarePress`), and actually generates an answer, then scores it.

No mocked tensors, no fabricated metrics. Memory and latency numbers come
from the real `DynamicCache` and `time.time()` around real forward passes.
"""
from collections import Counter
from typing import List

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
