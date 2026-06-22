"""Interactive end-to-end demo: real model, real KV cache, real compression.

Loads a small instruction-tuned LLM, tags a synthetic clinical note's tokens
by section, compresses the KV cache during prefill with `DomainAwarePress`
versus an uncompressed oracle, and prints the real generated answer, real
per-section retention, and real memory savings for both.

    PYTHONPATH=src python run_demo.py
    PYTHONPATH=src python run_demo.py --model Qwen/Qwen2.5-0.5B-Instruct --ratio 0.5
"""
import argparse
import contextlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import KnormPress

from domain_kv.section_parser import extract_sections, tag_token_sections
from domain_kv.allocator import allocate_token_budgets
from domain_kv.press import DomainAwarePress
from domain_kv import metrics as dkv_metrics

from benchmarks.runner import generate_answer

SAMPLE_NOTE = """Chief Complaint:
Chest pain for 2 days.

History of Present Illness:
Patient is a 65-year-old male presenting with acute chest pain radiating to the left arm. Symptoms started at rest this morning. Associated with dyspnea but no nausea. Denies recent trauma or falls. Reports history of hypertension.

Past Medical History:
Hypertension x 10 years, controlled on lisinopril. Diabetes type 2, on metformin. Hyperlipidemia. Prior myocardial infarction in 2015, treated with PCI.

Medications:
Lisinopril 10 mg daily. Metformin 500 mg twice daily. Atorvastatin 80 mg nightly. Aspirin 81 mg daily.

Allergies:
No known drug allergies.

Vital Signs:
Blood pressure 155/95, heart rate 102, respiratory rate 18, temperature 98.6F.

Physical Exam:
Alert, anxious-appearing male. Cardiac exam reveals regular rate and rhythm, no murmurs. Lungs clear to auscultation bilaterally. No peripheral edema.

Assessment and Plan:
Acute coronary syndrome versus musculoskeletal pain. Will admit for cardiac workup. Obtain EKG, troponin, CBC, CMP. Start heparin drip. Cardiology consult pending.
"""

QUESTION = "What medication was started for this patient's acute presentation?"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--ratio", type=float, default=0.5, help="Fraction of KV tokens to evict.")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    print("=" * 78)
    print("Domain-Aware KV-Cache Compression for Clinical Notes — live demo")
    print("=" * 78)
    print(f"\nLoading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    note = extract_sections(SAMPLE_NOTE)
    section_texts = note.section_texts()
    print(f"\nExtracted {len(section_texts)} sections: {list(section_texts.keys())}")

    section_ids, names = tag_token_sections(note, tokenizer)
    context_ids = tokenizer(note.text, return_tensors="pt", add_special_tokens=True).input_ids
    n_tokens = context_ids.shape[1]
    question_ids = tokenizer(QUESTION + "\nAnswer:", return_tensors="pt", add_special_tokens=False).input_ids

    total_budget = max(1, int(round(n_tokens * (1 - args.ratio))))
    budgets = allocate_token_budgets(section_ids, names, total_budget)
    print(f"\nTotal tokens: {n_tokens}  ->  budget at ratio={args.ratio}: {total_budget}")
    print("Per-section budgets (tokens kept / tokens available):")
    counts = {i: int((section_ids == i).sum()) for i in range(len(names))}
    for i, name in enumerate(names):
        print(f"   {name:24s}: {budgets[i]:4d} / {counts[i]:4d}")

    def run(press, label):
        cache = DynamicCache()
        cm = press(model) if press is not None else contextlib.nullcontext()
        with cm:
            with torch.no_grad():
                model.model(input_ids=context_ids, past_key_values=cache)
        kept = dkv_metrics.cache_num_tokens(cache)
        mem_mb = dkv_metrics.cache_bytes(cache) / 1e6
        answer = generate_answer(model, tokenizer, question_ids, cache, n_tokens, args.max_new_tokens)
        print(f"\n[{label}]")
        print(f"   KV tokens kept : {kept} / {n_tokens} ({kept / n_tokens:.1%})")
        print(f"   KV cache size  : {mem_mb:.3f} MB")
        print(f"   Generated      : {answer!r}")
        return mem_mb

    print("\n" + "-" * 78)
    print(f"Question: {QUESTION}")
    print("-" * 78)

    oracle_mb = run(None, "Oracle (no compression)")
    press = DomainAwarePress(base_press=KnormPress())
    press.set_document(section_ids, budgets)
    compressed_mb = run(press, f"Domain-Aware (ratio={args.ratio})")

    print("\n" + "=" * 78)
    print(f"Memory saved: {oracle_mb - compressed_mb:.3f} MB ({(1 - compressed_mb / oracle_mb):.1%} reduction)")
    print("=" * 78)


if __name__ == "__main__":
    main()
