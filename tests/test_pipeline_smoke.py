"""Slow integration test: actually loads a real model and runs a real
compressed generation end to end. Opt-in (downloads ~1GB on first run):

    PYTHONPATH=src:. pytest -m slow tests/test_pipeline_smoke.py -v
"""
import pytest
import torch

pytestmark = pytest.mark.slow

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def test_real_compressed_generation_runs_end_to_end():
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from kvpress import KnormPress

    from domain_kv.section_parser import extract_sections, tag_token_sections
    from domain_kv.allocator import allocate_token_budgets
    from domain_kv.press import DomainAwarePress
    from domain_kv import metrics as dkv_metrics
    from benchmarks.runner import generate_answer

    text = (
        "Chief complaint:\nChest pain.\n\n"
        "History of present illness:\nPatient reports chest pain for 2 days, radiating to the left arm.\n\n"
        "Medications:\nAspirin 81 mg daily.\n\n"
        "Assessment and plan:\nObtain EKG and troponin. Start heparin drip.\n"
    )
    note = extract_sections(text)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()

    section_ids, names = tag_token_sections(note, tokenizer)
    context_ids = tokenizer(note.text, return_tensors="pt", add_special_tokens=True).input_ids
    n_tokens = context_ids.shape[1]
    budgets = allocate_token_budgets(section_ids, names, total_budget=n_tokens // 2)

    press = DomainAwarePress(base_press=KnormPress())
    press.set_document(section_ids, budgets)

    cache = DynamicCache()
    with press(model):
        with torch.no_grad():
            model.model(input_ids=context_ids, past_key_values=cache)

    kept = dkv_metrics.cache_num_tokens(cache)
    assert 0 < kept <= n_tokens

    question_ids = tokenizer("What medication was started?\nAnswer:", return_tensors="pt", add_special_tokens=False).input_ids
    answer = generate_answer(model, tokenizer, question_ids, cache, n_tokens, max_new_tokens=10)
    assert isinstance(answer, str) and len(answer) > 0
