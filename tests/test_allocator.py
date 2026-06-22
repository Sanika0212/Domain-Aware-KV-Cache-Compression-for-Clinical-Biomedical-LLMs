import numpy as np

from domain_kv.allocator import allocate_token_budgets, allocate_budgets


def test_allocate_token_budgets_sums_to_total():
    section_ids = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    names = ["allergies", "hpi"]  # low weight vs high weight
    budgets = allocate_token_budgets(section_ids, names, total_budget=4)
    assert sum(budgets.values()) == 4


def test_allocate_token_budgets_never_exceeds_section_size():
    section_ids = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    names = ["allergies", "hpi"]
    budgets = allocate_token_budgets(section_ids, names, total_budget=100)
    assert budgets[0] <= 2
    assert budgets[1] <= 6
    assert sum(budgets.values()) == 8  # capped at seq_len


def test_allocate_token_budgets_weights_favor_important_sections():
    # Equal-size sections, but hpi (weight 2.0) should get more budget than
    # allergies (weight 0.5) under a tight budget.
    section_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    names = ["allergies", "hpi"]
    budgets = allocate_token_budgets(section_ids, names, total_budget=4)
    assert budgets[1] > budgets[0]


def test_allocate_budgets_string_api_matches_token_api_proportions():
    sections = {"allergies": "a b c d", "hpi": "a b c d e f g h"}
    budgets = allocate_budgets(sections, total_budget=6)
    assert sum(budgets.values()) == 6
    assert budgets["hpi"] >= budgets["allergies"]
