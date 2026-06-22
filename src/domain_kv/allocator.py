"""Allocate per-section KV budgets using configurable heuristics."""
from typing import Dict

def allocate_budgets(sections: Dict[str, str], total_budget: int, weights: Dict[str, float] = None) -> Dict[str, int]:
    """Allocate integer budgets (number of KV pairs) per section.

    Args:
        sections: mapping section name -> text
        total_budget: total KV budget (pairs) allowed
        weights: optional mapping section name -> importance weight (overrides default sizing)

    Returns mapping section -> budget (int). Ensures budgets sum <= total_budget.
    """
    n = len(sections)
    if n == 0:
        return {}

    if weights is None:
        # default: proportional to section token count (approx by word count)
        wc = {s: max(1, len(t.split())) for s, t in sections.items()}
        total_wc = sum(wc.values())
        budgets = {s: max(1, int(total_budget * wc[s] / total_wc)) for s in sections}
    else:
        # normalize weights
        total_w = sum(weights.get(s, 1.0) for s in sections)
        budgets = {s: max(1, int(total_budget * weights.get(s, 1.0) / total_w)) for s in sections}

    # correct rounding to fit budget
    assigned = sum(budgets.values())
    i = 0
    keys = list(budgets.keys())
    while assigned > total_budget and i < len(keys):
        k = keys[i % len(keys)]
        if budgets[k] > 1:
            budgets[k] -= 1
            assigned -= 1
        i += 1

    while assigned < total_budget:
        # add to largest section (by current budget)
        k = max(budgets, key=lambda x: budgets[x])
        budgets[k] += 1
        assigned += 1

    return budgets
