"""Allocate per-section KV-cache token budgets.

Budgets are computed over real token counts (from `section_ids`, as produced by
`section_parser.tag_token_sections`), not word counts. A clinical importance
profile lets sections that matter more for downstream reasoning (HPI, labs,
vitals, medications) keep proportionally more of the budget than low-signal
sections (demographics, allergies boilerplate), matching the RQ4 hypothesis in
EVALUATION_PLAN.md.
"""
from typing import Dict, List
import numpy as np

# Relative importance weight per canonical section name. Sections not listed
# fall back to a neutral weight of 1.0.
CLINICAL_IMPORTANCE_PROFILE: Dict[str, float] = {
    "hpi": 2.0,
    "labs": 1.8,
    "vitals": 1.6,
    "medications": 1.4,
    "assessment_and_plan": 1.8,
    "chief_complaint": 1.3,
    "pmh": 1.1,
    "physical_exam": 1.1,
    "social_history": 0.6,
    "family_history": 0.6,
    "allergies": 0.5,
    "review_of_systems": 0.7,
    "preamble": 0.5,
    "full_note": 1.0,
    # PubMedQA-style structured-abstract sections (see
    # section_parser.from_labeled_paragraphs).
    "results": 1.8,
    "conclusions": 1.7,
    "methods": 1.2,
    "objective": 1.3,
    "background": 0.9,
}


def allocate_token_budgets(
    section_ids: np.ndarray,
    canonical_names: List[str],
    total_budget: int,
    weights: Dict[str, float] = None,
) -> Dict[int, int]:
    """Allocate an integer token budget per section index.

    Args:
        section_ids: shape (seq_len,) section index per token, as returned by
            `tag_token_sections`.
        canonical_names: canonical_names[i] is the section name for index i.
        total_budget: total number of tokens allowed to be kept.
        weights: optional override of CLINICAL_IMPORTANCE_PROFILE.

    Returns:
        Mapping section_index -> token budget (int >= 0). Sums to
        min(total_budget, seq_len). Never allocates more tokens to a section
        than it actually has.
    """
    weights = weights if weights is not None else CLINICAL_IMPORTANCE_PROFILE
    seq_len = len(section_ids)
    total_budget = min(total_budget, seq_len)

    counts = np.bincount(section_ids, minlength=len(canonical_names))
    section_weights = np.array(
        [weights.get(canonical_names[i], 1.0) for i in range(len(canonical_names))],
        dtype=np.float64,
    )

    raw = counts.astype(np.float64) * section_weights
    if raw.sum() <= 0:
        raw = counts.astype(np.float64)

    proportions = raw / raw.sum()
    budgets = np.floor(proportions * total_budget).astype(np.int64)
    budgets = np.minimum(budgets, counts)

    # Distribute the rounding remainder to sections with the highest
    # weighted fractional remainder that still have spare capacity.
    remainder = total_budget - int(budgets.sum())
    if remainder > 0:
        frac = (proportions * total_budget) - budgets
        spare = counts - budgets
        order = np.argsort(-frac)
        for idx in order:
            if remainder <= 0:
                break
            give = min(remainder, int(spare[idx]))
            if give > 0:
                budgets[idx] += give
                remainder -= give

    return {i: int(budgets[i]) for i in range(len(canonical_names))}
