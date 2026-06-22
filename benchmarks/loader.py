"""Dataset loader utilities for benchmark experiments.

This module provides a tiny placeholder dataset loader that yields (id, text)
pairs. Replace or extend with real clinical datasets (MIMIC-CXR notes, discharge
summaries, or public medical QA datasets) and proper license checks before use.
"""
from typing import Iterator, Tuple, List


def load_sample_notes() -> Iterator[Tuple[str, str]]:
    """Yield small set of sample clinical notes for quick experiments."""
    notes: List[Tuple[str, str]] = [
        ("note1", "Chief complaint:\nChest pain\nHistory of present illness:\n...\nMedications:\nAspirin"),
        ("note2", "History of present illness:\nFever and cough.\nAssessment and plan:\nTest and treat."),
    ]
    for n in notes:
        yield n
