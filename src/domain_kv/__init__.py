"""Domain-aware KV-cache compression for clinical/biomedical LLMs.

Exported modules:
- section_parser: regex-based clinical section segmentation + tokenizer-level
  section tagging.
- allocator: per-section KV token budget allocation from clinical importance
  weights.
- press: DomainAwarePress, a kvpress ScorerPress enforcing per-section budgets.
- metrics: real KV-cache size/retention metrics.
"""

from .section_parser import extract_sections, tag_token_sections, from_labeled_paragraphs, SectionedNote, Section
from .allocator import allocate_token_budgets, allocate_budgets, CLINICAL_IMPORTANCE_PROFILE
from .press import DomainAwarePress
from . import metrics

__all__ = [
    "extract_sections",
    "tag_token_sections",
    "from_labeled_paragraphs",
    "SectionedNote",
    "Section",
    "allocate_token_budgets",
    "allocate_budgets",
    "CLINICAL_IMPORTANCE_PROFILE",
    "DomainAwarePress",
    "metrics",
]
