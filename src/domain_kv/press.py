"""DomainAwarePress: per-section KV-cache budgets for clinical documents.

This is the core research contribution of the project: a `kvpress.ScorerPress`
that enforces a hard per-section token budget (derived from clinical document
structure via `allocator.allocate_token_budgets`) instead of pruning purely by
a global importance score. Within each section, tokens are still ranked by a
delegated base scorer (e.g. SnapKV's attention-based score, or key-norm), so
the method is "structure-aware budgeting + learned-style intra-section
ranking", not a naive heuristic.

Usage
-----
    press = DomainAwarePress(base_press=SnapKVPress(), compression_ratio=0.5)
    press.set_document(section_ids, budgets)
    with press(model):
        outputs = model(input_ids, past_key_values=cache)

`section_ids` and `budgets` must be set before each new document/prompt (the
press instance is reused across all attention layers within one forward pass,
so this is set once per document, not per layer).
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.knorm_press import KnormPress


@dataclass
class DomainAwarePress(ScorerPress):
    """Per-section KV budget allocation with delegated intra-section ranking.

    Parameters
    ----------
    base_press : ScorerPress, default=KnormPress()
        Used only for its `.score()` method to rank tokens *within* a section.
        Any ScorerPress works (SnapKVPress, ExpectedAttentionPress, ...).
    compression_ratio : float, default=0.0
        Unused directly (kept for ScorerPress/BasePress compatibility); the
        effective ratio is implied by the budgets passed to `set_document`.
    """

    base_press: ScorerPress = field(default_factory=KnormPress)
    compression_ratio: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self._section_ids: Optional[torch.Tensor] = None
        self._budgets: Optional[Dict[int, int]] = None

    def set_document(self, section_ids, budgets: Dict[int, int]):
        """Register the per-token section assignment and per-section budgets
        for the document about to be processed. Must be called before each
        new prefill.
        """
        if not torch.is_tensor(section_ids):
            section_ids = torch.as_tensor(section_ids, dtype=torch.long)
        self._section_ids = section_ids
        self._budgets = budgets

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        return self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ):
        if self._section_ids is None or self._budgets is None:
            raise RuntimeError(
                "DomainAwarePress.set_document(section_ids, budgets) must be called "
                "before running a forward pass."
            )

        batch_size, _, k_len, _ = keys.shape
        if batch_size != 1:
            raise NotImplementedError(
                "DomainAwarePress currently supports batch_size=1 (one document's KV cache "
                "compressed at a time), matching this project's benchmark usage."
            )

        section_ids = self._section_ids.to(keys.device)
        if section_ids.shape[0] != k_len:
            raise ValueError(
                f"section_ids has length {section_ids.shape[0]} but keys have seq_len {k_len}; "
                "did the tokenizer/section tagging run on the same prompt as the model input?"
            )

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        # scores: (1, num_kv_heads, k_len) -> one ranking per token, averaged over kv heads.
        token_scores = scores.mean(dim=1)[0]  # (k_len,)

        keep_mask = torch.zeros(k_len, dtype=torch.bool, device=keys.device)
        for sec_id, budget in self._budgets.items():
            sec_positions = (section_ids == sec_id).nonzero(as_tuple=True)[0]
            if budget <= 0 or sec_positions.numel() == 0:
                continue
            budget = min(budget, sec_positions.numel())
            top = token_scores[sec_positions].topk(budget).indices
            keep_mask[sec_positions[top]] = True

        if not keep_mask.any():
            # Degenerate (all-zero budgets) - keep everything rather than crash generation.
            return keys, values

        kept_indices = keep_mask.nonzero(as_tuple=True)[0]  # ascending, chronological order
        gather_indices = (
            kept_indices.view(1, 1, -1, 1).expand(1, keys.shape[1], -1, module.head_dim)
        )
        keys = keys.gather(2, gather_indices).contiguous()
        values = values.gather(2, gather_indices).contiguous()
        return keys, values
