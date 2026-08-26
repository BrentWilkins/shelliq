"""Prompt-grounded candidate selection with post-command count prediction."""

from __future__ import annotations

import torch

from shelliq_training.semantic_action_pretrained_candidate_model import (
    SemanticActionPretrainedCandidateModel,
)
from shelliq_training.semantic_action_typed_model import TypedActionBatch


class SemanticActionGroundedCountModel(SemanticActionPretrainedCandidateModel):
    """Mask grounded training targets to prompt-supported argument candidates."""

    def _candidate_loss_mask(self, batch: TypedActionBatch, role_mask: torch.Tensor) -> torch.Tensor:
        grounded = batch.candidate_features[:, :, :2].bool().any(dim=-1).unsqueeze(1)
        labels = batch.candidate_labels.clamp_min(0)
        target_is_grounded = grounded.expand_as(role_mask).gather(2, labels.unsqueeze(-1)).squeeze(-1)
        argument_role = batch.word_role_labels == 0
        use_grounding = (target_is_grounded & argument_role).unsqueeze(-1)
        return torch.where(use_grounding, role_mask & grounded, role_mask)
