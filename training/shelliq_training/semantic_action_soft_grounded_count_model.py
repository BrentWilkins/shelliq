"""Soft prompt-grounding prior with post-command count prediction."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional

from shelliq_training.semantic_action_pretrained_candidate_model import (
    SemanticActionPretrainedCandidateModel,
)
from shelliq_training.semantic_action_typed_model import TypedActionBatch, TypedActionOutput


class SemanticActionSoftGroundedCountModel(SemanticActionPretrainedCandidateModel):
    """Score every candidate with a supervised grounded/global class prior."""

    def __init__(self, *args, argument_role_index: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0 <= argument_role_index < self.role_count:
            raise ValueError('argument_role_index is outside role vocabulary')
        self.argument_role_index = argument_role_index
        self.grounding_head = nn.Linear(self.config.d_model, 2)
        nn.init.xavier_uniform_(self.grounding_head.weight)
        nn.init.zeros_(self.grounding_head.bias)

    def distributions(
        self,
        batch: TypedActionBatch,
        hidden: torch.Tensor,
        memory: torch.Tensor,
    ) -> TypedActionOutput:
        output = super().distributions(batch, hidden, memory)
        assert output.role_candidate_logits is not None
        grounding_logits = self.grounding_head(hidden)
        grounding_log_probs = functional.log_softmax(grounding_logits, dim=-1)
        candidate_classes = batch.candidate_features[:, :, :2].bool().any(dim=-1).long()
        class_indices = candidate_classes.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        prior = grounding_log_probs.gather(2, class_indices)
        adjustment = torch.zeros_like(output.role_candidate_logits)
        adjustment[:, :, self.argument_role_index, :] = prior
        role_logits = output.role_candidate_logits + adjustment
        return replace(
            output,
            candidate_logits=role_logits.mean(dim=2),
            role_candidate_logits=role_logits,
            grounding_logits=grounding_logits,
        )
