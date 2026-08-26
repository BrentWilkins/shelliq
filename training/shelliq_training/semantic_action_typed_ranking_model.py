"""Order-sensitive, role-conditioned candidate ranking for typed semantic actions."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional

from shelliq_training.semantic_action_candidate_model import CANDIDATE_IGNORE_INDEX
from shelliq_training.semantic_action_model import LABEL_IGNORE_INDEX, ActionDecoderConfig
from shelliq_training.semantic_action_pointer_model import _action_embedding_initialization
from shelliq_training.semantic_action_typed import BIGRAM_BUCKETS
from shelliq_training.semantic_action_typed_model import (
    COUNT_IGNORE_INDEX,
    SemanticActionTypedModel,
    TypedActionBatch,
    TypedActionOutput,
)


class SemanticActionTypedRankingModel(SemanticActionTypedModel):
    """Typed model with byte-bigram keys and a query scale for each word role."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        config: ActionDecoderConfig,
        *,
        maximum_source_bytes: int,
        role_count: int,
        initial_action_embedding: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            encoder,
            decoder,
            config,
            maximum_source_bytes=maximum_source_bytes,
            initial_action_embedding=initial_action_embedding,
        )
        if role_count <= 0:
            raise ValueError('role_count must be positive')
        self.role_count = role_count
        self.candidate_bigram_embedding = nn.Parameter(torch.empty(BIGRAM_BUCKETS, config.d_model))
        self.role_query_scale = nn.Parameter(torch.ones(role_count, config.d_model))
        nn.init.normal_(self.candidate_bigram_embedding, mean=0, std=config.d_model**-0.5)

    @classmethod
    def from_codet5(
        cls,
        pretrained,
        tokenizer,
        *,
        maximum_source_bytes: int,
        role_count: int,
        dropout: float = 0.1,
    ) -> SemanticActionTypedRankingModel:
        config = ActionDecoderConfig(dropout=dropout)
        encoder = pretrained.get_encoder()
        decoder = pretrained.get_decoder()
        decoder.config = copy.deepcopy(decoder.config)
        decoder.block = nn.ModuleList(list(decoder.block[: config.num_layers]))
        decoder.config.num_layers = config.num_layers
        for module in decoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
        initial = _action_embedding_initialization(pretrained.shared.weight.detach(), tokenizer, config)
        return cls(
            encoder,
            decoder,
            config,
            maximum_source_bytes=maximum_source_bytes,
            role_count=role_count,
            initial_action_embedding=initial,
        )

    def _content_keys(self, batch: TypedActionBatch) -> torch.Tensor:
        dtype = self.source_byte_embedding.weight.dtype
        unigram = torch.matmul(batch.candidate_byte_histogram.to(dtype), self.source_byte_embedding.weight)
        bigram = torch.matmul(batch.candidate_bigram_histogram.to(dtype), self.candidate_bigram_embedding)
        global_feature = batch.candidate_features[:, :, 2:3] * self.candidate_feature_embedding[2].view(1, 1, -1)
        return self.candidate_norm(unigram + bigram + global_feature)

    def _typed_candidate_keys(self, batch: TypedActionBatch, memory: torch.Tensor) -> torch.Tensor:
        unigram = torch.matmul(
            batch.candidate_byte_histogram.to(self.source_byte_embedding.weight.dtype),
            self.source_byte_embedding.weight,
        )
        bigram = torch.matmul(
            batch.candidate_bigram_histogram.to(self.candidate_bigram_embedding.dtype),
            self.candidate_bigram_embedding,
        )
        context = self._candidate_keys(batch, self._byte_keys(batch, memory))
        context = context * batch.candidate_features[:, :, :1]
        provenance = torch.einsum('bcf,fd->bcd', batch.candidate_features, self.candidate_feature_embedding)
        return self.candidate_norm(unigram + bigram + context + provenance)

    def _role_logits(self, hidden: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        query = self.candidate_query(hidden).unsqueeze(2) * self.role_query_scale.view(1, 1, self.role_count, -1)
        keys = self.candidate_key(candidates)
        logits = torch.einsum('btrd,bcd->btrc', query, keys) / math.sqrt(self.config.d_model)
        return logits

    def distributions(
        self,
        batch: TypedActionBatch,
        hidden: torch.Tensor,
        memory: torch.Tensor,
    ) -> TypedActionOutput:
        role_logits = self._role_logits(hidden, self._typed_candidate_keys(batch, memory))
        content_logits = self._role_logits(hidden, self._content_keys(batch))
        candidate_mask = batch.candidate_mask[:, None, None, :]
        role_logits = role_logits.masked_fill(~candidate_mask, -torch.inf)
        content_logits = content_logits.masked_fill(~candidate_mask, -torch.inf)
        action_logits = self.output_projection(hidden * self.config.d_model**-0.5)
        return TypedActionOutput(
            action_logits,
            role_logits.mean(dim=2),
            self.argument_count_head(hidden),
            role_logits,
            content_logits,
        )

    def loss_components(self, batch: TypedActionBatch) -> dict[str, torch.Tensor]:
        output = self(batch)
        action_loss = functional.cross_entropy(
            output.action_logits.reshape(-1, self.config.vocab_size),
            batch.labels.reshape(-1),
            ignore_index=LABEL_IGNORE_INDEX,
        )
        assert output.role_candidate_logits is not None
        role_indices = (
            batch.word_role_labels.clamp_min(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(*batch.word_role_labels.shape, 1, output.candidate_logits.shape[-1])
        )
        role_logits = output.role_candidate_logits.gather(2, role_indices).squeeze(2)
        role_mask_indices = batch.word_role_labels.clamp_min(0).unsqueeze(-1).expand_as(role_logits)
        role_mask = batch.candidate_role_mask.gather(1, role_mask_indices)
        typed_logits = role_logits.masked_fill(~role_mask, -torch.inf)
        copyable = batch.candidate_labels != CANDIDATE_IGNORE_INDEX
        candidate_loss = (
            functional.cross_entropy(typed_logits[copyable], batch.candidate_labels[copyable])
            if bool(copyable.any())
            else action_loss.new_zeros(())
        )

        counted = batch.argument_count_labels != COUNT_IGNORE_INDEX
        count_loss = (
            functional.cross_entropy(output.argument_count_logits[counted], batch.argument_count_labels[counted])
            if bool(counted.any())
            else action_loss.new_zeros(())
        )

        assert output.content_role_candidate_logits is not None
        content_logits = output.content_role_candidate_logits.gather(2, role_indices).squeeze(2)
        labels = batch.candidate_labels.clamp_min(0)
        target_is_global = batch.candidate_features[:, :, 2].gather(1, labels)
        global_targets = copyable & target_is_global.bool()
        global_mask = role_mask & batch.candidate_features[:, :, 2].bool().unsqueeze(1)
        content_logits = content_logits.masked_fill(~global_mask, -torch.inf)
        content_global_loss = (
            functional.cross_entropy(content_logits[global_targets], batch.candidate_labels[global_targets])
            if bool(global_targets.any())
            else action_loss.new_zeros(())
        )
        total = action_loss + 0.25 * candidate_loss + 0.25 * count_loss + 0.10 * content_global_loss
        return {
            'action': action_loss,
            'candidate': candidate_loss,
            'argument_count': count_loss,
            'content_global': content_global_loss,
            'total': total,
        }
