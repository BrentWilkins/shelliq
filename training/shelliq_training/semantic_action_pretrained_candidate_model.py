"""Typed candidate ranking with pretrained CodeT5 lexical embeddings."""

from __future__ import annotations

import torch

from shelliq_training.semantic_action_typed_model import TypedActionBatch
from shelliq_training.semantic_action_typed_ranking_model import SemanticActionTypedRankingModel


class SemanticActionPretrainedCandidateModel(SemanticActionTypedRankingModel):
    """Add the encoder's pretrained token space to candidate keys."""

    def _pretrained_candidate_keys(self, batch: TypedActionBatch) -> torch.Tensor:
        embedding = getattr(self.encoder, 'embed_tokens', None)
        if embedding is None:
            embedding = getattr(self.encoder, 'embedding', None)
        if embedding is None:
            raise TypeError('encoder does not expose its input-token embedding')
        values = embedding(batch.candidate_token_ids)
        mask = batch.candidate_token_mask.unsqueeze(-1).to(values.dtype)
        return (values * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)

    def _content_keys(self, batch: TypedActionBatch) -> torch.Tensor:
        dtype = self.source_byte_embedding.weight.dtype
        unigram = torch.matmul(batch.candidate_byte_histogram.to(dtype), self.source_byte_embedding.weight)
        bigram = torch.matmul(batch.candidate_bigram_histogram.to(dtype), self.candidate_bigram_embedding)
        global_feature = batch.candidate_features[:, :, 2:3] * self.candidate_feature_embedding[2].view(1, 1, -1)
        pretrained = self._pretrained_candidate_keys(batch)
        return self.candidate_norm(unigram + bigram + pretrained + global_feature)

    def _typed_candidate_keys(self, batch: TypedActionBatch, memory: torch.Tensor) -> torch.Tensor:
        dtype = self.source_byte_embedding.weight.dtype
        unigram = torch.matmul(batch.candidate_byte_histogram.to(dtype), self.source_byte_embedding.weight)
        bigram = torch.matmul(batch.candidate_bigram_histogram.to(dtype), self.candidate_bigram_embedding)
        pretrained = self._pretrained_candidate_keys(batch)
        context = self._candidate_keys(batch, self._byte_keys(batch, memory))
        context = context * batch.candidate_features[:, :, :1]
        provenance = torch.einsum('bcf,fd->bcd', batch.candidate_features, self.candidate_feature_embedding)
        return self.candidate_norm(unigram + bigram + pretrained + context + provenance)
