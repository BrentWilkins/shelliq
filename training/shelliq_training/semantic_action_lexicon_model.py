"""Global-plus-local candidate model for semantic-action decoding."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional

from shelliq_training.semantic_action_candidate_model import (
    CANDIDATE_IGNORE_INDEX,
    CandidateActionBatch,
    CandidateActionOutput,
    SemanticActionCandidateModel,
)
from shelliq_training.semantic_action_model import LABEL_IGNORE_INDEX, ActionDecoderConfig
from shelliq_training.semantic_action_pointer_model import _action_embedding_initialization


class SemanticActionLexiconModel(SemanticActionCandidateModel):
    """Select training-derived global words or prompt-local lexical spans."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        config: ActionDecoderConfig,
        *,
        maximum_source_bytes: int,
        global_words: Sequence[bytes],
        initial_action_embedding: torch.Tensor | None = None,
        initial_global_embedding: torch.Tensor | None = None,
    ) -> None:
        if not global_words or len(set(global_words)) != len(global_words):
            raise ValueError('global words must be nonempty and unique')
        if tuple(sorted(global_words)) != tuple(global_words):
            raise ValueError('global words must use stable byte ordering')
        super().__init__(
            encoder,
            decoder,
            config,
            maximum_source_bytes=maximum_source_bytes,
            initial_action_embedding=initial_action_embedding,
        )
        self.global_words = tuple(global_words)
        self.global_candidate_embedding = nn.Embedding(len(global_words), config.d_model)
        nn.init.normal_(self.global_candidate_embedding.weight, mean=0, std=config.d_model**-0.5)
        if initial_global_embedding is not None:
            if initial_global_embedding.shape != self.global_candidate_embedding.weight.shape:
                raise ValueError('initial global candidate embedding has wrong shape')
            self.global_candidate_embedding.weight.data.copy_(initial_global_embedding)

    @classmethod
    def from_codet5(
        cls,
        pretrained,
        tokenizer,
        *,
        maximum_source_bytes: int,
        global_words: Sequence[bytes],
        dropout: float = 0.1,
    ) -> SemanticActionLexiconModel:
        config = ActionDecoderConfig(dropout=dropout)
        encoder = pretrained.get_encoder()
        decoder = pretrained.get_decoder()
        decoder.config = copy.deepcopy(decoder.config)
        decoder.block = nn.ModuleList(list(decoder.block[: config.num_layers]))
        decoder.config.num_layers = config.num_layers
        for module in decoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
        action_initial = _action_embedding_initialization(pretrained.shared.weight.detach(), tokenizer, config)
        global_initial = _global_embedding_initialization(pretrained.shared.weight.detach(), tokenizer, global_words)
        return cls(
            encoder,
            decoder,
            config,
            maximum_source_bytes=maximum_source_bytes,
            global_words=global_words,
            initial_action_embedding=action_initial,
            initial_global_embedding=global_initial,
        )

    def distributions(
        self,
        batch: CandidateActionBatch,
        hidden: torch.Tensor,
        memory: torch.Tensor,
    ) -> CandidateActionOutput:
        local = self._candidate_keys(batch, self._byte_keys(batch, memory))
        global_keys = self.global_candidate_embedding.weight.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        candidates = torch.cat((global_keys, local), dim=1)
        candidate_mask = torch.cat(
            (
                torch.ones(
                    (hidden.shape[0], len(self.global_words)),
                    dtype=torch.bool,
                    device=hidden.device,
                ),
                batch.candidate_mask,
            ),
            dim=1,
        )
        candidate_logits = torch.einsum(
            'btd,bcd->btc',
            self.candidate_query(hidden),
            self.candidate_key(candidates),
        ) / math.sqrt(self.config.d_model)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask[:, None, :], -torch.inf)
        action_logits = self.output_projection(hidden * self.config.d_model**-0.5)
        return CandidateActionOutput(action_logits, candidate_logits, self.copy_gate(hidden).squeeze(-1))

    def candidate_bytes(self, batch: CandidateActionBatch, choice: int) -> bytes:
        if choice < len(self.global_words):
            return self.global_words[choice]
        return super().candidate_bytes(batch, choice - len(self.global_words))

    def loss_components(self, batch: CandidateActionBatch) -> dict[str, torch.Tensor]:
        output = self(batch)
        action_loss = functional.cross_entropy(
            output.action_logits.reshape(-1, self.config.vocab_size),
            batch.labels.reshape(-1),
            ignore_index=LABEL_IGNORE_INDEX,
        )
        copyable = batch.candidate_labels != CANDIDATE_IGNORE_INDEX
        candidate_loss = (
            functional.cross_entropy(output.candidate_logits[copyable], batch.candidate_labels[copyable])
            if bool(copyable.any())
            else action_loss.new_zeros(())
        )
        return {
            'action': action_loss,
            'candidate': candidate_loss,
            'total': action_loss + 0.25 * candidate_loss,
        }


def _global_embedding_initialization(
    shared_embedding: torch.Tensor,
    tokenizer,
    global_words: Sequence[bytes],
) -> torch.Tensor:
    rows = []
    for raw in global_words:
        token_ids = tokenizer.encode(raw.decode('utf-8'), add_special_tokens=False)
        if not token_ids:
            raise ValueError('global word produced no CodeT5 tokens')
        rows.append(shared_embedding[torch.tensor(token_ids, device=shared_embedding.device)].mean(dim=0))
    return torch.stack(rows)
