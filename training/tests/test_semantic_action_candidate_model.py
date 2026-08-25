from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from shelliq_training.semantic_action_candidate_model import (
    CandidateActionBatch,
    SemanticActionCandidateModel,
)
from shelliq_training.semantic_action_model import ActionDecoderConfig
from shelliq_training.semantic_actions import ActionGrammar


class TinyEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, width)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        assert attention_mask.dtype == torch.bool
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens: nn.Module

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        assert attention_mask.dtype == torch.bool
        assert encoder_attention_mask.dtype == torch.bool
        return SimpleNamespace(last_hidden_state=self.embed_tokens(input_ids) + encoder_hidden_states.mean(dim=1, keepdim=True))


def grammar() -> ActionGrammar:
    return ActionGrammar(
        {
            'schema_version': 1,
            'vocab_size': 320,
            'start_state': 0,
            'complete_state': 2,
            'tokens': [{'id': 1, 'name': 'BOS'}, {'id': 2, 'name': 'EOS'}],
            'states': [
                {'id': 0, 'name': 'start', 'fixed': [{'token': 1, 'next_state': 1}], 'byte_payload': None},
                {'id': 1, 'name': 'end', 'fixed': [{'token': 2, 'next_state': 2}], 'byte_payload': None},
                {'id': 2, 'name': 'complete', 'fixed': [], 'byte_payload': None},
            ],
        }
    )


def batch() -> CandidateActionBatch:
    alignment = torch.zeros(1, 3, 3)
    alignment[0, 0, 0] = 1
    alignment[0, 1, 1] = 1
    alignment[0, 2, 1] = 1
    return CandidateActionBatch(
        source_ids=torch.tensor([[4, 5, 2]]),
        source_attention_mask=torch.ones(1, 3, dtype=torch.bool),
        source_bytes=torch.tensor([[ord('h'), ord('i'), ord('!')]]),
        source_byte_mask=torch.ones(1, 3, dtype=torch.bool),
        source_byte_alignment=alignment,
        candidate_starts=torch.tensor([[0, 2]]),
        candidate_ends=torch.tensor([[2, 3]]),
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
        decoder_input_ids=torch.tensor([[1, 64 + ord('h')]]),
        decoder_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        labels=torch.tensor([[64 + ord('h'), 2]]),
        candidate_labels=torch.tensor([[0, -100]]),
        word_starts=torch.tensor([[True, False]]),
    )


def test_candidate_model_cpu_forward_backward_and_masked_generation() -> None:
    torch.manual_seed(23)
    config = ActionDecoderConfig(
        d_model=16,
        num_heads=2,
        num_layers=1,
        feedforward_size=32,
        dropout=0,
        max_target_length=4,
    )
    model = SemanticActionCandidateModel(
        TinyEncoder(config.d_model),
        TinyDecoder(),
        config,
        maximum_source_bytes=4,
    )
    value = batch()
    output = model(value)
    assert output.action_logits.shape == (1, 2, 320)
    assert output.candidate_logits.shape == (1, 2, 2)
    components = model.loss_components(value)
    components['total'].backward()
    assert all(torch.isfinite(item) for item in components.values())
    assert model.candidate_query.weight.grad is not None
    assert model.generate(value, grammar(), max_new_tokens=2) == [(1, 2)]


def test_candidate_pooling_uses_fixed_span_boundaries() -> None:
    torch.manual_seed(29)
    config = ActionDecoderConfig(d_model=8, num_heads=2, num_layers=1, feedforward_size=16)
    model = SemanticActionCandidateModel(TinyEncoder(8), TinyDecoder(), config, maximum_source_bytes=4)
    value = batch()
    memory = model.encode(value)
    byte_keys = model._byte_keys(value, memory)
    candidates = model._candidate_keys(value, byte_keys)
    assert torch.allclose(candidates[0, 0], byte_keys[0, :2].mean(dim=0))
    assert torch.allclose(candidates[0, 1], byte_keys[0, 2])
