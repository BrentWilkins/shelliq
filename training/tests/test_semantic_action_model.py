from types import SimpleNamespace

import torch
from torch import nn

from shelliq_training.data import Corpus
from shelliq_training.semantic_action_model import (
    ActionDecoderConfig,
    SemanticActionModel,
    collate_actions,
)
from shelliq_training.semantic_actions import ActionExample, ActionGrammar


class TinyEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, width)

    def forward(self, *, input_ids, attention_mask, return_dict):
        assert return_dict
        assert attention_mask.dtype == torch.bool
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def tiny_manifest() -> dict[str, object]:
    return {
        'schema_version': 1,
        'vocab_size': 8,
        'start_state': 0,
        'complete_state': 2,
        'tokens': [{'id': 1, 'name': 'BOS'}, {'id': 2, 'name': 'EOS'}],
        'states': [
            {'id': 0, 'name': 'start', 'fixed': [{'token': 1, 'next_state': 1}]},
            {'id': 1, 'name': 'document', 'fixed': [{'token': 2, 'next_state': 2}]},
            {'id': 2, 'name': 'complete', 'fixed': []},
        ],
    }


def example() -> ActionExample:
    return ActionExample('tiny:1', Corpus.DISTRIBUTABLE, (4, 5, 2), (1, 2))


def test_cpu_forward_backward_and_manifest_masked_generation() -> None:
    torch.manual_seed(7)
    config = ActionDecoderConfig(
        vocab_size=8,
        d_model=16,
        num_heads=2,
        num_layers=1,
        feedforward_size=32,
        dropout=0,
        max_target_length=4,
    )
    model = SemanticActionModel(TinyEncoder(config.d_model), config)
    batch = collate_actions([example()], source_length=4, target_length=4, source_pad_token_id=0)
    loss = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.action_embedding.weight.grad is not None
    generated = model.generate(
        batch.source_ids,
        batch.source_attention_mask,
        ActionGrammar(tiny_manifest()),
        max_new_tokens=2,
    )
    assert generated == [(1, 2)]


def test_action_collator_shifts_bos_and_masks_padding() -> None:
    batch = collate_actions([example()], source_length=5, target_length=4, source_pad_token_id=0)
    assert batch.decoder_input_ids.tolist() == [[1, 0, 0, 0]]
    assert batch.labels.tolist() == [[2, -100, -100, -100]]
    assert batch.source_attention_mask.dtype == torch.bool
