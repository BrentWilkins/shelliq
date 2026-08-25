from types import SimpleNamespace

import torch
from torch import nn

from shelliq_training.semantic_action_model import ActionDecoderConfig
from shelliq_training.semantic_action_pointer_model import PointerActionBatch, SemanticActionPointerModel
from shelliq_training.semantic_actions import ActionGrammar


class TinyEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, width)

    def forward(self, *, input_ids, attention_mask, return_dict):
        assert return_dict
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens: nn.Module | None = None

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache,
        return_dict,
    ):
        assert self.embed_tokens is not None
        assert not use_cache and return_dict
        assert attention_mask.dtype == torch.bool
        context = encoder_hidden_states.mean(dim=1, keepdim=True)
        return SimpleNamespace(last_hidden_state=self.embed_tokens(input_ids) + context)


def grammar() -> ActionGrammar:
    return ActionGrammar(
        {
            'schema_version': 1,
            'vocab_size': 320,
            'start_state': 0,
            'complete_state': 2,
            'tokens': [{'id': 1, 'name': 'BOS'}, {'id': 2, 'name': 'EOS'}],
            'states': [
                {'id': 0, 'name': 'start', 'fixed': [{'token': 1, 'next_state': 1}]},
                {'id': 1, 'name': 'document', 'fixed': [{'token': 2, 'next_state': 2}]},
                {'id': 2, 'name': 'complete', 'fixed': []},
            ],
        }
    )


def batch() -> PointerActionBatch:
    alignment = torch.zeros(1, 3, 3)
    alignment[0, 0, 0] = 1
    alignment[0, 1, 1] = 1
    alignment[0, 2, 1] = 1
    return PointerActionBatch(
        source_ids=torch.tensor([[4, 5, 2]]),
        source_attention_mask=torch.ones(1, 3, dtype=torch.bool),
        source_bytes=torch.tensor([[ord('h'), ord('i'), ord('!')]]),
        source_byte_mask=torch.ones(1, 3, dtype=torch.bool),
        source_byte_alignment=alignment,
        decoder_input_ids=torch.tensor([[1, 64 + ord('h')]]),
        decoder_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        labels=torch.tensor([[64 + ord('h'), 2]]),
        copy_labels=torch.tensor([[0, -100]]),
    )


def test_pointer_generator_cpu_forward_backward_and_masked_generation() -> None:
    torch.manual_seed(13)
    config = ActionDecoderConfig(
        d_model=16,
        num_heads=2,
        num_layers=1,
        feedforward_size=32,
        dropout=0,
        max_target_length=4,
    )
    model = SemanticActionPointerModel(
        TinyEncoder(config.d_model),
        TinyDecoder(),
        config,
        maximum_source_bytes=4,
    )
    value = batch()
    output = model(value)
    assert output.log_probabilities.shape == (1, 2, 320)
    loss = model.loss(value)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.pointer_query.weight.grad is not None
    generated = model.generate(value, grammar(), max_new_tokens=2)
    assert generated == [(1, 2)]


def test_pointer_probability_only_copies_source_bytes() -> None:
    torch.manual_seed(17)
    config = ActionDecoderConfig(d_model=16, num_heads=2, num_layers=1, feedforward_size=32)
    model = SemanticActionPointerModel(TinyEncoder(config.d_model), TinyDecoder(), config, maximum_source_bytes=4)
    value = batch()
    output = model(value)
    copied_actions = {64 + ord('h'), 64 + ord('i'), 64 + ord('!')}
    assert torch.isfinite(output.pointer_logits).all()
    assert copied_actions.issubset(set(range(output.log_probabilities.shape[-1])))
