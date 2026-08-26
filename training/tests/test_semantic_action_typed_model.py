from __future__ import annotations

import torch
from test_semantic_action_candidate_model import TinyDecoder, TinyEncoder

from shelliq_training.semantic_action_model import ActionDecoderConfig
from shelliq_training.semantic_action_typed_model import (
    SemanticActionTypedModel,
    TypedActionBatch,
    TypedActionOutput,
)
from shelliq_training.semantic_actions import ActionGrammar


def grammar() -> ActionGrammar:
    return ActionGrammar(
        {
            'schema_version': 1,
            'vocab_size': 320,
            'start_state': 0,
            'complete_state': 5,
            'tokens': [
                {'id': 1, 'name': 'BOS'},
                {'id': 2, 'name': 'EOS'},
                {'id': 8, 'name': 'COMMAND_END'},
                {'id': 10, 'name': 'ARGUMENT_START'},
                {'id': 11, 'name': 'REDIRECT_START'},
                {'id': 23, 'name': 'WORD_END'},
            ],
            'states': [
                {'id': 0, 'name': 'start', 'fixed': [{'token': 1, 'next_state': 1}], 'byte_payload': None},
                {
                    'id': 1,
                    'name': 'command_name_bytes',
                    'fixed': [],
                    'byte_payload': {
                        'byte_offset': 64,
                        'byte_count': 256,
                        'end_token': 23,
                        'next_state': 2,
                        'require_nonempty': True,
                        'require_valid_utf8': True,
                    },
                },
                {
                    'id': 2,
                    'name': 'command_body',
                    'fixed': [{'token': 8, 'next_state': 4}, {'token': 10, 'next_state': 3}],
                    'byte_payload': None,
                },
                {
                    'id': 3,
                    'name': 'argument_bytes',
                    'fixed': [],
                    'byte_payload': {
                        'byte_offset': 64,
                        'byte_count': 256,
                        'end_token': 23,
                        'next_state': 2,
                        'require_nonempty': True,
                        'require_valid_utf8': True,
                    },
                },
                {'id': 4, 'name': 'end', 'fixed': [{'token': 2, 'next_state': 5}], 'byte_payload': None},
                {'id': 5, 'name': 'complete', 'fixed': [], 'byte_payload': None},
            ],
        }
    )


def batch() -> TypedActionBatch:
    alignment = torch.zeros(1, 3, 3)
    alignment[0, 0, 0] = 1
    alignment[0, 1, 1] = 1
    alignment[0, 2, 1] = 1
    return TypedActionBatch(
        source_ids=torch.tensor([[4, 5, 2]]),
        source_attention_mask=torch.ones(1, 3, dtype=torch.bool),
        source_bytes=torch.tensor([[ord('c'), ord('m'), ord('d')]]),
        source_byte_mask=torch.ones(1, 3, dtype=torch.bool),
        source_byte_alignment=alignment,
        candidate_bytes=torch.tensor([[[ord('c'), ord('m'), ord('d')], [ord('x'), 0, 0]]]),
        candidate_byte_mask=torch.tensor([[[True, True, True], [True, False, False]]]),
        candidate_byte_histogram=torch.nn.functional.one_hot(torch.tensor([[ord('c'), ord('x')]]), 256).float(),
        candidate_starts=torch.tensor([[0, 0]]),
        candidate_ends=torch.tensor([[3, 1]]),
        candidate_features=torch.tensor([[[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]),
        candidate_role_mask=torch.tensor([[[True, False], [False, True]]]),
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
        decoder_input_ids=torch.tensor([[1, 64 + ord('c')]]),
        decoder_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        labels=torch.tensor([[64 + ord('c'), 2]]),
        candidate_labels=torch.tensor([[0, 1]]),
        argument_count_labels=torch.tensor([[1, -100]]),
        word_role_labels=torch.tensor([[0, 1]]),
    )


def test_typed_model_has_finite_candidate_and_count_losses() -> None:
    torch.manual_seed(37)
    config = ActionDecoderConfig(d_model=8, num_heads=2, num_layers=1, feedforward_size=16)
    model = SemanticActionTypedModel(TinyEncoder(8), TinyDecoder(), config, maximum_source_bytes=4)
    components = model.loss_components(batch())
    components['total'].backward()
    assert set(components) == {'action', 'candidate', 'argument_count', 'total'}
    assert all(torch.isfinite(value) for value in components.values())
    assert model.argument_count_head.weight.grad is not None
    assert model.candidate_feature_embedding.grad is not None


def test_typed_beam_obeys_predicted_argument_count() -> None:
    config = ActionDecoderConfig(
        d_model=8,
        num_heads=2,
        num_layers=1,
        feedforward_size=16,
        dropout=0,
        max_target_length=16,
    )
    model = SemanticActionTypedModel(TinyEncoder(8), TinyDecoder(), config, maximum_source_bytes=4)

    def distributions(value, hidden, memory):
        del value, memory
        action_logits = torch.zeros((*hidden.shape[:2], 320), device=hidden.device)
        candidate_logits = torch.tensor([[[3.0, 2.0]]], device=hidden.device)
        counts = torch.zeros((*hidden.shape[:2], 13), device=hidden.device)
        counts[..., 1] = 4
        return TypedActionOutput(action_logits, candidate_logits, counts)

    model.distributions = distributions  # type: ignore[method-assign]
    generated = model.generate_typed_beam(
        batch(),
        grammar(),
        ('command_name_bytes', 'argument_bytes'),
        beam_width=2,
        max_new_tokens=15,
    )
    assert generated == [
        (
            1,
            64 + ord('c'),
            64 + ord('m'),
            64 + ord('d'),
            23,
            10,
            64 + ord('x'),
            23,
            8,
            2,
        )
    ]
    forced = model.generate_typed_beam(
        batch(),
        grammar(),
        ('command_name_bytes', 'argument_bytes'),
        beam_width=2,
        max_new_tokens=15,
        forced_argument_counts=(1,),
        forced_words=(b'cmd', b'x'),
    )
    assert forced == generated
    no_arguments = model.generate_typed_beam(
        batch(),
        grammar(),
        ('command_name_bytes', 'argument_bytes'),
        beam_width=2,
        max_new_tokens=15,
        forced_argument_counts=(0,),
        forced_words=(b'cmd',),
    )
    assert no_arguments == [(1, 64 + ord('c'), 64 + ord('m'), 64 + ord('d'), 23, 8, 2)]
