from __future__ import annotations

import torch
from test_semantic_action_candidate_model import TinyDecoder, TinyEncoder, batch

from shelliq_training.semantic_action_lexicon_model import SemanticActionLexiconModel
from shelliq_training.semantic_action_model import ActionDecoderConfig


def test_lexicon_model_scores_global_then_local_candidates() -> None:
    torch.manual_seed(31)
    config = ActionDecoderConfig(d_model=8, num_heads=2, num_layers=1, feedforward_size=16)
    model = SemanticActionLexiconModel(
        TinyEncoder(8),
        TinyDecoder(),
        config,
        maximum_source_bytes=4,
        global_words=(b'docker', b'pytest'),
    )
    value = batch()
    output = model(value)
    assert output.candidate_logits.shape == (1, 2, 4)
    assert model.candidate_bytes(value, 0) == b'docker'
    assert model.candidate_bytes(value, 2) == b'hi'
    components = model.loss_components(value)
    components['total'].backward()
    assert set(components) == {'action', 'candidate', 'total'}
    assert all(torch.isfinite(item) for item in components.values())
    assert model.global_candidate_embedding.weight.grad is not None


def test_lexicon_words_must_be_stably_sorted() -> None:
    config = ActionDecoderConfig(d_model=8, num_heads=2, num_layers=1, feedforward_size=16)
    try:
        SemanticActionLexiconModel(
            TinyEncoder(8),
            TinyDecoder(),
            config,
            maximum_source_bytes=4,
            global_words=(b'z', b'a'),
        )
    except ValueError as error:
        assert 'stable byte ordering' in str(error)
    else:
        raise AssertionError('unsorted global lexicon was accepted')
