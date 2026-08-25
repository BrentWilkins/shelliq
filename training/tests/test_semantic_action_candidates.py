from __future__ import annotations

from shelliq_training.semantic_action_candidates import (
    CANDIDATE_IGNORE_INDEX,
    align_actions_to_candidates,
    lexical_candidates,
    merge_candidate_counts,
)
from shelliq_training.semantic_actions import ActionGrammar


def grammar() -> ActionGrammar:
    return ActionGrammar(
        {
            'schema_version': 1,
            'vocab_size': 320,
            'start_state': 0,
            'complete_state': 3,
            'tokens': [
                {'id': 1, 'name': 'BOS'},
                {'id': 2, 'name': 'EOS'},
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
                {'id': 2, 'name': 'end', 'fixed': [{'token': 2, 'next_state': 3}], 'byte_payload': None},
                {'id': 3, 'name': 'complete', 'fixed': [], 'byte_payload': None},
            ],
        }
    )


def test_lexical_candidates_are_complete_deterministic_spans() -> None:
    source = 'Use docker --format="{{json .}}" and café.'
    candidates = lexical_candidates(source)
    values = [source.encode()[item.start : item.end].decode() for item in candidates]
    assert 'docker' in values
    assert '--format=' in values
    assert '{{json .}}' in values
    assert 'café' in values
    assert candidates == lexical_candidates(source)


def test_candidate_alignment_labels_only_first_word_byte() -> None:
    source = b'try cafe then cafe'
    candidates = lexical_candidates(source.decode())
    actions = (1, 64 + ord('c'), 64 + ord('a'), 64 + ord('f'), 64 + ord('e'), 23, 2)
    alignment = align_actions_to_candidates(source, actions, grammar(), candidates)
    expected = next(index for index, item in enumerate(candidates) if source[item.start : item.end] == b'cafe')
    assert alignment.labels == (CANDIDATE_IGNORE_INDEX, expected, -100, -100, -100, -100, -100)
    assert alignment.word_starts == (False, True, False, False, False, False, False)
    assert alignment.word_counts == {'command_name_bytes': (1, 1)}
    assert merge_candidate_counts([alignment])['all'] == {'copyable': 1, 'total': 1, 'rate': 1.0}


def test_candidate_alignment_uses_generator_fallback_for_missing_word() -> None:
    actions = (1, 64 + ord('x'), 23, 2)
    alignment = align_actions_to_candidates(b'source', actions, grammar(), lexical_candidates('source'))
    assert alignment.labels == (-100, -100, -100, -100)
    assert alignment.word_starts[1]
