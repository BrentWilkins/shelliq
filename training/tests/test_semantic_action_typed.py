from __future__ import annotations

from shelliq_training.semantic_action_typed import (
    align_typed_actions,
    byte_roles,
    command_argument_counts,
    role_word_lexicon,
    typed_candidates,
)
from shelliq_training.semantic_actions import ActionGrammar


def grammar() -> ActionGrammar:
    return ActionGrammar(
        {
            'schema_version': 1,
            'vocab_size': 320,
            'start_state': 0,
            'complete_state': 4,
            'tokens': [
                {'id': 1, 'name': 'BOS'},
                {'id': 2, 'name': 'EOS'},
                {'id': 10, 'name': 'ARGUMENT_START'},
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
                {'id': 2, 'name': 'body', 'fixed': [{'token': 10, 'next_state': 3}], 'byte_payload': None},
                {
                    'id': 3,
                    'name': 'argument_bytes',
                    'fixed': [],
                    'byte_payload': {
                        'byte_offset': 64,
                        'byte_count': 256,
                        'end_token': 23,
                        'next_state': 4,
                        'require_nonempty': True,
                        'require_valid_utf8': True,
                    },
                },
                {'id': 4, 'name': 'complete', 'fixed': [], 'byte_payload': None},
            ],
        }
    )


def actions() -> tuple[int, ...]:
    return (
        1,
        64 + ord('c'),
        64 + ord('m'),
        64 + ord('d'),
        23,
        10,
        64 + ord('2'),
        23,
    )


def test_role_lexicon_and_typed_alignment() -> None:
    value = grammar()
    lexicon = role_word_lexicon([actions()], value)
    assert lexicon == {b'cmd': ('command_name_bytes',), b'2': ('argument_bytes',)}
    candidates = typed_candidates('Run cmd with --maxfail and 2 failures.', value, lexicon)
    labels, counts, roles = align_typed_actions(actions(), value, candidates, (1,))
    assert labels[1] >= 0
    assert labels[6] >= 0
    assert counts[1] == 1
    assert roles[1] == byte_roles(value).index('command_name_bytes')
    assert roles[6] == byte_roles(value).index('argument_bytes')


def test_typed_derivations_compose_numbered_options() -> None:
    value = grammar()
    candidates = typed_candidates(
        'Use --maxfail and stop after 2 failures.',
        value,
        {b'--maxfail=3': ('argument_bytes',)},
    )
    by_value = {candidate.value: candidate for candidate in candidates}
    assert by_value[b'--maxfail=2'].derived
    assert set(by_value[b'--maxfail=3'].roles) == {byte_roles(value).index('argument_bytes')}


def test_command_argument_counts_support_pipelines() -> None:
    document = {
        's': [
            {
                't': 'p',
                'c': [
                    {'n': {'s': 'one'}, 'a': [{'s': 'a'}]},
                    {'n': {'s': 'two'}, 'a': [{'s': 'b'}, {'s': 'c'}]},
                ],
            }
        ]
    }
    assert command_argument_counts(document) == (1, 2)
