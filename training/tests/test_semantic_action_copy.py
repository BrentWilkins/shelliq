from shelliq_training.semantic_action_copy import COPY_IGNORE_INDEX, align_action_bytes_to_source, merge_copy_counts
from shelliq_training.semantic_actions import ActionGrammar


def manifest() -> dict[str, object]:
    return {
        'schema_version': 1,
        'vocab_size': 320,
        'start_state': 0,
        'complete_state': 3,
        'tokens': [{'id': 1, 'name': 'BOS'}, {'id': 2, 'name': 'EOS'}, {'id': 23, 'name': 'WORD_END'}],
        'states': [
            {'id': 0, 'name': 'start', 'fixed': [{'token': 1, 'next_state': 1}]},
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
            {'id': 2, 'name': 'after_word', 'fixed': [{'token': 2, 'next_state': 3}]},
            {'id': 3, 'name': 'complete', 'fixed': []},
        ],
    }


def test_aligns_utf8_bytes_to_exact_source_span() -> None:
    grammar = ActionGrammar(manifest())
    actions = (1, 64 + 0x63, 64 + 0x61, 64 + 0x66, 64 + 0xC3, 64 + 0xA9, 23, 2)
    alignment = align_action_bytes_to_source('Run café now.', actions, grammar)
    start = len(b'Run ')
    assert alignment.labels == (COPY_IGNORE_INDEX, start, start + 1, start + 2, start + 3, start + 4, -100, -100)
    assert alignment.span_end_labels == (COPY_IGNORE_INDEX, start + 4, -100, -100, -100, -100, -100, -100)
    assert alignment.word_counts == {'command_name_bytes': (1, 1)}
    assert alignment.byte_counts == {'command_name_bytes': (5, 5)}


def test_marks_absent_word_as_generator_only() -> None:
    grammar = ActionGrammar(manifest())
    actions = (1, 64 + ord('x'), 23, 2)
    alignment = align_action_bytes_to_source('no match', actions, grammar)
    assert alignment.labels == (-100, -100, -100, -100)
    assert alignment.span_end_labels == (-100, -100, -100, -100)
    merged = merge_copy_counts([alignment], 'words')
    assert merged['all'] == {'copyable': 0, 'total': 1, 'rate': 0.0}
