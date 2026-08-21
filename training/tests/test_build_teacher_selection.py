import json
from pathlib import Path

from scripts.build_teacher_selection import prompt_determines_expected, semantic_words, word_is_visible


def test_semantic_words_preserve_document_order():
    document = {
        'v': 2,
        'd': 'zsh',
        's': [
            {
                't': 'p',
                'c': [
                    {
                        'n': {'s': 'cp'},
                        'a': [{'s': '-av'}, {'s': 'source/'}, {'s': 'dest/'}],
                    }
                ],
            }
        ],
    }
    assert semantic_words(document) == ['cp', '-av', 'source/', 'dest/']


def test_visible_word_accepts_exact_literals_and_documented_short_clusters():
    prompt = 'cp: -a archives and -v reports files. Copy source/ to dest/.'
    assert word_is_visible('cp', prompt)
    assert word_is_visible('-av', prompt)
    assert word_is_visible('source/', prompt)
    assert not word_is_visible('hidden/', prompt)


def test_visible_word_allows_derived_quoted_program_but_not_hidden_quoted_operand():
    prompt = 'awk prints the third field from access.log.'
    assert word_is_visible("'{print $3}'", prompt)
    assert word_is_visible('access.log', prompt)
    assert not word_is_visible("'hidden.log'", prompt)


def test_prompt_determined_rejects_hidden_operands():
    row = {
        'instruction': 'Copy the source to the destination.',
        'context': 'cp -a preserves metadata.',
        'semantic_target': {
            'v': 2,
            'd': 'zsh',
            's': [
                {
                    't': 'p',
                    'c': [
                        {
                            'n': {'s': 'cp'},
                            'a': [{'s': '-a'}, {'s': 'old-tree/'}, {'s': 'new-tree/'}],
                        }
                    ],
                }
            ],
        },
    }
    assert not prompt_determines_expected(row)


def test_prompt_determined_accepts_fully_visible_reference():
    row = {
        'instruction': 'Copy old-tree/ to new-tree/.',
        'context': 'cp -a preserves metadata.',
        'semantic_target': {
            'v': 2,
            'd': 'zsh',
            's': [
                {
                    't': 'p',
                    'c': [
                        {
                            'n': {'s': 'cp'},
                            'a': [{'s': '-a'}, {'s': 'old-tree/'}, {'s': 'new-tree/'}],
                        }
                    ],
                }
            ],
        },
    }
    assert prompt_determines_expected(row)


def test_frozen_v2_set_is_small_balanced_and_prompt_determined():
    path = Path(__file__).resolve().parents[1] / 'evaluation' / 'teacher-selection-v2.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(rows) == 20
    assert len({row['challenge_id'] for row in rows}) == 20
    assert {row['category'] for row in rows} == {'precise', 'multi-constraint', 'pipeline', 'compositional'}
    assert sum(row['platform'] == 'darwin' for row in rows) == 4
    for row in rows:
        assert row['challenge_id'].startswith('teacher-selection:v2:')
        assert prompt_determines_expected(
            {
                'instruction': row['instruction'],
                'context': row['context'],
                'semantic_target': row['expected'],
            }
        )
