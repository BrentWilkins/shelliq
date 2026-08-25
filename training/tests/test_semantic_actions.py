import json
from dataclasses import replace

import pytest

from shelliq_training.data import Corpus, Platform, SequenceTooLongError, SFTRecord
from shelliq_training.prompt import PromptContract
from shelliq_training.semantic_actions import ActionGrammar, action_examples


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [3 + byte % 13 for byte in text.encode()]


def record() -> SFTRecord:
    return SFTRecord(
        record_id='actions:echo:1',
        corpus=Corpus.DISTRIBUTABLE,
        source='reviewed',
        license='MIT',
        provenance='fixture',
        command='echo',
        platform=Platform.LINUX,
        instruction='Print café.',
        response=json.dumps({'v': 2}),
        context='echo prints text.',
    )


def manifest() -> dict[str, object]:
    return {
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
            {'id': 0, 'name': 'start', 'fixed': [{'token': 1, 'next_state': 1}]},
            {
                'id': 1,
                'name': 'word',
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


def test_manifest_drives_utf8_mask_and_transitions() -> None:
    grammar = ActionGrammar(manifest())
    cursor = grammar.cursor()
    cursor.advance(grammar.tokens['BOS'])
    assert 64 + 0xC3 in cursor.allowed()
    cursor.advance(64 + 0xC3)
    assert 23 not in cursor.allowed()
    assert 64 + 0x28 not in cursor.allowed()
    cursor.advance(64 + 0xA9)
    assert 23 in cursor.allowed()
    cursor.advance(23)
    cursor.advance(2)
    assert cursor.complete


def test_manifest_rejects_unlisted_transition() -> None:
    cursor = ActionGrammar(manifest()).cursor()
    with pytest.raises(ValueError, match='not allowed'):
        cursor.advance(2)


def test_action_examples_preserve_prompt_and_reject_truncation() -> None:
    actions = [(1, 3, 4, 2)]
    examples = action_examples(
        [record()],
        actions,
        FakeTokenizer(),
        source_length=512,
        target_length=4,
        prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
    )
    assert examples[0].action_ids == actions[0]
    assert examples[0].source_ids[-1] == 2
    with pytest.raises(SequenceTooLongError, match='target has'):
        action_examples(
            [replace(record(), record_id='actions:echo:2')],
            actions,
            FakeTokenizer(),
            source_length=512,
            target_length=3,
            prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
        )
