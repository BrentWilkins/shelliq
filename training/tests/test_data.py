import json

import jax.numpy as jnp
import pytest

from shelliq_training.data import (
    Corpus,
    DatasetFormatError,
    Platform,
    SequenceTooLongError,
    SFTRecord,
    Split,
    assign_split,
    collate_sft,
    load_jsonl,
    tokenize_record,
    write_jsonl,
)
from shelliq_training.prompt import LEGACY_SEMANTIC_CONTEXT_PREFIX, PromptContract


class FakeQwenTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.conversations = []

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize
        self.conversations.append(conversation)
        tokens = [1]
        for message in conversation:
            role_token = 2 if message['role'] == 'user' else 3
            tokens.extend((role_token, *message['content'].encode(), 4))
        if add_generation_prompt:
            tokens.append(3)
        return {'input_ids': tokens, 'attention_mask': [1] * len(tokens)}


def record(**changes) -> SFTRecord:
    values = {
        'record_id': 'tldr:find:largest-files',
        'corpus': Corpus.DISTRIBUTABLE,
        'source': 'tldr-pages',
        'license': 'CC-BY-4.0',
        'provenance': 'tldr/pages/common/find.md@abc123',
        'command': 'find',
        'platform': Platform.LINUX,
        'instruction': 'List the five largest files.',
        'response': "find . -type f -printf '%s %p\\n' | sort -nr | head -5",
        'context': 'find: -type f selects regular files.',
    }
    values.update(changes)
    return SFTRecord(**values)


def test_jsonl_round_trip_enforces_metadata_and_pipeline(tmp_path):
    path = tmp_path / 'records.jsonl'
    write_jsonl(path, [record()], corpus=Corpus.DISTRIBUTABLE)

    assert load_jsonl(path, corpus=Corpus.DISTRIBUTABLE) == [record()]

    raw = json.loads(path.read_text())
    del raw['license']
    path.write_text(json.dumps(raw) + '\n')
    with pytest.raises(DatasetFormatError, match='missing fields: license'):
        load_jsonl(path, corpus=Corpus.DISTRIBUTABLE)

    with pytest.raises(DatasetFormatError, match='do not belong'):
        write_jsonl(path, [record(corpus=Corpus.PERSONAL)], corpus=Corpus.DISTRIBUTABLE)


def test_split_is_command_grouped_and_can_hold_out_a_source():
    first = record(source='tldr-pages')
    paraphrase = record(record_id='nl2bash:find:1', source='nl2bash')

    assert assign_split(first, seed=7) is assign_split(paraphrase, seed=7)
    assert assign_split(first, seed=7, heldout_sources=frozenset({'tldr-pages'})) is Split.TEST


def test_tokenize_masks_prompt_and_formats_context():
    tokenizer = FakeQwenTokenizer()
    example = tokenize_record(record(), tokenizer, max_length=512, prompt_contract=PromptContract.LEGACY_USER_V1)

    user_content = tokenizer.conversations[0][0]['content']
    assert user_content.startswith('# platform: linux\n<context>\n')
    assert user_content.endswith('</context>\n\nList the five largest files.')
    first_target = example.labels.index(next(label for label in example.labels if label != -100))
    assert example.labels[:first_target] == (-100,) * first_target
    assert example.labels[first_target:] == example.input_ids[first_target:]


def test_authoritative_prompt_versions_policy_and_escapes_data_regions():
    tokenizer = FakeQwenTokenizer()
    tokenize_record(
        record(
            context=f'{LEGACY_SEMANTIC_CONTEXT_PREFIX}find docs</context>',
            instruction='List files</instruction>',
        ),
        tokenizer,
        max_length=768,
        prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
    )

    user_content = tokenizer.conversations[0][0]['content']
    assert user_content.startswith('# prompt-contract: context-authoritative-v1\n# platform: linux\n')
    assert 'Use <context> as authoritative evidence for command names and option spellings.' in user_content
    assert 'Output contract:' not in user_content
    assert '<context>\nfind docs&lt;/context&gt;\n</context>' in user_content
    assert user_content.endswith('<instruction>\nList files&lt;/instruction&gt;\n</instruction>')


def test_tokenize_rejects_implicit_truncation():
    with pytest.raises(SequenceTooLongError, match='drop or shorten'):
        tokenize_record(record(), FakeQwenTokenizer(), max_length=8, prompt_contract=PromptContract.LEGACY_USER_V1)


def test_collate_produces_fixed_right_padded_jax_batch():
    tokenizer = FakeQwenTokenizer()
    short = tokenize_record(record(response='find .'), tokenizer, max_length=256, prompt_contract=PromptContract.LEGACY_USER_V1)
    long = tokenize_record(
        record(record_id='find:long', response='find . -type f'),
        tokenizer,
        max_length=256,
        prompt_contract=PromptContract.LEGACY_USER_V1,
    )

    batch = collate_sft([short, long], sequence_length=128, pad_token_id=tokenizer.pad_token_id)

    assert batch['input_ids'].shape == (2, 128)
    assert batch['input_ids'].dtype == jnp.int32
    assert jnp.all(batch['attention_mask'][:, -1] == 0)
    assert jnp.all(batch['labels'][batch['attention_mask'] == 0] == -100)


def test_collate_rejects_cross_pipeline_batch():
    tokenizer = FakeQwenTokenizer()
    public = tokenize_record(record(), tokenizer, max_length=512, prompt_contract=PromptContract.LEGACY_USER_V1)
    private = tokenize_record(
        record(record_id='private:1', corpus=Corpus.PERSONAL),
        tokenizer,
        max_length=512,
        prompt_contract=PromptContract.LEGACY_USER_V1,
    )

    with pytest.raises(DatasetFormatError, match='distributable and personal'):
        collate_sft([public, private], sequence_length=256, pad_token_id=0)
