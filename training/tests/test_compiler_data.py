import pytest
import torch

from shelliq_training.compiler_data import (
    LABEL_IGNORE_INDEX,
    CompilerSpecialTokens,
    collate_compiler,
    compiler_special_tokens,
    tokenize_compiler_record,
)
from shelliq_training.data import Corpus, DatasetFormatError, Platform, SequenceTooLongError, SFTRecord
from shelliq_training.prompt import PromptContract


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = None

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [3 + byte % 23 for byte in text.encode()]


def record(**changes) -> SFTRecord:
    values = {
        'record_id': 'compiler:find:1',
        'corpus': Corpus.DISTRIBUTABLE,
        'source': 'reviewed',
        'license': 'MIT',
        'provenance': 'fixture',
        'command': 'find',
        'platform': Platform.LINUX,
        'instruction': 'List files.',
        'response': '{"v":2}',
        'context': 'find: list files.',
    }
    values.update(changes)
    return SFTRecord(**values)


def test_tokenize_compiler_keeps_source_and_target_separate():
    tokenizer = FakeTokenizer()
    example = tokenize_compiler_record(
        record(),
        tokenizer,
        max_source_length=512,
        max_target_length=64,
        prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
    )

    assert example.source_ids[-1] == tokenizer.eos_token_id
    assert example.target_ids[-1] == tokenizer.eos_token_id
    assert compiler_special_tokens(tokenizer) == CompilerSpecialTokens(0, 2, 0)


def test_tokenize_compiler_rejects_implicit_truncation():
    with pytest.raises(SequenceTooLongError, match='source has'):
        tokenize_compiler_record(
            record(),
            FakeTokenizer(),
            max_source_length=2,
            max_target_length=64,
            prompt_contract=PromptContract.LEGACY_USER_V1,
        )


def test_collate_compiler_shifts_targets_and_masks_padding():
    public = tokenize_compiler_record(
        record(),
        FakeTokenizer(),
        max_source_length=512,
        max_target_length=64,
        prompt_contract=PromptContract.LEGACY_USER_V1,
    )
    batch = collate_compiler(
        [public],
        source_length=512,
        target_length=64,
        special_tokens=CompilerSpecialTokens(0, 2, 1),
    )

    assert batch.decoder_input_ids[0, 0].item() == 1
    assert torch.equal(batch.decoder_input_ids[0, 1 : len(public.target_ids)], batch.labels[0, : len(public.target_ids) - 1])
    assert torch.all(batch.labels[~batch.decoder_attention_mask] == LABEL_IGNORE_INDEX)
    assert batch.source_attention_mask.dtype is torch.bool


def test_collate_compiler_rejects_mixed_corpora():
    tokenizer = FakeTokenizer()
    examples = [
        tokenize_compiler_record(
            record(),
            tokenizer,
            max_source_length=512,
            max_target_length=64,
            prompt_contract=PromptContract.LEGACY_USER_V1,
        ),
        tokenize_compiler_record(
            record(record_id='personal:1', corpus=Corpus.PERSONAL),
            tokenizer,
            max_source_length=512,
            max_target_length=64,
            prompt_contract=PromptContract.LEGACY_USER_V1,
        ),
    ]
    with pytest.raises(DatasetFormatError, match='distributable and personal'):
        collate_compiler(
            examples,
            source_length=512,
            target_length=64,
            special_tokens=CompilerSpecialTokens(0, 2, 1),
        )
