"""Separate-source/target batches for the custom semantic compiler."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from shelliq_training.data import Corpus, DatasetFormatError, SequenceTooLongError, SFTRecord, format_user_message
from shelliq_training.prompt import PromptContract

LABEL_IGNORE_INDEX = -100


class CompilerTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None
    bos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class CompilerSpecialTokens:
    pad_token_id: int
    eos_token_id: int
    decoder_start_token_id: int


@dataclass(frozen=True, slots=True)
class CompilerExample:
    record_id: str
    corpus: Corpus
    source_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CompilerBatch:
    source_ids: torch.Tensor
    source_attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> CompilerBatch:
        return CompilerBatch(
            source_ids=self.source_ids.to(device),
            source_attention_mask=self.source_attention_mask.to(device),
            decoder_input_ids=self.decoder_input_ids.to(device),
            decoder_attention_mask=self.decoder_attention_mask.to(device),
            labels=self.labels.to(device),
        )


def compiler_special_tokens(tokenizer: CompilerTokenizer) -> CompilerSpecialTokens:
    """Resolve an explicit decoder start without assuming a tokenizer family."""
    if tokenizer.pad_token_id is None:
        raise ValueError('tokenizer has no pad_token_id')
    if tokenizer.eos_token_id is None:
        raise ValueError('tokenizer has no eos_token_id')
    decoder_start = tokenizer.bos_token_id
    if decoder_start is None:
        decoder_start = tokenizer.pad_token_id
    return CompilerSpecialTokens(tokenizer.pad_token_id, tokenizer.eos_token_id, decoder_start)


def tokenize_compiler_record(
    record: SFTRecord,
    tokenizer: CompilerTokenizer,
    *,
    max_source_length: int,
    max_target_length: int,
    prompt_contract: PromptContract,
) -> CompilerExample:
    """Tokenize encoder input and semantic target without implicit truncation."""
    if max_source_length < 2 or max_target_length < 2:
        raise ValueError('source and target lengths must be at least 2')
    special = compiler_special_tokens(tokenizer)
    source_ids = tuple(
        int(token)
        for token in tokenizer.encode(
            format_user_message(record, prompt_contract=prompt_contract),
            add_special_tokens=False,
        )
    ) + (special.eos_token_id,)
    target_ids = tuple(int(token) for token in tokenizer.encode(record.response, add_special_tokens=False)) + (
        special.eos_token_id,
    )
    if len(source_ids) > max_source_length:
        raise SequenceTooLongError(
            f'{record.record_id}: source has {len(source_ids)} tokens, exceeding max_source_length={max_source_length}'
        )
    if len(target_ids) > max_target_length:
        raise SequenceTooLongError(
            f'{record.record_id}: target has {len(target_ids)} tokens, exceeding max_target_length={max_target_length}'
        )
    return CompilerExample(record.record_id, record.corpus, source_ids, target_ids)


def collate_compiler(
    examples: Sequence[CompilerExample],
    *,
    source_length: int,
    target_length: int,
    special_tokens: CompilerSpecialTokens,
) -> CompilerBatch:
    """Right-pad fixed-shape encoder and shifted decoder batches."""
    if not examples:
        raise ValueError('cannot collate an empty batch')
    corpus = examples[0].corpus
    if any(example.corpus is not corpus for example in examples):
        raise DatasetFormatError('cannot collate distributable and personal examples together')

    source_rows: list[tuple[int, ...]] = []
    source_masks: list[tuple[int, ...]] = []
    decoder_rows: list[tuple[int, ...]] = []
    decoder_masks: list[tuple[int, ...]] = []
    label_rows: list[tuple[int, ...]] = []
    for example in examples:
        if len(example.source_ids) > source_length:
            raise SequenceTooLongError(f'{example.record_id}: source exceeds source_length={source_length}')
        if len(example.target_ids) > target_length:
            raise SequenceTooLongError(f'{example.record_id}: target exceeds target_length={target_length}')
        source_padding = source_length - len(example.source_ids)
        target_padding = target_length - len(example.target_ids)
        decoder_ids = (special_tokens.decoder_start_token_id,) + example.target_ids[:-1]
        source_rows.append(example.source_ids + (special_tokens.pad_token_id,) * source_padding)
        source_masks.append((1,) * len(example.source_ids) + (0,) * source_padding)
        decoder_rows.append(decoder_ids + (special_tokens.pad_token_id,) * target_padding)
        decoder_masks.append((1,) * len(decoder_ids) + (0,) * target_padding)
        label_rows.append(example.target_ids + (LABEL_IGNORE_INDEX,) * target_padding)

    return CompilerBatch(
        source_ids=torch.tensor(source_rows, dtype=torch.long),
        source_attention_mask=torch.tensor(source_masks, dtype=torch.bool),
        decoder_input_ids=torch.tensor(decoder_rows, dtype=torch.long),
        decoder_attention_mask=torch.tensor(decoder_masks, dtype=torch.bool),
        labels=torch.tensor(label_rows, dtype=torch.long),
    )
