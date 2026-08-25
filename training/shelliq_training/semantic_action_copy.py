"""Source-copy supervision for semantic actions, derived from the Rust grammar."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tokenizers import ByteLevelBPETokenizer

from shelliq_training.data import Corpus, DatasetFormatError, SequenceTooLongError, SFTRecord, format_user_message
from shelliq_training.prompt import PromptContract
from shelliq_training.semantic_action_model import LABEL_IGNORE_INDEX
from shelliq_training.semantic_action_pointer_model import PointerActionBatch
from shelliq_training.semantic_actions import ActionGrammar

COPY_IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class CopyAlignment:
    labels: tuple[int, ...]
    roles: tuple[str | None, ...]
    word_counts: dict[str, tuple[int, int]]
    byte_counts: dict[str, tuple[int, int]]


@dataclass(frozen=True, slots=True)
class PointerActionExample:
    record_id: str
    corpus: Corpus
    source_ids: tuple[int, ...]
    source_bytes: tuple[int, ...]
    byte_token_indices: tuple[tuple[int, ...], ...]
    action_ids: tuple[int, ...]
    copy_labels: tuple[int, ...]


class AlignedCodeT5Tokenizer:
    """CodeT5 byte-BPE IDs plus source offsets required by the pointer."""

    def __init__(self, vocab: str, merges: str, *, pad_token_id: int = 0, eos_token_id: int = 2) -> None:
        self.backend = ByteLevelBPETokenizer(vocab, merges)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def encode_with_byte_alignment(self, source: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
        encoded = self.backend.encode(source, add_special_tokens=False)
        source_ids = tuple(encoded.ids) + (self.eos_token_id,)
        raw = tuple(source.encode('utf-8'))
        alignment: list[list[int]] = [[] for _ in raw]
        byte_offsets = [0]
        for character in source:
            byte_offsets.append(byte_offsets[-1] + len(character.encode('utf-8')))
        for token_index, (start, end) in enumerate(encoded.offsets):
            for byte_index in range(byte_offsets[start], byte_offsets[end]):
                alignment[byte_index].append(token_index)
        if any(not token_indices for token_indices in alignment):
            missing = next(index for index, token_indices in enumerate(alignment) if not token_indices)
            raise ValueError(f'CodeT5 tokenizer did not align source byte {missing}')
        return source_ids, raw, tuple(tuple(indices) for indices in alignment)


def align_action_bytes_to_source(source: str, actions: Sequence[int], grammar: ActionGrammar) -> CopyAlignment:
    """Align complete action words to their first exact UTF-8 source span."""
    source_bytes = source.encode('utf-8')
    labels = [COPY_IGNORE_INDEX] * len(actions)
    roles: list[str | None] = [None] * len(actions)
    word_total: Counter[str] = Counter()
    word_copyable: Counter[str] = Counter()
    byte_total: Counter[str] = Counter()
    byte_copyable: Counter[str] = Counter()
    cursor = grammar.cursor()
    position = 0
    while position < len(actions):
        state = grammar.states[cursor.state]
        payload = state.byte_payload
        if payload is None:
            cursor.advance(actions[position])
            position += 1
            continue
        begin = position
        raw = bytearray()
        while position < len(actions) and actions[position] != payload.end_token:
            token = actions[position]
            if not payload.byte_offset <= token < payload.byte_offset + payload.byte_count:
                raise ValueError(f'non-byte action {token} inside {state.name}')
            raw.append(token - payload.byte_offset)
            roles[position] = state.name
            cursor.advance(token)
            position += 1
        if position == len(actions):
            raise ValueError(f'incomplete action word in {state.name}')
        raw.decode('utf-8')
        word_total[state.name] += 1
        byte_total[state.name] += len(raw)
        source_position = source_bytes.find(raw)
        if source_position >= 0:
            word_copyable[state.name] += 1
            byte_copyable[state.name] += len(raw)
            for offset in range(len(raw)):
                labels[begin + offset] = source_position + offset
        cursor.advance(actions[position])
        position += 1
    if not cursor.complete:
        raise ValueError('action sequence did not complete grammar during copy alignment')
    role_names = sorted(word_total)
    return CopyAlignment(
        tuple(labels),
        tuple(roles),
        {role: (word_copyable[role], word_total[role]) for role in role_names},
        {role: (byte_copyable[role], byte_total[role]) for role in role_names},
    )


def merge_copy_counts(alignments: Sequence[CopyAlignment], field: str) -> dict[str, dict[str, int | float]]:
    """Aggregate role counts into stable report rows."""
    copied: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for alignment in alignments:
        values = alignment.word_counts if field == 'words' else alignment.byte_counts
        for role, (role_copied, role_total) in values.items():
            copied[role] += role_copied
            total[role] += role_total
    copied['all'] = sum(value for role, value in copied.items() if role != 'all')
    total['all'] = sum(value for role, value in total.items() if role != 'all')
    return {
        role: {
            'copyable': copied[role],
            'total': total[role],
            'rate': copied[role] / total[role] if total[role] else 0.0,
        }
        for role in sorted(total)
    }


def pointer_action_examples(
    records: Sequence[SFTRecord],
    actions: Sequence[Sequence[int]],
    tokenizer: AlignedCodeT5Tokenizer,
    grammar: ActionGrammar,
    *,
    source_length: int,
    source_byte_length: int,
    target_length: int,
    prompt_contract: PromptContract,
) -> list[PointerActionExample]:
    if len(records) != len(actions):
        raise ValueError('records and action targets are not aligned')
    examples = []
    for record, target in zip(records, actions, strict=True):
        source = format_user_message(record, prompt_contract=prompt_contract)
        source_ids, source_bytes, byte_token_indices = tokenizer.encode_with_byte_alignment(source)
        target_ids = tuple(target)
        copy = align_action_bytes_to_source(source, target_ids, grammar)
        if len(source_ids) > source_length:
            raise SequenceTooLongError(f'{record.record_id}: source exceeds {source_length} CodeT5 tokens')
        if len(source_bytes) > source_byte_length:
            raise SequenceTooLongError(f'{record.record_id}: source exceeds {source_byte_length} UTF-8 bytes')
        if len(target_ids) > target_length:
            raise SequenceTooLongError(f'{record.record_id}: target exceeds {target_length} actions')
        examples.append(
            PointerActionExample(
                record.record_id,
                record.corpus,
                source_ids,
                source_bytes,
                byte_token_indices,
                target_ids,
                copy.labels,
            )
        )
    return examples


def collate_pointer_actions(
    examples: Sequence[PointerActionExample],
    *,
    source_length: int,
    source_byte_length: int,
    target_length: int,
    source_pad_token_id: int,
) -> PointerActionBatch:
    if not examples:
        raise ValueError('cannot collate an empty pointer batch')
    if any(example.corpus is not examples[0].corpus for example in examples):
        raise DatasetFormatError('cannot mix corpus privacy classes')
    source_rows = []
    source_masks = []
    byte_rows = []
    byte_masks = []
    alignments = []
    decoder_rows = []
    decoder_masks = []
    labels = []
    copy_labels = []
    for example in examples:
        if (
            len(example.source_ids) > source_length
            or len(example.source_bytes) > source_byte_length
            or len(example.action_ids) > target_length
        ):
            raise SequenceTooLongError(f'{example.record_id}: pointer batch shape would truncate data')
        source_padding = source_length - len(example.source_ids)
        byte_padding = source_byte_length - len(example.source_bytes)
        shifted = example.action_ids[:-1]
        expected = example.action_ids[1:]
        expected_copy = example.copy_labels[1:]
        target_padding = target_length - len(expected)
        source_rows.append(example.source_ids + (source_pad_token_id,) * source_padding)
        source_masks.append((1,) * len(example.source_ids) + (0,) * source_padding)
        byte_rows.append(example.source_bytes + (0,) * byte_padding)
        byte_masks.append((1,) * len(example.source_bytes) + (0,) * byte_padding)
        alignment = torch.zeros(source_byte_length, source_length)
        for byte_index, token_indices in enumerate(example.byte_token_indices):
            weight = 1 / len(token_indices)
            alignment[byte_index, list(token_indices)] = weight
        alignments.append(alignment)
        decoder_rows.append(shifted + (0,) * target_padding)
        decoder_masks.append((1,) * len(shifted) + (0,) * target_padding)
        labels.append(expected + (LABEL_IGNORE_INDEX,) * target_padding)
        copy_labels.append(expected_copy + (COPY_IGNORE_INDEX,) * target_padding)
    return PointerActionBatch(
        torch.tensor(source_rows, dtype=torch.long),
        torch.tensor(source_masks, dtype=torch.bool),
        torch.tensor(byte_rows, dtype=torch.long),
        torch.tensor(byte_masks, dtype=torch.bool),
        torch.stack(alignments),
        torch.tensor(decoder_rows, dtype=torch.long),
        torch.tensor(decoder_masks, dtype=torch.bool),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(copy_labels, dtype=torch.long),
    )
