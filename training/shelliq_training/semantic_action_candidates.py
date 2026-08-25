"""Deterministic lexical candidates for atomic semantic-action copying."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from shelliq_training.data import (
    Corpus,
    DatasetFormatError,
    SequenceTooLongError,
    SFTRecord,
    format_user_message,
)
from shelliq_training.prompt import PromptContract
from shelliq_training.semantic_action_candidate_model import CandidateActionBatch
from shelliq_training.semantic_action_copy import AlignedCodeT5Tokenizer
from shelliq_training.semantic_action_model import LABEL_IGNORE_INDEX
from shelliq_training.semantic_actions import ActionGrammar

CANDIDATE_IGNORE_INDEX = -100
_WRAPPERS = ' \t\r\n.,;:!?()[]{}<>"\'`'
_SHELL_ATOM = re.compile(r'[A-Za-z0-9_./~$%+@=-]+')
_NONSPACE = re.compile(r'\S+')


@dataclass(frozen=True, slots=True)
class LexicalCandidate:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class CandidateAlignment:
    labels: tuple[int, ...]
    word_starts: tuple[bool, ...]
    roles: tuple[str | None, ...]
    word_counts: dict[str, tuple[int, int]]


@dataclass(frozen=True, slots=True)
class CandidateActionExample:
    record_id: str
    corpus: Corpus
    source_ids: tuple[int, ...]
    source_bytes: tuple[int, ...]
    byte_token_indices: tuple[tuple[int, ...], ...]
    candidates: tuple[LexicalCandidate, ...]
    action_ids: tuple[int, ...]
    candidate_labels: tuple[int, ...]
    word_starts: tuple[bool, ...]


def lexical_candidates(source: str, *, maximum_bytes: int = 128) -> tuple[LexicalCandidate, ...]:
    """Extract stable, complete source spans without learned boundaries."""
    if maximum_bytes <= 0:
        raise ValueError('maximum_bytes must be positive')
    byte_offsets = [0]
    for character in source:
        byte_offsets.append(byte_offsets[-1] + len(character.encode('utf-8')))

    spans: set[tuple[int, int]] = set()

    def add(start: int, end: int, *, trim: bool = True) -> None:
        if trim:
            while start < end and source[start] in _WRAPPERS:
                start += 1
            while end > start and source[end - 1] in _WRAPPERS:
                end -= 1
        byte_start, byte_end = byte_offsets[start], byte_offsets[end]
        if 0 < byte_end - byte_start <= maximum_bytes:
            spans.add((byte_start, byte_end))

    for match in _NONSPACE.finditer(source):
        add(*match.span())
    for match in _SHELL_ATOM.finditer(source):
        add(*match.span())
    for quote in ('"', "'", '`'):
        pattern = re.compile(re.escape(quote) + r'([^' + re.escape(quote) + r']+)' + re.escape(quote))
        for match in pattern.finditer(source):
            add(match.start(1), match.end(1), trim=False)

    return tuple(LexicalCandidate(start, end) for start, end in sorted(spans))


def align_actions_to_candidates(
    source_bytes: bytes,
    actions: Sequence[int],
    grammar: ActionGrammar,
    candidates: Sequence[LexicalCandidate],
    global_words: Sequence[bytes] = (),
) -> CandidateAlignment:
    """Supervise one candidate choice at the first byte of each word."""
    labels = [CANDIDATE_IGNORE_INDEX] * len(actions)
    word_starts = [False] * len(actions)
    roles: list[str | None] = [None] * len(actions)
    copied: Counter[str] = Counter()
    total: Counter[str] = Counter()
    values = [source_bytes[item.start : item.end] for item in candidates]
    global_indices = {value: index for index, value in enumerate(global_words)}
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
        while position < len(actions) and payload.byte_offset <= actions[position] < payload.byte_offset + payload.byte_count:
            raw.append(actions[position] - payload.byte_offset)
            roles[position] = state.name
            cursor.advance(actions[position])
            position += 1
        if not raw or position == len(actions):
            raise ValueError(f'incomplete action word in {state.name}')
        raw.decode('utf-8')
        word_starts[begin] = True
        total[state.name] += 1
        value = bytes(raw)
        if value in global_indices:
            labels[begin] = global_indices[value]
            copied[state.name] += 1
        else:
            try:
                labels[begin] = len(global_words) + values.index(value)
            except ValueError:
                pass
            else:
                copied[state.name] += 1
        cursor.advance(actions[position])
        position += 1
    if not cursor.complete:
        raise ValueError('action sequence did not complete grammar during candidate alignment')
    role_names = sorted(total)
    return CandidateAlignment(
        tuple(labels),
        tuple(word_starts),
        tuple(roles),
        {role: (copied[role], total[role]) for role in role_names},
    )


def action_words(actions: Sequence[int], grammar: ActionGrammar) -> tuple[bytes, ...]:
    """Return complete semantic word payloads in action order."""
    words: list[bytes] = []
    cursor = grammar.cursor()
    position = 0
    while position < len(actions):
        state = grammar.states[cursor.state]
        payload = state.byte_payload
        if payload is None:
            cursor.advance(actions[position])
            position += 1
            continue
        raw = bytearray()
        while position < len(actions) and payload.byte_offset <= actions[position] < payload.byte_offset + payload.byte_count:
            raw.append(actions[position] - payload.byte_offset)
            cursor.advance(actions[position])
            position += 1
        if not raw or position == len(actions):
            raise ValueError(f'incomplete action word in {state.name}')
        raw.decode('utf-8')
        words.append(bytes(raw))
        cursor.advance(actions[position])
        position += 1
    if not cursor.complete:
        raise ValueError('action sequence did not complete grammar')
    return tuple(words)


def global_word_lexicon(sequences: Sequence[Sequence[int]], grammar: ActionGrammar) -> tuple[bytes, ...]:
    """Build a stable candidate vocabulary from training targets only."""
    return tuple(sorted({word for actions in sequences for word in action_words(actions, grammar)}))


def relabel_with_global_words(
    examples: Sequence[CandidateActionExample],
    grammar: ActionGrammar,
    global_words: Sequence[bytes],
) -> list[CandidateActionExample]:
    """Prefer stable global candidates, then exact prompt-local candidates."""
    return [
        replace(
            example,
            candidate_labels=align_actions_to_candidates(
                bytes(example.source_bytes),
                example.action_ids,
                grammar,
                example.candidates,
                global_words,
            ).labels,
        )
        for example in examples
    ]


def merge_candidate_counts(alignments: Sequence[CandidateAlignment]) -> dict[str, dict[str, int | float]]:
    copied: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for alignment in alignments:
        for role, (role_copied, role_total) in alignment.word_counts.items():
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


def candidate_action_examples(
    records: Sequence[SFTRecord],
    actions: Sequence[Sequence[int]],
    tokenizer: AlignedCodeT5Tokenizer,
    grammar: ActionGrammar,
    *,
    source_length: int,
    source_byte_length: int,
    target_length: int,
    prompt_contract: PromptContract,
) -> list[CandidateActionExample]:
    if len(records) != len(actions):
        raise ValueError('records and semantic actions are not aligned')
    examples: list[CandidateActionExample] = []
    for record, target in zip(records, actions, strict=True):
        source = format_user_message(record, prompt_contract=prompt_contract)
        source_ids, source_bytes, byte_token_indices = tokenizer.encode_with_byte_alignment(source)
        if len(source_ids) > source_length:
            raise SequenceTooLongError(f'{record.record_id}: CodeT5 source exceeds {source_length} tokens')
        if len(source_bytes) > source_byte_length:
            raise SequenceTooLongError(f'{record.record_id}: source exceeds {source_byte_length} bytes')
        if len(target) > target_length:
            raise SequenceTooLongError(f'{record.record_id}: actions exceed {target_length} tokens')
        candidates = lexical_candidates(source)
        alignment = align_actions_to_candidates(bytes(source_bytes), target, grammar, candidates)
        examples.append(
            CandidateActionExample(
                record.record_id,
                record.corpus,
                source_ids,
                source_bytes,
                byte_token_indices,
                candidates,
                tuple(target),
                alignment.labels,
                alignment.word_starts,
            )
        )
    return examples


def collate_candidate_actions(
    examples: Sequence[CandidateActionExample],
    *,
    source_length: int,
    source_byte_length: int,
    candidate_count: int,
    target_length: int,
    source_pad_token_id: int,
) -> CandidateActionBatch:
    if not examples:
        raise ValueError('cannot collate an empty candidate batch')
    if candidate_count <= 0:
        raise ValueError('candidate_count must be positive')
    if any(example.corpus is not examples[0].corpus for example in examples):
        raise DatasetFormatError('cannot mix corpus privacy classes')
    source_rows, source_masks, byte_rows, byte_masks, alignments = [], [], [], [], []
    candidate_starts, candidate_ends, candidate_masks = [], [], []
    decoder_rows, decoder_masks, labels, candidate_labels, word_starts = [], [], [], [], []
    for example in examples:
        if (
            len(example.source_ids) > source_length
            or len(example.source_bytes) > source_byte_length
            or len(example.candidates) > candidate_count
            or len(example.action_ids) > target_length
        ):
            raise SequenceTooLongError(f'{example.record_id}: candidate batch shape would truncate data')
        source_padding = source_length - len(example.source_ids)
        byte_padding = source_byte_length - len(example.source_bytes)
        candidate_padding = candidate_count - len(example.candidates)
        shifted, expected = example.action_ids[:-1], example.action_ids[1:]
        target_padding = target_length - len(expected)
        source_rows.append(example.source_ids + (source_pad_token_id,) * source_padding)
        source_masks.append((1,) * len(example.source_ids) + (0,) * source_padding)
        byte_rows.append(example.source_bytes + (0,) * byte_padding)
        byte_masks.append((1,) * len(example.source_bytes) + (0,) * byte_padding)
        alignment = torch.zeros(source_byte_length, source_length)
        for byte_index, token_indices in enumerate(example.byte_token_indices):
            alignment[byte_index, list(token_indices)] = 1 / len(token_indices)
        alignments.append(alignment)
        candidate_starts.append(tuple(item.start for item in example.candidates) + (0,) * candidate_padding)
        candidate_ends.append(tuple(item.end for item in example.candidates) + (0,) * candidate_padding)
        candidate_masks.append((1,) * len(example.candidates) + (0,) * candidate_padding)
        decoder_rows.append(shifted + (0,) * target_padding)
        decoder_masks.append((1,) * len(shifted) + (0,) * target_padding)
        labels.append(expected + (LABEL_IGNORE_INDEX,) * target_padding)
        candidate_labels.append(example.candidate_labels[1:] + (CANDIDATE_IGNORE_INDEX,) * target_padding)
        word_starts.append(example.word_starts[1:] + (False,) * target_padding)
    return CandidateActionBatch(
        torch.tensor(source_rows, dtype=torch.long),
        torch.tensor(source_masks, dtype=torch.bool),
        torch.tensor(byte_rows, dtype=torch.long),
        torch.tensor(byte_masks, dtype=torch.bool),
        torch.stack(alignments),
        torch.tensor(candidate_starts, dtype=torch.long),
        torch.tensor(candidate_ends, dtype=torch.long),
        torch.tensor(candidate_masks, dtype=torch.bool),
        torch.tensor(decoder_rows, dtype=torch.long),
        torch.tensor(decoder_masks, dtype=torch.bool),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(candidate_labels, dtype=torch.long),
        torch.tensor(word_starts, dtype=torch.bool),
    )
