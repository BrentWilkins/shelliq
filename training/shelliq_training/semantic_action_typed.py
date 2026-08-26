"""Typed semantic-word candidates and argument-count supervision."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from shelliq_training.data import Corpus, DatasetFormatError, SequenceTooLongError, SFTRecord, format_user_message
from shelliq_training.prompt import PromptContract
from shelliq_training.semantic_action_candidates import (
    CANDIDATE_IGNORE_INDEX,
    AlignedCodeT5Tokenizer,
    LexicalCandidate,
    action_words,
    lexical_candidates,
)
from shelliq_training.semantic_action_model import LABEL_IGNORE_INDEX
from shelliq_training.semantic_action_typed_model import TypedActionBatch
from shelliq_training.semantic_actions import ActionGrammar

COUNT_IGNORE_INDEX = -100
MAXIMUM_ARGUMENTS = 12
BIGRAM_BUCKETS = 512
_NUMBER = re.compile(r'\d+')
_LONG_OPTION = re.compile(r'--[A-Za-z][A-Za-z0-9-]*')
_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_SINGLE_FLAG = re.compile(r'(?<![\w-])-[A-Za-z](?![A-Za-z])')
_NUMBER_WORDS = {
    'zero': 0,
    'one': 1,
    'single': 1,
    'once': 1,
    'first': 1,
    'two': 2,
    'second': 2,
    'three': 3,
    'third': 3,
    'four': 4,
    'fourth': 4,
    'five': 5,
    'fifth': 5,
    'six': 6,
    'sixth': 6,
    'seven': 7,
    'seventh': 7,
    'eight': 8,
    'eighth': 8,
    'nine': 9,
    'ninth': 9,
    'ten': 10,
    'tenth': 10,
    'eleven': 11,
    'eleventh': 11,
    'twelve': 12,
    'twelfth': 12,
    'thirteen': 13,
    'thirteenth': 13,
    'fourteen': 14,
    'fourteenth': 14,
    'fifteen': 15,
    'fifteenth': 15,
    'sixteen': 16,
    'sixteenth': 16,
    'seventeen': 17,
    'seventeenth': 17,
    'eighteen': 18,
    'eighteenth': 18,
    'nineteen': 19,
    'nineteenth': 19,
    'twenty': 20,
    'twentieth': 20,
}


@dataclass(frozen=True, slots=True)
class TypedCandidate:
    value: bytes
    local_span: LexicalCandidate | None
    local: bool
    derived: bool
    global_: bool
    roles: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TypedActionExample:
    record_id: str
    corpus: Corpus
    source_ids: tuple[int, ...]
    source_bytes: tuple[int, ...]
    byte_token_indices: tuple[tuple[int, ...], ...]
    candidates: tuple[TypedCandidate, ...]
    action_ids: tuple[int, ...]
    candidate_labels: tuple[int, ...]
    argument_count_labels: tuple[int, ...]
    word_role_labels: tuple[int, ...]


def byte_roles(grammar: ActionGrammar) -> tuple[str, ...]:
    return tuple(sorted(state.name for state in grammar.states.values() if state.byte_payload is not None))


def role_word_lexicon(sequences: Sequence[Sequence[int]], grammar: ActionGrammar) -> dict[bytes, tuple[str, ...]]:
    observed: dict[bytes, set[str]] = defaultdict(set)
    for actions in sequences:
        cursor = grammar.cursor()
        position = 0
        for word in action_words(actions, grammar):
            while grammar.states[cursor.state].byte_payload is None:
                cursor.advance(actions[position])
                position += 1
            role = grammar.states[cursor.state].name
            observed[word].add(role)
            payload = grammar.states[cursor.state].byte_payload
            assert payload is not None
            while payload.byte_offset <= actions[position] < payload.byte_offset + payload.byte_count:
                cursor.advance(actions[position])
                position += 1
            cursor.advance(actions[position])
            position += 1
    return {word: tuple(sorted(roles)) for word, roles in observed.items()}


def typed_candidates(
    source: str,
    grammar: ActionGrammar,
    global_roles: Mapping[bytes, Sequence[str]],
    *,
    maximum_bytes: int = 128,
    normalize_number_words: bool = False,
) -> tuple[TypedCandidate, ...]:
    roles = byte_roles(grammar)
    role_ids = {role: index for index, role in enumerate(roles)}
    source_bytes = source.encode('utf-8')
    local_spans: dict[bytes, LexicalCandidate] = {}
    for span in lexical_candidates(source, maximum_bytes=maximum_bytes):
        value = source_bytes[span.start : span.end]
        local_spans.setdefault(value, span)
    local_text = {value.decode('utf-8') for value in local_spans}
    derived_text = _derived_values(source, local_text, tuple(value.decode('utf-8') for value in global_roles))
    if normalize_number_words:
        lowered = source.lower()
        derived_text.update(
            str(number) for word, number in _NUMBER_WORDS.items() if re.search(rf'(?<![a-z]){re.escape(word)}(?![a-z])', lowered)
        )
    derived = {value.encode('utf-8') for value in derived_text if 0 < len(value.encode('utf-8')) <= maximum_bytes}
    values = set(local_spans) | derived | set(global_roles)
    all_role_ids = tuple(range(len(roles)))
    result = []
    for value in sorted(values):
        local = value in local_spans
        is_derived = value in derived
        is_global = value in global_roles
        allowed = all_role_ids if local or is_derived else tuple(role_ids[role] for role in global_roles[value])
        result.append(TypedCandidate(value, local_spans.get(value), local, is_derived, is_global, tuple(sorted(allowed))))
    return tuple(result)


def _derived_values(source: str, local_values: set[str], global_values: Sequence[str]) -> set[str]:
    values = set(local_values)
    for value in tuple(values):
        values.update(part for part in re.split(r'(?<!^)[-–]', value) if part)
    numbers = {value for value in values if _NUMBER.fullmatch(value)}
    for known in global_values:
        if _NUMBER.search(known):
            values.update(_NUMBER.sub(number, known) for number in numbers)
    paths = {value for value in values if value.endswith('.py')}
    identifiers = {value for value in values if '_' in value and _IDENTIFIER.fullmatch(value)}
    values.update(path + '::' + name for path in paths for name in identifiers)
    values.update('@' + value for value in tuple(values) if re.search(r'\.[A-Za-z0-9]+$', value))
    long_options = {value for value in values if _LONG_OPTION.fullmatch(value)}
    values.update(option + '=' + number for option in long_options for number in numbers)
    short_flags = [match.group()[1:] for match in _SINGLE_FLAG.finditer(source)]
    for start in range(len(short_flags)):
        for end in range(start + 2, min(len(short_flags), start + 5) + 1):
            values.add('-' + ''.join(short_flags[start:end]))
    return values - local_values


def command_argument_counts(document: Mapping[str, object]) -> tuple[int, ...]:
    counts = []
    statements = document.get('s')
    if not isinstance(statements, list):
        raise ValueError('semantic document statements are missing')
    for statement in statements:
        if not isinstance(statement, dict):
            raise ValueError('semantic statement is not an object')
        commands = statement.get('c', [])
        if not isinstance(commands, list):
            raise ValueError('semantic commands are not a list')
        for command in commands:
            if not isinstance(command, dict) or not isinstance(command.get('a', []), list):
                raise ValueError('semantic command arguments are malformed')
            count = len(command.get('a', []))
            if count > MAXIMUM_ARGUMENTS:
                raise ValueError(f'argument count {count} exceeds {MAXIMUM_ARGUMENTS}')
            counts.append(count)
    return tuple(counts)


def align_typed_actions(
    actions: Sequence[int],
    grammar: ActionGrammar,
    candidates: Sequence[TypedCandidate],
    argument_counts: Sequence[int],
    *,
    post_command_counts: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    candidate_labels = [CANDIDATE_IGNORE_INDEX] * len(actions)
    count_labels = [COUNT_IGNORE_INDEX] * len(actions)
    word_role_labels = [CANDIDATE_IGNORE_INDEX] * len(actions)
    candidate_indices = {candidate.value: index for index, candidate in enumerate(candidates)}
    roles = byte_roles(grammar)
    role_ids = {role: index for index, role in enumerate(roles)}
    cursor = grammar.cursor()
    position = 0
    command_index = 0
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
            cursor.advance(actions[position])
            position += 1
        if not raw or position == len(actions):
            raise ValueError(f'incomplete action word in {state.name}')
        value = bytes(raw)
        word_role_labels[begin] = role_ids[state.name]
        candidate_index = candidate_indices.get(value)
        if candidate_index is not None and role_ids[state.name] in candidates[candidate_index].roles:
            candidate_labels[begin] = candidate_index
        if state.name == 'command_name_bytes':
            if command_index >= len(argument_counts):
                raise ValueError('more command action words than semantic commands')
            count_position = position + 1 if post_command_counts else begin
            if count_position >= len(actions):
                raise ValueError('command count label has no post-command action')
            count_labels[count_position] = argument_counts[command_index]
            command_index += 1
        cursor.advance(actions[position])
        position += 1
    if command_index != len(argument_counts):
        raise ValueError('semantic command count does not match action words')
    if not cursor.complete:
        raise ValueError('typed action alignment did not complete grammar')
    return tuple(candidate_labels), tuple(count_labels), tuple(word_role_labels)


def typed_action_examples(
    records: Sequence[SFTRecord],
    actions: Sequence[Sequence[int]],
    tokenizer: AlignedCodeT5Tokenizer,
    grammar: ActionGrammar,
    global_roles: Mapping[bytes, Sequence[str]],
    *,
    source_length: int,
    source_byte_length: int,
    target_length: int,
    prompt_contract: PromptContract,
    normalize_number_words: bool = False,
    post_command_counts: bool = False,
) -> list[TypedActionExample]:
    if len(records) != len(actions):
        raise ValueError('records and actions are not aligned')
    examples = []
    for record, target in zip(records, actions, strict=True):
        source = format_user_message(record, prompt_contract=prompt_contract)
        source_ids, source_bytes, byte_token_indices = tokenizer.encode_with_byte_alignment(source)
        if len(source_ids) > source_length or len(source_bytes) > source_byte_length or len(target) > target_length:
            raise SequenceTooLongError(f'{record.record_id}: typed example exceeds a fixed length')
        candidates = typed_candidates(
            source,
            grammar,
            global_roles,
            normalize_number_words=normalize_number_words,
        )
        candidate_labels, count_labels, word_role_labels = align_typed_actions(
            target,
            grammar,
            candidates,
            command_argument_counts(json.loads(record.response)),
            post_command_counts=post_command_counts,
        )
        examples.append(
            TypedActionExample(
                record.record_id,
                record.corpus,
                source_ids,
                source_bytes,
                byte_token_indices,
                candidates,
                tuple(target),
                candidate_labels,
                count_labels,
                word_role_labels,
            )
        )
    return examples


def collate_typed_actions(
    examples: Sequence[TypedActionExample],
    *,
    tokenizer: AlignedCodeT5Tokenizer,
    source_length: int,
    source_byte_length: int,
    candidate_count: int,
    candidate_byte_length: int,
    role_count: int,
    target_length: int,
    source_pad_token_id: int,
) -> TypedActionBatch:
    if not examples:
        raise ValueError('cannot collate an empty typed batch')
    if any(example.corpus is not examples[0].corpus for example in examples):
        raise DatasetFormatError('cannot mix corpus privacy classes')
    encoded_candidates = {
        candidate.value: tuple(tokenizer.encode(candidate.value.decode('utf-8'), add_special_tokens=False))
        for example in examples
        for candidate in example.candidates
    }
    candidate_token_length = max(len(token_ids) for token_ids in encoded_candidates.values())
    rows: dict[str, list] = defaultdict(list)
    alignments = []
    for example in examples:
        if (
            len(example.source_ids) > source_length
            or len(example.source_bytes) > source_byte_length
            or len(example.candidates) > candidate_count
            or len(example.action_ids) > target_length
        ):
            raise SequenceTooLongError(f'{example.record_id}: typed batch shape would truncate data')
        source_padding = source_length - len(example.source_ids)
        byte_padding = source_byte_length - len(example.source_bytes)
        candidate_padding = candidate_count - len(example.candidates)
        shifted, expected = example.action_ids[:-1], example.action_ids[1:]
        target_padding = target_length - len(expected)
        rows['source'].append(example.source_ids + (source_pad_token_id,) * source_padding)
        rows['source_mask'].append((1,) * len(example.source_ids) + (0,) * source_padding)
        rows['source_bytes'].append(example.source_bytes + (0,) * byte_padding)
        rows['source_byte_mask'].append((1,) * len(example.source_bytes) + (0,) * byte_padding)
        alignment = torch.zeros(source_byte_length, source_length)
        for byte_index, token_indices in enumerate(example.byte_token_indices):
            alignment[byte_index, list(token_indices)] = 1 / len(token_indices)
        alignments.append(alignment)
        candidate_bytes, candidate_byte_masks, histograms, bigrams, token_ids, token_masks, starts, ends, features, role_masks = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for item in example.candidates:
            padding = candidate_byte_length - len(item.value)
            candidate_bytes.append(tuple(item.value) + (0,) * padding)
            candidate_byte_masks.append((1,) * len(item.value) + (0,) * padding)
            histogram = [0.0] * 256
            for byte in item.value:
                histogram[byte] += 1 / len(item.value)
            histograms.append(tuple(histogram))
            bigram = [0.0] * BIGRAM_BUCKETS
            pair_count = max(1, len(item.value) - 1)
            for first, second in zip(item.value, item.value[1:], strict=False):
                bigram[(first * 257 + second) % BIGRAM_BUCKETS] += 1 / pair_count
            bigrams.append(tuple(bigram))
            encoded = encoded_candidates[item.value]
            token_padding = candidate_token_length - len(encoded)
            token_ids.append(encoded + (tokenizer.pad_token_id,) * token_padding)
            token_masks.append((1,) * len(encoded) + (0,) * token_padding)
            starts.append(item.local_span.start if item.local_span else 0)
            ends.append(item.local_span.end if item.local_span else 1)
            features.append((item.local, item.derived, item.global_))
            role_masks.append(tuple(role in item.roles for role in range(role_count)))
        candidate_bytes.extend([(0,) * candidate_byte_length] * candidate_padding)
        candidate_byte_masks.extend([(0,) * candidate_byte_length] * candidate_padding)
        histograms.extend([(0.0,) * 256] * candidate_padding)
        bigrams.extend([(0.0,) * BIGRAM_BUCKETS] * candidate_padding)
        token_ids.extend([(tokenizer.pad_token_id,) * candidate_token_length] * candidate_padding)
        token_masks.extend([(0,) * candidate_token_length] * candidate_padding)
        starts.extend([0] * candidate_padding)
        ends.extend([1] * candidate_padding)
        features.extend([(False, False, False)] * candidate_padding)
        role_masks.extend([(False,) * role_count] * candidate_padding)
        rows['candidate_bytes'].append(candidate_bytes)
        rows['candidate_byte_mask'].append(candidate_byte_masks)
        rows['candidate_byte_histogram'].append(histograms)
        rows['candidate_bigram_histogram'].append(bigrams)
        rows['candidate_token_ids'].append(token_ids)
        rows['candidate_token_mask'].append(token_masks)
        rows['candidate_starts'].append(starts)
        rows['candidate_ends'].append(ends)
        rows['candidate_features'].append(features)
        rows['candidate_role_mask'].append(role_masks)
        rows['candidate_mask'].append((1,) * len(example.candidates) + (0,) * candidate_padding)
        rows['decoder'].append(shifted + (0,) * target_padding)
        rows['decoder_mask'].append((1,) * len(shifted) + (0,) * target_padding)
        rows['labels'].append(expected + (LABEL_IGNORE_INDEX,) * target_padding)
        rows['candidate_labels'].append(example.candidate_labels[1:] + (CANDIDATE_IGNORE_INDEX,) * target_padding)
        rows['count_labels'].append(example.argument_count_labels[1:] + (COUNT_IGNORE_INDEX,) * target_padding)
        rows['word_role_labels'].append(example.word_role_labels[1:] + (CANDIDATE_IGNORE_INDEX,) * target_padding)
    return TypedActionBatch(
        torch.tensor(rows['source'], dtype=torch.long),
        torch.tensor(rows['source_mask'], dtype=torch.bool),
        torch.tensor(rows['source_bytes'], dtype=torch.long),
        torch.tensor(rows['source_byte_mask'], dtype=torch.bool),
        torch.stack(alignments),
        torch.tensor(rows['candidate_bytes'], dtype=torch.long),
        torch.tensor(rows['candidate_byte_mask'], dtype=torch.bool),
        torch.tensor(rows['candidate_byte_histogram'], dtype=torch.float),
        torch.tensor(rows['candidate_bigram_histogram'], dtype=torch.float),
        torch.tensor(rows['candidate_token_ids'], dtype=torch.long),
        torch.tensor(rows['candidate_token_mask'], dtype=torch.bool),
        torch.tensor(rows['candidate_starts'], dtype=torch.long),
        torch.tensor(rows['candidate_ends'], dtype=torch.long),
        torch.tensor(rows['candidate_features'], dtype=torch.float),
        torch.tensor(rows['candidate_role_mask'], dtype=torch.bool).transpose(1, 2),
        torch.tensor(rows['candidate_mask'], dtype=torch.bool),
        torch.tensor(rows['decoder'], dtype=torch.long),
        torch.tensor(rows['decoder_mask'], dtype=torch.bool),
        torch.tensor(rows['labels'], dtype=torch.long),
        torch.tensor(rows['candidate_labels'], dtype=torch.long),
        torch.tensor(rows['count_labels'], dtype=torch.long),
        torch.tensor(rows['word_role_labels'], dtype=torch.long),
    )
