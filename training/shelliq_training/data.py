"""Auditable supervised-fine-tuning records and Qwen batch construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from shelliq_training.prompt import LEGACY_SEMANTIC_CONTEXT_PREFIX, PromptContract
from shelliq_training.prompt import format_user_message as format_prompt_user_message

if TYPE_CHECKING:
    from shelliq_training.training import CausalLMBatch

IGNORE_INDEX = -100

SCHEMA_VERSION = 1


class Corpus(StrEnum):
    """Datasets that must remain structurally separate."""

    DISTRIBUTABLE = 'distributable'
    PERSONAL = 'personal'


class Platform(StrEnum):
    LINUX = 'linux'
    DARWIN = 'darwin'


class Split(StrEnum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


class DatasetFormatError(ValueError):
    """A record does not satisfy the auditable JSONL contract."""


class SequenceTooLongError(ValueError):
    """A record cannot fit without silently discarding training text."""


class ChatTokenizer(Protocol):
    pad_token_id: int | None

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Mapping[str, object] | Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class SFTRecord:
    """One training pair with enough metadata to audit or exclude it."""

    record_id: str
    corpus: Corpus
    source: str
    license: str
    provenance: str
    command: str
    platform: Platform
    instruction: str
    response: str
    context: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> SFTRecord:
        expected = {
            'schema_version',
            'record_id',
            'corpus',
            'source',
            'license',
            'provenance',
            'command',
            'platform',
            'instruction',
            'response',
            'context',
        }
        missing = expected - raw.keys()
        unknown = raw.keys() - expected
        if missing:
            raise DatasetFormatError(f'missing fields: {", ".join(sorted(missing))}')
        if unknown:
            raise DatasetFormatError(f'unknown fields: {", ".join(sorted(unknown))}')
        if raw['schema_version'] != SCHEMA_VERSION:
            raise DatasetFormatError(f'unsupported schema_version: {raw["schema_version"]!r}')

        try:
            corpus = Corpus(_text(raw, 'corpus'))
            platform = Platform(_text(raw, 'platform'))
        except ValueError as error:
            raise DatasetFormatError(str(error)) from error

        return cls(
            record_id=_text(raw, 'record_id'),
            corpus=corpus,
            source=_text(raw, 'source'),
            license=_text(raw, 'license'),
            provenance=_text(raw, 'provenance'),
            command=_text(raw, 'command'),
            platform=platform,
            instruction=_text(raw, 'instruction'),
            response=_text(raw, 'response'),
            context=_text(raw, 'context'),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            'schema_version': SCHEMA_VERSION,
            'record_id': self.record_id,
            'corpus': self.corpus.value,
            'source': self.source,
            'license': self.license,
            'provenance': self.provenance,
            'command': self.command,
            'platform': self.platform.value,
            'instruction': self.instruction,
            'response': self.response,
            'context': self.context,
        }


@dataclass(frozen=True, slots=True)
class TokenizedExample:
    record_id: str
    corpus: Corpus
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


def _text(raw: Mapping[str, object], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value.strip():
        raise DatasetFormatError(f'{field} must be a non-empty string')
    return value.strip()


def load_jsonl(path: str | Path, *, corpus: Corpus | None = None) -> list[SFTRecord]:
    """Load strict JSONL, rejecting duplicate IDs and cross-pipeline rows."""
    records: list[SFTRecord] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise DatasetFormatError('record must be a JSON object')
                record = SFTRecord.from_dict(raw)
            except (json.JSONDecodeError, DatasetFormatError) as error:
                raise DatasetFormatError(f'{path}:{line_number}: {error}') from error
            if record.record_id in seen_ids:
                raise DatasetFormatError(f'{path}:{line_number}: duplicate record_id {record.record_id!r}')
            if corpus is not None and record.corpus is not corpus:
                raise DatasetFormatError(f'{path}:{line_number}: {record.corpus.value} record in {corpus.value} pipeline')
            seen_ids.add(record.record_id)
            records.append(record)
    return records


def load_semantic_jsonl(path: str | Path) -> list[SFTRecord]:
    """Load validated semantic-conversion rows as canonical JSON targets."""
    expected = {
        'conversion_schema_version',
        'record_id',
        'corpus',
        'source',
        'license',
        'provenance',
        'command',
        'platform',
        'instruction',
        'shell_response',
        'context',
        'semantic_target',
    }
    records = []
    seen_ids = set()
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetFormatError(f'{path}:{line_number}: {error}') from error
            if not isinstance(raw, dict) or raw.keys() != expected:
                raise DatasetFormatError(f'{path}:{line_number}: invalid semantic row fields')
            target = raw['semantic_target']
            if (
                raw['conversion_schema_version'] != 1
                or raw['corpus'] != Corpus.DISTRIBUTABLE.value
                or not isinstance(target, dict)
                or target.get('v') != 2
                or target.get('d') != 'zsh'
            ):
                raise DatasetFormatError(f'{path}:{line_number}: invalid semantic schema')
            record_id = raw['record_id']
            if record_id in seen_ids:
                raise DatasetFormatError(f'{path}:{line_number}: duplicate record_id {record_id!r}')
            semantic_response = json.dumps(target, ensure_ascii=False, separators=(',', ':'))
            try:
                record = SFTRecord.from_dict(
                    {
                        'schema_version': SCHEMA_VERSION,
                        'record_id': record_id,
                        'corpus': raw['corpus'],
                        'source': raw['source'],
                        'license': raw['license'],
                        'provenance': raw['provenance'],
                        'command': raw['command'],
                        'platform': raw['platform'],
                        'instruction': raw['instruction'],
                        'response': semantic_response,
                        'context': f'{LEGACY_SEMANTIC_CONTEXT_PREFIX}{raw["context"]}',
                    }
                )
            except (TypeError, ValueError) as error:
                raise DatasetFormatError(f'{path}:{line_number}: {error}') from error
            records.append(record)
            seen_ids.add(record_id)
    return records


def write_jsonl(path: str | Path, records: Iterable[SFTRecord], *, corpus: Corpus) -> None:
    """Write one pipeline, refusing to create a mixed-derived artifact."""
    materialized = list(records)
    _require_corpus(materialized, corpus)
    with Path(path).open('w', encoding='utf-8') as stream:
        for record in materialized:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            stream.write('\n')


def assign_split(
    record: SFTRecord,
    *,
    seed: int,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    heldout_sources: frozenset[str] = frozenset(),
    heldout_platforms: frozenset[Platform] = frozenset(),
) -> Split:
    """Hash by command, so paraphrases and sources cannot leak across splits."""
    _validate_fractions(validation_fraction, test_fraction)
    if record.source in heldout_sources or record.platform in heldout_platforms:
        return Split.TEST
    digest = hashlib.sha256(f'{seed}\0{record.command}'.encode()).digest()
    unit_interval = int.from_bytes(digest[:8], 'big') / 2**64
    if unit_interval < test_fraction:
        return Split.TEST
    if unit_interval < test_fraction + validation_fraction:
        return Split.VALIDATION
    return Split.TRAIN


def split_records(
    records: Iterable[SFTRecord],
    *,
    corpus: Corpus,
    seed: int,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    heldout_sources: frozenset[str] = frozenset(),
    heldout_platforms: frozenset[Platform] = frozenset(),
) -> dict[Split, list[SFTRecord]]:
    materialized = list(records)
    _require_corpus(materialized, corpus)
    forced_test_commands = {
        record.command for record in materialized if record.source in heldout_sources or record.platform in heldout_platforms
    }
    result = {split: [] for split in Split}
    for record in materialized:
        if record.command in forced_test_commands:
            split = Split.TEST
        else:
            split = assign_split(
                record,
                seed=seed,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                heldout_sources=heldout_sources,
                heldout_platforms=heldout_platforms,
            )
        result[split].append(record)
    return result


def format_user_message(record: SFTRecord, *, prompt_contract: PromptContract) -> str:
    """Match the selected versioned retrieval-augmented inference shape."""
    return format_prompt_user_message(
        platform=record.platform.value,
        context=record.context,
        instruction=record.instruction,
        contract=prompt_contract,
    )


def tokenize_record(
    record: SFTRecord,
    tokenizer: ChatTokenizer,
    *,
    max_length: int,
    prompt_contract: PromptContract,
) -> TokenizedExample:
    """Apply Qwen's chat template and mask every token before the answer."""
    if max_length < 2:
        raise ValueError('max_length must be at least 2')
    user_message = {'role': 'user', 'content': format_user_message(record, prompt_contract=prompt_contract)}
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            [user_message],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            [user_message, {'role': 'assistant', 'content': record.response}],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise DatasetFormatError('chat template does not produce a stable assistant-prefix boundary')
    if len(full_ids) <= len(prompt_ids):
        raise DatasetFormatError('chat template produced no assistant tokens')
    if len(full_ids) > max_length:
        raise SequenceTooLongError(
            f'{record.record_id}: {len(full_ids)} tokens exceeds max_length={max_length}; drop or shorten the record explicitly'
        )
    labels = (IGNORE_INDEX,) * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return TokenizedExample(
        record_id=record.record_id,
        corpus=record.corpus,
        input_ids=full_ids,
        labels=labels,
    )


def collate_sft(
    examples: Sequence[TokenizedExample],
    *,
    sequence_length: int,
    pad_token_id: int,
) -> CausalLMBatch:
    """Right-pad to a fixed shape, avoiding a JAX recompilation per batch."""
    import jax.numpy as jnp

    from shelliq_training.training import CausalLMBatch

    if not examples:
        raise ValueError('cannot collate an empty batch')
    if sequence_length < 2:
        raise ValueError('sequence_length must be at least 2')
    if pad_token_id < 0:
        raise ValueError('pad_token_id must be non-negative')
    corpus = examples[0].corpus
    if any(example.corpus is not corpus for example in examples):
        raise DatasetFormatError('cannot collate distributable and personal examples together')

    input_rows: list[tuple[int, ...]] = []
    label_rows: list[tuple[int, ...]] = []
    attention_rows: list[tuple[int, ...]] = []
    for example in examples:
        if len(example.input_ids) != len(example.labels):
            raise DatasetFormatError(f'{example.record_id}: input_ids and labels differ in length')
        if len(example.input_ids) > sequence_length:
            raise SequenceTooLongError(
                f'{example.record_id}: {len(example.input_ids)} tokens exceeds sequence_length={sequence_length}'
            )
        padding = sequence_length - len(example.input_ids)
        input_rows.append(example.input_ids + (pad_token_id,) * padding)
        label_rows.append(example.labels + (IGNORE_INDEX,) * padding)
        attention_rows.append((1,) * len(example.input_ids) + (0,) * padding)

    return cast(
        CausalLMBatch,
        {
            'input_ids': jnp.asarray(input_rows, dtype=jnp.int32),
            'attention_mask': jnp.asarray(attention_rows, dtype=jnp.int32),
            'labels': jnp.asarray(label_rows, dtype=jnp.int32),
        },
    )


def tokenizer_pad_id(tokenizer: ChatTokenizer) -> int:
    if tokenizer.pad_token_id is None:
        raise ValueError('tokenizer has no pad_token_id')
    return tokenizer.pad_token_id


def _token_ids(values: object) -> tuple[int, ...]:
    if isinstance(values, Mapping):
        values = values.get('input_ids')
    if isinstance(values, (str, bytes)):
        raise DatasetFormatError('chat template returned text instead of token IDs')
    try:
        return tuple(int(value) for value in cast(Iterable[object], values))
    except (TypeError, ValueError) as error:
        raise DatasetFormatError('chat template returned invalid token IDs') from error


def _require_corpus(records: Sequence[SFTRecord], corpus: Corpus) -> None:
    wrong = [record.record_id for record in records if record.corpus is not corpus]
    if wrong:
        raise DatasetFormatError(f'{len(wrong)} record(s) do not belong to the {corpus.value} pipeline: {", ".join(wrong[:3])}')


def _validate_fractions(validation_fraction: float, test_fraction: float) -> None:
    if not 0 <= validation_fraction < 1:
        raise ValueError('validation_fraction must be in [0, 1)')
    if not 0 <= test_fraction < 1:
        raise ValueError('test_fraction must be in [0, 1)')
    if validation_fraction + test_fraction >= 1:
        raise ValueError('validation_fraction + test_fraction must be less than 1')
