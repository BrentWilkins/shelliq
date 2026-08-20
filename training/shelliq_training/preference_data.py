"""Strict reviewed chosen/rejected records for future preference training."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from shelliq_training.data import Corpus, DatasetFormatError, Platform

PREFERENCE_SCHEMA_VERSION = 1
_REVIEW_DATE = re.compile(r'\d{4}-\d{2}-\d{2}')


class FailureMode(StrEnum):
    WRONG_COMMAND = 'wrong-command'
    WRONG_SUBCOMMAND = 'wrong-subcommand'
    MISSING_FLAG = 'missing-flag'
    EXTRA_FLAG = 'extra-flag'
    WRONG_FLAG = 'wrong-flag'
    WRONG_OPTION_VALUE = 'wrong-option-value'
    WRONG_OPTION_BINDING = 'wrong-option-binding'
    CHANGED_OPERAND = 'changed-operand'
    MISSING_OPERAND = 'missing-operand'
    INVENTED_OPERAND = 'invented-operand'
    WRONG_ORDERING = 'wrong-ordering'
    WRONG_SEPARATOR = 'wrong-separator'
    WRONG_REDIRECTION = 'wrong-redirection'
    WRONG_PIPELINE = 'wrong-pipeline'
    FRAMING_MISMATCH = 'framing-mismatch'
    UNSUPPORTED_PLATFORM_SPELLING = 'unsupported-platform-spelling'
    UNGROUNDED_LITERAL = 'ungrounded-literal'


def _text(raw: Mapping[str, object], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value.strip():
        raise DatasetFormatError(f'{field} must be a non-empty string')
    return value.strip()


def _semantic_document(raw: Mapping[str, object], field: str) -> dict[str, object]:
    value = raw[field]
    if not isinstance(value, dict):
        raise DatasetFormatError(f'{field} must be a SemanticDocumentV2 object')
    if set(value) != {'v', 'd', 's'}:
        raise DatasetFormatError(f'{field} must contain exactly v, d, and s')
    if value['v'] != 2 or value['d'] != 'zsh':
        raise DatasetFormatError(f'{field} must use SemanticDocumentV2 zsh envelope')
    if not isinstance(value['s'], list) or not value['s']:
        raise DatasetFormatError(f'{field}.s must be a non-empty statement list')
    return value


@dataclass(frozen=True, slots=True)
class PreferenceRecord:
    pair_id: str
    corpus: Corpus
    source: str
    license: str
    provenance: str
    platform: Platform
    instruction: str
    context: str
    chosen: dict[str, object]
    rejected: dict[str, object]
    failure_modes: tuple[FailureMode, ...]
    verifier_evidence: tuple[str, ...]
    reviewer: str
    reviewed_at: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> PreferenceRecord:
        expected = {
            'schema_version',
            'pair_id',
            'corpus',
            'source',
            'license',
            'provenance',
            'platform',
            'instruction',
            'context',
            'chosen',
            'rejected',
            'failure_modes',
            'verifier_evidence',
            'reviewer',
            'reviewed_at',
        }
        missing = expected - set(raw)
        unknown = set(raw) - expected
        if missing:
            raise DatasetFormatError(f'missing fields: {", ".join(sorted(missing))}')
        if unknown:
            raise DatasetFormatError(f'unknown fields: {", ".join(sorted(unknown))}')
        if raw['schema_version'] != PREFERENCE_SCHEMA_VERSION:
            raise DatasetFormatError(f'unsupported preference schema_version: {raw["schema_version"]!r}')
        try:
            corpus = Corpus(_text(raw, 'corpus'))
            platform = Platform(_text(raw, 'platform'))
        except ValueError as error:
            raise DatasetFormatError(str(error)) from error
        chosen = _semantic_document(raw, 'chosen')
        rejected = _semantic_document(raw, 'rejected')
        if chosen == rejected:
            raise DatasetFormatError('chosen and rejected documents must differ')
        raw_modes = raw['failure_modes']
        if not isinstance(raw_modes, list) or not raw_modes:
            raise DatasetFormatError('failure_modes must be a non-empty list')
        try:
            failure_modes = tuple(FailureMode(mode) for mode in raw_modes)
        except (TypeError, ValueError) as error:
            raise DatasetFormatError(f'invalid failure mode: {error}') from error
        if len(failure_modes) != len(set(failure_modes)):
            raise DatasetFormatError('failure_modes must be unique')
        raw_evidence = raw['verifier_evidence']
        if (
            not isinstance(raw_evidence, list)
            or not raw_evidence
            or not all(isinstance(item, str) and item.strip() for item in raw_evidence)
        ):
            raise DatasetFormatError('verifier_evidence must be non-empty strings')
        verifier_evidence = tuple(item.strip() for item in raw_evidence)
        if len(verifier_evidence) != len(set(verifier_evidence)):
            raise DatasetFormatError('verifier_evidence must be unique')
        reviewed_at = _text(raw, 'reviewed_at')
        if _REVIEW_DATE.fullmatch(reviewed_at) is None:
            raise DatasetFormatError('reviewed_at must be YYYY-MM-DD')
        return cls(
            pair_id=_text(raw, 'pair_id'),
            corpus=corpus,
            source=_text(raw, 'source'),
            license=_text(raw, 'license'),
            provenance=_text(raw, 'provenance'),
            platform=platform,
            instruction=_text(raw, 'instruction'),
            context=_text(raw, 'context'),
            chosen=chosen,
            rejected=rejected,
            failure_modes=failure_modes,
            verifier_evidence=verifier_evidence,
            reviewer=_text(raw, 'reviewer'),
            reviewed_at=reviewed_at,
        )


def load_preference_jsonl(path: str | Path) -> list[PreferenceRecord]:
    records = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise DatasetFormatError('record must be a JSON object')
                record = PreferenceRecord.from_dict(raw)
            except (json.JSONDecodeError, DatasetFormatError) as error:
                raise DatasetFormatError(f'{path}:{line_number}: {error}') from error
            if record.pair_id in seen_ids:
                raise DatasetFormatError(f'{path}:{line_number}: duplicate pair_id {record.pair_id!r}')
            seen_ids.add(record.pair_id)
            records.append(record)
    return records
