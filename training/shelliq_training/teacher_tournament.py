"""Provider-neutral records and scoring for teacher-model tournaments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from shelliq_training.data import Corpus, DatasetFormatError, Platform, SFTRecord
from shelliq_training.evaluation import ModelPrediction
from shelliq_training.semantic_evaluation import SemanticEvaluationMetrics, evaluate_semantic_predictions

CHALLENGE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1


class ChallengeCategory(StrEnum):
    PRECISE = 'precise'
    MULTI_CONSTRAINT = 'multi-constraint'
    PIPELINE = 'pipeline'
    COMPOSITIONAL = 'compositional'


class RiskClass(StrEnum):
    READ_ONLY = 'read-only'
    FIXTURE_MUTATION = 'fixture-mutation'
    STATIC_ONLY = 'static-only'


def _text(raw: dict[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetFormatError(f'{field} must be a non-empty string')
    return value.strip()


def _text_list(raw: dict[str, object], field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    value = raw.get(field)
    if not isinstance(value, list) or (not value and not allow_empty):
        raise DatasetFormatError(f'{field} must be a{" possibly empty" if allow_empty else " non-empty"} list')
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise DatasetFormatError(f'{field} must contain non-empty strings')
    normalized = tuple(item.strip() for item in value)
    if len(normalized) != len(set(normalized)):
        raise DatasetFormatError(f'{field} must not contain duplicates')
    return normalized


@dataclass(frozen=True, slots=True)
class TeacherChallenge:
    challenge_id: str
    category: ChallengeCategory
    platform: Platform
    command: str
    instruction: str
    context: str
    expected: dict[str, object]
    grounding_paths: tuple[str, ...]
    constraints: tuple[str, ...]
    risk: RiskClass
    functional_eligible: bool

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> TeacherChallenge:
        expected_fields = {
            'schema_version',
            'challenge_id',
            'category',
            'platform',
            'command',
            'instruction',
            'context',
            'expected',
            'grounding_paths',
            'constraints',
            'risk',
            'functional_eligible',
        }
        if set(raw) != expected_fields:
            missing = expected_fields - set(raw)
            unknown = set(raw) - expected_fields
            detail = []
            if missing:
                detail.append(f'missing fields: {", ".join(sorted(missing))}')
            if unknown:
                detail.append(f'unknown fields: {", ".join(sorted(unknown))}')
            raise DatasetFormatError('; '.join(detail))
        if raw['schema_version'] != CHALLENGE_SCHEMA_VERSION:
            raise DatasetFormatError(f'unsupported challenge schema: {raw["schema_version"]!r}')
        expected = raw['expected']
        if not isinstance(expected, dict) or expected.get('v') != 2 or expected.get('d') != 'zsh':
            raise DatasetFormatError('expected must be SemanticDocumentV2')
        if not isinstance(raw['functional_eligible'], bool):
            raise DatasetFormatError('functional_eligible must be boolean')
        try:
            category = ChallengeCategory(_text(raw, 'category'))
            platform = Platform(_text(raw, 'platform'))
            risk = RiskClass(_text(raw, 'risk'))
        except ValueError as error:
            raise DatasetFormatError(str(error)) from error
        grounding_paths = _text_list(raw, 'grounding_paths', allow_empty=True)
        if not all(path.startswith('/') for path in grounding_paths):
            raise DatasetFormatError('grounding_paths must be JSON pointers')
        functional_eligible = raw['functional_eligible']
        assert isinstance(functional_eligible, bool)
        if functional_eligible and risk is RiskClass.STATIC_ONLY:
            raise DatasetFormatError('static-only challenge cannot be functional_eligible')
        return cls(
            challenge_id=_text(raw, 'challenge_id'),
            category=category,
            platform=platform,
            command=_text(raw, 'command'),
            instruction=_text(raw, 'instruction'),
            context=_text(raw, 'context'),
            expected=expected,
            grounding_paths=grounding_paths,
            constraints=_text_list(raw, 'constraints'),
            risk=risk,
            functional_eligible=functional_eligible,
        )

    def as_sft_record(self) -> SFTRecord:
        return SFTRecord(
            record_id=self.challenge_id,
            corpus=Corpus.DISTRIBUTABLE,
            source='shelliq-teacher-selection',
            license='MIT OR Apache-2.0',
            provenance=f'teacher-selection-v1:{self.challenge_id}',
            command=self.command,
            platform=self.platform,
            instruction=self.instruction,
            response=json.dumps(self.expected, ensure_ascii=False, separators=(',', ':')),
            context=self.context,
        )


@dataclass(frozen=True, slots=True)
class CandidateResult:
    challenge_id: str
    provider: str
    model: str
    prompt_hash: str
    sample: int
    temperature: float
    seed: int | None
    generated: str | None
    latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    error: str | None

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> CandidateResult:
        expected = {'schema_version', *cls.__dataclass_fields__}
        if set(raw) != expected:
            raise DatasetFormatError('candidate result has missing or unknown fields')
        if raw['schema_version'] != RESULT_SCHEMA_VERSION:
            raise DatasetFormatError('unsupported candidate result schema')
        for field in ('challenge_id', 'provider', 'model', 'prompt_hash'):
            _text(raw, field)
        if not isinstance(raw['sample'], int) or raw['sample'] < 0:
            raise DatasetFormatError('sample must be a non-negative integer')
        if not isinstance(raw['temperature'], int | float) or raw['temperature'] < 0:
            raise DatasetFormatError('temperature must be non-negative')
        for field in ('seed', 'input_tokens', 'output_tokens'):
            if raw[field] is not None and (not isinstance(raw[field], int) or raw[field] < 0):
                raise DatasetFormatError(f'{field} must be a non-negative integer or null')
        for field in ('latency_ms', 'cost_usd'):
            if raw[field] is not None and (not isinstance(raw[field], int | float) or raw[field] < 0):
                raise DatasetFormatError(f'{field} must be non-negative or null')
        generated = raw['generated']
        error = raw['error']
        if (generated is None) == (error is None):
            raise DatasetFormatError('candidate result must contain exactly one of generated or error')
        if generated is not None and not isinstance(generated, str):
            raise DatasetFormatError('generated must be string or null')
        if error is not None and (not isinstance(error, str) or not error.strip()):
            raise DatasetFormatError('error must be non-empty string or null')
        values = {field: raw[field] for field in cls.__dataclass_fields__}
        values['temperature'] = float(values['temperature'])
        for field in ('latency_ms', 'cost_usd'):
            if values[field] is not None:
                values[field] = float(values[field])
        return cls(**values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {'schema_version': RESULT_SCHEMA_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class TournamentScore:
    model: str
    provider: str
    metrics: SemanticEvaluationMetrics
    completed: int
    errors: int
    total_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


def load_challenges(path: Path) -> list[TeacherChallenge]:
    return _load_jsonl(path, TeacherChallenge.from_dict, 'challenge_id')


def load_results(path: Path) -> list[CandidateResult]:
    return _load_jsonl(path, CandidateResult.from_dict, None)


def _load_jsonl(path: Path, loader: Any, unique_field: str | None) -> list[Any]:
    records = []
    seen: set[str] = set()
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise DatasetFormatError('record must be object')
                record = loader(raw)
            except (json.JSONDecodeError, DatasetFormatError) as error:
                raise DatasetFormatError(f'{path}:{line_number}: {error}') from error
            if unique_field is not None:
                value = getattr(record, unique_field)
                if value in seen:
                    raise DatasetFormatError(f'{path}:{line_number}: duplicate {unique_field} {value!r}')
                seen.add(value)
            records.append(record)
    if not records:
        raise DatasetFormatError(f'{path}: no records')
    return records


def prompt_hash(system_prompt: str | None, user_message: str) -> str:
    document = json.dumps(
        {'system': system_prompt, 'user': user_message}, ensure_ascii=False, separators=(',', ':'), sort_keys=True
    )
    return hashlib.sha256(document.encode()).hexdigest()


def score_results(challenges: list[TeacherChallenge], results: list[CandidateResult]) -> TournamentScore:
    by_id = {challenge.challenge_id: challenge for challenge in challenges}
    successful = [result for result in results if result.generated is not None]
    identities = {(result.provider, result.model, result.sample) for result in results}
    if len(identities) != 1:
        raise ValueError('score one provider/model/sample result set at a time')
    if len({result.challenge_id for result in results}) != len(results):
        raise ValueError('result set contains duplicate challenge IDs')
    unknown = {result.challenge_id for result in results} - set(by_id)
    if unknown:
        raise ValueError(f'unknown challenge IDs: {sorted(unknown)!r}')
    # Keep failed requests in the denominator. Treating them as absent would make
    # an unreliable provider look better than one that completed every challenge.
    scored_challenges = [by_id[result.challenge_id] for result in results]
    records = [challenge.as_sft_record() for challenge in scored_challenges]
    predictions = [ModelPrediction(result.challenge_id, result.generated or '', result.latency_ms or 0.0) for result in results]
    audit = {challenge.challenge_id: challenge.grounding_paths for challenge in scored_challenges}
    metrics = evaluate_semantic_predictions(records, predictions, audit)
    provider, model, _ = next(iter(identities))
    return TournamentScore(
        model=model,
        provider=provider,
        metrics=metrics,
        completed=len(successful),
        errors=len(results) - len(successful),
        total_latency_ms=sum(result.latency_ms or 0 for result in results),
        total_input_tokens=sum(result.input_tokens or 0 for result in results),
        total_output_tokens=sum(result.output_tokens or 0 for result in results),
        total_cost_usd=sum(result.cost_usd or 0 for result in results),
    )
