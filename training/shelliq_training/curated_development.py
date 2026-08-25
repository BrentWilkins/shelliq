"""Validation and leakage audits for the curated-development benchmark."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from shelliq_training.data import SFTRecord
from shelliq_training.semantic_evaluation import GroundingAudit

SCHEMA_VERSION = 1
SUITE_ID = 'curated-development-v1'
SOURCE = 'shelliq-curated-development'
RECORD_PREFIX = 'curated-development:v1:'
MINIMUM_FROZEN_RECORDS = 100
MINIMUM_FROZEN_FAMILIES = 20
REQUIRED_DIMENSIONS = frozenset(
    {
        'flag-selection',
        'option-value',
        'operand-binding',
        'ordering',
        'pipeline-semantics',
    }
)


@dataclass(frozen=True, slots=True)
class ChallengeAnnotation:
    family: str
    dimensions: frozenset[str]
    constraint_count: int


@dataclass(frozen=True, slots=True)
class CuratedDevelopmentManifest:
    status: str
    records: Mapping[str, ChallengeAnnotation]


@dataclass(frozen=True, slots=True)
class CuratedDevelopmentAudit:
    records: int
    families: int
    commands: int
    dimensions: Mapping[str, int]
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            'schema_version': SCHEMA_VERSION,
            'suite_id': SUITE_ID,
            'status': self.status,
            'records': self.records,
            'families': self.families,
            'commands': self.commands,
            'dimensions': dict(sorted(self.dimensions.items())),
        }


def load_manifest(path: Path) -> CuratedDevelopmentManifest:
    """Load the closed annotation set for the curated-development suite."""
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path}: manifest must be an object')
    expected = {'schema_version', 'suite_id', 'status', 'records'}
    if set(value) != expected:
        raise ValueError(f'{path}: expected fields {sorted(expected)}')
    if value['schema_version'] != SCHEMA_VERSION or value['suite_id'] != SUITE_ID:
        raise ValueError(f'{path}: unsupported curated-development manifest')
    status = value['status']
    if status not in {'draft', 'frozen'}:
        raise ValueError(f'{path}: status must be draft or frozen')
    raw_records = value['records']
    if not isinstance(raw_records, dict):
        raise ValueError(f'{path}: records must be an object')

    records: dict[str, ChallengeAnnotation] = {}
    for record_id, raw in raw_records.items():
        if not isinstance(record_id, str) or not isinstance(raw, dict):
            raise ValueError(f'{path}: invalid challenge annotation')
        if set(raw) != {'family', 'dimensions', 'constraint_count'}:
            raise ValueError(f'{path}: {record_id}: invalid annotation fields')
        family = raw['family']
        dimensions = raw['dimensions']
        constraint_count = raw['constraint_count']
        if not isinstance(family, str) or not family.strip():
            raise ValueError(f'{path}: {record_id}: invalid family')
        if (
            not isinstance(dimensions, list)
            or not dimensions
            or not all(isinstance(item, str) and item for item in dimensions)
            or len(dimensions) != len(set(dimensions))
        ):
            raise ValueError(f'{path}: {record_id}: invalid dimensions')
        if not isinstance(constraint_count, int) or constraint_count < 2:
            raise ValueError(f'{path}: {record_id}: requires at least two constraints')
        records[record_id] = ChallengeAnnotation(
            family=family,
            dimensions=frozenset(dimensions),
            constraint_count=constraint_count,
        )
    return CuratedDevelopmentManifest(status=status, records=records)


def _normalized(value: str) -> str:
    return ' '.join(value.casefold().split())


def audit_curated_development(
    records: Sequence[SFTRecord],
    manifest: CuratedDevelopmentManifest,
    grounding_audit: GroundingAudit,
    *,
    training_records: Iterable[SFTRecord] = (),
    behavioral_holdouts: Iterable[SFTRecord] = (),
) -> CuratedDevelopmentAudit:
    """Reject contamination and incomplete frozen-suite coverage."""
    if not records:
        raise ValueError('curated-development suite must not be empty')
    ids = {record.record_id for record in records}
    if len(ids) != len(records):
        raise ValueError('curated-development suite has duplicate record IDs')
    if ids != set(manifest.records):
        raise ValueError('curated-development manifest does not exactly cover suite IDs')
    if ids != set(grounding_audit):
        raise ValueError('curated-development grounding audit does not exactly cover suite IDs')
    if any(record.source != SOURCE for record in records):
        raise ValueError(f'curated-development records must use source {SOURCE!r}')
    if any(not record.record_id.startswith(RECORD_PREFIX) for record in records):
        raise ValueError(f'curated-development record IDs must start with {RECORD_PREFIX!r}')

    command_families: dict[str, set[str]] = {}
    for record in records:
        annotation = manifest.records[record.record_id]
        command_families.setdefault(record.command, set()).add(annotation.family)
    ambiguous = {command: values for command, values in command_families.items() if len(values) != 1}
    if ambiguous:
        raise ValueError(f'commands assigned to multiple families: {ambiguous}')

    training = list(training_records)
    holdouts = list(behavioral_holdouts)
    suite_commands = {record.command for record in records}
    curated_training_commands = {record.command for record in training if record.source == 'shelliq-curated'}
    overlapping_training_commands = suite_commands & curated_training_commands
    if overlapping_training_commands:
        raise ValueError(f'curated-development commands overlap curated training: {sorted(overlapping_training_commands)}')
    holdout_commands = {record.command for record in holdouts}
    overlapping_holdout_commands = suite_commands & holdout_commands
    if overlapping_holdout_commands:
        raise ValueError(f'curated-development commands overlap behavioral holdouts: {sorted(overlapping_holdout_commands)}')

    comparison_records = training + holdouts
    comparison_instructions = {_normalized(record.instruction) for record in comparison_records}
    comparison_responses = {_normalized(record.response) for record in comparison_records}
    duplicate_instructions = sorted(
        record.record_id for record in records if _normalized(record.instruction) in comparison_instructions
    )
    duplicate_responses = sorted(record.record_id for record in records if _normalized(record.response) in comparison_responses)
    if duplicate_instructions or duplicate_responses:
        raise ValueError(
            f'curated-development exact leakage: instructions={duplicate_instructions}, responses={duplicate_responses}'
        )

    dimensions = Counter(dimension for annotation in manifest.records.values() for dimension in annotation.dimensions)
    families = {annotation.family for annotation in manifest.records.values()}
    if manifest.status == 'frozen':
        if len(records) < MINIMUM_FROZEN_RECORDS:
            raise ValueError(f'frozen suite requires at least {MINIMUM_FROZEN_RECORDS} records')
        if len(families) < MINIMUM_FROZEN_FAMILIES:
            raise ValueError(f'frozen suite requires at least {MINIMUM_FROZEN_FAMILIES} families')
        missing_dimensions = REQUIRED_DIMENSIONS - dimensions.keys()
        if missing_dimensions:
            raise ValueError(f'frozen suite missing dimensions: {sorted(missing_dimensions)}')

    return CuratedDevelopmentAudit(
        records=len(records),
        families=len(families),
        commands=len(suite_commands),
        dimensions=dimensions,
        status=manifest.status,
    )
