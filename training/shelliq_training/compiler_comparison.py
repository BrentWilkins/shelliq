"""Frozen split and paired metrics for semantic compiler comparisons."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from shelliq_training.data import Corpus, DatasetFormatError, SFTRecord, Split, split_records

COMPARISON_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FrozenSplit:
    name: Split
    record_ids: tuple[str, ...]
    commands: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            'record_count': len(self.record_ids),
            'command_count': len(self.commands),
            'record_ids': list(self.record_ids),
            'commands': list(self.commands),
        }


@dataclass(frozen=True, slots=True)
class ComparisonManifest:
    experiment: str
    dataset_sha256: str
    split_seed: int
    validation_fraction: float
    test_fraction: float
    splits: tuple[FrozenSplit, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            'schema_version': COMPARISON_SCHEMA_VERSION,
            'experiment': self.experiment,
            'dataset_sha256': self.dataset_sha256,
            'split_seed': self.split_seed,
            'validation_fraction': self.validation_fraction,
            'test_fraction': self.test_fraction,
            'splits': {split.name.value: split.to_dict() for split in self.splits},
        }

    def records_by_split(self, records: Sequence[SFTRecord]) -> dict[Split, list[SFTRecord]]:
        by_id = {record.record_id: record for record in records}
        if len(by_id) != len(records):
            raise DatasetFormatError('dataset contains duplicate record IDs')
        result: dict[Split, list[SFTRecord]] = {}
        for frozen in self.splits:
            try:
                selected = [by_id[record_id] for record_id in frozen.record_ids]
            except KeyError as error:
                raise DatasetFormatError(f'manifest record is missing from dataset: {error.args[0]}') from error
            if {record.command for record in selected} != set(frozen.commands):
                raise DatasetFormatError(f'{frozen.name.value} commands do not match the frozen manifest')
            result[frozen.name] = selected
        manifest_ids = {record_id for split in self.splits for record_id in split.record_ids}
        if manifest_ids != by_id.keys():
            raise DatasetFormatError('manifest record IDs do not exactly cover the dataset')
        return result


def freeze_comparison_manifest(
    records: Sequence[SFTRecord],
    *,
    dataset_sha256: str,
    experiment: str,
    split_seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> ComparisonManifest:
    split = split_records(
        records,
        corpus=Corpus.DISTRIBUTABLE,
        seed=split_seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    frozen = tuple(
        FrozenSplit(
            name,
            tuple(record.record_id for record in split[name]),
            tuple(sorted({record.command for record in split[name]})),
        )
        for name in Split
    )
    manifest = ComparisonManifest(
        experiment,
        dataset_sha256,
        split_seed,
        validation_fraction,
        test_fraction,
        frozen,
    )
    _validate_manifest(manifest)
    return manifest


def write_comparison_manifest(path: str | Path, manifest: ComparisonManifest) -> None:
    Path(path).write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + '\n')


def load_comparison_manifest(path: str | Path) -> ComparisonManifest:
    raw = json.loads(Path(path).read_text())
    expected = {
        'schema_version',
        'experiment',
        'dataset_sha256',
        'split_seed',
        'validation_fraction',
        'test_fraction',
        'splits',
    }
    if not isinstance(raw, dict) or raw.keys() != expected or raw['schema_version'] != COMPARISON_SCHEMA_VERSION:
        raise DatasetFormatError('invalid comparison manifest envelope')
    raw_splits = raw['splits']
    if not isinstance(raw_splits, dict) or raw_splits.keys() != {split.value for split in Split}:
        raise DatasetFormatError('comparison manifest must contain every split')
    splits: list[FrozenSplit] = []
    for name in Split:
        split = raw_splits[name.value]
        if not isinstance(split, dict) or split.keys() != {'record_count', 'command_count', 'record_ids', 'commands'}:
            raise DatasetFormatError(f'invalid {name.value} split')
        record_ids = _text_tuple(split['record_ids'], f'{name.value}.record_ids')
        commands = _text_tuple(split['commands'], f'{name.value}.commands')
        if split['record_count'] != len(record_ids) or split['command_count'] != len(commands):
            raise DatasetFormatError(f'invalid {name.value} split counts')
        splits.append(FrozenSplit(name, record_ids, commands))
    manifest = ComparisonManifest(
        experiment=_text(raw, 'experiment'),
        dataset_sha256=_sha256_text(raw, 'dataset_sha256'),
        split_seed=_integer(raw, 'split_seed'),
        validation_fraction=_fraction(raw, 'validation_fraction'),
        test_fraction=_fraction(raw, 'test_fraction'),
        splits=tuple(splits),
    )
    _validate_manifest(manifest)
    return manifest


def require_matching_dataset(path: str | Path, manifest: ComparisonManifest) -> None:
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != manifest.dataset_sha256:
        raise DatasetFormatError(f'dataset SHA-256 mismatch: expected {manifest.dataset_sha256}, got {actual}')


def paired_binary_counts(left: Mapping[str, bool], right: Mapping[str, bool]) -> dict[str, int | float]:
    """Return paired outcomes and an exact two-sided McNemar p-value."""
    if left.keys() != right.keys() or not left:
        raise ValueError('paired outcomes must have the same non-empty record IDs')
    left_only = sum(left[key] and not right[key] for key in left)
    right_only = sum(right[key] and not left[key] for key in left)
    both = sum(left[key] and right[key] for key in left)
    neither = len(left) - left_only - right_only - both
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, value) for value in range(min(left_only, right_only) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    return {
        'both': both,
        'left_only': left_only,
        'right_only': right_only,
        'neither': neither,
        'discordant': discordant,
        'mcnemar_exact_two_sided_p': p_value,
    }


def _validate_manifest(manifest: ComparisonManifest) -> None:
    if manifest.validation_fraction + manifest.test_fraction >= 1:
        raise DatasetFormatError('comparison fractions leave no training data')
    all_ids: set[str] = set()
    all_commands: set[str] = set()
    for split in manifest.splits:
        if not split.record_ids or not split.commands:
            raise DatasetFormatError(f'{split.name.value} split must not be empty')
        if len(set(split.record_ids)) != len(split.record_ids):
            raise DatasetFormatError(f'duplicate record ID in {split.name.value} split')
        if len(set(split.commands)) != len(split.commands):
            raise DatasetFormatError(f'duplicate command in {split.name.value} split')
        if all_ids.intersection(split.record_ids):
            raise DatasetFormatError('record ID occurs in multiple splits')
        if all_commands.intersection(split.commands):
            raise DatasetFormatError('command occurs in multiple splits')
        all_ids.update(split.record_ids)
        all_commands.update(split.commands)


def _text(raw: Mapping[str, object], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value:
        raise DatasetFormatError(f'{field} must be non-empty text')
    return value


def _sha256_text(raw: Mapping[str, object], field: str) -> str:
    value = _text(raw, field)
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise DatasetFormatError(f'{field} must be a lowercase SHA-256 digest')
    return value


def _integer(raw: Mapping[str, object], field: str) -> int:
    value = raw[field]
    if not isinstance(value, int):
        raise DatasetFormatError(f'{field} must be an integer')
    return value


def _fraction(raw: Mapping[str, object], field: str) -> float:
    value = raw[field]
    if not isinstance(value, int | float) or not 0 <= value < 1:
        raise DatasetFormatError(f'{field} must be a fraction in [0, 1)')
    return float(value)


def _text_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise DatasetFormatError(f'{field} must be a non-empty text list')
    return tuple(value)
