"""Reproducible quality and split audit for a distributable training candidate."""

from __future__ import annotations

import hashlib
import re
import shlex
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from shelliq_training.data import Corpus, DatasetFormatError, Platform, SFTRecord, Split, load_jsonl, split_records
from shelliq_training.evaluation import validate_heldout_splits

AUDIT_SCHEMA_VERSION = 1
_NEGATIVE_NUMBER = re.compile(r'^-\d+(?:\.\d+)?$')


def audit_distributable_corpus(
    dataset: str | Path,
    *,
    seed: int,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    heldout_sources: frozenset[str] = frozenset(),
    heldout_platforms: frozenset[Platform] = frozenset(),
) -> dict[str, object]:
    """Load, preflight, split, and summarize one immutable candidate artifact."""
    dataset = Path(dataset)
    records = load_jsonl(dataset, corpus=Corpus.DISTRIBUTABLE)
    if not records:
        raise DatasetFormatError(f'{dataset}: dataset contains no records')
    accepted, rejected = preflight_records(records)
    if not accepted:
        raise DatasetFormatError(f'{dataset}: preflight accepted no records')

    splits = split_records(
        accepted,
        corpus=Corpus.DISTRIBUTABLE,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        heldout_sources=heldout_sources,
        heldout_platforms=heldout_platforms,
    )
    validate_heldout_splits(splits)

    return {
        'audit_schema_version': AUDIT_SCHEMA_VERSION,
        'dataset': {'name': dataset.name, 'sha256': _sha256(dataset)},
        'records': {
            'total': len(records),
            'preflight_accepted': len(accepted),
            'preflight_rejected': len(rejected),
            'rejected_by_reason': dict(sorted(Counter(rejected.values()).items())),
            'rejected_record_ids': sorted(rejected),
        },
        'composition': {
            'commands': len({record.command for record in records}),
            'sources': _counter(record.source for record in records),
            'licenses': _counter(record.license for record in records),
            'platforms': _counter(record.platform.value for record in records),
            'top_commands': _top_counter(record.command for record in records),
        },
        'duplicates': {
            'instruction_response': _duplicate_summary(records, lambda record: (record.instruction, record.response)),
            'response': _duplicate_summary(records, lambda record: record.response),
            'normalized_instruction': _duplicate_summary(records, lambda record: _normalize_instruction(record.instruction)),
        },
        'character_lengths': {
            field: _length_summary([len(getattr(record, field)) for record in records])
            for field in ('instruction', 'response', 'context')
        },
        'lexical_options': _lexical_option_summary(accepted),
        'split': {
            'seed': seed,
            'validation_fraction': validation_fraction,
            'test_fraction': test_fraction,
            'heldout_sources': sorted(heldout_sources),
            'heldout_platforms': sorted(platform.value for platform in heldout_platforms),
            'partitions': {
                split.value: {
                    'records': len(splits[split]),
                    'commands': len({record.command for record in splits[split]}),
                    'sources': _counter(record.source for record in splits[split]),
                    'platforms': _counter(record.platform.value for record in splits[split]),
                }
                for split in Split
            },
        },
    }


def preflight_records(records: Sequence[SFTRecord]) -> tuple[list[SFTRecord], dict[str, str]]:
    """Reject unresolved templates and invalid shell quoting before splitting."""
    accepted: list[SFTRecord] = []
    rejected: dict[str, str] = {}
    for record in records:
        if _contains_unresolved_placeholder(record.response):
            rejected[record.record_id] = 'unresolved tldr placeholder'
            continue
        try:
            shlex.split(record.response)
        except ValueError as error:
            rejected[record.record_id] = f'invalid shell quoting: {error}'
            continue
        accepted.append(record)
    return accepted, rejected


def _contains_unresolved_placeholder(value: str) -> bool:
    """Recognize unescaped ``{{...}}`` without rejecting literal brace formats."""
    openings = [index for index in range(len(value) - 1) if value[index : index + 2] == '{{' and _is_unescaped(value, index)]
    for opening in openings:
        if any(value[index : index + 2] == '}}' and _is_unescaped(value, index) for index in range(opening + 2, len(value) - 1)):
            return True
    return False


def _is_unescaped(value: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and value[index] == '\\':
        backslashes += 1
        index -= 1
    return backslashes % 2 == 0


def _duplicate_summary(records: Sequence[SFTRecord], key: Callable[[SFTRecord], object]) -> dict[str, object]:
    groups: defaultdict[object, list[str]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record.record_id)
    duplicates = sorted((ids for ids in groups.values() if len(ids) > 1), key=lambda ids: (-len(ids), ids))
    return {
        'groups': len(duplicates),
        'records': sum(len(ids) for ids in duplicates),
        'sample_record_ids': [ids[:10] for ids in duplicates[:20]],
    }


def _normalize_instruction(value: str) -> str:
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', value.casefold()).split())


def _lexical_option_summary(records: Sequence[SFTRecord]) -> dict[str, object]:
    counts: Counter[str] = Counter()
    command_options: set[tuple[str, str]] = set()
    for record in records:
        for token in shlex.split(record.response):
            if len(token) < 2 or token == '--' or not token.startswith('-') or _NEGATIVE_NUMBER.fullmatch(token):
                continue
            option = token.split('=', maxsplit=1)[0]
            counts[option] += 1
            command_options.add((record.command, option))
    return {
        'distinct_tokens': len(counts),
        'distinct_command_tokens': len(command_options),
        'top_tokens': [{'token': token, 'records': count} for token, count in _ordered_counts(counts)[:30]],
    }


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _top_counter(values: Iterable[str]) -> list[dict[str, object]]:
    return [{'name': name, 'records': count} for name, count in _ordered_counts(Counter(values))[:20]]


def _ordered_counts(counts: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _length_summary(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {
        'min': ordered[0],
        'p50': ordered[(len(ordered) - 1) * 50 // 100],
        'p95': ordered[(len(ordered) - 1) * 95 // 100],
        'max': ordered[-1],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
