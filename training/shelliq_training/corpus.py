"""Deterministic integration of distributable schema-v1 corpus files."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from shelliq_training.data import Corpus, DatasetFormatError, SFTRecord, load_jsonl


def supervised_corpus_paths(directory: str | Path) -> tuple[Path, ...]:
    """Return only JSONL files using the supervised record/response schema."""
    return tuple(path for path in sorted(Path(directory).glob('*.jsonl')) if not path.name.startswith('reviewed-preference-'))


@dataclass(frozen=True, slots=True)
class CorpusInputSummary:
    """Auditable facts about one input file."""

    kind: str
    name: str
    record_count: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            'kind': self.kind,
            'name': self.name,
            'record_count': self.record_count,
            'sha256': self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CorpusMergeReport:
    """Stable manifest data for a merged distributable artifact."""

    inputs: tuple[CorpusInputSummary, ...]
    record_count: int
    command_count: int
    source_counts: tuple[tuple[str, int], ...]
    platform_counts: tuple[tuple[str, int], ...]
    curated_family_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            'schema_version': 1,
            'record_count': self.record_count,
            'command_count': self.command_count,
            'source_counts': dict(self.source_counts),
            'platform_counts': dict(self.platform_counts),
            'curated_family_counts': dict(self.curated_family_counts),
            'inputs': [input_summary.to_dict() for input_summary in self.inputs],
        }


def merge_distributable_corpus(
    tldr_path: str | Path,
    curated_directory: str | Path,
) -> tuple[list[SFTRecord], CorpusMergeReport]:
    """Load TLDR then curated families, rejecting IDs repeated across files."""
    tldr_file = Path(tldr_path)
    curated_directory = Path(curated_directory)
    curated_files = supervised_corpus_paths(curated_directory)
    if not curated_files:
        raise DatasetFormatError(f'{curated_directory}: no curated JSONL files found')

    inputs = [('tldr', tldr_file), *(('curated', path) for path in curated_files)]
    merged: list[SFTRecord] = []
    input_summaries: list[CorpusInputSummary] = []
    family_counts: list[tuple[str, int]] = []
    record_origins: dict[str, str] = {}

    for kind, path in inputs:
        records = load_jsonl(path, corpus=Corpus.DISTRIBUTABLE)
        if not records:
            raise DatasetFormatError(f'{path}: corpus input contains no records')
        origin = f'{kind}:{path.name}'
        for record in records:
            previous = record_origins.get(record.record_id)
            if previous is not None:
                raise DatasetFormatError(
                    f'duplicate record_id {record.record_id!r} across corpus inputs: {previous} and {origin}'
                )
            record_origins[record.record_id] = origin
        merged.extend(records)
        input_summaries.append(
            CorpusInputSummary(
                kind=kind,
                name=path.name,
                record_count=len(records),
                sha256=_sha256(path),
            )
        )
        if kind == 'curated':
            family_counts.append((path.stem, len(records)))

    source_counts = Counter(record.source for record in merged)
    platform_counts = Counter(record.platform.value for record in merged)
    report = CorpusMergeReport(
        inputs=tuple(input_summaries),
        record_count=len(merged),
        command_count=len({record.command for record in merged}),
        source_counts=tuple(sorted(source_counts.items())),
        platform_counts=tuple(sorted(platform_counts.items())),
        curated_family_counts=tuple(family_counts),
    )
    return merged, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
