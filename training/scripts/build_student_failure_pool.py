#!/usr/bin/env python3
"""Build a leak-free, balanced corpus pool for harvesting student failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_teacher_selection import category  # noqa: E402
from shelliq_training.data import Corpus, Split, load_semantic_jsonl, split_records  # noqa: E402

PILOT_QUOTAS = {
    ('linux', 'precise'): 24,
    ('darwin', 'precise'): 8,
    ('linux', 'multi-constraint'): 24,
    ('darwin', 'multi-constraint'): 8,
    ('linux', 'pipeline'): 32,
    ('linux', 'compositional'): 32,
}
MAX_RECORDS_PER_COMMAND = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grounding-output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source', default='shelliq-curated')
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


def _collect_frozen(value: object, record_ids: set[str], instructions: set[str]) -> None:
    if isinstance(value, dict):
        record_id = value.get('record_id')
        instruction = value.get('instruction')
        if isinstance(record_id, str):
            record_ids.add(record_id)
        if isinstance(instruction, str):
            instructions.add(instruction.strip().casefold())
        for child in value.values():
            _collect_frozen(child, record_ids, instructions)
    elif isinstance(value, list):
        for child in value:
            _collect_frozen(child, record_ids, instructions)


def load_frozen_identifiers(evaluation_dir: Path) -> tuple[set[str], set[str]]:
    record_ids: set[str] = set()
    instructions: set[str] = set()
    for path in sorted((*evaluation_dir.rglob('*.json'), *evaluation_dir.rglob('*.jsonl'))):
        try:
            if path.suffix == '.jsonl':
                values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            else:
                values = [json.loads(path.read_text())]
        except json.JSONDecodeError as error:
            raise ValueError(f'{path}: invalid evaluation JSON: {error}') from error
        for value in values:
            _collect_frozen(value, record_ids, instructions)
    return record_ids, instructions


def _stable_key(record_id: str, seed: int) -> str:
    return hashlib.sha256(f'{seed}\0{record_id}'.encode()).hexdigest()


def select_pool(
    rows: list[dict[str, object]],
    train_ids: set[str],
    frozen_ids: set[str],
    frozen_instructions: set[str],
    *,
    source: str,
    seed: int,
) -> list[dict[str, object]]:
    quotas = dict(PILOT_QUOTAS)
    selected: list[dict[str, object]] = []
    command_counts: Counter[tuple[str, str]] = Counter()
    eligible = [
        row
        for row in rows
        if row.get('record_id') in train_ids
        and row.get('record_id') not in frozen_ids
        and str(row.get('instruction', '')).strip().casefold() not in frozen_instructions
        and row.get('source') == source
    ]
    for row in sorted(eligible, key=lambda item: _stable_key(str(item['record_id']), seed)):
        quota = (str(row['platform']), category(row))
        command = (str(row['platform']), str(row['command']))
        if quotas.get(quota, 0) <= 0 or command_counts[command] >= MAX_RECORDS_PER_COMMAND:
            continue
        selected.append(row)
        quotas[quota] -= 1
        command_counts[command] += 1
        if not any(quotas.values()):
            break
    if any(quotas.values()):
        raise ValueError(f'corpus cannot satisfy failure-pool quotas: {quotas}')
    return selected


def grounding_document(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        'schema_version': 1,
        'records': {str(row['record_id']): {'variable_literal_paths': []} for row in rows},
    }


def main() -> None:
    args = parse_args()
    outputs = (args.output, args.grounding_output, args.manifest)
    for output in outputs:
        if output.exists():
            raise FileExistsError(f'output already exists: {output}')
        if not output.parent.is_dir():
            raise FileNotFoundError(f'output parent does not exist: {output.parent}')
    raw_rows = [json.loads(line) for line in args.semantic_dataset.read_text().splitlines() if line.strip()]
    records = load_semantic_jsonl(args.semantic_dataset)
    train_ids = {record.record_id for record in split_records(records, corpus=Corpus.DISTRIBUTABLE, seed=args.seed)[Split.TRAIN]}
    frozen_ids, frozen_instructions = load_frozen_identifiers(args.evaluation_dir)
    selected = select_pool(
        raw_rows,
        train_ids,
        frozen_ids,
        frozen_instructions,
        source=args.source,
        seed=args.seed,
    )
    args.output.write_text(''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n' for row in selected))
    args.grounding_output.write_text(json.dumps(grounding_document(selected), indent=2, sort_keys=True) + '\n')
    manifest = {
        'schema_version': 1,
        'semantic_dataset': str(args.semantic_dataset),
        'evaluation_dir': str(args.evaluation_dir),
        'seed': args.seed,
        'source': args.source,
        'records': len(selected),
        'frozen_record_ids_excluded': len(frozen_ids),
        'frozen_instructions_excluded': len(frozen_instructions),
        'category_counts': dict(sorted(Counter(category(row) for row in selected).items())),
        'platform_counts': dict(sorted(Counter(str(row['platform']) for row in selected).items())),
        'record_ids': [row['record_id'] for row in selected],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: manifest[key] for key in ('records', 'category_counts', 'platform_counts')}, sort_keys=True))
    print(f'pool: {args.output}')
    print(f'grounding: {args.grounding_output}')
    print(f'manifest: {args.manifest}')


if __name__ == '__main__':
    main()
