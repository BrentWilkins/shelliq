#!/usr/bin/env python3
"""Validate a targeted corpus against release holdout and convert it to semantic JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.data import Corpus, Split, load_jsonl, load_semantic_jsonl, split_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--holdout-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--converter', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


def validate_no_holdout_leakage(dataset: Path, holdout_report: Path, *, seed: int) -> dict[str, object]:
    records = load_jsonl(dataset, corpus=Corpus.DISTRIBUTABLE)
    report = json.loads(holdout_report.read_text())
    examples = report.get('examples')
    if not isinstance(examples, list) or not examples:
        raise ValueError(f'{holdout_report}: missing examples')
    holdout_ids = {example['record_id'] for example in examples}
    holdout_instructions = {example['instruction'].strip().casefold() for example in examples}
    holdout_targets = {example['expected'] for example in examples}
    holdout_commands = {json.loads(example['expected'])['s'][0]['c'][0]['n']['s'] for example in examples}
    shared_ids = sorted(record.record_id for record in records if record.record_id in holdout_ids)
    shared_instructions = sorted(
        record.record_id for record in records if record.instruction.strip().casefold() in holdout_instructions
    )
    if shared_ids or shared_instructions:
        raise ValueError(f'holdout leakage: record_ids={shared_ids}, instructions={shared_instructions}')
    shared_commands = sorted({record.command for record in records} & holdout_commands)
    if shared_commands:
        raise ValueError(f'holdout command families are forbidden: {shared_commands}')
    splits = split_records(records, corpus=Corpus.DISTRIBUTABLE, seed=seed)
    split_counts = {split.value: len(split_records_) for split, split_records_ in splits.items()}
    if not split_counts[Split.TRAIN.value] or not split_counts[Split.TEST.value]:
        raise ValueError(f'targeted dataset needs train and test rows: {split_counts}')
    return {
        'records': len(records),
        'commands': len({record.command for record in records}),
        'command_counts': dict(sorted(Counter(record.command for record in records).items())),
        'split_counts': split_counts,
        'holdout_record_ids_checked': len(holdout_ids),
        'holdout_commands_excluded': sorted(holdout_commands),
        'holdout_semantic_targets_checked': len(holdout_targets),
    }


def main() -> None:
    args = parse_args()
    for output in (args.output, args.manifest):
        if output.exists():
            raise FileExistsError(f'output already exists: {output}')
        if not output.parent.is_dir():
            raise FileNotFoundError(f'output parent does not exist: {output.parent}')
    leakage_audit = validate_no_holdout_leakage(args.dataset, args.holdout_report, seed=args.seed)
    subprocess.run(
        [str(args.converter), str(args.dataset), str(args.output), str(args.manifest)],
        check=True,
    )
    converted = load_semantic_jsonl(args.output)
    if len(converted) != leakage_audit['records']:
        raise ValueError(f'semantic conversion rejected rows: {len(converted)} != {leakage_audit["records"]}')
    manifest = json.loads(args.manifest.read_text())
    manifest['targeted_finishing'] = leakage_audit
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(leakage_audit, sort_keys=True))
    print(f'dataset: {args.output}')
    print(f'manifest: {args.manifest}')


if __name__ == '__main__':
    main()
