#!/usr/bin/env python3
"""Export a complete manual-review ledger into strict preference JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.preference_data import PreferenceRecord, load_preference_jsonl  # noqa: E402
from shelliq_training.teacher_verification import rust_validate_documents  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--decisions', type=Path, nargs='+', required=True)
    parser.add_argument('--prior-preferences', type=Path, nargs='*', default=[])
    parser.add_argument('--validator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def export_records(
    queue: list[dict[str, object]],
    decisions: list[dict[str, object]],
    validations: list[object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    queue_by_id = {str(row['record_id']): row for row in queue}
    decision_by_id = {str(row['record_id']): row for row in decisions}
    if len(queue_by_id) != len(queue) or len(decision_by_id) != len(decisions):
        raise ValueError('queue and decision record IDs must be unique')
    missing = set(queue_by_id) - set(decision_by_id)
    unknown = set(decision_by_id) - set(queue_by_id)
    if missing or unknown:
        raise ValueError(f'decisions must cover queue exactly; missing={sorted(missing)}, unknown={sorted(unknown)}')
    if len(validations) != len(queue) * 2 or not all(getattr(value, 'valid', False) for value in validations):
        raise ValueError('every chosen and rejected document must pass Rust semantic validation')

    exported: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()
    for record_id, candidate in queue_by_id.items():
        decision = decision_by_id[record_id]
        state = decision.get('decision')
        if state == 'exclude':
            reason = decision.get('reason')
            if not isinstance(reason, str) or not reason:
                raise ValueError(f'{record_id}: excluded decision requires reason')
            exclusions[reason] += 1
            continue
        if state != 'include':
            raise ValueError(f'{record_id}: decision must be include or exclude')
        failure_modes = decision.get('failure_modes')
        evidence = decision.get('evidence')
        if not isinstance(failure_modes, list) or not isinstance(evidence, str) or not evidence:
            raise ValueError(f'{record_id}: included decision requires failure_modes and evidence')
        row = {
            'schema_version': 1,
            'pair_id': f'preference:v1:{record_id}',
            'corpus': candidate['corpus'],
            'source': 'shelliq-reviewed-preference',
            'license': candidate['license'],
            'provenance': (
                f'{candidate["provenance"]}; rejected=lora-rank8-scale0.5-lr2e-4-step10000; review=student-failure-v2@2026-08-21'
            ),
            'platform': candidate['platform'],
            'instruction': candidate['instruction'],
            'context': candidate['context'],
            'chosen': candidate['chosen'],
            'rejected': candidate['rejected'],
            'failure_modes': failure_modes,
            'verifier_evidence': [
                'chosen and rejected pass SemanticDocumentV2 decode, render, reparse, and relower checks',
                evidence,
            ],
            'reviewer': 'codex-manual-review',
            'reviewed_at': '2026-08-21',
        }
        PreferenceRecord.from_dict(row)
        exported.append(row)
    summary: dict[str, object] = {
        'schema_version': 1,
        'reviewed': len(queue),
        'included': len(exported),
        'excluded': len(queue) - len(exported),
        'exclusion_reasons': dict(sorted(exclusions.items())),
        'failure_modes': dict(sorted(Counter(mode for row in exported for mode in row['failure_modes']).items())),
    }
    return exported, summary


def main() -> None:
    args = parse_args()
    for path in (args.queue, *args.decisions, *args.prior_preferences, args.validator):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output, args.summary):
        if path.exists():
            raise FileExistsError(path)
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
    queue = _load_jsonl(args.queue)
    queue_ids = {str(row['record_id']) for row in queue}
    all_decisions = [row for path in args.decisions for row in _load_jsonl(path)]
    decisions = [row for row in all_decisions if str(row['record_id']) in queue_ids]
    documents = [json.dumps(row[key], separators=(',', ':')) for row in queue for key in ('chosen', 'rejected')]
    validations = rust_validate_documents(documents, args.validator)
    exported, summary = export_records(queue, decisions, validations)
    prior_rows = [row for path in args.prior_preferences for row in _load_jsonl(path)]
    for path in args.prior_preferences:
        load_preference_jsonl(path)
    combined = {str(row['pair_id']): row for row in prior_rows}
    if len(combined) != len(prior_rows):
        raise ValueError('prior preference pair IDs must be unique')
    carried_forward = set(combined) - {str(row['pair_id']) for row in exported}
    for row in exported:
        pair_id = str(row['pair_id'])
        prior = combined.get(pair_id)
        if prior is not None and prior != row:
            raise ValueError(f'prior preference pair changed: {pair_id}')
        combined[pair_id] = row
    exported = [combined[pair_id] for pair_id in sorted(combined)]
    summary['included'] = len(exported)
    summary['carried_forward'] = len(carried_forward)
    summary['failure_modes'] = dict(sorted(Counter(mode for row in exported for mode in row['failure_modes']).items()))
    args.output.write_text(''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n' for row in exported))
    load_preference_jsonl(args.output)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
