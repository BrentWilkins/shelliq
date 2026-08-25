#!/usr/bin/env python3
"""Prepare or summarize a closed manual audit of curated-development outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.semantic_error_analysis import classify_example, load_report_examples
from shelliq_training.semantic_evaluation import load_grounding_audit

TRAINING_ROOT = Path(__file__).resolve().parents[1]
FAILURE_MODES = frozenset(
    {
        'semantic_alternative',
        'wrong_command',
        'missing_or_wrong_flags',
        'operand_binding',
        'ordering',
        'pipeline_structure',
        'json_contract',
        'truncation',
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    parser.add_argument(
        '--report',
        action='append',
        default=[],
        metavar='LABEL=PATH',
        help='Evaluation report to adjudicate; repeat for the 0.5B and 1.5B checkpoints.',
    )
    parser.add_argument('--worksheet', type=Path, required=True)
    parser.add_argument(
        '--decisions',
        type=Path,
        help='Versioned reviewer decisions for non-exact rows; strict grounded rows are accepted automatically.',
    )
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--grounding-audit',
        type=Path,
        default=TRAINING_ROOT / 'evaluation' / 'curated-development-grounding-v1.json',
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_report_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs = []
    labels = set()
    for value in values:
        label, separator, raw_path = value.partition('=')
        if not separator or not label or not raw_path:
            raise ValueError(f'invalid report specification {value!r}; expected LABEL=PATH')
        if label in labels:
            raise ValueError(f'duplicate report label {label!r}')
        labels.add(label)
        specs.append((label, Path(raw_path)))
    if not specs:
        raise ValueError('at least one --report is required with --prepare')
    return specs


def _suggested_failure_modes(categories: tuple[str, ...]) -> list[str]:
    suggestions = set()
    if 'invalid_json' in categories or 'invalid_envelope' in categories:
        suggestions.add('json_contract')
    if 'wrong_command' in categories:
        suggestions.add('wrong_command')
    if any(item in categories for item in ('missing_flags', 'extra_flags')):
        suggestions.add('missing_or_wrong_flags')
    if 'reordered_flags' in categories:
        suggestions.add('ordering')
    if 'operand_or_structure_mismatch' in categories:
        suggestions.add('operand_binding')
    return sorted(suggestions)


def prepare_worksheet(report_specs: list[tuple[str, Path]], grounding_path: Path) -> dict[str, object]:
    audit = load_grounding_audit(grounding_path)
    rows = []
    common_ids: set[str] | None = None
    reports = []
    for label, path in report_specs:
        examples = load_report_examples(path)
        ids = set(examples)
        if common_ids is not None and ids != common_ids:
            raise ValueError('evaluation reports do not contain the same record IDs')
        common_ids = ids
        reports.append({'label': label, 'path': str(path), 'sha256': _sha256(path)})
        for record_id, (instruction, expected, generated) in examples.items():
            result = classify_example(record_id, expected, generated, audit)
            rows.append(
                {
                    'checkpoint': label,
                    'record_id': record_id,
                    'instruction': instruction,
                    'expected': expected,
                    'generated': generated,
                    'strict_grounded': result.grounded_exact,
                    'suggested_failure_modes': _suggested_failure_modes(result.categories),
                    'human_correct': None,
                    'failure_modes': [],
                    'notes': '',
                }
            )
    return {
        'schema_version': 1,
        'suite': 'curated-development-v1',
        'grounding_audit': {'path': str(grounding_path), 'sha256': _sha256(grounding_path)},
        'allowed_failure_modes': sorted(FAILURE_MODES),
        'reports': reports,
        'rows': rows,
    }


def apply_decisions(document: dict[str, object], decisions: dict[str, object]) -> dict[str, object]:
    """Overlay explicit non-exact review decisions without mutating the source worksheet."""
    if decisions.get('schema_version') != 1 or decisions.get('suite') != document.get('suite'):
        raise ValueError('unsupported adjudication decisions')
    entries = decisions.get('decisions')
    if isinstance(entries, dict):
        expanded = []
        for mode, keys in entries.items():
            if mode not in FAILURE_MODES or not isinstance(keys, list):
                raise ValueError('grouped adjudication decisions must map failure modes to key lists')
            for raw_key in keys:
                if not isinstance(raw_key, str) or '|' not in raw_key:
                    raise ValueError('grouped adjudication key must be CHECKPOINT|RECORD_ID')
                checkpoint, record_id = raw_key.split('|', 1)
                expanded.append(
                    {
                        'checkpoint': checkpoint,
                        'record_id': record_id,
                        'human_correct': mode == 'semantic_alternative',
                        'failure_modes': [mode],
                        'notes': f'reviewer classified as {mode}',
                    }
                )
        entries = expanded
    if not isinstance(entries, list):
        raise ValueError('adjudication decisions must contain a decision list or grouped mode mapping')
    indexed: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('adjudication decision must be an object')
        key = (entry.get('checkpoint'), entry.get('record_id'))
        if not all(isinstance(item, str) for item in key):
            raise ValueError('adjudication decision is missing checkpoint or record_id')
        if key in indexed:
            raise ValueError(f'duplicate adjudication decision {key[0]}/{key[1]}')
        indexed[key] = entry  # type: ignore[index]

    completed = json.loads(json.dumps(document))
    rows = completed.get('rows')
    assert isinstance(rows, list)
    expected_keys = {
        (row['checkpoint'], row['record_id']) for row in rows if isinstance(row, dict) and row.get('strict_grounded') is not True
    }
    extra = set(indexed) - expected_keys
    missing = expected_keys - set(indexed)
    if extra:
        checkpoint, record_id = sorted(extra)[0]
        raise ValueError(f'decision does not match a non-exact worksheet row: {checkpoint}/{record_id}')
    if missing:
        checkpoint, record_id = sorted(missing)[0]
        raise ValueError(f'missing adjudication decision {checkpoint}/{record_id}')
    for row in rows:
        assert isinstance(row, dict)
        if row.get('strict_grounded') is True:
            row.update({'human_correct': True, 'failure_modes': [], 'notes': 'accepted by strict grounded match'})
            continue
        decision = indexed[(row['checkpoint'], row['record_id'])]
        row.update(
            {
                'human_correct': decision.get('human_correct'),
                'failure_modes': decision.get('failure_modes'),
                'notes': decision.get('notes', ''),
            }
        )
    completed['decision_source'] = {
        'reviewer': decisions.get('reviewer'),
        'method': decisions.get('method'),
    }
    return completed


def summarize_worksheet(document: dict[str, object]) -> dict[str, object]:
    if document.get('schema_version') != 1 or document.get('suite') != 'curated-development-v1':
        raise ValueError('unsupported adjudication worksheet')
    rows = document.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('worksheet has no rows')
    by_checkpoint: dict[str, list[dict[str, object]]] = {}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('worksheet row must be an object')
        checkpoint = row.get('checkpoint')
        record_id = row.get('record_id')
        human_correct = row.get('human_correct')
        failure_modes = row.get('failure_modes')
        if not isinstance(checkpoint, str) or not isinstance(record_id, str):
            raise ValueError('worksheet row is missing checkpoint or record_id')
        if (checkpoint, record_id) in seen:
            raise ValueError(f'duplicate adjudication row {checkpoint}/{record_id}')
        seen.add((checkpoint, record_id))
        if not isinstance(human_correct, bool):
            raise ValueError(f'{checkpoint}/{record_id}: human_correct must be completed')
        if (
            not isinstance(failure_modes, list)
            or len(failure_modes) != len(set(failure_modes))
            or any(mode not in FAILURE_MODES for mode in failure_modes)
        ):
            raise ValueError(f'{checkpoint}/{record_id}: invalid failure_modes')
        if human_correct and any(mode != 'semantic_alternative' for mode in failure_modes):
            raise ValueError(f'{checkpoint}/{record_id}: correct output has non-semantic failure mode')
        if not human_correct and not failure_modes:
            raise ValueError(f'{checkpoint}/{record_id}: incorrect output requires a failure mode')
        by_checkpoint.setdefault(checkpoint, []).append(row)

    populations = [{row['record_id'] for row in checkpoint_rows} for checkpoint_rows in by_checkpoint.values()]
    if any(population != populations[0] for population in populations[1:]):
        raise ValueError('checkpoint adjudications do not cover the same record IDs')
    summaries = {}
    for checkpoint, checkpoint_rows in sorted(by_checkpoint.items()):
        failure_counts = Counter(
            mode
            for row in checkpoint_rows
            for mode in row['failure_modes']  # type: ignore[union-attr]
        )
        summaries[checkpoint] = {
            'total_examples': len(checkpoint_rows),
            'human_correct': sum(row['human_correct'] is True for row in checkpoint_rows),
            'strict_grounded': sum(row.get('strict_grounded') is True for row in checkpoint_rows),
            'explicitly_reviewed': sum(row.get('strict_grounded') is not True for row in checkpoint_rows),
            'failure_counts': dict(sorted(failure_counts.items())),
        }
    return {
        'schema_version': 1,
        'suite': document['suite'],
        'worksheet_reports': document.get('reports'),
        'decision_source': document.get('decision_source'),
        'adjudication_complete': True,
        'checkpoints': summaries,
    }


def main() -> None:
    args = parse_args()
    if args.prepare:
        if args.decisions is not None:
            raise SystemExit('--decisions is only valid with --summarize')
        if args.worksheet.exists():
            raise SystemExit(f'worksheet already exists: {args.worksheet}')
        try:
            document = prepare_worksheet(parse_report_specs(args.report), args.grounding_audit)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        args.worksheet.write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
        print(f'worksheet: {args.worksheet} ({len(document["rows"])} rows)')
        return
    if args.report:
        raise SystemExit('--report is only valid with --prepare')
    if args.output is None:
        raise SystemExit('--summarize requires --output')
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')
    try:
        document = json.loads(args.worksheet.read_text())
        if args.decisions is not None:
            document = apply_decisions(document, json.loads(args.decisions.read_text()))
        summary = summarize_worksheet(document)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary['checkpoints'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
