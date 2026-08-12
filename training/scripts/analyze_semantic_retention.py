#!/usr/bin/env python3
"""Compare two reports on the frozen semantic retention suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.retention_analysis import retention_failures  # noqa: E402
from shelliq_training.semantic_error_analysis import (  # noqa: E402
    analyze_examples,
    compare_results,
    load_report_examples,
)
from shelliq_training.semantic_evaluation import load_grounding_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--grounding-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f'output already exists: {args.output}')
    audit = load_grounding_audit(args.grounding_audit)
    baseline_examples = load_report_examples(args.baseline)
    candidate_examples = load_report_examples(args.candidate)
    if baseline_examples.keys() != candidate_examples.keys():
        raise ValueError('baseline and candidate retention record IDs differ')
    for record_id in baseline_examples:
        old_instruction, old_expected, _ = baseline_examples[record_id]
        new_instruction, new_expected, _ = candidate_examples[record_id]
        if (old_instruction, old_expected) != (new_instruction, new_expected):
            raise ValueError(f'{record_id}: retention instruction or target differs')
    baseline_metrics, baseline_results = analyze_examples(baseline_examples, audit)
    candidate_metrics, candidate_results = analyze_examples(candidate_examples, audit)
    failures = retention_failures(baseline_metrics, candidate_metrics)
    comparison = compare_results(baseline_results, candidate_results)
    report = {
        'analysis_schema_version': 1,
        'baseline': {'path': str(args.baseline), 'metrics': baseline_metrics},
        'candidate': {'path': str(args.candidate), 'metrics': candidate_metrics},
        'comparison': comparison,
        'gate': {'passed': not failures, 'failures': failures},
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report['gate'], sort_keys=True))
    if args.gate and failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
