#!/usr/bin/env python3
"""Compare semantic evaluation reports and enforce the release promotion gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.semantic_error_analysis import (  # noqa: E402
    PromotionThresholds,
    analysis_document,
    gate_metrics,
)
from shelliq_training.semantic_evaluation import load_grounding_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--grounding-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', action='store_true', help='exit nonzero unless the candidate beats incumbent thresholds')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')
    if not args.output.parent.is_dir():
        raise SystemExit(f'output parent does not exist: {args.output.parent}')
    document = analysis_document(args.baseline, args.candidate, load_grounding_audit(args.grounding_audit))
    metrics = document['candidate']['metrics']
    failures = gate_metrics(metrics, PromotionThresholds())
    document['promotion_gate'] = {
        'evaluated': True,
        'passed': not failures,
        'failures': failures,
    }
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'metrics': metrics, 'failure_counts': document['candidate']['failure_counts']}, sort_keys=True))
    print(f'report: {args.output}')
    if args.gate and failures:
        raise SystemExit('promotion gate failed: ' + '; '.join(failures))


if __name__ == '__main__':
    main()
