"""Write a reproducible audit report for a merged distributable corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.corpus_audit import audit_distributable_corpus  # noqa: E402
from shelliq_training.data import Platform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--heldout-source', action='append', default=[])
    parser.add_argument('--heldout-platform', type=Platform, action='append', default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f'output already exists: {args.output}')
    report = audit_distributable_corpus(
        args.dataset,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        heldout_sources=frozenset(args.heldout_source),
        heldout_platforms=frozenset(args.heldout_platform),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    records = report['records']
    composition = report['composition']
    print(
        f'audited {records["total"]} records ({records["preflight_accepted"]} accepted, '
        f'{records["preflight_rejected"]} rejected) across {composition["commands"]} commands'
    )
    print(f'report: {args.output}')


if __name__ == '__main__':
    main()
