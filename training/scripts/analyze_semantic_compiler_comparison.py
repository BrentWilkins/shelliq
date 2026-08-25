#!/usr/bin/env python3
"""Analyze frozen custom-versus-CodeT5 reports without retaining generations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.compiler_comparison import analyze_comparison_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--custom-train', type=Path, required=True)
    parser.add_argument('--custom-test', type=Path, required=True)
    parser.add_argument('--codet5-train', type=Path, required=True)
    parser.add_argument('--codet5-test', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return document


def main() -> None:
    args = parse_args()
    result = analyze_comparison_reports(
        load(args.custom_train),
        load(args.custom_test),
        load(args.codet5_train),
        load(args.codet5_test),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result['decision'], sort_keys=True))


if __name__ == '__main__':
    main()
