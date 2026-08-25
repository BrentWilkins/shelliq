#!/usr/bin/env python3
"""Build closed metadata and grounding sidecars for curated development."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.curated_development import SCHEMA_VERSION, SUITE_ID
from shelliq_training.data import Corpus, load_jsonl

TRAINING_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--suite',
        type=Path,
        default=TRAINING_ROOT / 'evaluation' / 'curated-development-v1.jsonl',
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        default=TRAINING_ROOT / 'evaluation' / 'curated-development-v1.manifest.json',
    )
    parser.add_argument(
        '--grounding-audit',
        type=Path,
        default=TRAINING_ROOT / 'evaluation' / 'curated-development-grounding-v1.json',
    )
    parser.add_argument('--status', choices=('draft', 'frozen'), required=True)
    return parser.parse_args()


def dimensions(response: str) -> list[str]:
    tokens = shlex.split(response, posix=True)
    result = {'operand-binding'}
    if any(token.startswith('-') for token in tokens[1:]):
        result.add('flag-selection')
    if re.search(r'--[^\s=]+=|\s-[A-Za-z]\s+\S', response):
        result.add('option-value')
    if len(tokens) >= 4:
        result.add('ordering')
    if any(operator in tokens for operator in ('|', '|&')):
        result.add('pipeline-semantics')
    return sorted(result)


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.suite, corpus=Corpus.DISTRIBUTABLE)
    annotations = {}
    grounding = {}
    for record in records:
        record_dimensions = dimensions(record.response)
        annotations[record.record_id] = {
            'family': record.command,
            'dimensions': record_dimensions,
            'constraint_count': max(2, len(record_dimensions)),
        }
        grounding[record.record_id] = {'variable_literal_paths': []}
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'suite_id': SUITE_ID,
        'status': args.status,
        'records': annotations,
    }
    audit = {'schema_version': 1, 'records': grounding}
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    args.grounding_audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
