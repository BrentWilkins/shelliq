#!/usr/bin/env python3
"""Audit the curated-development suite without evaluating a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.corpus import supervised_corpus_paths  # noqa: E402
from shelliq_training.curated_development import (  # noqa: E402
    audit_curated_development,
    load_manifest,
)
from shelliq_training.data import Corpus, SFTRecord, load_jsonl  # noqa: E402
from shelliq_training.semantic_evaluation import load_grounding_audit  # noqa: E402

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
    return parser.parse_args()


def _load_paths(paths: list[Path]) -> list[SFTRecord]:
    return [record for path in paths for record in load_jsonl(path, corpus=Corpus.DISTRIBUTABLE)]


def main() -> None:
    args = parse_args()
    corpus_paths = [*supervised_corpus_paths(TRAINING_ROOT / 'corpus')]
    holdout_paths = [
        TRAINING_ROOT / 'evaluation' / 'pipeline-compatibility-v1.jsonl',
        TRAINING_ROOT / 'evaluation' / 'semantic-retention-v1.jsonl',
        TRAINING_ROOT / 'evaluation' / 'semantic-shadow-release-v1.jsonl',
    ]
    report = audit_curated_development(
        load_jsonl(args.suite, corpus=Corpus.DISTRIBUTABLE),
        load_manifest(args.manifest),
        load_grounding_audit(args.grounding_audit),
        training_records=_load_paths(corpus_paths),
        behavioral_holdouts=_load_paths(holdout_paths),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
