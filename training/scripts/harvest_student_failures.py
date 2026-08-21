#!/usr/bin/env python3
"""Harvest balanced, verifier-backed near misses from a student evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_teacher_selection import category  # noqa: E402
from shelliq_training.semantic_evaluation import parse_semantic_document  # noqa: E402
from shelliq_training.teacher_verification import rust_validate_documents, verify_against_reference  # noqa: E402

FAILURE_QUOTAS = {'precise': 16, 'multi-constraint': 16, 'pipeline': 16, 'compositional': 16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--validator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


def _stable_key(record_id: str, seed: int) -> str:
    return hashlib.sha256(f'{seed}\0{record_id}'.encode()).hexdigest()


def harvest(
    pool_rows: list[dict[str, object]],
    examples: list[dict[str, object]],
    validations: list[object],
    *,
    seed: int,
) -> list[dict[str, object]]:
    pool_by_id = {str(row['record_id']): row for row in pool_rows}
    candidates: list[dict[str, object]] = []
    for example, rust_validation in zip(examples, validations, strict=True):
        record_id = str(example['record_id'])
        row = pool_by_id.get(record_id)
        if row is None:
            raise ValueError(f'evaluation record not in pool: {record_id}')
        generated = example.get('trained')
        expected_text = example.get('expected')
        if not isinstance(generated, str) or not isinstance(expected_text, str):
            raise ValueError(f'{record_id}: malformed evaluation example')
        expected = parse_semantic_document(expected_text)
        rejected = parse_semantic_document(generated)
        if expected is None:
            raise ValueError(f'{record_id}: invalid expected SemanticDocumentV2')
        verification = verify_against_reference(generated, expected, rust_validation)
        # Preference candidates must be valid semantic near misses with the right
        # routed command. Invalid framing is useful for metrics, not DPO pairs.
        if verification.accepted or rejected is None or not verification.rust_round_trip or not verification.first_command_match:
            continue
        group = category(row)
        candidates.append(
            {
                'schema_version': 1,
                'candidate_id': f'student-failure:v1:{record_id}',
                'record_id': record_id,
                'category': group,
                'corpus': row['corpus'],
                'source': row['source'],
                'license': row['license'],
                'provenance': row['provenance'],
                'platform': row['platform'],
                'command': row['command'],
                'instruction': row['instruction'],
                'context': row['context'],
                'chosen': expected,
                'chosen_rendered': row['shell_response'],
                'rejected': rejected,
                'student_raw': generated,
                'student_rendered': verification.rendered,
                'verification': asdict(verification),
                'review_state': 'pending',
            }
        )
    quotas = dict(FAILURE_QUOTAS)
    selected: list[dict[str, object]] = []
    command_counts: Counter[tuple[str, str]] = Counter()
    for candidate in sorted(candidates, key=lambda item: _stable_key(str(item['record_id']), seed)):
        group = str(candidate['category'])
        command = (str(candidate['platform']), str(candidate['command']))
        if quotas[group] <= 0 or command_counts[command] >= 4:
            continue
        selected.append(candidate)
        quotas[group] -= 1
        command_counts[command] += 1
        if not any(quotas.values()):
            break
    if any(quotas.values()):
        raise ValueError(f'not enough verified near misses for failure quotas: {quotas}')
    return selected


def main() -> None:
    args = parse_args()
    for output in (args.output, args.manifest):
        if output.exists():
            raise FileExistsError(f'output already exists: {output}')
        if not output.parent.is_dir():
            raise FileNotFoundError(f'output parent does not exist: {output.parent}')
    pool_rows = [json.loads(line) for line in args.pool.read_text().splitlines() if line.strip()]
    report = json.loads(args.evaluation.read_text())
    examples = report.get('examples')
    if not isinstance(examples, list) or not examples:
        raise ValueError('evaluation report has no examples')
    generated = [example['trained'] for example in examples]
    if not all(isinstance(item, str) for item in generated):
        raise ValueError('evaluation report contains malformed trained output')
    validations = rust_validate_documents(generated, args.validator)
    selected = harvest(pool_rows, examples, validations, seed=args.seed)
    args.output.write_text(''.join(json.dumps(candidate, sort_keys=True, separators=(',', ':')) + '\n' for candidate in selected))
    manifest = {
        'schema_version': 1,
        'pool': str(args.pool),
        'evaluation': str(args.evaluation),
        'validator': str(args.validator),
        'seed': args.seed,
        'records': len(selected),
        'category_counts': dict(sorted(Counter(str(item['category']) for item in selected).items())),
        'platform_counts': dict(sorted(Counter(str(item['platform']) for item in selected).items())),
        'review_state_counts': dict(sorted(Counter(str(item['review_state']) for item in selected).items())),
        'candidate_ids': [item['candidate_id'] for item in selected],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: manifest[key] for key in ('records', 'category_counts', 'platform_counts')}, sort_keys=True))
    print(f'queue: {args.output}')
    print(f'manifest: {args.manifest}')


if __name__ == '__main__':
    main()
