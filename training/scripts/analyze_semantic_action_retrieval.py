#!/usr/bin/env python3
"""Measure training-prompt retrieval ranks for typed semantic candidates."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_semantic_action_candidate as candidate
import run_semantic_action_typed as typed

from shelliq_training.semantic_action_typed import CANDIDATE_IGNORE_INDEX, byte_roles

EXPERIMENT = 'semantic-action-training-prompt-retrieval-v1'
_TOKEN = re.compile(r"--?[a-z0-9][a-z0-9-]*|[a-z0-9_./:+@='%{}]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--outer-manifest', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--copy-oracle', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    return parser.parse_args()


def tokenize(value: str) -> Counter[str]:
    return Counter(_TOKEN.findall(value.lower()))


def vectorize(tokens: Counter[str], inverse_document_frequency: dict[str, float]) -> dict[str, float]:
    values = {
        token: (1 + math.log(count)) * inverse_document_frequency.get(token, 0.0)
        for token, count in tokens.items()
        if token in inverse_document_frequency
    }
    norm = math.sqrt(sum(value * value for value in values.values()))
    return {token: value / norm for token, value in values.items()} if norm else {}


def similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def main() -> None:
    args = parse_args()
    _, _, _, _, _, _, manifest = candidate.load_inputs(args)
    train_ids = manifest['inner_splits']['train']['record_ids']
    evaluation_ids = manifest['inner_splits']['validation']['record_ids']
    typed.NORMALIZE_NUMBER_WORDS = True
    records, _, examples, _, _, grammar, _, _ = typed.load_typed(args, train_ids)
    del records
    roles = byte_roles(grammar)

    train_examples = [examples[record_id] for record_id in train_ids]
    document_tokens = [tokenize(bytes(example.source_bytes).decode('utf-8')) for example in train_examples]
    document_frequency: Counter[str] = Counter()
    for tokens in document_tokens:
        document_frequency.update(tokens)
    inverse_document_frequency = {
        token: math.log((1 + len(document_tokens)) / (1 + frequency)) + 1 for token, frequency in document_frequency.items()
    }
    document_vectors = [vectorize(tokens, inverse_document_frequency) for tokens in document_tokens]

    candidate_documents: dict[tuple[int, bytes], set[int]] = defaultdict(set)
    for document_index, example in enumerate(train_examples):
        for label, role in zip(example.candidate_labels, example.word_role_labels, strict=True):
            if label == CANDIDATE_IGNORE_INDEX or role == CANDIDATE_IGNORE_INDEX:
                continue
            candidate_documents[(role, example.candidates[label].value)].add(document_index)

    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, float]] = defaultdict(_summary)
    for record_id in evaluation_ids:
        example = examples[record_id]
        query = vectorize(tokenize(bytes(example.source_bytes).decode('utf-8')), inverse_document_frequency)
        document_scores = [similarity(query, document) for document in document_vectors]
        for position, (label, role) in enumerate(zip(example.candidate_labels, example.word_role_labels, strict=True)):
            if label == CANDIDATE_IGNORE_INDEX or role == CANDIDATE_IGNORE_INDEX:
                continue
            candidate_scores = []
            for index, item in enumerate(example.candidates):
                if role not in item.roles:
                    continue
                sources = candidate_documents.get((role, item.value), ())
                score = max((document_scores[source] for source in sources), default=-1.0)
                candidate_scores.append((score, index))
            candidate_scores.sort(key=lambda item: (-item[0], item[1]))
            rank = next(index for index, (_, candidate_index) in enumerate(candidate_scores, start=1) if candidate_index == label)
            gold = example.candidates[label]
            provenance = (
                '+'.join(
                    name
                    for name, active in zip(
                        ('local', 'derived', 'global'),
                        (gold.local, gold.derived, gold.global_),
                        strict=True,
                    )
                    if active
                )
                or 'none'
            )
            _add(summaries[f'role:{roles[role]}'], rank)
            _add(summaries[f'provenance:{provenance}'], rank)
            rows.append(
                {
                    'record_id': record_id,
                    'position': position,
                    'role': roles[role],
                    'provenance': provenance,
                    'gold': gold.value.decode('utf-8'),
                    'rank': rank,
                    'top8': [example.candidates[index].value.decode('utf-8') for _, index in candidate_scores[:8]],
                }
            )

    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'train_records': len(train_examples),
        'evaluation_records': len(evaluation_ids),
        'summaries': {name: _finish(row) for name, row in sorted(summaries.items())},
        'decisions': rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({**report, 'decisions': f'{len(rows)} rows'}, indent=2, sort_keys=True))


def _summary() -> dict[str, float]:
    return {'total': 0, 'top1': 0, 'top4': 0, 'top8': 0, 'reciprocal_rank': 0.0}


def _add(row: dict[str, float], rank: int) -> None:
    row['total'] += 1
    row['top1'] += int(rank <= 1)
    row['top4'] += int(rank <= 4)
    row['top8'] += int(rank <= 8)
    row['reciprocal_rank'] += 1 / rank


def _finish(row: dict[str, float]) -> dict[str, int | float]:
    total = int(row['total'])
    return {
        'total': total,
        'top1': int(row['top1']),
        'top1_rate': row['top1'] / total,
        'top4': int(row['top4']),
        'top4_rate': row['top4'] / total,
        'top8': int(row['top8']),
        'top8_rate': row['top8'] / total,
        'mean_reciprocal_rank': row['reciprocal_rank'] / total,
    }


if __name__ == '__main__':
    main()
