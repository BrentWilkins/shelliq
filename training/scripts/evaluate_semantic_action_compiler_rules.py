#!/usr/bin/env python3
"""Evaluate compiler-rule overrides on a locked grounded-count inner report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_semantic_action_candidate as candidate
import run_semantic_action_grounded_count_v2 as grounded_count
import run_semantic_action_typed as typed
from run_semantic_action_decoder import _evaluate

from shelliq_training.semantic_action_compiler_rules import compile_semantic_rule

EXPERIMENT = 'semantic-action-compiler-rules-v2'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--outer-manifest', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--copy-oracle', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, _, _, _, _, manifest = candidate.load_inputs(args)
    train_ids = manifest['inner_splits']['train']['record_ids']
    evaluation_ids = manifest['inner_splits']['validation']['record_ids']
    typed.NORMALIZE_NUMBER_WORDS = True
    typed.POST_COMMAND_COUNTS = True
    records, _, examples, _, client, _, _, _ = typed.load_typed(args, train_ids)
    by_record = {record.record_id: record for record in records}
    evaluation_records = [by_record[record_id] for record_id in evaluation_ids]
    evaluation_examples = [examples[record_id] for record_id in evaluation_ids]

    baseline = json.loads(args.baseline_report.read_text())
    if baseline.get('experiment') != grounded_count.EXPERIMENT or baseline.get('phase') != 'inner':
        raise ValueError('baseline report is not grounded-count v2 inner evaluation')
    baseline_outcomes = {row['record_id']: row for row in baseline['outcomes']}
    neural_actions = [tuple(baseline_outcomes[record_id]['generated_actions']) for record_id in evaluation_ids]

    compilations = [
        compile_semantic_rule(
            command=record.command,
            platform=record.platform.value,
            instruction=record.instruction,
            context=record.context,
        )
        for record in evaluation_records
    ]
    matched = [item for item in compilations if item is not None]
    encoded = iter(client.encode([item.document for item in matched]))
    generated = [next(encoded) if item is not None else neural for item, neural in zip(compilations, neural_actions, strict=True)]
    metrics, outcomes = _evaluate(
        evaluation_records,
        evaluation_examples,
        generated,
        client,
    )

    rule_indices: dict[str, list[int]] = defaultdict(list)
    fallback_indices: list[int] = []
    for index, item in enumerate(compilations):
        if item is None:
            fallback_indices.append(index)
        else:
            rule_indices[item.rule_id].append(index)
    per_rule = {
        rule_id: _subset_metrics(indices, evaluation_records, evaluation_examples, generated, client)
        for rule_id, indices in sorted(rule_indices.items())
    }
    per_rule['fallback'] = _subset_metrics(
        fallback_indices,
        evaluation_records,
        evaluation_examples,
        generated,
        client,
    )
    count = len(evaluation_records)
    gate_passed = (
        metrics['rust_valid'] / count >= 0.90
        and metrics['first_command_match'] / count >= 0.80
        and metrics['reference_accepted'] >= 5
    )
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'source_experiment': grounded_count.EXPERIMENT,
        'source_best_epoch': baseline['best_epoch'],
        'evaluation_records': count,
        'rule_coverage': len(matched),
        'rust_encoded_rules': len(matched),
        'gate_passed': gate_passed,
        'floors': {
            'minimum_rust_valid': math.ceil(0.90 * count),
            'minimum_first_command_match': math.ceil(0.80 * count),
            'minimum_reference_accepted': 5,
        },
        'metrics': metrics,
        'per_rule': per_rule,
        'outcomes': outcomes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'outcomes'}, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit('compiler-rule inner gate failed')


def _subset_metrics(indices, records, examples, generated, client):
    if not indices:
        return {'examples': 0, 'exact_actions': 0, 'rust_valid': 0, 'first_command_match': 0, 'reference_accepted': 0}
    metrics, _ = _evaluate(
        [records[index] for index in indices],
        [examples[index] for index in indices],
        [generated[index] for index in indices],
        client,
    )
    return metrics


if __name__ == '__main__':
    main()
