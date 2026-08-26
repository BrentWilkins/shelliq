#!/usr/bin/env python3
"""Attribute typed-slot errors with ranks and gold-decision counterfactuals."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_semantic_action_candidate as candidate
import run_semantic_action_decoder as v1
import run_semantic_action_typed as typed
import torch

from shelliq_training.semantic_action_typed import (
    CANDIDATE_IGNORE_INDEX,
    action_words,
    byte_roles,
    command_argument_counts,
)

EXPERIMENT = 'semantic-action-typed-attribution-v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--outer-manifest', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--copy-oracle', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, _, _, _, _, manifest = candidate.load_inputs(args)
    train_ids = manifest['inner_splits']['train']['record_ids']
    evaluation_ids = manifest['inner_splits']['validation']['record_ids']
    records, _, examples, tokenizer, client, grammar, _, _ = typed.load_typed(args, train_ids)
    by_record = {record.record_id: record for record in records}
    evaluation_examples = [examples[item] for item in evaluation_ids]
    evaluation_records = [by_record[item] for item in evaluation_ids]
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('experiment') != typed.EXPERIMENT:
        raise ValueError('typed checkpoint belongs to another experiment')
    if checkpoint.get('metadata', {}).get('best_epoch') != 7:
        raise ValueError('typed checkpoint is not the frozen epoch-7 selection')
    device = v1._device(args.device)
    model = typed.build_model(tokenizer).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    roles = byte_roles(grammar)
    role_metrics: dict[str, dict[str, float]] = defaultdict(_rank_row)
    provenance_metrics: dict[str, dict[str, float]] = defaultdict(_rank_row)
    count_metrics = _rank_row()
    fully_covered = []
    started = time.monotonic()

    with torch.no_grad():
        for index, example in enumerate(evaluation_examples, start=1):
            batch = typed.make_batch([example], tokenizer, len(roles)).to(device)
            output = model(batch)
            covered = True
            for position in range(batch.labels.shape[1]):
                role = int(batch.word_role_labels[0, position])
                if role == CANDIDATE_IGNORE_INDEX:
                    continue
                label = int(batch.candidate_labels[0, position])
                if label == CANDIDATE_IGNORE_INDEX:
                    covered = False
                    continue
                mask = batch.candidate_role_mask[0, role]
                logits = output.candidate_logits[0, position].masked_fill(~mask, -torch.inf)
                rank = 1 + int((logits > logits[label]).sum())
                _add_rank(role_metrics[roles[role]], rank)
                feature = batch.candidate_features[0, label].bool().tolist()
                provenance = '+'.join(
                    name for name, active in zip(('local', 'derived', 'global'), feature, strict=True) if active
                )
                _add_rank(provenance_metrics[provenance or 'none'], rank)
            counted = (batch.argument_count_labels[0] != -100).nonzero().flatten()
            for position_tensor in counted:
                position = int(position_tensor)
                label = int(batch.argument_count_labels[0, position])
                logits = output.argument_count_logits[0, position]
                rank = 1 + int((logits > logits[label]).sum())
                _add_rank(count_metrics, rank)
                count_metrics['absolute_error'] += abs(int(logits.argmax()) - label)
            if covered:
                fully_covered.append(index - 1)
            if index % 10 == 0:
                print(f'ranked={index}/{len(evaluation_examples)}', flush=True)

    gold_counts = _counterfactual(
        model,
        evaluation_examples,
        evaluation_records,
        tokenizer,
        grammar,
        client,
        device,
        force_counts=True,
        force_words=False,
    )
    covered_examples = [evaluation_examples[index] for index in fully_covered]
    covered_records = [evaluation_records[index] for index in fully_covered]
    gold_words = _counterfactual(
        model,
        covered_examples,
        covered_records,
        tokenizer,
        grammar,
        client,
        device,
        force_counts=False,
        force_words=True,
    )
    gold_both = _counterfactual(
        model,
        covered_examples,
        covered_records,
        tokenizer,
        grammar,
        client,
        device,
        force_counts=True,
        force_words=True,
    )
    baseline = json.loads(args.baseline_report.read_text())
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'source_experiment': typed.EXPERIMENT,
        'evaluation_records': len(evaluation_examples),
        'fully_candidate_covered_records': len(fully_covered),
        'baseline_metrics': baseline['metrics'],
        'candidate_ranks_by_role': {name: _finish(row) for name, row in sorted(role_metrics.items())},
        'candidate_ranks_by_provenance': {name: _finish(row) for name, row in sorted(provenance_metrics.items())},
        'argument_count_ranks': _finish(count_metrics, include_error=True),
        'gold_count_metrics': gold_counts,
        'fully_covered_gold_word_metrics': gold_words,
        'fully_covered_gold_count_and_word_metrics': gold_both,
        'elapsed_seconds': time.monotonic() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


def _rank_row() -> dict[str, float]:
    return {'total': 0, 'top1': 0, 'top4': 0, 'top8': 0, 'reciprocal_rank': 0.0, 'absolute_error': 0}


def _add_rank(row: dict[str, float], rank: int) -> None:
    row['total'] += 1
    row['top1'] += int(rank <= 1)
    row['top4'] += int(rank <= 4)
    row['top8'] += int(rank <= 8)
    row['reciprocal_rank'] += 1 / rank


def _finish(row: dict[str, float], *, include_error: bool = False) -> dict[str, int | float]:
    total = int(row['total'])
    result: dict[str, int | float] = {
        'total': total,
        'top1': int(row['top1']),
        'top1_rate': row['top1'] / total if total else 0.0,
        'top4': int(row['top4']),
        'top4_rate': row['top4'] / total if total else 0.0,
        'top8': int(row['top8']),
        'top8_rate': row['top8'] / total if total else 0.0,
        'mean_reciprocal_rank': row['reciprocal_rank'] / total if total else 0.0,
    }
    if include_error:
        result['top1_mean_absolute_error'] = row['absolute_error'] / total if total else 0.0
    return result


@torch.no_grad()
def _counterfactual(
    model,
    examples,
    records,
    tokenizer,
    grammar,
    client,
    device,
    *,
    force_counts: bool,
    force_words: bool,
):
    generated = []
    roles = byte_roles(grammar)
    for example, record in zip(examples, records, strict=True):
        batch = typed.make_batch([example], tokenizer, len(roles)).to(device)
        counts = command_argument_counts(json.loads(record.response)) if force_counts else None
        words = action_words(example.action_ids, grammar) if force_words else None
        generated.extend(
            model.generate_typed_beam(
                batch,
                grammar,
                roles,
                beam_width=typed.BEAM_WIDTH,
                forced_argument_counts=counts,
                forced_words=words,
            )
        )
    metrics, _ = v1._evaluate(records, examples, generated, client)
    return metrics


if __name__ == '__main__':
    main()
