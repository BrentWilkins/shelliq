#!/usr/bin/env python3
"""Run the one-shot outer-validation gate for the frozen compiler hybrid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_semantic_action_candidate as candidate
import run_semantic_action_grounded_count_v2 as grounded_count
import run_semantic_action_typed as typed
import torch
from evaluate_semantic_action_compiler_rules import _subset_metrics
from run_semantic_action_decoder import _evaluate, _ordered_batches
from torch import nn

from shelliq_training.semantic_action_compiler_rules import compile_semantic_rule

EXPERIMENT = 'semantic-action-compiler-rules-v2'
PHASE = 'outer'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--outer-manifest', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--copy-oracle', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--inner-report', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    return parser.parse_args()


def load_inner_authorization(path: Path) -> tuple[dict[str, object], int, int]:
    report = json.loads(path.read_text())
    if report.get('experiment') != EXPERIMENT or not report.get('gate_passed'):
        raise ValueError('compiler-rule inner gate did not pass')
    epochs = report.get('source_best_epoch')
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError('compiler-rule inner report has no valid source_best_epoch')
    seed = report.get('source_seed', typed.v1.INNER_SEED)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('compiler-rule inner report has no valid source_seed')
    return report, epochs, seed


def outer_gate_floors(count: int) -> dict[str, int]:
    if count <= 0:
        raise ValueError('outer validation must contain records')
    return {
        'minimum_rust_valid': math.ceil(0.90 * count),
        'minimum_first_command_match': math.ceil(0.80 * count),
        'minimum_reference_accepted': math.ceil(0.10 * count),
    }


def main() -> None:
    args = parse_args()
    authorization, epochs, seed = load_inner_authorization(args.inner_report)
    manifest = json.loads(args.manifest.read_text())
    train_ids = manifest['outer_splits']['train']['record_ids']
    evaluation_ids = manifest['outer_splits']['validation']['record_ids']

    records, _, examples, tokenizer, client, grammar, _, global_roles = typed.load_typed(args, train_ids)
    by_record = {record.record_id: record for record in records}
    train_examples = [examples[item] for item in train_ids]
    evaluation_examples = [examples[item] for item in evaluation_ids]
    evaluation_records = [by_record[item] for item in evaluation_ids]

    device = typed.v1._device(args.device)
    random.seed(seed)
    torch.manual_seed(seed)
    model = grounded_count.build_model(tokenizer).to(device)
    optimizer = candidate.optimizer_for(model)
    roles = len(typed.byte_roles(grammar))
    history: list[dict[str, object]] = []
    started = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        weighted = 0.0
        tokens_seen = 0
        for chosen in _ordered_batches(train_examples, args.batch_size, seed + epoch):
            batch = typed.make_batch(chosen, tokenizer, roles).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == 'cuda',
            ):
                loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite typed loss in epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = int((batch.labels != -100).sum())
            weighted += float(loss.detach()) * tokens
            tokens_seen += tokens
        row: dict[str, object] = {
            'epoch': epoch,
            'train_loss': weighted / tokens_seen,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    neural_actions = typed.generate(
        model,
        evaluation_examples,
        tokenizer,
        grammar,
        device,
    )
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
        rule_id: _subset_metrics(
            indices,
            evaluation_records,
            evaluation_examples,
            generated,
            client,
        )
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
    floors = outer_gate_floors(count)
    gate_passed = (
        metrics['rust_valid'] >= floors['minimum_rust_valid']
        and metrics['first_command_match'] >= floors['minimum_first_command_match']
        and metrics['reference_accepted'] >= floors['minimum_reference_accepted']
    )
    report: dict[str, object] = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': PHASE,
        'gate_passed': gate_passed,
        'source_experiment': grounded_count.EXPERIMENT,
        'source_epochs': epochs,
        'source_seed': seed,
        'inner_authorization_sha256': hashlib.sha256(args.inner_report.read_bytes()).hexdigest(),
        'train_records': len(train_examples),
        'evaluation_records': count,
        'rule_coverage': len(matched),
        'rust_encoded_rules': len(matched),
        'elapsed_seconds': time.monotonic() - started,
        'floors': floors,
        'metrics': metrics,
        'per_rule': per_rule,
        'history': history,
        'outcomes': outcomes,
    }
    typed.save_checkpoint(args.checkpoint, model, global_roles, report)
    typed.write_json(args.report, report)
    print(
        json.dumps(
            {
                'gate_passed': gate_passed,
                'rule_coverage': len(matched),
                **metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not gate_passed:
        raise SystemExit('compiler-rule outer gate failed')


if __name__ == '__main__':
    main()
