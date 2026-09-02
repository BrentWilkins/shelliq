#!/usr/bin/env python3
"""Train/calibrate or test the frozen documentation cross-encoder experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from shelliq_training.documentation_cross_encoder import (
    EXPERIMENT,
    EXPERIMENT_V2,
    REVISION,
    FrozenCodeT5CrossEncoder,
    RankCase,
    candidate_pool,
    compile_selected,
    embed_texts,
    load_templates,
    pair_text,
    rank_pool,
    threshold_for_complete_abstention,
    tokenizer,
)
from shelliq_training.semantic_actions import SemanticActionClient

LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SEED = 20260901


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'test'))
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--train-data', type=Path)
    parser.add_argument('--development-data', type=Path)
    parser.add_argument('--test-data', type=Path)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='auto')
    parser.add_argument('--encoder-batch-size', type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f'report already exists: {args.report}')
    if args.phase == 'train' and args.checkpoint.exists():
        raise FileExistsError(f'checkpoint already exists: {args.checkpoint}')
    manifest = json.loads(args.manifest.read_text())
    _validate_manifest(manifest, args.documentation_index)
    device = _device(args.device)
    if args.phase == 'train':
        if args.train_data is None or args.development_data is None:
            raise ValueError('train phase requires --train-data and --development-data')
        train(args, manifest, device)
    else:
        if args.test_data is None:
            raise ValueError('test phase requires --test-data')
        test(args, manifest, device)


def train(args: argparse.Namespace, manifest: dict[str, object], device: torch.device) -> None:
    _validate_split_hash(args.train_data, manifest, 'train')
    _validate_split_hash(args.development_data, manifest, 'development')
    train_cases = _load_cases(args.train_data)
    development_cases = _load_cases(args.development_data)
    typed_by_id, raw_by_id, grouped = load_templates(args.documentation_index)
    train_commands = tuple(manifest['splits']['train']['commands'])
    development_commands = tuple(manifest['splits']['development']['commands'])
    _assert_cases(train_cases, train_commands)
    _assert_cases(development_cases, development_commands)

    torch.manual_seed(SEED)
    model = FrozenCodeT5CrossEncoder().to(device)
    tokenizer_ = tokenizer()
    actions = SemanticActionClient(args.actions)

    train_texts, labels = _training_pairs(train_cases, train_commands, grouped, raw_by_id)
    started = time.monotonic()
    embeddings, train_truncated = embed_texts(
        model,
        tokenizer_,
        train_texts,
        device=device,
        batch_size=args.encoder_batch_size,
    )
    development_cache, development_truncated = _embed_pools(
        model,
        tokenizer_,
        development_cases,
        development_commands,
        grouped,
        raw_by_id,
        device,
        args.encoder_batch_size,
    )

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=LEARNING_RATE, weight_decay=0)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(SEED)
    best: dict[str, object] | None = None
    history: list[dict[str, object]] = []
    for epoch in range(1, EPOCHS + 1):
        model.head.train()
        permutation = torch.randperm(len(labels_tensor), generator=generator)
        losses: list[float] = []
        for start in range(0, len(permutation), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            logits = model.score_embeddings(embeddings[indices].to(device))
            loss = loss_fn(logits, labels_tensor[indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = _evaluate_cached(
            model,
            development_cases,
            development_cache,
            typed_by_id,
            actions,
            calibrate=True,
        )
        row = {'epoch': epoch, 'train_loss': statistics.fmean(losses), **metrics}
        history.append(row)
        selection = (
            float(metrics['macro_accuracy']),
            int(metrics['sufficient_ready']),
            -epoch,
        )
        if best is None or selection > best['selection']:
            best = {
                'selection': selection,
                'epoch': epoch,
                'head_state_dict': copy.deepcopy(model.head.state_dict()),
                'metrics': metrics,
            }

    assert best is not None
    model.head.load_state_dict(best['head_state_dict'])
    metrics = dict(best['metrics'])
    authorized = (
        int(metrics['sufficient_ready']) >= 45
        and float(metrics['sufficient_ready_rate']) >= 0.70
        and int(metrics['insufficient_abstained']) == len(development_cases)
    )
    checkpoint = {
        'schema_version': 1,
        'experiment': manifest['experiment'],
        'codet5_revision': REVISION,
        'seed': SEED,
        'selected_epoch': best['epoch'],
        'threshold': metrics['threshold'],
        'development_authorized': authorized,
        'manifest_sha256': _sha256(args.manifest),
        'documentation_index_sha256': _sha256(args.documentation_index),
        'head_state_dict': {name: value.cpu() for name, value in model.head.state_dict().items()},
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'development',
        'development_authorized': authorized,
        'selected_epoch': best['epoch'],
        'train_pairs': len(labels),
        'positive_train_pairs': sum(labels),
        'train_truncated': train_truncated,
        'development_truncated': development_truncated,
        'embedding_seconds': time.monotonic() - started,
        'metrics': metrics,
        'history': history,
        'checkpoint_sha256': _sha256(args.checkpoint),
    }
    _write_report(args.report, report)
    print(
        json.dumps(
            {key: report[key] for key in ('development_authorized', 'selected_epoch', 'metrics')}, indent=2, sort_keys=True
        )
    )
    if not authorized:
        raise SystemExit('development authorization gate failed; sealed test remains unscored')


def test(args: argparse.Namespace, manifest: dict[str, object], device: torch.device) -> None:
    _validate_split_hash(args.test_data, manifest, 'test')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if not checkpoint.get('development_authorized'):
        raise ValueError('checkpoint lacks development authorization')
    if checkpoint.get('manifest_sha256') != _sha256(args.manifest):
        raise ValueError('checkpoint manifest hash mismatch')
    cases = _load_cases(args.test_data)
    commands = tuple(manifest['splits']['test']['commands'])
    _assert_cases(cases, commands)
    typed_by_id, raw_by_id, grouped = load_templates(args.documentation_index)
    actions = SemanticActionClient(args.actions)
    model = FrozenCodeT5CrossEncoder().to(device)
    model.head.load_state_dict(checkpoint['head_state_dict'])
    tokenizer_ = tokenizer()
    threshold = float(checkpoint['threshold'])

    outcomes: list[dict[str, object]] = []
    latencies: list[float] = []
    truncated = 0
    for case_index, case in enumerate(cases):
        started = time.perf_counter()
        sufficient_ids = candidate_pool(case, commands, grouped, sufficient=True)
        sufficient_embeddings, count = embed_texts(
            model,
            tokenizer_,
            [pair_text(case.query, raw_by_id[record_id]) for record_id in sufficient_ids],
            device=device,
            batch_size=args.encoder_batch_size,
        )
        truncated += count
        sufficient = rank_pool(model, sufficient_embeddings, sufficient_ids, case.source_record_id)
        compiled = (
            sufficient.top_record_id == case.source_record_id
            and sufficient.top_score >= threshold
            and compile_selected(case, sufficient.top_record_id, typed_by_id, actions)
        )
        elapsed = (time.perf_counter() - started) * 1_000
        if case_index > 0:
            latencies.append(elapsed)

        insufficient_ids = candidate_pool(case, commands, grouped, sufficient=False)
        insufficient_embeddings, count = embed_texts(
            model,
            tokenizer_,
            [pair_text(case.query, raw_by_id[record_id]) for record_id in insufficient_ids],
            device=device,
            batch_size=args.encoder_batch_size,
        )
        truncated += count
        insufficient = rank_pool(model, insufficient_embeddings, insufficient_ids, case.source_record_id)
        abstained = insufficient.top_score < threshold
        outcomes.append(
            {
                'command': case.command,
                'source_record_id': case.source_record_id,
                'sufficient_top_record_id': sufficient.top_record_id,
                'sufficient_top_score': sufficient.top_score,
                'sufficient_ready': compiled,
                'insufficient_top_record_id': insufficient.top_record_id,
                'insufficient_top_score': insufficient.top_score,
                'insufficient_abstained': abstained,
                'latency_ms': elapsed,
            }
        )

    sufficient_ready = sum(bool(row['sufficient_ready']) for row in outcomes)
    insufficient_abstained = sum(bool(row['insufficient_abstained']) for row in outcomes)
    p95 = _percentile95(latencies)
    passed = sufficient_ready >= 90 and insufficient_abstained == 128 and p95 <= 5_000
    report = {
        'schema_version': 1,
        'experiment': manifest['experiment'],
        'phase': 'test',
        'gate_passed': passed,
        'threshold': threshold,
        'metrics': {
            'cases': len(cases),
            'sufficient_ready': sufficient_ready,
            'sufficient_ready_rate': sufficient_ready / len(cases),
            'insufficient_abstained': insufficient_abstained,
            'insufficient_abstain_rate': insufficient_abstained / len(cases),
            'truncated_pairs': truncated,
            'warm_cpu_p95_ms': p95,
        },
        'outcomes': outcomes,
        'checkpoint_sha256': _sha256(args.checkpoint),
    }
    _write_report(args.report, report)
    print(json.dumps({'gate_passed': passed, 'metrics': report['metrics']}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit('sealed test gate failed')


def _training_pairs(cases, commands, grouped, raw_by_id):
    texts: list[str] = []
    labels: list[int] = []
    command_indices = {command: index for index, command in enumerate(commands)}
    for case in cases:
        own = list(grouped[case.command])
        position = own.index(case.source_record_id)
        same_negative = own[(position + 1) % len(own)]
        cross_commands = [commands[(command_indices[case.command] + offset) % len(commands)] for offset in (1, 2)]
        record_ids = [case.source_record_id, same_negative, *(grouped[command][0] for command in cross_commands)]
        for index, record_id in enumerate(record_ids):
            texts.append(pair_text(case.query, raw_by_id[record_id]))
            labels.append(1 if index == 0 else 0)
    return texts, labels


def _embed_pools(model, tokenizer_, cases, commands, grouped, raw_by_id, device, batch_size):
    cache = []
    truncated = 0
    for case in cases:
        sufficient_ids = candidate_pool(case, commands, grouped, sufficient=True)
        insufficient_ids = candidate_pool(case, commands, grouped, sufficient=False)
        all_ids = (*sufficient_ids, *insufficient_ids)
        embeddings, count = embed_texts(
            model,
            tokenizer_,
            [pair_text(case.query, raw_by_id[record_id]) for record_id in all_ids],
            device=device,
            batch_size=batch_size,
        )
        truncated += count
        cache.append((sufficient_ids, insufficient_ids, embeddings[: len(sufficient_ids)], embeddings[len(sufficient_ids) :]))
    return cache, truncated


def _evaluate_cached(model, cases, cache, typed_by_id, actions, *, calibrate):
    sufficient_pools = []
    insufficient_pools = []
    for case, (sufficient_ids, insufficient_ids, sufficient_embeddings, insufficient_embeddings) in zip(
        cases, cache, strict=True
    ):
        sufficient_pools.append(rank_pool(model, sufficient_embeddings, sufficient_ids, case.source_record_id))
        insufficient_pools.append(rank_pool(model, insufficient_embeddings, insufficient_ids, case.source_record_id))
    threshold = threshold_for_complete_abstention([pool.top_score for pool in insufficient_pools]) if calibrate else 0.0
    sufficient_ready = sum(
        pool.top_record_id == case.source_record_id
        and pool.top_score >= threshold
        and compile_selected(case, pool.top_record_id, typed_by_id, actions)
        for case, pool in zip(cases, sufficient_pools, strict=True)
    )
    insufficient_abstained = sum(pool.top_score < threshold for pool in insufficient_pools)
    total = len(cases)
    return {
        'threshold': threshold,
        'sufficient_ready': sufficient_ready,
        'sufficient_ready_rate': sufficient_ready / total,
        'insufficient_abstained': insufficient_abstained,
        'insufficient_abstain_rate': insufficient_abstained / total,
        'macro_accuracy': (sufficient_ready + insufficient_abstained) / (2 * total),
    }


def _load_cases(path: Path) -> list[RankCase]:
    cases = []
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        if raw.get('schema_version') != 1:
            raise ValueError(f'{path}: unsupported case schema')
        cases.append(RankCase(raw['command'], raw['source_record_id'], raw['query']))
    return cases


def _assert_cases(cases, commands):
    if {case.command for case in cases} != set(commands):
        raise ValueError('case commands do not match manifest split')


def _validate_manifest(manifest, documentation_index):
    if manifest.get('experiment') not in {EXPERIMENT, EXPERIMENT_V2} or manifest.get('schema_version') != 1:
        raise ValueError('unexpected cross-encoder manifest')
    if manifest.get('documentation_index_sha256') != _sha256(documentation_index):
        raise ValueError('documentation index hash mismatch')
    splits = manifest['splits']
    if manifest.get('experiment') == EXPERIMENT_V2:
        if set(splits) != {'test'} or len(set(splits['test']['commands'])) != 128:
            raise ValueError('invalid v2 test manifest')
        return
    sets = {name: set(splits[name]['commands']) for name in ('train', 'development', 'test')}
    if sets['train'] & sets['development'] or sets['train'] & sets['test'] or sets['development'] & sets['test']:
        raise ValueError('command-disjoint split invariant failed')


def _validate_split_hash(path, manifest, split):
    if path is None or _sha256(path) != manifest['splits'][split]['sha256']:
        raise ValueError(f'{split} data hash mismatch')


def _device(requested):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return torch.device(requested)


def _percentile95(values):
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)] if ordered else 0.0


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
