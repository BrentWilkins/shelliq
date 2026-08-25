#!/usr/bin/env python3
"""Run the gated CodeT5-encoder semantic-action decoder experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoModelForSeq2SeqLM, RobertaTokenizer

from shelliq_training.compiler_comparison import load_comparison_manifest, require_matching_dataset
from shelliq_training.data import SFTRecord, Split, load_semantic_jsonl
from shelliq_training.prompt import PromptContract
from shelliq_training.semantic_action_model import (
    LABEL_IGNORE_INDEX,
    ActionDecoderConfig,
    SemanticActionModel,
    collate_actions,
)
from shelliq_training.semantic_actions import ActionExample, ActionGrammar, SemanticActionClient, action_examples
from shelliq_training.teacher_verification import RustValidation, verify_against_reference

EXPERIMENT = 'semantic-action-decoder-v1'
CODET5_MODEL_ID = 'Salesforce/codet5-small'
CODET5_REVISION = 'b1ee9570c289f21b5922b9c768a1ce12957bf968'
PROMPT_CONTRACT = PromptContract.CONTEXT_AUTHORITATIVE_V1
INNER_SEED = 20260826
SOURCE_LENGTH = 256
TARGET_LENGTH = 192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'inspect', 'cpu', 'overfit', 'inner', 'outer', 'compare'))
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--outer-manifest', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--actions', type=Path, default=Path('../target/debug/semantic-actions'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--inner-report', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=INNER_SEED)
    parser.add_argument('--log-interval', type=int, default=100)
    return parser.parse_args()


def load_tokenizer() -> RobertaTokenizer:
    vocab = hf_hub_download(CODET5_MODEL_ID, 'vocab.json', revision=CODET5_REVISION, local_files_only=True)
    merges = hf_hub_download(CODET5_MODEL_ID, 'merges.txt', revision=CODET5_REVISION, local_files_only=True)
    return RobertaTokenizer(vocab=vocab, merges=merges)


def build_model(*, dropout: float = 0.1) -> SemanticActionModel:
    pretrained = AutoModelForSeq2SeqLM.from_pretrained(CODET5_MODEL_ID, revision=CODET5_REVISION, local_files_only=True)
    encoder = pretrained.get_encoder()
    config = ActionDecoderConfig(dropout=dropout)
    model = SemanticActionModel(encoder, config)
    del pretrained
    encoder_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if encoder_parameters != 35_316_480:
        raise ValueError(f'unexpected CodeT5 encoder parameter count: {encoder_parameters}')
    if not 52_000_000 <= total_parameters <= 53_000_000:
        raise ValueError(f'unexpected hybrid parameter count: {total_parameters}')
    return model


def load_inputs(args: argparse.Namespace):
    outer = load_comparison_manifest(args.outer_manifest)
    require_matching_dataset(args.semantic_dataset, outer)
    records = load_semantic_jsonl(args.semantic_dataset)
    frozen = outer.records_by_split(records)
    client = SemanticActionClient(args.actions)
    rust_manifest = client.manifest()
    grammar = ActionGrammar(rust_manifest)
    tokenizer = load_tokenizer()
    encoded = client.encode([json.loads(record.response) for record in records])
    examples = action_examples(
        records,
        encoded,
        tokenizer,
        source_length=SOURCE_LENGTH,
        target_length=TARGET_LENGTH,
        prompt_contract=PROMPT_CONTRACT,
    )
    by_id = {example.record_id: example for example in examples}
    split_examples = {split: [by_id[record.record_id] for record in selected] for split, selected in frozen.items()}
    return records, frozen, split_examples, tokenizer, client, grammar, rust_manifest


def prepare(args: argparse.Namespace) -> None:
    records, frozen, examples, _, client, grammar, rust_manifest = load_inputs(args)
    inner_train, inner_validation = _command_disjoint_inner(frozen[Split.TRAIN], 0.15, INNER_SEED)
    encoded = client.encode([json.loads(record.response) for record in records])
    canonical = client.decode(encoded)
    if not all(item.valid for item in canonical):
        raise ValueError('Rust action decode failed during preparation')
    for index, sequence in enumerate(encoded, start=1):
        cursor = grammar.cursor()
        for token in sequence:
            if token not in cursor.allowed():
                raise ValueError(f'Python grammar rejected Rust action at row {index}')
            cursor.advance(token)
        if not cursor.complete:
            raise ValueError(f'Python grammar did not complete Rust action at row {index}')
    data = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'dataset_sha256': hashlib.sha256(args.semantic_dataset.read_bytes()).hexdigest(),
        'outer_manifest_sha256': hashlib.sha256(args.outer_manifest.read_bytes()).hexdigest(),
        'action_manifest_sha256': _json_sha256(rust_manifest),
        'codet5_model_id': CODET5_MODEL_ID,
        'codet5_revision': CODET5_REVISION,
        'prompt_contract': PROMPT_CONTRACT.value,
        'source_length': SOURCE_LENGTH,
        'target_length': TARGET_LENGTH,
        'action_vocab_size': rust_manifest['vocab_size'],
        'action_lengths': {
            'maximum': max(map(len, encoded)),
            'over_limit': sum(len(row) > TARGET_LENGTH for row in encoded),
        },
        'outer_splits': {split.value: {'record_ids': [record.record_id for record in frozen[split]]} for split in Split},
        'inner_seed': INNER_SEED,
        'inner_splits': {
            'train': {'record_ids': [record.record_id for record in inner_train]},
            'validation': {'record_ids': [record.record_id for record in inner_validation]},
        },
        'maximum_source_tokens': max(len(example.source_ids) for split in examples.values() for example in split),
    }
    _write_json(args.manifest, data)
    print(json.dumps(_manifest_summary(data), indent=2, sort_keys=True))


def inspect(args: argparse.Namespace) -> None:
    _, _, examples, _, _, _, rust_manifest = load_inputs(args)
    model = build_model()
    print(
        json.dumps(
            {
                'encoder_parameters': sum(parameter.numel() for parameter in model.encoder.parameters()),
                'decoder_parameters': sum(parameter.numel() for parameter in model.parameters())
                - sum(parameter.numel() for parameter in model.encoder.parameters()),
                'total_parameters': sum(parameter.numel() for parameter in model.parameters()),
                'action_vocab_size': rust_manifest['vocab_size'],
                'maximum_source_tokens': max(len(item.source_ids) for group in examples.values() for item in group),
                'maximum_target_actions': max(len(item.action_ids) for group in examples.values() for item in group),
            },
            indent=2,
            sort_keys=True,
        )
    )


def cpu_gate(args: argparse.Namespace) -> None:
    _, _, examples, tokenizer, _, grammar, _ = load_inputs(args)
    torch.manual_seed(args.seed)
    model = build_model(dropout=0).cpu()
    selected = [examples[Split.TRAIN][0]]
    batch = _batch(selected, tokenizer).to(torch.device('cpu'))
    initial = float(model.loss(batch).detach())
    model.loss(batch).backward()
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise RuntimeError('non-finite CPU gradient')
    # Masked generation is exercised portably on the same implementation by the
    # focused tiny-model test; advance the real grammar here to bind the manifest.
    cursor = grammar.cursor()
    for token in selected[0].action_ids:
        if token not in cursor.allowed():
            raise RuntimeError('Rust target is rejected by Python manifest interpreter')
        cursor.advance(token)
    if not cursor.complete:
        raise RuntimeError('real action target did not complete the manifest grammar')
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'cpu',
        'record_id': selected[0].record_id,
        'loss': initial,
        'finite_gradients': True,
        'manifest_target_complete': True,
        'focused_masked_generation_test': 'training/tests/test_semantic_action_model.py',
    }
    if args.report is None:
        raise ValueError('--report is required')
    _write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def overfit(args: argparse.Namespace) -> None:
    records, frozen, examples, tokenizer, client, grammar, _ = load_inputs(args)
    selected_records = _select_distinct_commands(frozen[Split.TRAIN], 8, args.seed)
    selected_ids = {record.record_id for record in selected_records}
    selected = [example for example in examples[Split.TRAIN] if example.record_id in selected_ids]
    selected.sort(key=lambda item: next(i for i, record in enumerate(selected_records) if record.record_id == item.record_id))
    device = _device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(dropout=0).to(device)
    optimizer = _optimizer(model)
    batch = _batch(selected, tokenizer).to(device)
    initial_loss = 0.0
    final_loss = 0.0
    exact = 0
    started = time.monotonic()
    completed_steps = args.steps
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss = model.loss(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite overfit loss at step {step}')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        if step == 1:
            initial_loss = final_loss
        should_evaluate = step == args.steps or step >= 100 and step % args.log_interval == 0
        if should_evaluate:
            generated = _generate(model, selected, tokenizer, grammar, device)
            exact = sum(actual == expected.action_ids for actual, expected in zip(generated, selected, strict=True))
            print(f'step={step} loss={final_loss:.6f} exact={exact}/8', flush=True)
            if exact == len(selected):
                completed_steps = step
                break
    generated = _generate(model, selected, tokenizer, grammar, device)
    metrics, outcomes = _evaluate(selected_records, selected, generated, client)
    passed = metrics['exact_actions'] == 8 and metrics['rust_valid'] == 8 and metrics['reference_accepted'] == 8
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'overfit',
        'gate_passed': passed,
        'seed': args.seed,
        'step_limit': args.steps,
        'completed_steps': completed_steps,
        'batch_size': len(selected),
        'encoder_learning_rate': 5e-5,
        'decoder_learning_rate': 5e-4,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'elapsed_seconds': time.monotonic() - started,
        'metrics': metrics,
        'outcomes': outcomes,
    }
    _require_output(args)
    _save_checkpoint(args.checkpoint, model, report)
    _write_json(args.report, report)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit('overfit gate failed')


def inner(args: argparse.Namespace) -> None:
    run_development(args, phase='inner')


def outer(args: argparse.Namespace) -> None:
    run_development(args, phase='outer')


def compare(args: argparse.Namespace) -> None:
    run_development(args, phase='compare')


def run_development(args: argparse.Namespace, *, phase: str) -> None:
    manifest = json.loads(args.manifest.read_text())
    records, frozen, examples, tokenizer, client, grammar, _ = load_inputs(args)
    by_record = {record.record_id: record for record in records}
    by_example = {example.record_id: example for group in examples.values() for example in group}
    if phase == 'inner':
        train_ids = manifest['inner_splits']['train']['record_ids']
        evaluation_ids = manifest['inner_splits']['validation']['record_ids']
        epochs = args.epochs
        select = True
    else:
        if args.inner_report is None:
            raise ValueError('--inner-report is required for outer/compare')
        inner_report = json.loads(args.inner_report.read_text())
        if not inner_report.get('gate_passed'):
            raise ValueError('inner gate did not pass')
        train_ids = manifest['outer_splits']['train']['record_ids']
        evaluation_split = 'validation' if phase == 'outer' else 'test'
        evaluation_ids = manifest['outer_splits'][evaluation_split]['record_ids']
        epochs = int(inner_report['best_epoch'])
        select = False
    train_examples = [by_example[record_id] for record_id in train_ids]
    evaluation_examples = [by_example[record_id] for record_id in evaluation_ids]
    evaluation_records = [by_record[record_id] for record_id in evaluation_ids]
    device = _device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model().to(device)
    optimizer = _optimizer(model)
    history = []
    best_loss = float('inf')
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        weighted = 0.0
        token_count = 0
        for chosen in _ordered_batches(train_examples, args.batch_size, args.seed + epoch):
            batch = _batch(chosen, tokenizer).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss in epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = int((batch.labels != LABEL_IGNORE_INDEX).sum())
            weighted += float(loss.detach()) * tokens
            token_count += tokens
        validation_loss = _mean_loss(model, evaluation_examples, tokenizer, args.batch_size, device)
        row = {'epoch': epoch, 'train_loss': weighted / token_count, 'validation_loss': validation_loss}
        history.append(row)
        if select and validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            _save_checkpoint(args.checkpoint, model, {'phase': phase, 'best_epoch': epoch, 'best_loss': best_loss})
        print(
            f'phase={phase} epoch={epoch}/{epochs} train_loss={row["train_loss"]:.6f} '
            f'evaluation_loss={validation_loss:.6f} best_epoch={best_epoch}',
            flush=True,
        )
    if select:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved['model_state_dict'])
    else:
        best_epoch = epochs
        best_loss = history[-1]['validation_loss']
        _save_checkpoint(args.checkpoint, model, {'phase': phase, 'epochs': epochs})
    generated = _generate(model, evaluation_examples, tokenizer, grammar, device)
    metrics, outcomes = _evaluate(evaluation_records, evaluation_examples, generated, client)
    count = len(evaluation_examples)
    thresholds = (0.90, 0.80, 0.05) if phase == 'inner' else (0.90, 0.80, 0.10)
    passed = (
        metrics['rust_valid'] / count >= thresholds[0]
        and metrics['first_command_match'] / count >= thresholds[1]
        and metrics['reference_accepted'] / count >= thresholds[2]
    )
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': phase,
        'gate_passed': passed,
        'seed': args.seed,
        'epochs': epochs,
        'best_epoch': best_epoch,
        'best_teacher_forced_action_loss': best_loss,
        'train_records': len(train_examples),
        'evaluation_records': count,
        'elapsed_seconds': time.monotonic() - started,
        'metrics': metrics,
        'history': history,
        'outcomes': outcomes,
    }
    _write_json(args.report, report)
    print(json.dumps({'gate_passed': passed, **metrics}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(f'{phase} gate failed')


def _optimizer(model: SemanticActionModel) -> torch.optim.AdamW:
    decoder = [parameter for name, parameter in model.named_parameters() if not name.startswith('encoder.')]
    return torch.optim.AdamW(
        [
            {'params': model.encoder.parameters(), 'lr': 5e-5},
            {'params': decoder, 'lr': 5e-4},
        ],
        weight_decay=0,
    )


def _batch(examples: Sequence[ActionExample], tokenizer) -> object:
    return collate_actions(
        examples,
        source_length=max(len(item.source_ids) for item in examples),
        target_length=max(len(item.action_ids) for item in examples),
        source_pad_token_id=tokenizer.pad_token_id,
    )


def _ordered_batches(examples: Sequence[ActionExample], batch_size: int, seed: int) -> list[list[ActionExample]]:
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    return [[examples[index] for index in order[offset : offset + batch_size]] for offset in range(0, len(order), batch_size)]


@torch.no_grad()
def _mean_loss(model, examples, tokenizer, batch_size, device) -> float:
    model.eval()
    weighted = 0.0
    count = 0
    for offset in range(0, len(examples), batch_size):
        batch = _batch(examples[offset : offset + batch_size], tokenizer).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss = model.loss(batch)
        tokens = int((batch.labels != LABEL_IGNORE_INDEX).sum())
        weighted += float(loss) * tokens
        count += tokens
    return weighted / count


@torch.no_grad()
def _generate(model, examples, tokenizer, grammar, device) -> list[tuple[int, ...]]:
    model.eval()
    generated = []
    for offset in range(0, len(examples), 8):
        batch = _batch(examples[offset : offset + 8], tokenizer).to(device)
        generated.extend(model.generate(batch.source_ids, batch.source_attention_mask, grammar))
    return generated


def _evaluate(records, examples, generated, client):
    decoded = client.decode(generated)
    metrics = {'examples': len(records), 'exact_actions': 0, 'rust_valid': 0, 'first_command_match': 0, 'reference_accepted': 0}
    outcomes = []
    for record, example, actual, result in zip(records, examples, generated, decoded, strict=True):
        generated_json = json.dumps(result.document, separators=(',', ':')) if result.document is not None else ''
        verification = verify_against_reference(
            generated_json,
            json.loads(record.response),
            RustValidation(result.valid, result.rendered, result.error),
        )
        exact = actual == example.action_ids
        metrics['exact_actions'] += exact
        metrics['rust_valid'] += result.valid
        metrics['first_command_match'] += verification.first_command_match
        metrics['reference_accepted'] += verification.accepted
        outcomes.append(
            {
                'record_id': record.record_id,
                'exact_actions': exact,
                'rust_valid': result.valid,
                'first_command_match': verification.first_command_match,
                'reference_accepted': verification.accepted,
                'failures': list(verification.failures),
                'generated_actions': list(actual),
                'error': result.error,
            }
        )
    return metrics, outcomes


def _select_distinct_commands(records: Sequence[SFTRecord], count: int, seed: int) -> list[SFTRecord]:
    ordered = sorted(records, key=lambda record: hashlib.sha256(f'{seed}\0{record.record_id}'.encode()).digest())
    selected = []
    commands = set()
    for record in ordered:
        if record.command not in commands:
            selected.append(record)
            commands.add(record.command)
            if len(selected) == count:
                return selected
    raise ValueError(f'could not select {count} distinct commands')


def _command_disjoint_inner(
    records: Sequence[SFTRecord], validation_fraction: float, seed: int
) -> tuple[list[SFTRecord], list[SFTRecord]]:
    """Choose a stable command subset whose record count is nearest the requested fraction."""
    if not 0 < validation_fraction < 1:
        raise ValueError('inner validation fraction must be between zero and one')
    groups: dict[str, list[SFTRecord]] = {}
    for record in records:
        groups.setdefault(record.command, []).append(record)
    commands = sorted(groups, key=lambda command: hashlib.sha256(f'{seed}\0{command}'.encode()).digest())
    target = round(len(records) * validation_fraction)
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for command in commands:
        count = len(groups[command])
        for current, selected in sorted(reachable.items(), reverse=True):
            reachable.setdefault(current + count, (*selected, command))
    selected_count = min((count for count in reachable if count > 0), key=lambda count: (abs(count - target), count))
    heldout_commands = set(reachable[selected_count])
    train = [record for record in records if record.command not in heldout_commands]
    validation = [record for record in records if record.command in heldout_commands]
    if {record.command for record in train}.intersection(record.command for record in validation):
        raise AssertionError('inner split leaked a command')
    return train, validation


def _device(name: str) -> torch.device:
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return torch.device(name)


def _save_checkpoint(path: Path | None, model: SemanticActionModel, metadata: Mapping[str, object]) -> None:
    if path is None:
        raise ValueError('--checkpoint is required')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(
        {
            'schema_version': 1,
            'experiment': EXPERIMENT,
            'codet5_revision': CODET5_REVISION,
            'model_config': model.config.to_dict(),
            'metadata': dict(metadata),
            'model_state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(path)


def _write_json(path: Path | None, value: Mapping[str, object]) -> None:
    if path is None:
        raise ValueError('--report or --manifest path is required')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _require_output(args: argparse.Namespace) -> None:
    if args.checkpoint is None or args.report is None:
        raise ValueError('--checkpoint and --report are required')


def _json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _manifest_summary(manifest: Mapping[str, object]) -> dict[str, object]:
    inner = manifest['inner_splits']
    outer = manifest['outer_splits']
    return {
        'action_vocab_size': manifest['action_vocab_size'],
        'action_lengths': manifest['action_lengths'],
        'maximum_source_tokens': manifest['maximum_source_tokens'],
        'inner_train': len(inner['train']['record_ids']),
        'inner_validation': len(inner['validation']['record_ids']),
        'outer_train': len(outer['train']['record_ids']),
        'outer_validation': len(outer['validation']['record_ids']),
        'outer_test': len(outer['test']['record_ids']),
    }


def main() -> None:
    args = parse_args()
    phases = {
        'prepare': prepare,
        'inspect': inspect,
        'cpu': cpu_gate,
        'overfit': overfit,
        'inner': inner,
        'outer': outer,
        'compare': compare,
    }
    phases[args.phase](args)


if __name__ == '__main__':
    main()
