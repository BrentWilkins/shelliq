#!/usr/bin/env python3
"""Run the gated transferred-decoder semantic-action pointer experiment."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_semantic_action_decoder as v1
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoModelForSeq2SeqLM, RobertaTokenizer

from shelliq_training.compiler_comparison import load_comparison_manifest, require_matching_dataset
from shelliq_training.data import Split, load_semantic_jsonl
from shelliq_training.prompt import PromptContract
from shelliq_training.semantic_action_copy import (
    AlignedCodeT5Tokenizer,
    PointerActionExample,
    collate_pointer_actions,
    pointer_action_examples,
)
from shelliq_training.semantic_action_pointer_model import SemanticActionPointerModel
from shelliq_training.semantic_actions import ActionGrammar, SemanticActionClient

EXPERIMENT = 'semantic-action-span-v1'
COPY_ORACLE_EXPERIMENT = 'semantic-action-pointer-v1'
MODEL_ID = v1.CODET5_MODEL_ID
REVISION = v1.CODET5_REVISION
PROMPT_CONTRACT = PromptContract.CONTEXT_AUTHORITATIVE_V1
SOURCE_LENGTH = 256
SOURCE_BYTE_LENGTH = 768
TARGET_LENGTH = 192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('inspect', 'cpu', 'overfit', 'inner', 'outer', 'compare'))
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--outer-manifest', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--copy-oracle', type=Path, required=True)
    parser.add_argument('--actions', type=Path, default=Path('../target/debug/semantic-actions'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--inner-report', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=v1.INNER_SEED)
    parser.add_argument('--log-interval', type=int, default=100)
    return parser.parse_args()


def tokenizer_files() -> tuple[str, str]:
    vocab = hf_hub_download(MODEL_ID, 'vocab.json', revision=REVISION, local_files_only=True)
    merges = hf_hub_download(MODEL_ID, 'merges.txt', revision=REVISION, local_files_only=True)
    return vocab, merges


def build_model(tokenizer: RobertaTokenizer, *, dropout: float = 0.1) -> SemanticActionPointerModel:
    pretrained = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    model = SemanticActionPointerModel.from_codet5(
        pretrained,
        tokenizer,
        maximum_source_bytes=SOURCE_BYTE_LENGTH,
        dropout=dropout,
    )
    encoder_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if encoder_parameters != 35_316_480:
        raise ValueError(f'unexpected CodeT5 encoder parameter count: {encoder_parameters}')
    if not 53_000_000 <= total_parameters <= 54_000_000:
        raise ValueError(f'unexpected pointer model parameter count: {total_parameters}')
    return model


def load_inputs(args: argparse.Namespace):
    copy_oracle = json.loads(args.copy_oracle.read_text())
    if copy_oracle.get('experiment') != COPY_ORACLE_EXPERIMENT or not copy_oracle.get('gate_passed'):
        raise ValueError('copy-oracle gate was not passed')
    experiment_manifest = json.loads(args.manifest.read_text())
    outer = load_comparison_manifest(args.outer_manifest)
    require_matching_dataset(args.semantic_dataset, outer)
    records = load_semantic_jsonl(args.semantic_dataset)
    frozen = outer.records_by_split(records)
    client = SemanticActionClient(args.actions)
    grammar = ActionGrammar(client.manifest())
    vocab, merges = tokenizer_files()
    tokenizer = RobertaTokenizer(vocab=vocab, merges=merges)
    aligned_tokenizer = AlignedCodeT5Tokenizer(vocab, merges)
    actions = client.encode([json.loads(record.response) for record in records])
    examples = pointer_action_examples(
        records,
        actions,
        aligned_tokenizer,
        grammar,
        source_length=SOURCE_LENGTH,
        source_byte_length=SOURCE_BYTE_LENGTH,
        target_length=TARGET_LENGTH,
        prompt_contract=PROMPT_CONTRACT,
    )
    by_id = {example.record_id: example for example in examples}
    split_examples = {split: [by_id[record.record_id] for record in selected] for split, selected in frozen.items()}
    if set(experiment_manifest['outer_splits']['train']['record_ids']) != {record.record_id for record in frozen[Split.TRAIN]}:
        raise ValueError('experiment and outer manifests disagree')
    return records, frozen, split_examples, tokenizer, client, grammar, experiment_manifest


def inspect(args: argparse.Namespace) -> None:
    _, _, examples, tokenizer, _, _, _ = load_inputs(args)
    model = build_model(tokenizer)
    encoder = sum(parameter.numel() for parameter in model.encoder.parameters())
    print(
        json.dumps(
            {
                'encoder_parameters': encoder,
                'decoder_pointer_parameters': sum(parameter.numel() for parameter in model.parameters()) - encoder,
                'total_parameters': sum(parameter.numel() for parameter in model.parameters()),
                'transferred_decoder_layers': len(model.decoder.block),
                'maximum_source_tokens': max(len(item.source_ids) for group in examples.values() for item in group),
                'maximum_source_bytes': max(len(item.source_bytes) for group in examples.values() for item in group),
                'maximum_target_actions': max(len(item.action_ids) for group in examples.values() for item in group),
            },
            indent=2,
            sort_keys=True,
        )
    )


def cpu(args: argparse.Namespace) -> None:
    _, _, examples, tokenizer, _, grammar, _ = load_inputs(args)
    torch.manual_seed(args.seed)
    model = build_model(tokenizer, dropout=0).cpu()
    chosen = [examples[Split.TRAIN][0]]
    batch = make_batch(chosen, tokenizer).to(torch.device('cpu'))
    components = model.loss_components(batch)
    components['total'].backward()
    finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    if not finite:
        raise RuntimeError('non-finite CPU pointer gradient')
    generated = model.generate(batch, grammar, max_new_tokens=8, atomic_span_copy=True)
    cursor = grammar.cursor()
    for token in generated[0]:
        if token not in cursor.allowed():
            raise RuntimeError('masked generation escaped the Rust manifest grammar')
        cursor.advance(token)
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'cpu',
        'finite_gradients': True,
        'losses': {name: float(value.detach()) for name, value in components.items()},
        'masked_generation_actions': len(generated[0]),
        'masked_generation_complete': cursor.complete,
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def overfit(args: argparse.Namespace) -> None:
    _, frozen, examples, tokenizer, client, grammar, _ = load_inputs(args)
    selected_records = v1._select_distinct_commands(frozen[Split.TRAIN], 8, args.seed)
    selected_by_id = {example.record_id: example for example in examples[Split.TRAIN]}
    selected = [selected_by_id[record.record_id] for record in selected_records]
    device = v1._device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(tokenizer, dropout=0).to(device)
    optimizer = optimizer_for(model)
    batch = make_batch(selected, tokenizer).to(device)
    initial_loss = 0.0
    final_loss = 0.0
    completed_steps = args.steps
    started = time.monotonic()
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
        if step == args.steps or step >= 100 and step % args.log_interval == 0:
            generated = generate(model, selected, tokenizer, grammar, device)
            exact = sum(actual == expected.action_ids for actual, expected in zip(generated, selected, strict=True))
            print(f'step={step} loss={final_loss:.6f} exact={exact}/8', flush=True)
            if exact == len(selected):
                completed_steps = step
                break
    generated = generate(model, selected, tokenizer, grammar, device)
    metrics, outcomes = v1._evaluate(selected_records, selected, generated, client)
    passed = metrics['exact_actions'] == 8 and metrics['rust_valid'] == 8 and metrics['reference_accepted'] == 8
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'overfit',
        'gate_passed': passed,
        'seed': args.seed,
        'step_limit': args.steps,
        'completed_steps': completed_steps,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'elapsed_seconds': time.monotonic() - started,
        'metrics': metrics,
        'outcomes': outcomes,
    }
    save_checkpoint(args.checkpoint, model, report)
    write_json(args.report, report)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit('pointer overfit gate failed')


def inner(args: argparse.Namespace) -> None:
    development(args, phase='inner')


def outer(args: argparse.Namespace) -> None:
    development(args, phase='outer')


def compare(args: argparse.Namespace) -> None:
    development(args, phase='compare')


def development(args: argparse.Namespace, *, phase: str) -> None:
    records, _, examples, tokenizer, client, grammar, manifest = load_inputs(args)
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
            raise ValueError('inner pointer gate did not pass')
        train_ids = manifest['outer_splits']['train']['record_ids']
        split = 'validation' if phase == 'outer' else 'test'
        evaluation_ids = manifest['outer_splits'][split]['record_ids']
        epochs = int(inner_report['best_epoch'])
        select = False
    train_examples = [by_example[record_id] for record_id in train_ids]
    evaluation_examples = [by_example[record_id] for record_id in evaluation_ids]
    evaluation_records = [by_record[record_id] for record_id in evaluation_ids]
    device = v1._device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(tokenizer).to(device)
    optimizer = optimizer_for(model)
    history = []
    best_loss = float('inf')
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        weighted = 0.0
        tokens_seen = 0
        for chosen in v1._ordered_batches(train_examples, args.batch_size, args.seed + epoch):
            batch = make_batch(chosen, tokenizer).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite pointer loss in epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = int((batch.labels != -100).sum())
            weighted += float(loss.detach()) * tokens
            tokens_seen += tokens
        action_loss = mean_action_loss(model, evaluation_examples, tokenizer, args.batch_size, device)
        row = {'epoch': epoch, 'train_total_loss': weighted / tokens_seen, 'validation_action_loss': action_loss}
        history.append(row)
        if select and action_loss < best_loss:
            best_loss = action_loss
            best_epoch = epoch
            save_checkpoint(args.checkpoint, model, {'phase': phase, 'best_epoch': epoch, 'best_loss': best_loss})
        print(
            f'phase={phase} epoch={epoch}/{epochs} train_loss={row["train_total_loss"]:.6f} '
            f'action_loss={action_loss:.6f} best_epoch={best_epoch}',
            flush=True,
        )
    if select:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved['model_state_dict'])
    else:
        best_epoch = epochs
        best_loss = history[-1]['validation_action_loss']
        save_checkpoint(args.checkpoint, model, {'phase': phase, 'epochs': epochs})
    generated = generate(model, evaluation_examples, tokenizer, grammar, device)
    metrics, outcomes = v1._evaluate(evaluation_records, evaluation_examples, generated, client)
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
    write_json(args.report, report)
    print(json.dumps({'gate_passed': passed, **metrics}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(f'pointer {phase} gate failed')


def optimizer_for(model: SemanticActionPointerModel) -> torch.optim.AdamW:
    pretrained = list(model.encoder.parameters()) + [
        parameter for name, parameter in model.decoder.named_parameters() if not name.startswith('embed_tokens.')
    ]
    pretrained_ids = {id(parameter) for parameter in pretrained}
    new = [parameter for parameter in model.parameters() if id(parameter) not in pretrained_ids]
    if len({id(parameter) for parameter in [*pretrained, *new]}) != len(list(model.parameters())):
        raise AssertionError('optimizer parameter groups overlap or omit parameters')
    return torch.optim.AdamW(
        [{'params': pretrained, 'lr': 5e-5}, {'params': new, 'lr': 5e-4}],
        weight_decay=0,
    )


def make_batch(examples: Sequence[PointerActionExample], tokenizer) -> object:
    return collate_pointer_actions(
        examples,
        source_length=max(len(item.source_ids) for item in examples),
        source_byte_length=max(len(item.source_bytes) for item in examples),
        target_length=max(len(item.action_ids) for item in examples),
        source_pad_token_id=tokenizer.pad_token_id,
    )


@torch.no_grad()
def mean_action_loss(model, examples, tokenizer, batch_size, device) -> float:
    model.eval()
    weighted = 0.0
    tokens_seen = 0
    for offset in range(0, len(examples), batch_size):
        batch = make_batch(examples[offset : offset + batch_size], tokenizer).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            action_loss = model.loss_components(batch)['action']
        tokens = int((batch.labels != -100).sum())
        weighted += float(action_loss) * tokens
        tokens_seen += tokens
    return weighted / tokens_seen


@torch.no_grad()
def generate(model, examples, tokenizer, grammar, device) -> list[tuple[int, ...]]:
    model.eval()
    generated = []
    for example in examples:
        batch = make_batch([example], tokenizer).to(device)
        generated.extend(model.generate(batch, grammar, atomic_span_copy=True))
    return generated


def save_checkpoint(path: Path | None, model: SemanticActionPointerModel, metadata: Mapping[str, object]) -> None:
    if path is None:
        raise ValueError('--checkpoint is required')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(
        {
            'schema_version': 1,
            'experiment': EXPERIMENT,
            'codet5_revision': REVISION,
            'model_config': model.config.to_dict(),
            'maximum_source_bytes': model.maximum_source_bytes,
            'metadata': dict(metadata),
            'model_state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(path)


def write_json(path: Path | None, value: Mapping[str, object]) -> None:
    if path is None:
        raise ValueError('--report is required')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main() -> None:
    args = parse_args()
    phases = {'inspect': inspect, 'cpu': cpu, 'overfit': overfit, 'inner': inner, 'outer': outer, 'compare': compare}
    phases[args.phase](args)


if __name__ == '__main__':
    main()
