#!/usr/bin/env python3
"""Run the gated role-masked typed semantic-slot experiment."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_semantic_action_candidate as candidate
import run_semantic_action_decoder as v1
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM

from shelliq_training.data import Split
from shelliq_training.semantic_action_copy import AlignedCodeT5Tokenizer
from shelliq_training.semantic_action_typed import (
    TypedActionExample,
    byte_roles,
    collate_typed_actions,
    role_word_lexicon,
    typed_action_examples,
)
from shelliq_training.semantic_action_typed_model import SemanticActionTypedModel

EXPERIMENT = 'semantic-action-typed-slots-v1'
BEAM_WIDTH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('cpu', 'overfit', 'inner'))
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--outer-manifest', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--copy-oracle', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=v1.INNER_SEED)
    parser.add_argument('--log-interval', type=int, default=100)
    return parser.parse_args()


def build_model(tokenizer, *, dropout: float = 0.1) -> SemanticActionTypedModel:
    pretrained = AutoModelForSeq2SeqLM.from_pretrained(
        candidate.MODEL_ID,
        revision=candidate.REVISION,
        local_files_only=True,
    )
    return SemanticActionTypedModel.from_codet5(
        pretrained,
        tokenizer,
        maximum_source_bytes=candidate.SOURCE_BYTE_LENGTH,
        dropout=dropout,
    )


def load_typed(args: argparse.Namespace, train_ids: Sequence[str]):
    records, frozen, _, tokenizer, client, grammar, manifest = candidate.load_inputs(args)
    actions = client.encode([json.loads(record.response) for record in records])
    by_action = {record.record_id: action for record, action in zip(records, actions, strict=True)}
    global_roles = role_word_lexicon([by_action[item] for item in train_ids], grammar)
    vocab, merges = candidate.tokenizer_files()
    aligned = AlignedCodeT5Tokenizer(vocab, merges)
    examples = typed_action_examples(
        records,
        actions,
        aligned,
        grammar,
        global_roles,
        source_length=candidate.SOURCE_LENGTH,
        source_byte_length=candidate.SOURCE_BYTE_LENGTH,
        target_length=candidate.TARGET_LENGTH,
        prompt_contract=candidate.PROMPT_CONTRACT,
    )
    return records, frozen, {item.record_id: item for item in examples}, tokenizer, client, grammar, manifest, global_roles


def cpu(args: argparse.Namespace) -> None:
    _, _, _, _, _, _, manifest = candidate.load_inputs(args)
    train_ids = manifest['inner_splits']['train']['record_ids']
    _, _, examples, tokenizer, _, grammar, _, global_roles = load_typed(args, train_ids)
    torch.manual_seed(args.seed)
    model = build_model(tokenizer, dropout=0).cpu()
    batch = make_batch([examples[train_ids[0]]], tokenizer, len(byte_roles(grammar))).to(torch.device('cpu'))
    components = model.loss_components(batch)
    components['total'].backward()
    finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    if not finite:
        raise RuntimeError('non-finite CPU typed gradient')
    generated = model.generate_typed_beam(
        batch,
        grammar,
        byte_roles(grammar),
        beam_width=BEAM_WIDTH,
        max_new_tokens=8,
    )
    cursor = grammar.cursor()
    for token in generated[0]:
        if token not in cursor.allowed():
            raise RuntimeError('typed beam escaped Rust manifest grammar')
        cursor.advance(token)
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'cpu',
        'gate_passed': True,
        'global_words': len(global_roles),
        'maximum_candidates': max(len(example.candidates) for example in examples.values()),
        'finite_gradients': finite,
        'losses': {name: float(value.detach()) for name, value in components.items()},
        'bounded_generation_actions': len(generated[0]),
        'bounded_generation_complete': cursor.complete,
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def overfit(args: argparse.Namespace) -> None:
    records, frozen, _, tokenizer, client, grammar, manifest = candidate.load_inputs(args)
    selected_records = v1._select_distinct_commands(frozen[Split.TRAIN], 8, args.seed)
    selected_ids = [record.record_id for record in selected_records]
    _, _, examples, tokenizer, _, grammar, _, global_roles = load_typed(args, selected_ids)
    selected = [examples[item] for item in selected_ids]
    device = v1._device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(tokenizer, dropout=0).to(device)
    optimizer = candidate.optimizer_for(model)
    batch = make_batch(selected, tokenizer, len(byte_roles(grammar))).to(device)
    initial_loss = final_loss = 0.0
    completed_steps = args.steps
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss = model.loss(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite typed overfit loss at step {step}')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        if step == 1:
            initial_loss = final_loss
        if step == args.steps or (step >= 100 and step % args.log_interval == 0):
            generated = generate(model, selected, tokenizer, grammar, device)
            exact = sum(actual == expected.action_ids for actual, expected in zip(generated, selected, strict=True))
            print(f'step={step} loss={final_loss:.6f} exact={exact}/8', flush=True)
            if exact == 8:
                completed_steps = step
                break
    generated = generate(model, selected, tokenizer, grammar, device)
    metrics, outcomes = v1._evaluate(selected_records, selected, generated, client)
    passed = metrics['exact_actions'] == metrics['rust_valid'] == metrics['reference_accepted'] == 8
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'overfit',
        'gate_passed': passed,
        'seed': args.seed,
        'step_limit': args.steps,
        'completed_steps': completed_steps,
        'global_words': len(global_roles),
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'elapsed_seconds': time.monotonic() - started,
        'metrics': metrics,
        'outcomes': outcomes,
    }
    save_checkpoint(args.checkpoint, model, global_roles, report)
    write_json(args.report, report)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit('typed overfit gate failed')


def inner(args: argparse.Namespace) -> None:
    records, _, _, _, _, _, manifest = candidate.load_inputs(args)
    train_ids = manifest['inner_splits']['train']['record_ids']
    evaluation_ids = manifest['inner_splits']['validation']['record_ids']
    records, _, examples, tokenizer, client, grammar, _, global_roles = load_typed(args, train_ids)
    by_record = {record.record_id: record for record in records}
    train_examples = [examples[item] for item in train_ids]
    evaluation_examples = [examples[item] for item in evaluation_ids]
    evaluation_records = [by_record[item] for item in evaluation_ids]
    device = v1._device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(tokenizer).to(device)
    optimizer = candidate.optimizer_for(model)
    history = []
    best_loss, best_epoch = float('inf'), 0
    started = time.monotonic()
    roles = len(byte_roles(grammar))
    for epoch in range(1, args.epochs + 1):
        model.train()
        weighted = tokens_seen = 0
        for chosen in v1._ordered_batches(train_examples, args.batch_size, args.seed + epoch):
            batch = make_batch(chosen, tokenizer, roles).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite typed loss in epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = int((batch.labels != -100).sum())
            weighted += float(loss.detach()) * tokens
            tokens_seen += tokens
        validation_loss = mean_total_loss(
            model,
            evaluation_examples,
            tokenizer,
            roles,
            args.batch_size,
            device,
        )
        row = {
            'epoch': epoch,
            'train_total_loss': weighted / tokens_seen,
            'validation_total_loss': validation_loss,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            save_checkpoint(
                args.checkpoint,
                model,
                global_roles,
                {'phase': 'inner', 'best_epoch': epoch, 'best_loss': best_loss},
            )
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    generated = generate(model, evaluation_examples, tokenizer, grammar, device)
    metrics, outcomes = v1._evaluate(evaluation_records, evaluation_examples, generated, client)
    count = len(evaluation_examples)
    progress = (
        metrics['rust_valid'] / count >= 0.90
        and metrics['first_command_match'] / count >= 0.80
        and metrics['reference_accepted'] >= 1
    )
    promotion = progress and metrics['reference_accepted'] >= 5
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'inner',
        'gate_passed': promotion,
        'progress_evidence': progress,
        'seed': args.seed,
        'epochs': args.epochs,
        'best_epoch': best_epoch,
        'best_teacher_forced_total_loss': best_loss,
        'global_words': len(global_roles),
        'train_records': len(train_examples),
        'evaluation_records': count,
        'elapsed_seconds': time.monotonic() - started,
        'metrics': metrics,
        'history': history,
        'outcomes': outcomes,
    }
    write_json(args.report, report)
    print(json.dumps({'gate_passed': promotion, 'progress_evidence': progress, **metrics}, indent=2, sort_keys=True))
    if not progress:
        raise SystemExit('typed inner progress gate failed')


def make_batch(examples: Sequence[TypedActionExample], tokenizer, role_count: int):
    return collate_typed_actions(
        examples,
        tokenizer=tokenizer,
        source_length=max(len(item.source_ids) for item in examples),
        source_byte_length=max(len(item.source_bytes) for item in examples),
        candidate_count=max(len(item.candidates) for item in examples),
        candidate_byte_length=max(len(candidate.value) for item in examples for candidate in item.candidates),
        role_count=role_count,
        target_length=max(len(item.action_ids) for item in examples),
        source_pad_token_id=tokenizer.pad_token_id,
    )


@torch.no_grad()
def mean_total_loss(model, examples, tokenizer, role_count, batch_size, device) -> float:
    model.eval()
    weighted = tokens_seen = 0
    for offset in range(0, len(examples), batch_size):
        batch = make_batch(examples[offset : offset + batch_size], tokenizer, role_count).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss = model.loss(batch)
        tokens = int((batch.labels != -100).sum())
        weighted += float(loss) * tokens
        tokens_seen += tokens
    return weighted / tokens_seen


@torch.no_grad()
def generate(model, examples, tokenizer, grammar, device):
    model.eval()
    generated = []
    roles = byte_roles(grammar)
    for example in examples:
        batch = make_batch([example], tokenizer, len(roles)).to(device)
        generated.extend(model.generate_typed_beam(batch, grammar, roles, beam_width=BEAM_WIDTH))
    return generated


def save_checkpoint(path, model, global_roles, metadata: Mapping[str, object]) -> None:
    if path is None:
        raise ValueError('--checkpoint is required')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(
        {
            'schema_version': 1,
            'experiment': EXPERIMENT,
            'codet5_revision': candidate.REVISION,
            'global_roles': {word.decode('utf-8'): list(roles) for word, roles in global_roles.items()},
            'metadata': dict(metadata),
            'model_state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(path)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main() -> None:
    args = parse_args()
    {'cpu': cpu, 'overfit': overfit, 'inner': inner}[args.phase](args)


if __name__ == '__main__':
    main()
