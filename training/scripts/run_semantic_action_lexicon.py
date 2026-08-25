#!/usr/bin/env python3
"""Run the gated global-plus-local semantic-action candidate experiment."""

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
from shelliq_training.semantic_action_candidates import (
    CandidateActionExample,
    global_word_lexicon,
    relabel_with_global_words,
)
from shelliq_training.semantic_action_lexicon_model import SemanticActionLexiconModel

EXPERIMENT = 'semantic-action-lexicon-beam-v1'
BEAM_WIDTH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('cpu', 'overfit', 'inner', 'outer'))
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--outer-manifest', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--copy-oracle', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--inner-report', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=v1.INNER_SEED)
    parser.add_argument('--log-interval', type=int, default=100)
    return parser.parse_args()


def build_model(tokenizer, global_words: Sequence[bytes], *, dropout: float = 0.1) -> SemanticActionLexiconModel:
    pretrained = AutoModelForSeq2SeqLM.from_pretrained(
        candidate.MODEL_ID,
        revision=candidate.REVISION,
        local_files_only=True,
    )
    return SemanticActionLexiconModel.from_codet5(
        pretrained,
        tokenizer,
        maximum_source_bytes=candidate.SOURCE_BYTE_LENGTH,
        global_words=global_words,
        dropout=dropout,
    )


def relabeled_examples(examples, grammar, train_ids: Sequence[str]):
    by_id = {example.record_id: example for group in examples.values() for example in group}
    global_words = global_word_lexicon([by_id[item].action_ids for item in train_ids], grammar)
    relabeled = relabel_with_global_words(list(by_id.values()), grammar, global_words)
    return {example.record_id: example for example in relabeled}, global_words


def cpu(args: argparse.Namespace) -> None:
    _, _, examples, tokenizer, _, grammar, manifest = candidate.load_inputs(args)
    train_ids = manifest['inner_splits']['train']['record_ids']
    by_id, global_words = relabeled_examples(examples, grammar, train_ids)
    torch.manual_seed(args.seed)
    model = build_model(tokenizer, global_words, dropout=0).cpu()
    batch = candidate.make_batch([by_id[train_ids[0]]], tokenizer).to(torch.device('cpu'))
    components = model.loss_components(batch)
    components['total'].backward()
    finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    if not finite:
        raise RuntimeError('non-finite CPU lexicon gradient')
    generated = model.generate_beam(batch, grammar, beam_width=BEAM_WIDTH, max_new_tokens=8)
    cursor = grammar.cursor()
    for token in generated[0]:
        if token not in cursor.allowed():
            raise RuntimeError('beam generation escaped Rust manifest grammar')
        cursor.advance(token)
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'cpu',
        'gate_passed': True,
        'global_lexicon_size': len(global_words),
        'finite_gradients': finite,
        'losses': {name: float(value.detach()) for name, value in components.items()},
        'bounded_generation_actions': len(generated[0]),
        'bounded_generation_complete': cursor.complete,
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def overfit(args: argparse.Namespace) -> None:
    _, frozen, examples, tokenizer, client, grammar, _ = candidate.load_inputs(args)
    selected_records = v1._select_distinct_commands(frozen[Split.TRAIN], 8, args.seed)
    selected_ids = [record.record_id for record in selected_records]
    by_id, global_words = relabeled_examples(examples, grammar, selected_ids)
    selected = [by_id[item] for item in selected_ids]
    device = v1._device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(tokenizer, global_words, dropout=0).to(device)
    optimizer = candidate.optimizer_for(model)
    batch = candidate.make_batch(selected, tokenizer).to(device)
    initial_loss = final_loss = 0.0
    completed_steps = args.steps
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss = model.loss(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite lexicon overfit loss at step {step}')
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
        'global_lexicon_size': len(global_words),
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'elapsed_seconds': time.monotonic() - started,
        'metrics': metrics,
        'outcomes': outcomes,
    }
    save_checkpoint(args.checkpoint, model, global_words, report)
    write_json(args.report, report)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit('lexicon overfit gate failed')


def inner(args: argparse.Namespace) -> None:
    development(args, phase='inner')


def outer(args: argparse.Namespace) -> None:
    development(args, phase='outer')


def development(args: argparse.Namespace, *, phase: str) -> None:
    records, _, examples, tokenizer, client, grammar, manifest = candidate.load_inputs(args)
    by_record = {record.record_id: record for record in records}
    if phase == 'inner':
        train_ids = manifest['inner_splits']['train']['record_ids']
        evaluation_ids = manifest['inner_splits']['validation']['record_ids']
        epochs = args.epochs
        select = True
    else:
        if args.inner_report is None:
            raise ValueError('--inner-report is required for outer validation')
        inner_report = json.loads(args.inner_report.read_text())
        if not inner_report.get('gate_passed'):
            raise ValueError('inner lexicon gate did not pass')
        train_ids = manifest['outer_splits']['train']['record_ids']
        evaluation_ids = manifest['outer_splits']['validation']['record_ids']
        epochs = int(inner_report['best_epoch'])
        select = False
    by_example, global_words = relabeled_examples(examples, grammar, train_ids)
    train_examples = [by_example[item] for item in train_ids]
    evaluation_examples = [by_example[item] for item in evaluation_ids]
    evaluation_records = [by_record[item] for item in evaluation_ids]
    device = v1._device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(tokenizer, global_words).to(device)
    optimizer = candidate.optimizer_for(model)
    history = []
    best_loss, best_epoch = float('inf'), 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        weighted = tokens_seen = 0
        for chosen in v1._ordered_batches(train_examples, args.batch_size, args.seed + epoch):
            batch = candidate.make_batch(chosen, tokenizer).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite lexicon loss in epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = int((batch.labels != -100).sum())
            weighted += float(loss.detach()) * tokens
            tokens_seen += tokens
        validation_loss = mean_total_loss(model, evaluation_examples, tokenizer, args.batch_size, device)
        row = {
            'epoch': epoch,
            'train_total_loss': weighted / tokens_seen,
            'validation_total_loss': validation_loss,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if select and validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            save_checkpoint(
                args.checkpoint,
                model,
                global_words,
                {'phase': phase, 'best_epoch': epoch, 'best_loss': best_loss},
            )
    if select:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved['model_state_dict'])
    else:
        best_epoch, best_loss = epochs, history[-1]['validation_total_loss']
        save_checkpoint(args.checkpoint, model, global_words, {'phase': phase, 'epochs': epochs})
    generated = generate(model, evaluation_examples, tokenizer, grammar, device)
    metrics, outcomes = v1._evaluate(evaluation_records, evaluation_examples, generated, client)
    count = len(evaluation_examples)
    acceptance = 0.05 if phase == 'inner' else 0.10
    passed = (
        metrics['rust_valid'] / count >= 0.90
        and metrics['first_command_match'] / count >= 0.80
        and metrics['reference_accepted'] / count >= acceptance
    )
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': phase,
        'gate_passed': passed,
        'seed': args.seed,
        'epochs': epochs,
        'best_epoch': best_epoch,
        'best_teacher_forced_total_loss': best_loss,
        'global_lexicon_size': len(global_words),
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
        raise SystemExit(f'lexicon {phase} gate failed')


@torch.no_grad()
def mean_total_loss(model, examples, tokenizer, batch_size, device) -> float:
    model.eval()
    weighted = tokens_seen = 0
    for offset in range(0, len(examples), batch_size):
        batch = candidate.make_batch(examples[offset : offset + batch_size], tokenizer).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss = model.loss(batch)
        tokens = int((batch.labels != -100).sum())
        weighted += float(loss) * tokens
        tokens_seen += tokens
    return weighted / tokens_seen


@torch.no_grad()
def generate(model, examples: Sequence[CandidateActionExample], tokenizer, grammar, device):
    model.eval()
    generated = []
    for example in examples:
        batch = candidate.make_batch([example], tokenizer).to(device)
        generated.extend(model.generate_beam(batch, grammar, beam_width=BEAM_WIDTH))
    return generated


def save_checkpoint(
    path: Path | None,
    model: SemanticActionLexiconModel,
    global_words: Sequence[bytes],
    metadata: Mapping[str, object],
) -> None:
    if path is None:
        raise ValueError('--checkpoint is required')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(
        {
            'schema_version': 1,
            'experiment': EXPERIMENT,
            'codet5_revision': candidate.REVISION,
            'global_words': [word.decode('utf-8') for word in global_words],
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
    {'cpu': cpu, 'overfit': overfit, 'inner': inner, 'outer': outer}[args.phase](args)


if __name__ == '__main__':
    main()
