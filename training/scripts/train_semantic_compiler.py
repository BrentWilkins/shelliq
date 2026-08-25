#!/usr/bin/env python3
"""Train and evaluate the from-scratch ShellIQ semantic compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from shelliq_training.compiler_data import (
    CompilerExample,
    collate_compiler,
    compiler_special_tokens,
    tokenize_compiler_record,
)
from shelliq_training.data import SFTRecord, load_semantic_jsonl
from shelliq_training.prompt import PromptContract
from shelliq_training.semantic_compiler import CompilerConfig, SemanticCompiler
from shelliq_training.teacher_verification import rust_validate_documents, verify_against_reference

MODEL_ID = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--model-id', default=MODEL_ID, help='Tokenizer only; neural weights are never loaded.')
    parser.add_argument(
        '--prompt-contract', type=PromptContract, choices=PromptContract, default=PromptContract.CONTEXT_AUTHORITATIVE_V1
    )
    parser.add_argument('--examples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=2_000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--source-length', type=int, default=192)
    parser.add_argument('--target-length', type=int, default=192)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--encoder-layers', type=int, default=2)
    parser.add_argument('--decoder-layers', type=int, default=2)
    parser.add_argument('--feedforward-size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--validator', type=Path, default=Path('../target/debug/validate-semantic-documents'))
    parser.add_argument('--selection-only', action='store_true')
    return parser.parse_args()


def deterministic_subset(records: Sequence[SFTRecord], *, count: int, seed: int) -> list[SFTRecord]:
    """Select command-disjoint rows with a stable hash order."""
    if count <= 0:
        raise ValueError('examples must be positive')
    ordered = sorted(records, key=lambda record: hashlib.sha256(f'{seed}\0{record.record_id}'.encode()).digest())
    selected: list[SFTRecord] = []
    commands: set[str] = set()
    for record in ordered:
        if record.command in commands:
            continue
        selected.append(record)
        commands.add(record.command)
        if len(selected) == count:
            return selected
    raise ValueError(f'requested {count} command-disjoint examples but only found {len(selected)}')


def train_model(
    model: SemanticCompiler,
    examples: Sequence[CompilerExample],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    special_tokens,
    device: torch.device,
    log_interval: int,
) -> tuple[float, float, float]:
    if steps <= 0 or batch_size <= 0:
        raise ValueError('steps and batch-size must be positive')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    use_amp = device.type == 'cuda'
    initial_loss = 0.0
    final_loss = 0.0
    started = time.monotonic()
    model.train()
    for step in range(steps):
        offset = (step * batch_size) % len(examples)
        chosen = [examples[(offset + index) % len(examples)] for index in range(min(batch_size, len(examples)))]
        batch = collate_compiler(
            chosen,
            source_length=model.config.max_source_length,
            target_length=model.config.max_target_length,
            special_tokens=special_tokens,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = model.loss(batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'non-finite loss at step {step + 1}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        final_loss = float(loss.detach())
        if step == 0:
            initial_loss = final_loss
        if log_interval > 0 and ((step + 1) % log_interval == 0 or step == 0 or step + 1 == steps):
            elapsed = time.monotonic() - started
            print(f'step={step + 1} loss={final_loss:.6f} elapsed_seconds={elapsed:.1f}', flush=True)
    return initial_loss, final_loss, time.monotonic() - started


@torch.no_grad()
def evaluate_overfit(
    model: SemanticCompiler,
    records: Sequence[SFTRecord],
    examples: Sequence[CompilerExample],
    tokenizer,
    *,
    special_tokens,
    device: torch.device,
    validator: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    generated_documents: list[str] = []
    for example in examples:
        batch = collate_compiler(
            [example],
            source_length=model.config.max_source_length,
            target_length=model.config.max_target_length,
            special_tokens=special_tokens,
        ).to(device)
        generated_ids = model.generate(
            batch.source_ids,
            batch.source_attention_mask,
            max_new_tokens=model.config.max_target_length,
        )[0]
        generated_documents.append(
            tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        )

    rust_results = rust_validate_documents(generated_documents, validator)
    rows: list[dict[str, object]] = []
    exact = 0
    valid = 0
    accepted = 0
    for record, generated, rust_result in zip(records, generated_documents, rust_results, strict=True):
        expected = json.loads(record.response)
        verification = verify_against_reference(generated, expected, rust_result)
        is_exact = generated == record.response
        exact += is_exact
        valid += rust_result.valid
        accepted += verification.accepted
        rows.append(
            {
                'record_id': record.record_id,
                'expected': record.response,
                'generated': generated,
                'exact': is_exact,
                'rust_round_trip': rust_result.valid,
                'accepted': verification.accepted,
                'failures': list(verification.failures),
                'rendered': rust_result.rendered,
            }
        )
    count = len(records)
    return (
        {
            'examples': count,
            'exact': exact,
            'rust_round_trip': valid,
            'reference_accepted': accepted,
            'overfit_gate_passed': exact == count and valid == count,
        },
        rows,
    )


def resolve_device(value: str) -> torch.device:
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if value == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return torch.device(value)


def main() -> None:
    args = parse_args()
    if not args.selection_only and (args.checkpoint is None or args.report is None):
        raise SystemExit('--checkpoint and --report are required unless --selection-only is used')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = deterministic_subset(
        load_semantic_jsonl(args.semantic_dataset),
        count=args.examples,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    special_tokens = compiler_special_tokens(tokenizer)
    examples = [
        tokenize_compiler_record(
            record,
            tokenizer,
            max_source_length=args.source_length,
            max_target_length=args.target_length,
            prompt_contract=args.prompt_contract,
        )
        for record in records
    ]
    selection = {
        'record_ids': [record.record_id for record in records],
        'source_token_lengths': [len(example.source_ids) for example in examples],
        'target_token_lengths': [len(example.target_ids) for example in examples],
    }
    if args.selection_only:
        print(json.dumps(selection, indent=2))
        return

    device = resolve_device(args.device)
    config = CompilerConfig(
        vocab_size=len(tokenizer),
        pad_token_id=special_tokens.pad_token_id,
        decoder_start_token_id=special_tokens.decoder_start_token_id,
        eos_token_id=special_tokens.eos_token_id,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        feedforward_size=args.feedforward_size,
        dropout=args.dropout,
        max_source_length=args.source_length,
        max_target_length=args.target_length,
    )
    model = SemanticCompiler(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f'device={device} parameters={parameter_count} records={len(records)}', flush=True)
    initial_loss, final_loss, elapsed = train_model(
        model,
        examples,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        special_tokens=special_tokens,
        device=device,
        log_interval=args.log_interval,
    )
    metrics, generations = evaluate_overfit(
        model,
        records,
        examples,
        tokenizer,
        special_tokens=special_tokens,
        device=device,
        validator=args.validator,
    )

    assert args.checkpoint is not None
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'schema_version': 1,
        'model_config': config.to_dict(),
        'model_state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
        'tokenizer_id': args.model_id,
        'prompt_contract': args.prompt_contract.value,
        'seed': args.seed,
        'record_ids': selection['record_ids'],
        'steps': args.steps,
    }
    temporary_checkpoint = args.checkpoint.with_suffix(args.checkpoint.suffix + '.tmp')
    torch.save(checkpoint, temporary_checkpoint)
    temporary_checkpoint.replace(args.checkpoint)
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    report = {
        'schema_version': 1,
        'experiment': 'custom-semantic-compiler-v1',
        'dataset': str(args.semantic_dataset),
        'dataset_sha256': hashlib.sha256(args.semantic_dataset.read_bytes()).hexdigest(),
        'tokenizer_id': args.model_id,
        'prompt_contract': args.prompt_contract.value,
        'device': str(device),
        'seed': args.seed,
        'selection': selection,
        'model_config': config.to_dict(),
        'parameter_count': parameter_count,
        'training': {
            'steps': args.steps,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'max_grad_norm': args.max_grad_norm,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'elapsed_seconds': elapsed,
        },
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': checkpoint_sha256,
        'metrics': metrics,
        'generations': generations,
    }
    assert args.report is not None
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
