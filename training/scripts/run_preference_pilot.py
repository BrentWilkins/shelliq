#!/usr/bin/env python3
"""Run matched chosen-SFT and cached-reference DPO pilots from one LoRA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from safetensors.flax import load_file
from transformers import AutoTokenizer

from scripts.smoke_finetune import automatic_preflight  # noqa: E402
from shelliq_training.checkpoint import (  # noqa: E402
    TargetFormat,
    restore_adapter_checkpoint,
    save_checkpoint,
)
from shelliq_training.config import Qwen2Config  # noqa: E402
from shelliq_training.data import (  # noqa: E402
    Corpus,
    SequenceTooLongError,
    Split,
    collate_sft,
    load_semantic_jsonl,
    split_records,
    tokenize_record,
    tokenizer_pad_id,
)
from shelliq_training.lora import inject_lora  # noqa: E402
from shelliq_training.model import Qwen2ForCausalLM  # noqa: E402
from shelliq_training.preference_data import load_preference_jsonl  # noqa: E402
from shelliq_training.preference_training import (  # noqa: E402
    TokenizedPreference,
    preference_log_probability_step,
    preference_train_step,
    sft_control_train_step,
    split_preference_records,
    tokenize_preference,
)
from shelliq_training.prompt import PromptContract  # noqa: E402
from shelliq_training.training import create_lora_optimizer, eval_step  # noqa: E402
from shelliq_training.weights import load_hf_state_dict  # noqa: E402

MODEL_ID = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preferences', type=Path, required=True)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--sequence-length', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--replay-weight', type=float, default=0.2)
    parser.add_argument('--replay-examples', type=int, default=128)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--split-seed', type=int, default=2026)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_batches(
    preferences: list[TokenizedPreference], *, sequence_length: int, pad_token_id: int
) -> list[tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
    return [
        (
            collate_sft([preference.chosen], sequence_length=sequence_length, pad_token_id=pad_token_id),
            collate_sft([preference.rejected], sequence_length=sequence_length, pad_token_id=pad_token_id),
        )
        for preference in preferences
    ]


def _score_pairs(model, batches):
    scores = []
    for chosen_batch, rejected_batch in batches:
        chosen, rejected = preference_log_probability_step(model, chosen_batch, rejected_batch)
        scores.append((float(np.asarray(chosen)[0]), float(np.asarray(rejected)[0])))
    return scores


def _score_summary(policy, reference) -> dict[str, float]:
    raw_margins = np.asarray([chosen - rejected for chosen, rejected in policy])
    reference_margins = np.asarray([chosen - rejected for chosen, rejected in reference])
    deltas = raw_margins - reference_margins
    return {
        'mean_chosen_logp': float(np.mean([chosen for chosen, _ in policy])),
        'mean_rejected_logp': float(np.mean([rejected for _, rejected in policy])),
        'mean_raw_margin': float(np.mean(raw_margins)),
        'mean_reference_margin_delta': float(np.mean(deltas)),
        'reference_preference_accuracy': float(np.mean(deltas > 0)),
        'raw_preference_accuracy': float(np.mean(raw_margins > 0)),
    }


def _validate_args(args: argparse.Namespace) -> None:
    for path in (args.preferences, args.semantic_dataset, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_root.exists() or args.report.exists():
        raise FileExistsError('output root and report must not already exist')
    if not args.output_root.parent.is_dir() or not args.report.parent.is_dir():
        raise FileNotFoundError('output parents must exist')
    if args.epochs <= 0 or args.sequence_length < 2 or args.replay_examples <= 0:
        raise ValueError('epochs, sequence length, and replay examples must be positive')
    for name in ('learning_rate', 'beta', 'replay_weight'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be finite and positive')


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if jax.default_backend() != 'gpu':
        raise SystemExit('run_preference_pilot.py requires the JAX GPU backend')
    console = Console()
    preferences = load_preference_jsonl(args.preferences)
    train_records, validation_records = split_preference_records(preferences, seed=args.split_seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    train_preferences = [tokenize_preference(record, tokenizer, max_length=args.sequence_length) for record in train_records]
    validation_preferences = [
        tokenize_preference(record, tokenizer, max_length=args.sequence_length) for record in validation_records
    ]
    pad_id = tokenizer_pad_id(tokenizer)
    train_batches = _pair_batches(train_preferences, sequence_length=args.sequence_length, pad_token_id=pad_id)
    validation_batches = _pair_batches(validation_preferences, sequence_length=args.sequence_length, pad_token_id=pad_id)

    clean_semantic, rejected_semantic = automatic_preflight(load_semantic_jsonl(args.semantic_dataset))
    semantic_splits = split_records(clean_semantic, corpus=Corpus.DISTRIBUTABLE, seed=args.split_seed)
    replay_records = [record for record in semantic_splits[Split.TRAIN] if record.source == 'shelliq-curated']
    replay_records.sort(key=lambda record: hashlib.sha256(f'{args.seed}\0{record.record_id}'.encode()).digest())
    replay_batches = []
    for record in replay_records:
        try:
            example = tokenize_record(
                record,
                tokenizer,
                max_length=args.sequence_length,
                prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
            )
        except SequenceTooLongError:
            continue
        replay_batches.append(collate_sft([example], sequence_length=args.sequence_length, pad_token_id=pad_id))
        if len(replay_batches) == args.replay_examples:
            break
    if len(replay_batches) < args.replay_examples:
        raise ValueError(f'only {len(replay_batches)} usable curated replay examples')

    model = Qwen2ForCausalLM(Qwen2Config(), param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    weights_path = hf_hub_download(MODEL_ID, 'model.safetensors', local_files_only=True)
    load_hf_state_dict(model, load_file(weights_path), param_dtype=jnp.bfloat16)
    inject_lora(model, rank=args.rank, alpha=args.alpha, rngs=nnx.Rngs(1))
    restore_adapter_checkpoint(
        args.checkpoint,
        model,
        model_id=MODEL_ID,
        corpus=Corpus.DISTRIBUTABLE,
        target_format=TargetFormat.SEMANTIC_DOCUMENT_V2,
        prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
    )

    console.print('[cyan]Caching frozen reference scores (no second model copy)[/cyan]')
    reference_train = _score_pairs(model, train_batches)
    reference_validation = _score_pairs(model, validation_batches)
    total_steps = args.epochs * len(train_batches)
    args.output_root.mkdir()
    condition_reports: dict[str, object] = {}
    beta = jnp.asarray(args.beta, dtype=jnp.float32)
    replay_weight = jnp.asarray(args.replay_weight, dtype=jnp.float32)

    for condition_index, condition in enumerate(('sft-control', 'dpo')):
        restore_adapter_checkpoint(
            args.checkpoint,
            model,
            model_id=MODEL_ID,
            corpus=Corpus.DISTRIBUTABLE,
            target_format=TargetFormat.SEMANTIC_DOCUMENT_V2,
            prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
        )
        optimizer = create_lora_optimizer(model, learning_rate=args.learning_rate, max_grad_norm=1.0)
        observations = []
        started = time.perf_counter()
        with Progress(console=console) as progress:
            task = progress.add_task(f'[cyan]{condition}', total=total_steps, loss='--')
            step = 0
            for epoch in range(args.epochs):
                order = np.random.default_rng(args.seed + condition_index * 1000 + epoch).permutation(len(train_batches))
                for batch_index in order:
                    chosen_batch, rejected_batch = train_batches[int(batch_index)]
                    replay_batch = replay_batches[step % len(replay_batches)]
                    if condition == 'dpo':
                        reference_chosen, reference_rejected = reference_train[int(batch_index)]
                        values = preference_train_step(
                            model,
                            optimizer,
                            chosen_batch,
                            rejected_batch,
                            jnp.asarray([reference_chosen]),
                            jnp.asarray([reference_rejected]),
                            replay_batch,
                            beta,
                            replay_weight,
                        )
                    else:
                        values = sft_control_train_step(model, optimizer, chosen_batch, replay_batch, replay_weight)
                    step += 1
                    scalar_values = [float(np.asarray(value)) for value in values]
                    observations.append(scalar_values)
                    progress.update(task, advance=1, loss=f'{scalar_values[0]:.4f}')
        train_scores = _score_pairs(model, train_batches)
        validation_scores = _score_pairs(model, validation_batches)
        replay_loss = float(np.asarray(eval_step(model, replay_batches[0])))
        checkpoint = args.output_root / condition
        metadata = save_checkpoint(
            checkpoint,
            model,
            optimizer,
            model_id=MODEL_ID,
            corpus=Corpus.DISTRIBUTABLE,
            target_format=TargetFormat.SEMANTIC_DOCUMENT_V2,
            prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
        )
        condition_reports[condition] = {
            'checkpoint': str(checkpoint),
            'checkpoint_step': metadata.step,
            'elapsed_seconds': time.perf_counter() - started,
            'final_objective': observations[-1][0],
            'replay_probe_loss': replay_loss,
            'train': _score_summary(train_scores, reference_train),
            'validation': _score_summary(validation_scores, reference_validation),
        }

    report = {
        'report_schema_version': 1,
        'model_id': MODEL_ID,
        'preference_dataset': str(args.preferences),
        'preference_sha256': _sha256(args.preferences),
        'semantic_dataset': str(args.semantic_dataset),
        'semantic_sha256': _sha256(args.semantic_dataset),
        'starting_checkpoint': str(args.checkpoint),
        'rank': args.rank,
        'alpha': args.alpha,
        'sequence_length': args.sequence_length,
        'train_pairs': len(train_records),
        'validation_pairs': len(validation_records),
        'replay_examples': len(replay_batches),
        'semantic_preflight_rejections': rejected_semantic,
        'epochs': args.epochs,
        'steps': total_steps,
        'learning_rate': args.learning_rate,
        'beta': args.beta,
        'replay_weight': args.replay_weight,
        'seed': args.seed,
        'split_seed': args.split_seed,
        'conditions': condition_reports,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    table = Table(title='Preference pilot')
    table.add_column('condition')
    table.add_column('validation margin Δ', justify='right')
    table.add_column('validation preference', justify='right')
    table.add_column('replay loss', justify='right')
    for condition, result in condition_reports.items():
        validation = result['validation']
        table.add_row(
            condition,
            f'{validation["mean_reference_margin_delta"]:.3f}',
            f'{validation["reference_preference_accuracy"]:.1%}',
            f'{result["replay_probe_loss"]:.4f}',
        )
    console.print(table)
    console.print(f'report: {args.report}')


if __name__ == '__main__':
    main()
