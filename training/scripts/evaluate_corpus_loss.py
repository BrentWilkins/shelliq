#!/usr/bin/env python3
"""Measure a LoRA checkpoint on an entire deterministic corpus split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors.flax import load_file
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_finetune import (  # noqa: E402
    MODEL_ID,
    automatic_preflight,
    make_batches,
    select_all_examples,
)
from shelliq_training.checkpoint import (  # noqa: E402
    TargetFormat,
    restore_adapter_checkpoint,
)
from shelliq_training.config import Qwen2Config  # noqa: E402
from shelliq_training.data import (  # noqa: E402
    Corpus,
    Split,
    load_semantic_jsonl,
    split_records,
    tokenizer_pad_id,
)
from shelliq_training.lora import inject_lora  # noqa: E402
from shelliq_training.model import Qwen2ForCausalLM  # noqa: E402
from shelliq_training.progress import evaluation_progress  # noqa: E402
from shelliq_training.prompt import PromptContract  # noqa: E402
from shelliq_training.training import eval_step  # noqa: E402
from shelliq_training.weights import load_hf_state_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', type=Split, choices=Split, default=Split.TEST)
    parser.add_argument('--sequence-length', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(record_ids: list[str]) -> str:
    return hashlib.sha256('\n'.join(record_ids).encode()).hexdigest()


def load_evaluation_split(
    path: Path,
    *,
    split: Split,
    split_seed: int,
):
    """Load and split semantic records through the semantic loader contract."""
    clean_records, rejected = automatic_preflight(load_semantic_jsonl(path))
    splits = split_records(
        clean_records,
        corpus=Corpus.DISTRIBUTABLE,
        seed=split_seed,
    )
    return splits[split], rejected


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')
    if not args.output.parent.is_dir():
        raise SystemExit(f'output parent does not exist: {args.output.parent}')
    if jax.default_backend() != 'gpu':
        raise SystemExit('evaluate_corpus_loss.py requires the JAX GPU backend')
    if args.batch_size <= 0:
        raise SystemExit('--batch-size must be positive')

    split_records_for_evaluation, rejected = load_evaluation_split(
        args.semantic_dataset,
        split=args.split,
        split_seed=args.split_seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    records, examples = select_all_examples(
        split_records_for_evaluation,
        tokenizer,
        max_length=args.sequence_length,
        seed=args.seed,
        prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
    )
    usable = len(examples) - (len(examples) % args.batch_size)
    records = records[:usable]
    examples = examples[:usable]
    batches = make_batches(
        examples,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        pad_token_id=tokenizer_pad_id(tokenizer),
    )

    model = Qwen2ForCausalLM(Qwen2Config(), param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    weights_path = hf_hub_download(MODEL_ID, 'model.safetensors', local_files_only=True)
    load_hf_state_dict(model, load_file(weights_path), param_dtype=jnp.bfloat16)
    inject_lora(model, rank=args.rank, alpha=args.alpha, rngs=nnx.Rngs(1))
    metadata = restore_adapter_checkpoint(
        args.checkpoint,
        model,
        model_id=MODEL_ID,
        corpus=Corpus.DISTRIBUTABLE,
        target_format=TargetFormat.SEMANTIC_DOCUMENT_V2,
        prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
    )

    started = time.perf_counter()
    losses: list[float] = []
    with evaluation_progress() as progress:
        task = progress.add_task(
            f'{args.split.value} loss',
            total=len(batches),
            record_id='starting',
        )
        for index, (batch, batch_records) in enumerate(
            zip(
                batches,
                (records[offset : offset + args.batch_size] for offset in range(0, len(records), args.batch_size)),
                strict=True,
            )
        ):
            loss = float(np.asarray(eval_step(model, batch)))
            if not np.isfinite(loss):
                raise SystemExit(f'non-finite loss at batch {index}')
            losses.append(loss)
            progress.update(task, advance=1, record_id=batch_records[-1].record_id)

    report = {
        'schema_version': 1,
        'dataset': {
            'path': str(args.semantic_dataset.resolve()),
            'sha256': _sha256(args.semantic_dataset),
            'rejected_record_ids': sorted(rejected),
        },
        'selection': {
            'split': args.split.value,
            'split_seed': args.split_seed,
            'selection_seed': args.seed,
            'usable_examples': len(records),
            'record_ids_sha256': _ids_sha256([record.record_id for record in records]),
            'sequence_length': args.sequence_length,
            'batch_size': args.batch_size,
        },
        'checkpoint': {
            'path': str(args.checkpoint.resolve()),
            **metadata.to_dict(),
        },
        'mean_loss': float(np.mean(losses)),
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'mean_loss': report['mean_loss'], 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
