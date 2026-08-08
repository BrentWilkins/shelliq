#!/usr/bin/env python3
"""Compare base and LoRA generations on held-out SemanticDocumentV2 targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors.flax import load_file
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_checkpoint import _generate  # noqa: E402
from scripts.smoke_finetune import GENERATION_TOKENS, MODEL_ID, select_examples  # noqa: E402
from shelliq_training.checkpoint import restore_checkpoint  # noqa: E402
from shelliq_training.config import Qwen2Config  # noqa: E402
from shelliq_training.data import Corpus, SFTRecord, Split, load_semantic_jsonl, split_records  # noqa: E402
from shelliq_training.evaluation import ModelPrediction  # noqa: E402
from shelliq_training.lora import inject_lora  # noqa: E402
from shelliq_training.model import Qwen2ForCausalLM  # noqa: E402
from shelliq_training.semantic_evaluation import evaluate_semantic_predictions  # noqa: E402
from shelliq_training.training import create_lora_optimizer  # noqa: E402
from shelliq_training.weights import load_hf_state_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--examples', type=int, default=35)
    parser.add_argument('--sequence-length', type=int, default=384)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--priority-source', default='shelliq-curated')
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(records: list[SFTRecord], predictions: list[ModelPrediction]) -> dict[str, object]:
    return asdict(evaluate_semantic_predictions(records, predictions))


def _by_source(records: list[SFTRecord], predictions: list[ModelPrediction]) -> dict[str, dict[str, object]]:
    prediction_by_id = {prediction.record_id: prediction for prediction in predictions}
    return {
        source: _metrics(
            source_records,
            [prediction_by_id[record.record_id] for record in source_records],
        )
        for source in sorted({record.source for record in records})
        if (source_records := [record for record in records if record.source == source])
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')
    if not args.output.parent.is_dir():
        raise SystemExit(f'output parent does not exist: {args.output.parent}')
    if args.examples <= 0:
        raise SystemExit('--examples must be positive')
    if jax.default_backend() != 'gpu':
        raise SystemExit('evaluate_semantic_checkpoint.py requires JAX GPU backend')

    records = load_semantic_jsonl(args.semantic_dataset)
    splits = split_records(records, corpus=Corpus.DISTRIBUTABLE, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    eval_records, eval_examples = select_examples(
        splits[Split.TEST],
        tokenizer,
        count=args.examples,
        max_length=args.sequence_length,
        seed=args.seed + 1,
        priority_source=args.priority_source,
    )
    print(f'evaluation selection: {len(eval_records)} rows, sources={dict(Counter(record.source for record in eval_records))}')

    model = Qwen2ForCausalLM(Qwen2Config(), param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    weights_path = hf_hub_download(MODEL_ID, 'model.safetensors', local_files_only=True)
    load_hf_state_dict(model, load_file(weights_path), param_dtype=jnp.bfloat16)
    inject_lora(model, rank=16, alpha=32, rngs=nnx.Rngs(1))
    optimizer = create_lora_optimizer(model)

    baseline_predictions, baseline_texts = _generate(
        model,
        eval_records,
        eval_examples,
        tokenizer,
        sequence_length=args.sequence_length,
    )
    metadata = restore_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        model_id=MODEL_ID,
        corpus=Corpus.DISTRIBUTABLE,
    )
    trained_predictions, trained_texts = _generate(
        model,
        eval_records,
        eval_examples,
        tokenizer,
        sequence_length=args.sequence_length,
    )

    report = {
        'evaluation_schema_version': 1,
        'target_format': 'semantic-document-v2-json',
        'model_id': MODEL_ID,
        'dataset': {
            'path': str(args.semantic_dataset),
            'sha256': _sha256(args.semantic_dataset),
            'accepted_records': len(records),
        },
        'checkpoint': {'path': str(args.checkpoint), 'step': metadata.step},
        'selection': {
            'seed': args.seed,
            'examples': len(eval_records),
            'generation_tokens': GENERATION_TOKENS,
            'priority_source': args.priority_source,
            'source_counts': dict(Counter(record.source for record in eval_records)),
        },
        'metric_notes': {
            'document_envelope_rate': 'Exact top-level v2/zsh/s JSON contract; not Rust round-trip validation.',
            'structural_exact_match': 'Decoded JSON equality, ignoring insignificant whitespace.',
            'first_command_accuracy': 'Literal first command name in the first pipeline stage.',
        },
        'baseline': {
            'overall': _metrics(eval_records, baseline_predictions),
            'by_source': _by_source(eval_records, baseline_predictions),
        },
        'trained': {
            'overall': _metrics(eval_records, trained_predictions),
            'by_source': _by_source(eval_records, trained_predictions),
        },
        'examples': [
            {
                'record_id': record.record_id,
                'source': record.source,
                'instruction': record.instruction,
                'expected': record.response,
                'baseline': baseline,
                'trained': trained,
            }
            for record, baseline, trained in zip(eval_records, baseline_texts, trained_texts, strict=True)
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'baseline': report['baseline'], 'trained': report['trained']}, sort_keys=True))
    print(f'report: {args.output}')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit(f'{type(error).__name__}: {error}') from error
