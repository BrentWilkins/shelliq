#!/usr/bin/env python3
"""Compare base and LoRA generations on held-out SemanticDocumentV2 targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from shelliq_training.checkpoint import (  # noqa: E402
    TargetFormat,
    restore_adapter_checkpoint,
)
from shelliq_training.config import Qwen2Config  # noqa: E402
from shelliq_training.data import (  # noqa: E402
    IGNORE_INDEX,
    Corpus,
    SFTRecord,
    Split,
    load_semantic_jsonl,
    split_records,
    tokenize_record,
)
from shelliq_training.evaluation import ModelPrediction  # noqa: E402
from shelliq_training.lora import inject_lora  # noqa: E402
from shelliq_training.model import Qwen2ForCausalLM  # noqa: E402
from shelliq_training.prompt import PromptContract  # noqa: E402
from shelliq_training.semantic_evaluation import (  # noqa: E402
    GroundingAudit,
    evaluate_semantic_predictions,
    load_grounding_audit,
)
from shelliq_training.weights import load_hf_state_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--model-id', default=MODEL_ID)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--examples', type=int, default=35)
    parser.add_argument(
        '--selection',
        choices=('test', 'all'),
        default='test',
        help='Evaluate the command-grouped test split or every row in a dedicated benchmark.',
    )
    parser.add_argument('--sequence-length', type=int, default=384)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=32.0)
    parser.add_argument(
        '--baseline-report',
        type=Path,
        help='Reuse base-model generations from an exact-matching prior report.',
    )
    parser.add_argument('--priority-source', default='shelliq-curated')
    parser.add_argument(
        '--prompt-contract',
        type=PromptContract,
        choices=PromptContract,
        default=PromptContract.CONTEXT_AUTHORITATIVE_V1,
    )
    parser.add_argument(
        '--grounding-audit',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'evaluation' / 'curated-grounding-v1.json',
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(records: list[SFTRecord], predictions: list[ModelPrediction], grounding_audit: GroundingAudit) -> dict[str, object]:
    return asdict(evaluate_semantic_predictions(records, predictions, grounding_audit))


def _by_source(
    records: list[SFTRecord], predictions: list[ModelPrediction], grounding_audit: GroundingAudit
) -> dict[str, dict[str, object]]:
    prediction_by_id = {prediction.record_id: prediction for prediction in predictions}
    return {
        source: _metrics(
            source_records,
            [prediction_by_id[record.record_id] for record in source_records],
            grounding_audit,
        )
        for source in sorted({record.source for record in records})
        if (source_records := [record for record in records if record.source == source])
    }


def _load_baseline_predictions(
    path: Path,
    records: list[SFTRecord],
    *,
    dataset_sha256: str,
    prompt_contract: PromptContract,
    selection: str,
    model_id: str,
) -> tuple[list[ModelPrediction], list[str]]:
    report = json.loads(path.read_text())
    if report.get('model_id') != model_id:
        raise ValueError(f'{path}: baseline model mismatch')
    if report.get('prompt_contract') != prompt_contract.value:
        raise ValueError(f'{path}: baseline prompt contract mismatch')
    dataset = report.get('dataset')
    if not isinstance(dataset, dict) or dataset.get('sha256') != dataset_sha256:
        raise ValueError(f'{path}: baseline dataset mismatch')
    selection_data = report.get('selection')
    if not isinstance(selection_data, dict) or selection_data.get('selection') != selection:
        raise ValueError(f'{path}: baseline selection mismatch')
    examples = report.get('examples')
    if not isinstance(examples, list) or len(examples) != len(records):
        raise ValueError(f'{path}: baseline example count mismatch')
    by_id = {record.record_id: record for record in records}
    texts: dict[str, str] = {}
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError(f'{path}: invalid baseline example')
        record_id = example.get('record_id')
        text = example.get('baseline')
        expected = example.get('expected')
        if not isinstance(record_id, str) or not isinstance(text, str):
            raise ValueError(f'{path}: invalid baseline prediction')
        record = by_id.get(record_id)
        if record is None or expected != record.response or record_id in texts:
            raise ValueError(f'{path}: baseline example drift for {record_id!r}')
        texts[record_id] = text
    if set(texts) != set(by_id):
        raise ValueError(f'{path}: baseline record IDs mismatch')
    ordered_texts = [texts[record.record_id] for record in records]
    predictions = [ModelPrediction(record.record_id, text) for record, text in zip(records, ordered_texts, strict=True)]
    return predictions, ordered_texts


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')
    if not args.output.parent.is_dir():
        raise SystemExit(f'output parent does not exist: {args.output.parent}')
    if args.examples <= 0:
        raise SystemExit('--examples must be positive')
    if args.rank <= 0:
        raise SystemExit('--rank must be positive')
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        raise SystemExit('--alpha must be finite and positive')
    if jax.default_backend() != 'gpu':
        raise SystemExit('evaluate_semantic_checkpoint.py requires JAX GPU backend')

    records = load_semantic_jsonl(args.semantic_dataset)
    splits = split_records(records, corpus=Corpus.DISTRIBUTABLE, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    if args.selection == 'all':
        eval_records = records
        eval_examples = [
            tokenize_record(
                record,
                tokenizer,
                max_length=args.sequence_length,
                prompt_contract=args.prompt_contract,
            )
            for record in eval_records
        ]
        too_long = [
            record.record_id
            for record, example in zip(eval_records, eval_examples, strict=True)
            if next(index for index, label in enumerate(example.labels) if label != IGNORE_INDEX) + GENERATION_TOKENS
            > args.sequence_length
        ]
        if too_long:
            raise SystemExit(f'benchmark prompts leave fewer than {GENERATION_TOKENS} generation tokens: {too_long}')
    else:
        eval_records, eval_examples = select_examples(
            splits[Split.TEST],
            tokenizer,
            count=args.examples,
            max_length=args.sequence_length,
            seed=args.seed + 1,
            priority_source=args.priority_source,
            prompt_contract=args.prompt_contract,
        )
    grounding_audit = load_grounding_audit(args.grounding_audit)
    selected_ids = {record.record_id for record in eval_records}
    audited_ids = set(grounding_audit)
    if selected_ids != audited_ids:
        missing = sorted(selected_ids - audited_ids)
        extra = sorted(audited_ids - selected_ids)
        raise SystemExit(f'grounding audit does not match selection: missing={missing}, extra={extra}')
    print(f'evaluation selection: {len(eval_records)} rows, sources={dict(Counter(record.source for record in eval_records))}')

    config_path = hf_hub_download(args.model_id, 'config.json', local_files_only=True)
    model = Qwen2ForCausalLM(
        Qwen2Config.from_json(Path(config_path)),
        param_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(0),
    )
    weights_path = hf_hub_download(args.model_id, 'model.safetensors', local_files_only=True)
    load_hf_state_dict(model, load_file(weights_path), param_dtype=jnp.bfloat16)
    inject_lora(model, rank=args.rank, alpha=args.alpha, rngs=nnx.Rngs(1))

    dataset_sha256 = _sha256(args.semantic_dataset)
    if args.baseline_report is None:
        baseline_predictions, baseline_texts = _generate(
            model,
            eval_records,
            eval_examples,
            tokenizer,
            sequence_length=args.sequence_length,
            label='base evaluation',
        )
    else:
        baseline_predictions, baseline_texts = _load_baseline_predictions(
            args.baseline_report,
            eval_records,
            dataset_sha256=dataset_sha256,
            prompt_contract=args.prompt_contract,
            selection=args.selection,
            model_id=args.model_id,
        )
    metadata = restore_adapter_checkpoint(
        args.checkpoint,
        model,
        model_id=args.model_id,
        corpus=Corpus.DISTRIBUTABLE,
        target_format=TargetFormat.SEMANTIC_DOCUMENT_V2,
        prompt_contract=args.prompt_contract,
    )
    trained_predictions, trained_texts = _generate(
        model,
        eval_records,
        eval_examples,
        tokenizer,
        sequence_length=args.sequence_length,
        label=f'rank {args.rank} evaluation',
    )

    report = {
        'evaluation_schema_version': 2,
        'target_format': 'semantic-document-v2-json',
        'prompt_contract': args.prompt_contract.value,
        'model_id': args.model_id,
        'dataset': {
            'path': str(args.semantic_dataset),
            'sha256': dataset_sha256,
            'accepted_records': len(records),
        },
        'checkpoint': {'path': str(args.checkpoint), 'step': metadata.step},
        'adapter': {
            'rank': metadata.rank,
            'alpha': metadata.alpha,
            'scale': metadata.alpha / metadata.rank,
        },
        'grounding_audit': {
            'path': str(args.grounding_audit),
            'sha256': _sha256(args.grounding_audit),
        },
        'selection': {
            'seed': args.seed,
            'examples': len(eval_records),
            'generation_tokens': GENERATION_TOKENS,
            'priority_source': args.priority_source,
            'selection': args.selection,
            'source_counts': dict(Counter(record.source for record in eval_records)),
        },
        'metric_notes': {
            'document_envelope_rate': 'Exact top-level v2/zsh/s JSON contract; not Rust round-trip validation.',
            'structural_exact_match': 'Decoded JSON equality, ignoring insignificant whitespace.',
            'grounded_document_exact_match': (
                'Decoded JSON equality after replacing only manually audited prompt-variable literals; '
                'all commands, flags, argument positions, redirects, and AST shape remain exact.'
            ),
            'first_command_accuracy': 'Literal first command name in the first pipeline stage.',
            'command_flag_sequence_exact_match': (
                'First command and ordered dash-prefixed arguments; ignores operand literals '
                'that may be unspecified by the prompt.'
            ),
        },
        'baseline': {
            'overall': _metrics(eval_records, baseline_predictions, grounding_audit),
            'by_source': _by_source(eval_records, baseline_predictions, grounding_audit),
        },
        'trained': {
            'overall': _metrics(eval_records, trained_predictions, grounding_audit),
            'by_source': _by_source(eval_records, trained_predictions, grounding_audit),
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
