"""Compare base and LoRA-checkpoint generations on held-out commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors.flax import load_file
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_finetune import (  # noqa: E402
    GENERATION_TOKENS,
    MODEL_ID,
    automatic_preflight,
    greedy_completion,
    select_examples,
)
from shelliq_training.checkpoint import TargetFormat, restore_checkpoint  # noqa: E402
from shelliq_training.config import Qwen2Config  # noqa: E402
from shelliq_training.data import (  # noqa: E402
    IGNORE_INDEX,
    Corpus,
    SFTRecord,
    Split,
    TokenizedExample,
    load_jsonl,
    split_records,
)
from shelliq_training.evaluation import (  # noqa: E402
    EvaluationExample,
    ModelPrediction,
    evaluate_predictions,
    parse_command,
)
from shelliq_training.lora import inject_lora  # noqa: E402
from shelliq_training.model import Qwen2ForCausalLM  # noqa: E402
from shelliq_training.training import create_lora_optimizer  # noqa: E402
from shelliq_training.weights import load_hf_state_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--examples', type=int, default=64)
    parser.add_argument('--sequence-length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--priority-source', default='shelliq-curated')
    parser.add_argument('--option-arities', type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _short_completion(example: TokenizedExample) -> bool:
    return sum(label != IGNORE_INDEX for label in example.labels) <= GENERATION_TOKENS


def _load_option_arities(path: Path | None) -> dict[str, dict[str, int]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {'schema_version', 'records'}:
        raise ValueError('option arity file must contain only schema_version and records')
    if raw['schema_version'] != 1 or not isinstance(raw['records'], dict):
        raise ValueError('unsupported option arity schema')
    result = {}
    for record_id, arities in raw['records'].items():
        if not isinstance(record_id, str) or not isinstance(arities, dict):
            raise ValueError('option arity records must map record IDs to objects')
        if not all(
            isinstance(flag, str) and flag.startswith('-') and not isinstance(arity, bool) and arity in {0, 1}
            for flag, arity in arities.items()
        ):
            raise ValueError(f'{record_id}: option arities must be 0 or 1 for flag keys')
        result[record_id] = arities
    return result


def _validate_option_arities(records: Sequence[SFTRecord], option_arities: dict[str, dict[str, int]]) -> None:
    records_by_id = {record.record_id: record for record in records}
    unknown = sorted(option_arities.keys() - records_by_id.keys())
    if unknown:
        raise ValueError(f'option arity records are absent from dataset: {unknown[:3]!r}')
    for record_id, arities in option_arities.items():
        parsed = parse_command(records_by_id[record_id].response, arities)
        if parsed is None:
            continue
        expected_flags = set(parsed.flags)
        annotated_flags = set(arities)
        if annotated_flags != expected_flags:
            raise ValueError(
                f'{record_id}: option arity flags differ from expected command; '
                f'missing={sorted(expected_flags - annotated_flags)!r}, '
                f'extra={sorted(annotated_flags - expected_flags)!r}'
            )


def _select_heldout(
    records: Sequence[SFTRecord],
    tokenizer: object,
    *,
    count: int,
    sequence_length: int,
    seed: int,
    priority_source: str | None,
) -> tuple[list[SFTRecord], list[TokenizedExample]]:
    candidate_count = min(max(count * 3, count), len({record.command for record in records}))
    candidate_records, candidate_examples = select_examples(
        records,
        tokenizer,
        count=candidate_count,
        max_length=sequence_length,
        seed=seed,
        priority_source=priority_source,
    )
    selected = [
        (record, example)
        for record, example in zip(candidate_records, candidate_examples, strict=True)
        if _short_completion(example)
    ][:count]
    if len(selected) != count:
        raise ValueError(f'only found {len(selected)} short held-out examples; requested {count}')
    return [record for record, _ in selected], [example for _, example in selected]


def _generate(
    model: Qwen2ForCausalLM,
    records: Sequence[SFTRecord],
    examples: Sequence[TokenizedExample],
    tokenizer: object,
    *,
    sequence_length: int,
) -> tuple[list[ModelPrediction], list[str]]:
    # Compile before measuring per-example latency.
    greedy_completion(model, examples[0], tokenizer, sequence_length=sequence_length)
    predictions = []
    texts = []
    for record, example in zip(records, examples, strict=True):
        started = time.perf_counter()
        text = greedy_completion(model, example, tokenizer, sequence_length=sequence_length)
        latency_ms = (time.perf_counter() - started) * 1000
        predictions.append(ModelPrediction(record.record_id, text, latency_ms))
        texts.append(text)
    return predictions, texts


def _native_zsh_valid(source: str) -> bool:
    try:
        result = subprocess.run(
            ['zsh', '-f', '-n', '-c', source],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def _metrics(
    records: Sequence[SFTRecord],
    predictions: Sequence[ModelPrediction],
    option_arities: dict[str, dict[str, int]],
) -> dict[str, object]:
    pairs = [
        (record, prediction)
        for record, prediction in zip(records, predictions, strict=True)
        if parse_command(record.response, {}) is not None
    ]
    basic_records = [record for record, _ in pairs]
    basic_predictions = [prediction for _, prediction in pairs]
    examples = [EvaluationExample(record, {}) for record in basic_records]
    metrics = asdict(evaluate_predictions(examples, basic_predictions))
    metrics['operand_exact_match_zero_arity'] = metrics.pop('operand_exact_match')
    metrics['option_argument_accuracy'] = None
    annotated_pairs = [(record, prediction) for record, prediction in pairs if record.record_id in option_arities]
    metrics['arity_annotated_examples'] = len(annotated_pairs)
    if annotated_pairs:
        annotated_metrics = evaluate_predictions(
            [EvaluationExample(record, option_arities[record.record_id]) for record, _ in annotated_pairs],
            [prediction for _, prediction in annotated_pairs],
        )
        metrics['option_argument_accuracy'] = annotated_metrics.option_argument_accuracy
        metrics['operand_exact_match'] = annotated_metrics.operand_exact_match
    else:
        metrics['operand_exact_match'] = None
    metrics['total_examples'] = len(records)
    metrics['basic_grammar_examples'] = len(basic_records)
    metrics['exact_match_all'] = sum(
        record.response.strip() == prediction.text.strip() for record, prediction in zip(records, predictions, strict=True)
    ) / len(records)
    metrics['native_zsh_rate'] = sum(_native_zsh_valid(item.text) for item in predictions) / len(predictions)
    return metrics


def _source_metrics(
    records: Sequence[SFTRecord],
    predictions: Sequence[ModelPrediction],
    option_arities: dict[str, dict[str, int]],
) -> dict[str, dict[str, object]]:
    prediction_by_id = {prediction.record_id: prediction for prediction in predictions}
    return {
        source: _metrics(
            source_records,
            [prediction_by_id[record.record_id] for record in source_records],
            option_arities,
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
        raise SystemExit('evaluate_checkpoint.py requires a JAX GPU backend')

    records = load_jsonl(args.dataset, corpus=Corpus.DISTRIBUTABLE)
    option_arities = _load_option_arities(args.option_arities)
    _validate_option_arities(records, option_arities)
    clean_records, rejected = automatic_preflight(records)
    splits = split_records(clean_records, corpus=Corpus.DISTRIBUTABLE, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    eval_records, eval_examples = _select_heldout(
        splits[Split.TEST],
        tokenizer,
        count=args.examples,
        sequence_length=args.sequence_length,
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
        target_format=TargetFormat.RAW_SHELL,
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
        'model_id': MODEL_ID,
        'dataset': {
            'path': str(args.dataset),
            'sha256': _sha256(args.dataset),
            'accepted_records': len(clean_records),
            'rejected_record_ids': sorted(rejected),
        },
        'checkpoint': {
            'path': str(args.checkpoint),
            'step': metadata.step,
        },
        'selection': {
            'seed': args.seed,
            'examples': len(eval_records),
            'generation_tokens': GENERATION_TOKENS,
            'priority_source': args.priority_source,
            'source_counts': dict(Counter(record.source for record in eval_records)),
        },
        'metric_notes': {
            'basic_grammar': 'Command, flag, and operand metrics exclude compound expected commands.',
            'option_arguments': 'Scored only on explicitly arity-annotated simple commands.',
            'operands': 'Semantic score uses annotations; zero-arity lexical floor is also retained.',
        },
        'option_arities': None
        if args.option_arities is None
        else {
            'path': str(args.option_arities),
            'sha256': _sha256(args.option_arities),
            'records': len(option_arities),
        },
        'baseline': {
            'overall': _metrics(eval_records, baseline_predictions, option_arities),
            'by_source': _source_metrics(eval_records, baseline_predictions, option_arities),
        },
        'trained': {
            'overall': _metrics(eval_records, trained_predictions, option_arities),
            'by_source': _source_metrics(eval_records, trained_predictions, option_arities),
        },
        'examples': [
            {
                'record_id': record.record_id,
                'source': record.source,
                'instruction': record.instruction,
                'expected': record.response,
                'baseline': baseline,
                'baseline_native_zsh_valid': _native_zsh_valid(baseline),
                'trained': trained,
                'trained_native_zsh_valid': _native_zsh_valid(trained),
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
