#!/usr/bin/env python3
"""Prepare, train, and evaluate the frozen custom-versus-CodeT5 comparison."""

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
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, RobertaTokenizer

from shelliq_training.compiler_comparison import (
    ComparisonManifest,
    freeze_comparison_manifest,
    load_comparison_manifest,
    require_matching_dataset,
    write_comparison_manifest,
)
from shelliq_training.compiler_data import (
    LABEL_IGNORE_INDEX,
    CompilerBatch,
    CompilerExample,
    CompilerSpecialTokens,
    collate_compiler,
    compiler_special_tokens,
    tokenize_compiler_record,
)
from shelliq_training.data import SFTRecord, Split, load_semantic_jsonl
from shelliq_training.prompt import PromptContract
from shelliq_training.semantic_compiler import CompilerConfig, SemanticCompiler
from shelliq_training.teacher_verification import rust_validate_documents, verify_against_reference

EXPERIMENT = 'semantic-compiler-heldout-v1'
CUSTOM_TOKENIZER_ID = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
CODET5_MODEL_ID = 'Salesforce/codet5-small'
PROMPT_CONTRACT = PromptContract.CONTEXT_AUTHORITATIVE_V1
CUSTOM_PARAMETER_TARGET = 60_207_680


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'inspect', 'train', 'evaluate'))
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--contender', choices=('custom', 'codet5'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--validator', type=Path, default=Path('../target/debug/validate-semantic-documents'))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--source-length', type=int, default=256)
    parser.add_argument('--target-length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20260825)
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> None:
    records = load_semantic_jsonl(args.semantic_dataset)
    digest = hashlib.sha256(args.semantic_dataset.read_bytes()).hexdigest()
    manifest = freeze_comparison_manifest(
        records,
        dataset_sha256=digest,
        experiment=EXPERIMENT,
        split_seed=args.seed,
        validation_fraction=0.1,
        test_fraction=0.1,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_comparison_manifest(args.manifest, manifest)
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))


def load_frozen_data(args: argparse.Namespace) -> tuple[ComparisonManifest, dict[Split, list[SFTRecord]]]:
    manifest = load_comparison_manifest(args.manifest)
    if manifest.experiment != EXPERIMENT:
        raise ValueError(f'expected experiment {EXPERIMENT!r}, got {manifest.experiment!r}')
    require_matching_dataset(args.semantic_dataset, manifest)
    records = load_semantic_jsonl(args.semantic_dataset)
    return manifest, manifest.records_by_split(records)


def load_tokenizer(contender: str):
    if contender == 'custom':
        return AutoTokenizer.from_pretrained(CUSTOM_TOKENIZER_ID)
    # Transformers 5 cannot deserialize this checkpoint's Transformers 4.10
    # AddedToken dictionaries. Constructing the documented Roberta tokenizer
    # from the official vocabulary and merges preserves the exact 32,100 IDs.
    vocab = hf_hub_download(CODET5_MODEL_ID, 'vocab.json')
    merges = hf_hub_download(CODET5_MODEL_ID, 'merges.txt')
    return RobertaTokenizer(
        vocab=vocab,
        merges=merges,
        additional_special_tokens=[f'<extra_id_{index}>' for index in range(99, -1, -1)],
    )


def tokenize_splits(
    records: dict[Split, list[SFTRecord]],
    tokenizer,
    *,
    source_length: int,
    target_length: int,
    included_splits: Sequence[Split] = tuple(Split),
) -> dict[Split, list[CompilerExample]]:
    return {
        split: [
            tokenize_compiler_record(
                record,
                tokenizer,
                max_source_length=source_length,
                max_target_length=target_length,
                prompt_contract=PROMPT_CONTRACT,
            )
            for record in records[split]
        ]
        for split in included_splits
    }


def custom_config(tokenizer, *, source_length: int, target_length: int) -> CompilerConfig:
    special = compiler_special_tokens(tokenizer)
    return CompilerConfig(
        vocab_size=len(tokenizer),
        pad_token_id=special.pad_token_id,
        decoder_start_token_id=special.decoder_start_token_id,
        eos_token_id=special.eos_token_id,
        d_model=320,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        feedforward_size=1280,
        dropout=0.1,
        max_source_length=source_length,
        max_target_length=target_length,
    )


def build_model(contender: str, tokenizer, *, source_length: int, target_length: int, pretrained: bool):
    if contender == 'custom':
        return SemanticCompiler(custom_config(tokenizer, source_length=source_length, target_length=target_length))
    if pretrained:
        return AutoModelForSeq2SeqLM.from_pretrained(CODET5_MODEL_ID)
    config = AutoConfig.from_pretrained(CODET5_MODEL_ID)
    return AutoModelForSeq2SeqLM.from_config(config)


def model_identity(contender: str, model) -> dict[str, object]:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if contender == 'custom' and parameter_count != CUSTOM_PARAMETER_TARGET:
        raise ValueError(f'custom parameter count drifted: expected {CUSTOM_PARAMETER_TARGET}, got {parameter_count}')
    ratio = parameter_count / CUSTOM_PARAMETER_TARGET
    if contender == 'codet5' and not 0.95 <= ratio <= 1.05:
        raise ValueError(f'CodeT5/custom parameter ratio {ratio:.4f} is outside the preregistered 5% tolerance')
    config = model.config.to_dict() if contender == 'codet5' else model.config.to_dict()
    return {
        'contender': contender,
        'model_id': CODET5_MODEL_ID if contender == 'codet5' else 'custom-scratch-v1',
        'resolved_revision': getattr(model.config, '_commit_hash', None) if contender == 'codet5' else None,
        'parameter_count': parameter_count,
        'trainable_parameter_count': trainable_parameters,
        'custom_parameter_ratio': ratio,
        'config': config,
    }


def inspect(args: argparse.Namespace) -> None:
    contender = require_contender(args)
    _, records = load_frozen_data(args)
    tokenizer = load_tokenizer(contender)
    examples = tokenize_splits(
        records,
        tokenizer,
        source_length=args.source_length,
        target_length=args.target_length,
    )
    model = build_model(
        contender,
        tokenizer,
        source_length=args.source_length,
        target_length=args.target_length,
        pretrained=contender == 'codet5',
    )
    result = {
        'model': model_identity(contender, model),
        'tokenizer_id': CUSTOM_TOKENIZER_ID if contender == 'custom' else CODET5_MODEL_ID,
        'splits': {
            split.value: {
                'records': len(examples[split]),
                'maximum_source_tokens': max(len(example.source_ids) for example in examples[split]),
                'maximum_target_tokens': max(len(example.target_ids) for example in examples[split]),
            }
            for split in Split
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def contender_loss(model, contender: str, batch: CompilerBatch) -> torch.Tensor:
    if contender == 'custom':
        return model.loss(batch)
    output = model(
        input_ids=batch.source_ids,
        attention_mask=batch.source_attention_mask,
        labels=batch.labels,
    )
    return output.loss


def ordered_batches(examples: Sequence[CompilerExample], *, batch_size: int, seed: int) -> list[list[CompilerExample]]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(examples), generator=generator).tolist()
    return [[examples[index] for index in order[offset : offset + batch_size]] for offset in range(0, len(order), batch_size)]


def mean_loss(
    model,
    contender: str,
    examples: Sequence[CompilerExample],
    *,
    batch_size: int,
    special_tokens: CompilerSpecialTokens,
    source_length: int,
    target_length: int,
    device: torch.device,
) -> float:
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    with torch.no_grad():
        for offset in range(0, len(examples), batch_size):
            batch = collate_compiler(
                examples[offset : offset + batch_size],
                source_length=source_length,
                target_length=target_length,
                special_tokens=special_tokens,
            ).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = contender_loss(model, contender, batch)
            tokens = int((batch.labels != LABEL_IGNORE_INDEX).sum())
            weighted_loss += float(loss) * tokens
            token_count += tokens
    return weighted_loss / token_count


def save_checkpoint(path: Path, model, metadata: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(
        {
            'schema_version': 1,
            'metadata': metadata,
            'model_state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(args: argparse.Namespace) -> None:
    contender = require_contender(args)
    checkpoint, report_path = require_outputs(args)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError('epochs and batch-size must be positive')
    manifest, records = load_frozen_data(args)
    tokenizer = load_tokenizer(contender)
    tokenized = tokenize_splits(
        records,
        tokenizer,
        source_length=args.source_length,
        target_length=args.target_length,
        included_splits=(Split.TRAIN, Split.VALIDATION),
    )
    special = compiler_special_tokens(tokenizer)
    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(
        contender,
        tokenizer,
        source_length=args.source_length,
        target_length=args.target_length,
        pretrained=contender == 'codet5',
    ).to(device)
    identity = model_identity(contender, model)
    learning_rate = 1e-3 if contender == 'custom' else 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    best_loss = float('inf')
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_weighted_loss = 0.0
        train_tokens = 0
        for examples in ordered_batches(tokenized[Split.TRAIN], batch_size=args.batch_size, seed=args.seed + epoch):
            batch = collate_compiler(
                examples,
                source_length=args.source_length,
                target_length=args.target_length,
                special_tokens=special,
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = contender_loss(model, contender, batch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f'non-finite loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = int((batch.labels != LABEL_IGNORE_INDEX).sum())
            train_weighted_loss += float(loss) * tokens
            train_tokens += tokens
        validation_loss = mean_loss(
            model,
            contender,
            tokenized[Split.VALIDATION],
            batch_size=args.batch_size,
            special_tokens=special,
            source_length=args.source_length,
            target_length=args.target_length,
            device=device,
        )
        train_loss = train_weighted_loss / train_tokens
        history.append({'epoch': epoch, 'train_loss': train_loss, 'validation_loss': validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            metadata = {
                'experiment': EXPERIMENT,
                'contender': contender,
                'dataset_sha256': manifest.dataset_sha256,
                'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                'tokenizer_id': CUSTOM_TOKENIZER_ID if contender == 'custom' else CODET5_MODEL_ID,
                'prompt_contract': PROMPT_CONTRACT.value,
                'seed': args.seed,
                'source_length': args.source_length,
                'target_length': args.target_length,
                'best_epoch': best_epoch,
                'best_validation_loss': best_loss,
                'model': identity,
            }
            save_checkpoint(checkpoint, model, metadata)
        elapsed = time.monotonic() - started
        print(
            f'contender={contender} epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} '
            f'validation_loss={validation_loss:.6f} best_epoch={best_epoch} elapsed_seconds={elapsed:.1f}',
            flush=True,
        )

    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'train',
        'contender': contender,
        'dataset_sha256': manifest.dataset_sha256,
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'model': identity,
        'tokenizer_id': CUSTOM_TOKENIZER_ID if contender == 'custom' else CODET5_MODEL_ID,
        'prompt_contract': PROMPT_CONTRACT.value,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'record_presentations': args.epochs * len(tokenized[Split.TRAIN]),
        'learning_rate': learning_rate,
        'best_epoch': best_epoch,
        'best_validation_loss': best_loss,
        'elapsed_seconds': time.monotonic() - started,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': checkpoint_sha256,
        'history': history,
    }
    write_report(report_path, report)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    contender = require_contender(args)
    checkpoint, report_path = require_outputs(args)
    manifest, records = load_frozen_data(args)
    tokenizer = load_tokenizer(contender)
    tokenized = tokenize_splits(
        records,
        tokenizer,
        source_length=args.source_length,
        target_length=args.target_length,
        included_splits=(Split.TEST,),
    )
    special = compiler_special_tokens(tokenizer)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata = saved['metadata']
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if (
        metadata['experiment'] != EXPERIMENT
        or metadata['contender'] != contender
        or metadata['dataset_sha256'] != manifest.dataset_sha256
        or metadata['manifest_sha256'] != manifest_sha256
    ):
        raise ValueError('checkpoint metadata does not match the frozen comparison')
    model = build_model(
        contender,
        tokenizer,
        source_length=args.source_length,
        target_length=args.target_length,
        pretrained=False,
    )
    model.load_state_dict(saved['model_state_dict'], strict=True)
    device = resolve_device(args.device)
    model.to(device).eval()
    test_records = records[Split.TEST]
    test_examples = tokenized[Split.TEST]
    generated_documents: list[str] = []
    started = time.monotonic()
    for offset in range(0, len(test_examples), args.batch_size):
        batch = collate_compiler(
            test_examples[offset : offset + args.batch_size],
            source_length=args.source_length,
            target_length=args.target_length,
            special_tokens=special,
        ).to(device)
        if contender == 'custom':
            generated_ids = model.generate(
                batch.source_ids,
                batch.source_attention_mask,
                max_new_tokens=args.target_length,
            )
        else:
            generated_ids = model.generate(
                input_ids=batch.source_ids,
                attention_mask=batch.source_attention_mask,
                max_new_tokens=args.target_length,
                do_sample=False,
                num_beams=1,
            )
        generated_documents.extend(
            tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        )
    generated_documents = [document.strip() for document in generated_documents]
    rust_results = rust_validate_documents(generated_documents, args.validator)
    outcomes: list[dict[str, object]] = []
    exact_count = 0
    valid_count = 0
    accepted_count = 0
    for record, generated, rust_result in zip(test_records, generated_documents, rust_results, strict=True):
        verification = verify_against_reference(generated, json.loads(record.response), rust_result)
        exact = generated == record.response
        exact_count += exact
        valid_count += rust_result.valid
        accepted_count += verification.accepted
        outcomes.append(
            {
                'record_id': record.record_id,
                'exact': exact,
                'rust_round_trip': rust_result.valid,
                'reference_accepted': verification.accepted,
                'failures': list(verification.failures),
                'expected': record.response,
                'generated': generated,
                'rendered': rust_result.rendered,
            }
        )
    count = len(outcomes)
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'test',
        'contender': contender,
        'dataset_sha256': manifest.dataset_sha256,
        'manifest_sha256': manifest_sha256,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'checkpoint_metadata': metadata,
        'elapsed_seconds': time.monotonic() - started,
        'metrics': {
            'examples': count,
            'exact': exact_count,
            'exact_rate': exact_count / count,
            'rust_round_trip': valid_count,
            'rust_round_trip_rate': valid_count / count,
            'reference_accepted': accepted_count,
            'reference_accepted_rate': accepted_count / count,
        },
        'outcomes': outcomes,
    }
    write_report(report_path, report)
    print(json.dumps(report['metrics'], sort_keys=True), flush=True)


def resolve_device(value: str) -> torch.device:
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if value == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return torch.device(value)


def require_contender(args: argparse.Namespace) -> str:
    if args.contender is None:
        raise ValueError(f'--contender is required for phase {args.phase}')
    return args.contender


def require_outputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.checkpoint is None or args.report is None:
        raise ValueError(f'--checkpoint and --report are required for phase {args.phase}')
    return args.checkpoint, args.report


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


def main() -> None:
    args = parse_args()
    if args.phase == 'prepare':
        prepare(args)
    elif args.phase == 'inspect':
        inspect(args)
    elif args.phase == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
