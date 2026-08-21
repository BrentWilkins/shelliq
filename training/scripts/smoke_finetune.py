"""Overfit a few real public tldr rows as an end-to-end GPU smoke test.

This deliberately measures plumbing, not model quality. It uses a deterministic
command-grouped split, keeps the base model frozen, and never reads personal data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors.flax import load_file
from transformers import AutoTokenizer

from shelliq_training.checkpoint import TargetFormat, restore_checkpoint, save_checkpoint
from shelliq_training.config import Qwen2Config
from shelliq_training.corpus_audit import preflight_records
from shelliq_training.data import (
    IGNORE_INDEX,
    Corpus,
    SequenceTooLongError,
    SFTRecord,
    Split,
    TokenizedExample,
    collate_sft,
    load_jsonl,
    load_semantic_jsonl,
    split_records,
    tokenize_record,
    tokenizer_pad_id,
)
from shelliq_training.lora import inject_lora, parameter_count
from shelliq_training.model import Qwen2ForCausalLM
from shelliq_training.progress import interactive_progress, training_progress
from shelliq_training.prompt import PromptContract
from shelliq_training.training import (
    CausalLMBatch,
    create_lora_optimizer,
    eval_step,
    train_step,
    train_step_with_gradient_norm,
    warmup_cosine_schedule,
)
from shelliq_training.tuning import EarlyStoppingTracker, LossObservation, best_observation
from shelliq_training.weights import load_hf_state_dict

MODEL_ID = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
GENERATION_TOKENS = 192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument('--dataset', type=Path)
    dataset.add_argument('--semantic-dataset', type=Path)
    parser.add_argument('--sequence-length', type=int, default=128)
    parser.add_argument('--train-examples', type=int, default=4)
    parser.add_argument('--all-train-examples', action='store_true')
    parser.add_argument('--eval-examples', type=int, default=4)
    parser.add_argument('--eval-split', type=Split, choices=Split, default=Split.TEST)
    parser.add_argument('--train-loss-eval-examples', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--lr-schedule', choices=('constant', 'warmup-cosine'), default='constant')
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--end-learning-rate', type=float, default=0.0)
    parser.add_argument('--eval-interval', type=int, default=0)
    parser.add_argument('--early-stopping-patience', type=int, default=0)
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0)
    parser.add_argument('--early-stopping-min-epochs', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--gradient-diagnostics', action='store_true')
    parser.add_argument('--shuffle-each-epoch', action='store_true')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=32.0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--split-seed', type=int)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--best-checkpoint-root', type=Path)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--heldout-probe', action='store_true')
    parser.add_argument('--priority-source')
    parser.add_argument(
        '--rehearsal-record-prefix',
        help='Repeat selected training rows whose record IDs start with this prefix.',
    )
    parser.add_argument('--rehearsal-weight', type=int, default=1)
    parser.add_argument('--prompt-contract', type=PromptContract, choices=PromptContract)
    return parser.parse_args()


def _ordered_records(records: Sequence[SFTRecord], seed: int) -> list[SFTRecord]:
    return sorted(
        records,
        key=lambda record: hashlib.sha256(f'{seed}\0{record.record_id}'.encode()).digest(),
    )


def automatic_preflight(records: Sequence[SFTRecord]) -> tuple[list[SFTRecord], dict[str, str]]:
    """Exclude obvious template and quoting defects from the unaudited smoke set."""
    return preflight_records(records)


def select_examples(
    records: Sequence[SFTRecord],
    tokenizer: object,
    *,
    count: int,
    max_length: int,
    seed: int,
    priority_source: str | None = None,
    prompt_contract: PromptContract = PromptContract.LEGACY_USER_V1,
) -> tuple[list[SFTRecord], list[TokenizedExample]]:
    """Choose deterministic rows, optionally retaining every priority-source row."""
    selected_records: list[SFTRecord] = []
    examples: list[TokenizedExample] = []
    seen_commands: set[str] = set()
    ordered = _ordered_records(records, seed)
    prioritized = [record for record in ordered if record.source == priority_source]
    remaining = [record for record in ordered if record.source != priority_source]

    def add_record(record: SFTRecord, *, require_new_command: bool) -> bool:
        if require_new_command and record.command in seen_commands:
            return False
        try:
            example = tokenize_record(  # type: ignore[arg-type]
                record,
                tokenizer,
                max_length=max_length,
                prompt_contract=prompt_contract,
            )
        except SequenceTooLongError:
            return False
        prompt_length = next(index for index, label in enumerate(example.labels) if label != IGNORE_INDEX)
        if prompt_length + GENERATION_TOKENS > max_length:
            return False
        selected_records.append(record)
        examples.append(example)
        seen_commands.add(record.command)
        return True

    for record in prioritized:
        if len(examples) >= count:
            break
        add_record(record, require_new_command=False)
    for record in remaining:
        if len(examples) >= count:
            break
        add_record(record, require_new_command=True)
    if len(examples) < count:
        raise ValueError(f'only found {len(examples)} usable examples; requested {count}')
    return selected_records, examples


def select_all_examples(
    records: Sequence[SFTRecord],
    tokenizer: object,
    *,
    max_length: int,
    seed: int,
    prompt_contract: PromptContract = PromptContract.LEGACY_USER_V1,
) -> tuple[list[SFTRecord], list[TokenizedExample]]:
    """Tokenize every usable row in deterministic order without command deduplication."""
    selected_records = []
    examples = []
    for record in _ordered_records(records, seed):
        try:
            example = tokenize_record(
                record,
                tokenizer,  # type: ignore[arg-type]
                max_length=max_length,
                prompt_contract=prompt_contract,
            )
        except SequenceTooLongError:
            continue
        prompt_length = next(index for index, label in enumerate(example.labels) if label != IGNORE_INDEX)
        if prompt_length + GENERATION_TOKENS > max_length:
            continue
        selected_records.append(record)
        examples.append(example)
    if not examples:
        raise ValueError('no usable training examples found')
    return selected_records, examples


def make_batches(
    examples: Sequence[TokenizedExample],
    *,
    batch_size: int,
    sequence_length: int,
    pad_token_id: int,
) -> list[CausalLMBatch]:
    if batch_size <= 0 or len(examples) % batch_size:
        raise ValueError('example counts must be divisible by a positive batch size')
    return [
        collate_sft(
            examples[offset : offset + batch_size],
            sequence_length=sequence_length,
            pad_token_id=pad_token_id,
        )
        for offset in range(0, len(examples), batch_size)
    ]


def apply_rehearsal_weight(
    records: Sequence[SFTRecord],
    examples: Sequence[TokenizedExample],
    *,
    record_prefix: str | None,
    weight: int,
    seed: int,
) -> tuple[list[SFTRecord], list[TokenizedExample]]:
    """Repeat a named curriculum slice without duplicating source records."""
    if len(records) != len(examples):
        raise ValueError('records and examples must have equal lengths')
    if weight < 1:
        raise ValueError('rehearsal weight must be positive')
    if record_prefix is None:
        if weight != 1:
            raise ValueError('rehearsal weight requires a record prefix')
        return list(records), list(examples)
    weighted: list[tuple[SFTRecord, TokenizedExample, int]] = []
    matched = 0
    for record, example in zip(records, examples, strict=True):
        repeats = weight if record.record_id.startswith(record_prefix) else 1
        matched += repeats > 1
        weighted.extend((record, example, occurrence) for occurrence in range(repeats))
    if not matched:
        raise ValueError(f'no selected training record IDs start with {record_prefix!r}')
    weighted.sort(key=lambda item: hashlib.sha256(f'{seed}\0{item[0].record_id}\0{item[2]}'.encode()).digest())
    return [item[0] for item in weighted], [item[1] for item in weighted]


def mean_loss(model: Qwen2ForCausalLM, batches: Sequence[CausalLMBatch]) -> float:
    losses = [float(np.asarray(eval_step(model, batch))) for batch in batches]
    return float(np.mean(losses))


@nnx.jit
def _greedy_generate(
    model: Qwen2ForCausalLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    eos_token_id: jax.Array,
    pad_token_id: jax.Array,
) -> jax.Array:
    prompt_length = jnp.sum(attention_mask[0], dtype=jnp.int32)

    def generate_token(
        offset: jax.Array,
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        token_ids, mask, finished = state
        write_index = prompt_length + offset
        logits = model(token_ids, mask)
        next_token = jnp.argmax(logits[0, write_index - 1].astype(jnp.float32))
        next_token = jnp.where(finished, pad_token_id, next_token)
        token_ids = token_ids.at[0, write_index].set(next_token)
        mask = mask.at[0, write_index].set(~finished)
        return token_ids, mask, finished | (next_token == eos_token_id)

    generated, _, _ = jax.lax.fori_loop(
        0,
        GENERATION_TOKENS,
        generate_token,
        (input_ids, attention_mask, jnp.asarray(False)),
    )
    return generated


def greedy_completion(
    model: Qwen2ForCausalLM,
    example: TokenizedExample,
    tokenizer: object,
    *,
    sequence_length: int,
) -> str:
    prompt_length = next(index for index, label in enumerate(example.labels) if label != IGNORE_INDEX)
    prompt_ids = example.input_ids[:prompt_length]
    pad_id = tokenizer_pad_id(tokenizer)  # type: ignore[arg-type]
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    if eos_id is None:
        raise ValueError('tokenizer has no eos_token_id')
    input_ids = np.full((1, sequence_length), pad_id, dtype=np.int32)
    attention_mask = np.zeros((1, sequence_length), dtype=np.int32)
    input_ids[0, :prompt_length] = prompt_ids
    attention_mask[0, :prompt_length] = 1
    generated = np.asarray(
        _greedy_generate(
            model,
            jnp.asarray(input_ids),
            jnp.asarray(attention_mask),
            jnp.asarray(eos_id, dtype=jnp.int32),
            jnp.asarray(pad_id, dtype=jnp.int32),
        )
    )[0, prompt_length : prompt_length + GENERATION_TOKENS]
    return tokenizer.decode(generated.tolist(), skip_special_tokens=True).strip()


def _print_generation(label: str, record: SFTRecord, generated: str) -> None:
    print(
        json.dumps(
            {
                'stage': label,
                'record_id': record.record_id,
                'instruction': record.instruction,
                'expected': record.response,
                'generated': generated,
            },
            ensure_ascii=False,
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _record_id_sha256(records: Sequence[SFTRecord]) -> str:
    payload = '\n'.join(record.record_id for record in records).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    if args.report is not None and args.report.exists():
        raise SystemExit(f'report already exists: {args.report}')
    if args.report is not None and not args.report.parent.is_dir():
        raise SystemExit(f'report parent does not exist: {args.report.parent}')
    if jax.default_backend() != 'gpu':
        raise SystemExit('smoke_finetune.py requires a JAX GPU backend')
    if args.sequence_length < GENERATION_TOKENS + 2:
        raise SystemExit(f'--sequence-length must exceed {GENERATION_TOKENS + 1}')
    if args.steps <= 0:
        raise SystemExit('--steps must be positive')
    if args.batch_size <= 0:
        raise SystemExit('--batch-size must be positive')
    if args.train_loss_eval_examples < 0:
        raise SystemExit('--train-loss-eval-examples must be non-negative')
    if args.train_loss_eval_examples % args.batch_size:
        raise SystemExit('--train-loss-eval-examples must be divisible by batch size')
    if args.epochs is not None and args.epochs <= 0:
        raise SystemExit('--epochs must be positive')
    if args.warmup_steps < 0:
        raise SystemExit('--warmup-steps must be non-negative')
    if not math.isfinite(args.end_learning_rate) or args.end_learning_rate < 0:
        raise SystemExit('--end-learning-rate must be finite and non-negative')
    if args.lr_schedule == 'constant' and (args.warmup_steps or args.end_learning_rate):
        raise SystemExit('warmup/end learning rates require --lr-schedule warmup-cosine')
    if args.eval_interval < 0:
        raise SystemExit('--eval-interval must be non-negative')
    if args.early_stopping_patience < 0:
        raise SystemExit('--early-stopping-patience must be non-negative')
    if args.early_stopping_patience and not args.eval_interval:
        raise SystemExit('--early-stopping-patience requires --eval-interval')
    if args.best_checkpoint_root is not None and not args.eval_interval:
        raise SystemExit('--best-checkpoint-root requires --eval-interval')
    if args.best_checkpoint_root is not None and args.best_checkpoint_root.exists():
        raise SystemExit(f'best-checkpoint root already exists: {args.best_checkpoint_root}')
    if not math.isfinite(args.early_stopping_min_delta) or args.early_stopping_min_delta < 0:
        raise SystemExit('--early-stopping-min-delta must be finite and non-negative')
    if not math.isfinite(args.early_stopping_min_epochs) or args.early_stopping_min_epochs < 0:
        raise SystemExit('--early-stopping-min-epochs must be finite and non-negative')
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise SystemExit('--max-grad-norm must be finite and positive')
    if args.rank <= 0:
        raise SystemExit('--rank must be positive')
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        raise SystemExit('--alpha must be finite and positive')
    if (
        args.checkpoint is not None
        and args.resume_checkpoint is not None
        and args.checkpoint.resolve() == args.resume_checkpoint.resolve()
    ):
        raise SystemExit('--checkpoint and --resume-checkpoint must be different paths')

    if args.semantic_dataset is not None:
        dataset_path = args.semantic_dataset
        clean_records = load_semantic_jsonl(dataset_path)
        rejected = {}
        target_format = TargetFormat.SEMANTIC_DOCUMENT_V2
        default_prompt_contract = PromptContract.CONTEXT_AUTHORITATIVE_V1
    else:
        dataset_path = args.dataset
        clean_records, rejected = automatic_preflight(load_jsonl(dataset_path, corpus=Corpus.DISTRIBUTABLE))
        target_format = TargetFormat.RAW_SHELL
        default_prompt_contract = PromptContract.LEGACY_USER_V1
    prompt_contract = args.prompt_contract or default_prompt_contract
    total_records = len(clean_records) + len(rejected)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    splits = split_records(clean_records, corpus=Corpus.DISTRIBUTABLE, seed=split_seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if args.all_train_examples:
        train_records, train_examples = select_all_examples(
            splits[Split.TRAIN],
            tokenizer,
            max_length=args.sequence_length,
            seed=args.seed,
            prompt_contract=prompt_contract,
        )
    else:
        train_records, train_examples = select_examples(
            splits[Split.TRAIN],
            tokenizer,
            count=args.train_examples,
            max_length=args.sequence_length,
            seed=args.seed,
            priority_source=args.priority_source,
            prompt_contract=prompt_contract,
        )
    unique_train_records = train_records
    train_records, train_examples = apply_rehearsal_weight(
        train_records,
        train_examples,
        record_prefix=args.rehearsal_record_prefix,
        weight=args.rehearsal_weight,
        seed=args.seed,
    )
    if args.all_train_examples:
        usable_count = len(train_examples) - (len(train_examples) % args.batch_size)
        train_records = train_records[:usable_count]
        train_examples = train_examples[:usable_count]
    eval_records, eval_examples = select_examples(
        splits[args.eval_split],
        tokenizer,
        count=args.eval_examples,
        max_length=args.sequence_length,
        seed=split_seed + 1,
        priority_source=args.priority_source,
        prompt_contract=prompt_contract,
    )
    pad_id = tokenizer_pad_id(tokenizer)
    train_batches = make_batches(
        train_examples,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        pad_token_id=pad_id,
    )
    eval_batches = make_batches(
        eval_examples,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        pad_token_id=pad_id,
    )
    train_loss_batches = train_batches
    if args.train_loss_eval_examples:
        train_loss_batch_count = args.train_loss_eval_examples // args.batch_size
        train_loss_batches = train_batches[:train_loss_batch_count]
        if len(train_loss_batches) != train_loss_batch_count:
            raise SystemExit('--train-loss-eval-examples exceeds selected training examples')
    total_steps = args.steps if args.epochs is None else args.epochs * len(train_batches)
    early_stopping_min_steps = math.ceil(args.early_stopping_min_epochs * len(train_batches))
    if args.warmup_steps >= total_steps:
        raise SystemExit('--warmup-steps must be less than total training steps')

    print(
        f'corpus: {total_records:,} records; automated preflight accepted {len(clean_records):,}, '
        f'rejected {len(rejected)}; split '
        f'{len(splits[Split.TRAIN]):,}/{len(splits[Split.VALIDATION]):,}/{len(splits[Split.TEST]):,} '
        '(train/validation/test)'
    )
    if rejected:
        print(f'preflight rejections: {json.dumps(rejected, sort_keys=True)}')
    print(f'device: {jax.devices()[0]}')
    print(f'train selection: {len(train_records):,} rows, sources={dict(Counter(record.source for record in train_records))}')
    print(f'eval selection: {len(eval_records):,} rows, sources={dict(Counter(record.source for record in eval_records))}')

    model = Qwen2ForCausalLM(Qwen2Config(), param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    weights_path = hf_hub_download(MODEL_ID, 'model.safetensors', local_files_only=True)
    load_hf_state_dict(model, load_file(weights_path), param_dtype=jnp.bfloat16)
    inject_lora(model, rank=args.rank, alpha=args.alpha, rngs=nnx.Rngs(1))
    learning_rate: float | Callable[[jax.Array], jax.Array] = args.learning_rate
    if args.lr_schedule == 'warmup-cosine':
        learning_rate = warmup_cosine_schedule(
            peak_learning_rate=args.learning_rate,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            end_learning_rate=args.end_learning_rate,
        )
    optimizer = create_lora_optimizer(
        model,
        learning_rate=learning_rate,
        max_grad_norm=args.max_grad_norm,
    )
    resumed_metadata = None
    if args.resume_checkpoint is not None:
        resumed_metadata = restore_checkpoint(
            args.resume_checkpoint,
            model,
            optimizer,
            model_id=MODEL_ID,
            corpus=Corpus.DISTRIBUTABLE,
            target_format=target_format,
            prompt_contract=prompt_contract,
        )
        print(f'resumed checkpoint: {args.resume_checkpoint} at step {resumed_metadata.step}')
    print(f'parameters: {parameter_count(model, nnx.Param):,} total; {parameter_count(model, nnx.LoRAParam):,} trainable LoRA')

    started = time.perf_counter()
    initial_train_loss = mean_loss(model, train_loss_batches)
    initial_eval_loss = mean_loss(model, eval_batches)
    baseline_generation = greedy_completion(
        model,
        train_examples[0],
        tokenizer,
        sequence_length=args.sequence_length,
    )
    baseline_heldout_generation = None
    if args.heldout_probe:
        baseline_heldout_generation = greedy_completion(
            model,
            eval_examples[0],
            tokenizer,
            sequence_length=args.sequence_length,
        )
    print(f'initial loss: train={initial_train_loss:.4f}, heldout={initial_eval_loss:.4f}')
    _print_generation('before', train_records[0], baseline_generation)
    if baseline_heldout_generation is not None:
        _print_generation('heldout-before', eval_records[0], baseline_heldout_generation)

    report_every = max(total_steps // 4, 1)
    show_progress = interactive_progress()
    observations: list[LossObservation] = []
    early_stopping = None
    if args.eval_interval:
        observations.append(
            LossObservation(
                step=0,
                examples_seen=0,
                train_loss=initial_train_loss,
                heldout_loss=initial_eval_loss,
            )
        )
        if args.early_stopping_patience:
            early_stopping = EarlyStoppingTracker(
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
            )
            early_stopping.observe(observations[0])
    gradient_norm_sum = 0.0
    gradient_norm_max = 0.0
    clipped_steps = 0
    completed_steps = 0
    stopped_early = False
    interval_loss_sum = 0.0
    interval_loss_steps = 0
    best_checkpoint: Path | None = None
    batch_orders: list[np.ndarray] = []
    if args.shuffle_each_epoch:
        epoch_count = math.ceil(total_steps / len(train_batches))
        batch_orders = [np.random.default_rng(args.seed + epoch).permutation(len(train_batches)) for epoch in range(epoch_count)]
    with training_progress() as progress:
        task = progress.add_task(
            f'rank {args.rank} training',
            total=total_steps,
            loss='—',
        )
        for step in range(total_steps):
            epoch, offset = divmod(step, len(train_batches))
            batch_index = int(batch_orders[epoch][offset]) if batch_orders else offset
            batch = train_batches[batch_index]
            if args.gradient_diagnostics:
                loss, gradient_norm = train_step_with_gradient_norm(model, optimizer, batch)
                gradient_norm_value = float(np.asarray(gradient_norm))
                gradient_norm_sum += gradient_norm_value
                gradient_norm_max = max(gradient_norm_max, gradient_norm_value)
                clipped_steps += int(gradient_norm_value > args.max_grad_norm)
            else:
                loss = train_step(model, optimizer, batch)
            loss_value = float(np.asarray(loss))
            completed_steps = step + 1
            interval_loss_sum += loss_value
            interval_loss_steps += 1
            progress.update(task, advance=1, loss=f'{loss_value:.4f}')
            if not show_progress and (step == 0 or (step + 1) % report_every == 0 or step + 1 == total_steps):
                print(f'step {step + 1}/{total_steps}: loss={loss_value:.4f}')
            should_evaluate = args.eval_interval and (completed_steps % args.eval_interval == 0 or completed_steps == total_steps)
            if should_evaluate:
                heldout_loss = mean_loss(model, eval_batches)
                observation = LossObservation(
                    step=completed_steps,
                    examples_seen=completed_steps * args.batch_size,
                    train_loss=interval_loss_sum / interval_loss_steps,
                    heldout_loss=heldout_loss,
                )
                is_new_best = observation.heldout_loss < min(previous.heldout_loss for previous in observations)
                observations.append(observation)
                interval_loss_sum = 0.0
                interval_loss_steps = 0
                progress.console.print(f'validation step {completed_steps}: heldout={heldout_loss:.4f}')
                if is_new_best and args.best_checkpoint_root is not None:
                    checkpoint_path = args.best_checkpoint_root / f'step-{completed_steps:08d}'
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        model_id=MODEL_ID,
                        corpus=Corpus.DISTRIBUTABLE,
                        target_format=target_format,
                        prompt_contract=prompt_contract,
                    )
                    if best_checkpoint is not None:
                        shutil.rmtree(best_checkpoint)
                    best_checkpoint = checkpoint_path
                    progress.console.print(f'best checkpoint: {best_checkpoint}')
                should_stop = early_stopping is not None and early_stopping.observe(observation)
                if should_stop and completed_steps >= early_stopping_min_steps and completed_steps < total_steps:
                    stopped_early = True
                    progress.console.print(f'early stop at step {completed_steps}; best step {early_stopping.best.step}')
                    progress.update(task, total=completed_steps, completed=completed_steps)
                    break

    final_train_loss = mean_loss(model, train_loss_batches)
    final_eval_loss = mean_loss(model, eval_batches)
    trained_generation = greedy_completion(
        model,
        train_examples[0],
        tokenizer,
        sequence_length=args.sequence_length,
    )
    trained_heldout_generation = None
    if args.heldout_probe:
        trained_heldout_generation = greedy_completion(
            model,
            eval_examples[0],
            tokenizer,
            sequence_length=args.sequence_length,
        )
    elapsed = time.perf_counter() - started
    print(f'final loss: train={final_train_loss:.4f}, heldout={final_eval_loss:.4f}')
    _print_generation('after', train_records[0], trained_generation)
    if trained_heldout_generation is not None:
        _print_generation('heldout-after', eval_records[0], trained_heldout_generation)
    print(f'elapsed: {elapsed:.2f}s')

    failure_message = None
    if not math.isfinite(final_train_loss) or final_train_loss >= initial_train_loss * 0.9:
        failure_message = f'smoke failed: train loss did not fall by 10% ({initial_train_loss:.4f} -> {final_train_loss:.4f})'
    if args.checkpoint is not None and failure_message is None:
        metadata = save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            model_id=MODEL_ID,
            corpus=Corpus.DISTRIBUTABLE,
            target_format=target_format,
            prompt_contract=prompt_contract,
        )
        print(f'checkpoint: {args.checkpoint} at step {metadata.step}')
    best_loss_observation = best_observation(observations) if observations else None
    if args.report is not None:
        report = {
            'report_schema_version': 3,
            'model_id': MODEL_ID,
            'adapter': {
                'rank': args.rank,
                'alpha': args.alpha,
                'scale': args.alpha / args.rank,
            },
            'dataset': {
                'path': str(dataset_path),
                'sha256': _file_sha256(dataset_path),
                'accepted_records': len(clean_records),
                'rejected_record_ids': sorted(rejected),
                'target_format': target_format.value,
                'prompt_contract': prompt_contract.value,
            },
            'resume_checkpoint': None
            if resumed_metadata is None
            else {
                'path': str(args.resume_checkpoint),
                'step': resumed_metadata.step,
            },
            'selection': {
                'seed': args.seed,
                'split_seed': split_seed,
                'priority_source': args.priority_source,
                'train_examples': len(unique_train_records),
                'train_record_ids_sha256': _record_id_sha256(unique_train_records),
                'train_source_counts': dict(Counter(record.source for record in unique_train_records)),
                'effective_train_examples': len(train_records),
                'effective_train_source_counts': dict(Counter(record.source for record in train_records)),
                'effective_rehearsal_examples': sum(
                    record.record_id.startswith(args.rehearsal_record_prefix)
                    for record in train_records
                    if args.rehearsal_record_prefix is not None
                ),
                'rehearsal_record_prefix': args.rehearsal_record_prefix,
                'rehearsal_weight': args.rehearsal_weight,
                'eval_examples': len(eval_records),
                'eval_split': args.eval_split.value,
                'eval_record_ids_sha256': _record_id_sha256(eval_records),
                'eval_source_counts': dict(Counter(record.source for record in eval_records)),
            },
            'training': {
                'starting_step': 0 if resumed_metadata is None else resumed_metadata.step,
                'device': str(jax.devices()[0]),
                'sequence_length': args.sequence_length,
                'batch_size': args.batch_size,
                'steps': completed_steps,
                'max_steps': total_steps,
                'epochs': args.epochs,
                'all_train_examples': args.all_train_examples,
                'shuffle_each_epoch': args.shuffle_each_epoch,
                'stopped_early': stopped_early,
                'generation_tokens': GENERATION_TOKENS,
                'learning_rate': args.learning_rate,
                'lr_schedule': args.lr_schedule,
                'warmup_steps': args.warmup_steps,
                'end_learning_rate': args.end_learning_rate,
                'eval_interval': args.eval_interval,
                'early_stopping_patience': args.early_stopping_patience,
                'early_stopping_min_delta': args.early_stopping_min_delta,
                'early_stopping_min_epochs': args.early_stopping_min_epochs,
                'max_grad_norm': args.max_grad_norm,
                'initial_train_loss': initial_train_loss,
                'final_train_loss': final_train_loss,
                'train_loss_eval_examples': len(train_loss_batches) * args.batch_size,
                'initial_heldout_loss': initial_eval_loss,
                'final_heldout_loss': final_eval_loss,
                'loss_observations': [
                    {
                        'step': observation.step,
                        'examples_seen': observation.examples_seen,
                        'train_loss': observation.train_loss,
                        'heldout_loss': observation.heldout_loss,
                    }
                    for observation in observations
                ],
                'best_observation': None
                if best_loss_observation is None
                else {
                    'step': best_loss_observation.step,
                    'examples_seen': best_loss_observation.examples_seen,
                    'train_loss': best_loss_observation.train_loss,
                    'heldout_loss': best_loss_observation.heldout_loss,
                },
                'best_checkpoint': None if best_checkpoint is None else str(best_checkpoint),
                'gradient_diagnostics': None
                if not args.gradient_diagnostics
                else {
                    'mean_preclip_global_norm': gradient_norm_sum / completed_steps,
                    'max_preclip_global_norm': gradient_norm_max,
                    'clipped_steps': clipped_steps,
                    'clipped_fraction': clipped_steps / completed_steps,
                },
                'elapsed_seconds': elapsed,
                'passed_loss_gate': failure_message is None,
            },
            'probe': {
                'record_id': train_records[0].record_id,
                'expected': train_records[0].response,
                'before': baseline_generation,
                'after': trained_generation,
                'exact_match_before': baseline_generation == train_records[0].response,
                'exact_match_after': trained_generation == train_records[0].response,
            },
            'heldout_probe': {
                'record_id': eval_records[0].record_id,
                'expected': eval_records[0].response,
                'before': baseline_heldout_generation,
                'after': trained_heldout_generation,
                'exact_match_before': baseline_heldout_generation == eval_records[0].response,
                'exact_match_after': trained_heldout_generation == eval_records[0].response,
            },
        }
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(f'report: {args.report}')
    if failure_message is not None:
        raise SystemExit(failure_message)
    print('SMOKE PASS: real-corpus LoRA training reduced tiny-set loss by at least 10%')


if __name__ == '__main__':
    main()
