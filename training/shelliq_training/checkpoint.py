"""Orbax checkpoints for resumable LoRA training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from shelliq_training.data import Corpus
from shelliq_training.lora import DEFAULT_TARGETS, LoRALinear
from shelliq_training.model import Qwen2ForCausalLM

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(ValueError):
    """A checkpoint is incompatible with the active training run."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    step: int
    model_id: str
    corpus: Corpus
    rank: int
    alpha: float
    targets: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'step': self.step,
            'model_id': self.model_id,
            'corpus': self.corpus.value,
            'rank': self.rank,
            'alpha': self.alpha,
            'targets': list(self.targets),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CheckpointMetadata:
        expected = {'schema_version', 'step', 'model_id', 'corpus', 'rank', 'alpha', 'targets'}
        if raw.keys() != expected:
            missing = expected - raw.keys()
            unknown = raw.keys() - expected
            details = []
            if missing:
                details.append(f'missing {", ".join(sorted(missing))}')
            if unknown:
                details.append(f'unknown {", ".join(sorted(unknown))}')
            raise CheckpointError(f'invalid checkpoint metadata: {"; ".join(details)}')
        if raw['schema_version'] != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(f'unsupported checkpoint schema: {raw["schema_version"]!r}')
        try:
            step = int(cast(int, raw['step']))
            rank = int(cast(int, raw['rank']))
            alpha = float(cast(float, raw['alpha']))
            model_id = cast(str, raw['model_id'])
            corpus = Corpus(cast(str, raw['corpus']))
            raw_targets = cast(list[object], raw['targets'])
            targets = tuple(cast(str, target) for target in raw_targets)
        except (TypeError, ValueError) as error:
            raise CheckpointError(f'invalid checkpoint metadata values: {error}') from error
        if step < 0 or rank <= 0 or alpha <= 0:
            raise CheckpointError('checkpoint step, rank, and alpha must be valid positive values')
        if not isinstance(model_id, str) or not model_id:
            raise CheckpointError('checkpoint model_id must be a non-empty string')
        if not isinstance(raw_targets, list) or not targets or any(not isinstance(target, str) for target in raw_targets):
            raise CheckpointError('checkpoint targets must be a non-empty string list')
        return cls(step=step, model_id=model_id, corpus=corpus, rank=rank, alpha=alpha, targets=targets)


def save_checkpoint(
    directory: str | Path,
    model: Qwen2ForCausalLM,
    optimizer: nnx.Optimizer,
    *,
    model_id: str,
    corpus: Corpus,
) -> CheckpointMetadata:
    """Atomically save adapters, optimizer moments, step, and compatibility data."""
    path = Path(directory).resolve()
    if path.exists():
        raise FileExistsError(f'checkpoint already exists: {path}')
    rank, alpha, targets = lora_signature(model)
    step = int(np.asarray(optimizer.step[...]))
    metadata = CheckpointMetadata(
        step=step,
        model_id=model_id,
        corpus=corpus,
        rank=rank,
        alpha=alpha,
        targets=targets,
    )
    state = {
        'adapters': nnx.to_pure_dict(nnx.state(model, nnx.LoRAParam)),
        'optimizer': nnx.to_pure_dict(nnx.state(optimizer)),
    }
    with ocp.Checkpointer(ocp.CompositeCheckpointHandler()) as checkpointer:
        checkpointer.save(
            path,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                metadata=ocp.args.JsonSave(metadata.to_dict()),
            ),
        )
    return metadata


def restore_checkpoint(
    directory: str | Path,
    model: Qwen2ForCausalLM,
    optimizer: nnx.Optimizer,
    *,
    model_id: str,
    corpus: Corpus,
) -> CheckpointMetadata:
    """Restore only after model, corpus, and LoRA configuration agree."""
    path = Path(directory).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'checkpoint does not exist: {path}')
    target = {
        'adapters': nnx.to_pure_dict(nnx.state(model, nnx.LoRAParam)),
        'optimizer': nnx.to_pure_dict(nnx.state(optimizer)),
    }
    with ocp.Checkpointer(ocp.CompositeCheckpointHandler()) as checkpointer:
        restored = checkpointer.restore(
            path,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(target),
                metadata=ocp.args.JsonRestore(),
            ),
        )
    metadata = CheckpointMetadata.from_dict(restored.metadata)
    _validate_compatibility(metadata, model, model_id=model_id, corpus=corpus)

    adapter_state = nnx.state(model, nnx.LoRAParam)
    optimizer_state = nnx.state(optimizer)
    try:
        nnx.replace_by_pure_dict(adapter_state, restored.state['adapters'])
        nnx.replace_by_pure_dict(optimizer_state, restored.state['optimizer'])
    except ValueError as error:
        raise CheckpointError(f'checkpoint state does not match active model: {error}') from error
    nnx.update(model, adapter_state)
    nnx.update(optimizer, optimizer_state)
    return metadata


def lora_signature(model: Qwen2ForCausalLM) -> tuple[int, float, tuple[str, ...]]:
    """Derive rank, alpha, and targets from the live modules rather than CLI claims."""
    ranks: set[int] = set()
    alphas: set[float] = set()
    target_sets: set[tuple[str, ...]] = set()
    for layer in model.model.layers:
        modules = {
            'q_proj': layer.self_attn.q_proj,
            'k_proj': layer.self_attn.k_proj,
            'v_proj': layer.self_attn.v_proj,
            'o_proj': layer.self_attn.o_proj,
            'gate_proj': layer.mlp.gate_proj,
            'up_proj': layer.mlp.up_proj,
            'down_proj': layer.mlp.down_proj,
        }
        targets = tuple(name for name in DEFAULT_TARGETS if isinstance(modules[name], LoRALinear))
        target_sets.add(targets)
        for name in targets:
            module = cast(LoRALinear, modules[name])
            rank = int(module.adapter.lora_a.shape[1])
            ranks.add(rank)
            alphas.add(module.scale * rank)
    if not ranks:
        raise CheckpointError('model has no LoRA adapters')
    if len(ranks) != 1 or len(alphas) != 1 or len(target_sets) != 1:
        raise CheckpointError('model has inconsistent LoRA configuration across layers')
    targets = target_sets.pop()
    if not targets:
        raise CheckpointError('model has no LoRA target modules')
    return ranks.pop(), alphas.pop(), targets


def _validate_compatibility(
    metadata: CheckpointMetadata,
    model: Qwen2ForCausalLM,
    *,
    model_id: str,
    corpus: Corpus,
) -> None:
    rank, alpha, targets = lora_signature(model)
    mismatches = []
    if metadata.model_id != model_id:
        mismatches.append(f'model {metadata.model_id!r} != {model_id!r}')
    if metadata.corpus is not corpus:
        mismatches.append(f'corpus {metadata.corpus.value!r} != {corpus.value!r}')
    if metadata.rank != rank:
        mismatches.append(f'rank {metadata.rank} != {rank}')
    if not math.isclose(metadata.alpha, alpha):
        mismatches.append(f'alpha {metadata.alpha} != {alpha}')
    if metadata.targets != targets:
        mismatches.append(f'targets {metadata.targets!r} != {targets!r}')
    if mismatches:
        raise CheckpointError(f'incompatible checkpoint: {"; ".join(mismatches)}')
