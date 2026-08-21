import math
from collections.abc import Callable
from typing import TypedDict

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from shelliq_training.model import Qwen2ForCausalLM

IGNORE_INDEX = -100


class CausalLMBatch(TypedDict):
    input_ids: jax.Array
    attention_mask: jax.Array
    labels: jax.Array


def next_token_loss(
    logits: jax.Array,
    labels: jax.Array,
    attention_mask: jax.Array | None = None,
) -> jax.Array:
    """Mean next-token cross entropy over labels not equal to -100."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError('logits and labels must share batch and sequence dimensions')
    shifted_logits = logits[:, :-1, :].astype(jnp.float32)
    shifted_labels = labels[:, 1:]
    loss_mask = shifted_labels != IGNORE_INDEX
    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError('attention_mask and labels must have the same shape')
        loss_mask &= attention_mask[:, 1:].astype(jnp.bool_)

    safe_labels = jnp.where(loss_mask, shifted_labels, 0)
    token_losses = optax.softmax_cross_entropy_with_integer_labels(shifted_logits, safe_labels)
    total = jnp.sum(jnp.where(loss_mask, token_losses, 0.0))
    count = jnp.sum(loss_mask)
    return total / jnp.maximum(count, 1)


def causal_lm_loss(model: Qwen2ForCausalLM, batch: CausalLMBatch) -> jax.Array:
    logits = model(batch['input_ids'], batch['attention_mask'])
    return next_token_loss(logits, batch['labels'], batch['attention_mask'])


@nnx.jit
def eval_step(model: Qwen2ForCausalLM, batch: CausalLMBatch) -> jax.Array:
    """Evaluate a fixed-shape batch without constructing optimizer gradients."""
    return causal_lm_loss(model, batch)


def create_lora_optimizer(
    model: Qwen2ForCausalLM,
    *,
    learning_rate: float | Callable[[jax.Array], jax.Array] = 2e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
) -> nnx.Optimizer:
    transform = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    return nnx.Optimizer(model, transform, wrt=nnx.LoRAParam)


def warmup_cosine_schedule(
    *,
    peak_learning_rate: float,
    total_steps: int,
    warmup_steps: int,
    end_learning_rate: float,
) -> Callable[[jax.Array], jax.Array]:
    """Create a bounded warmup/cosine schedule for a complete training run."""
    if total_steps <= 0:
        raise ValueError('total steps must be positive')
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError('warmup steps must be non-negative and less than total steps')
    if (
        not math.isfinite(peak_learning_rate)
        or not math.isfinite(end_learning_rate)
        or peak_learning_rate <= 0
        or end_learning_rate < 0
    ):
        raise ValueError('learning rates must be non-negative and peak must be positive')
    if end_learning_rate > peak_learning_rate:
        raise ValueError('end learning rate cannot exceed peak learning rate')
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=end_learning_rate,
    )


@nnx.jit
def train_step(
    model: Qwen2ForCausalLM,
    optimizer: nnx.Optimizer,
    batch: CausalLMBatch,
) -> jax.Array:
    loss, gradients = nnx.value_and_grad(
        causal_lm_loss,
        argnums=nnx.DiffState(0, nnx.LoRAParam),
    )(model, batch)
    optimizer.update(model, gradients)
    return loss


@nnx.jit
def train_step_with_gradient_norm(
    model: Qwen2ForCausalLM,
    optimizer: nnx.Optimizer,
    batch: CausalLMBatch,
) -> tuple[jax.Array, jax.Array]:
    """Train one batch and return loss plus the pre-clipping global gradient norm."""
    loss, gradients = nnx.value_and_grad(
        causal_lm_loss,
        argnums=nnx.DiffState(0, nnx.LoRAParam),
    )(model, batch)
    gradient_norm = optax.tree.norm(gradients)
    optimizer.update(model, gradients)
    return loss, gradient_norm
