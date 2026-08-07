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


def create_lora_optimizer(
    model: Qwen2ForCausalLM,
    *,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
) -> nnx.Optimizer:
    transform = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    return nnx.Optimizer(model, transform, wrt=nnx.LoRAParam)


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
