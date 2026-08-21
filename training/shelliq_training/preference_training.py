"""Memory-efficient offline DPO primitives with cached reference scores."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from shelliq_training.data import IGNORE_INDEX, SFTRecord, TokenizedExample, tokenize_record
from shelliq_training.preference_data import PreferenceRecord
from shelliq_training.prompt import PromptContract
from shelliq_training.semantic_evaluation import first_command
from shelliq_training.training import CausalLMBatch, causal_lm_loss


@dataclass(frozen=True, slots=True)
class TokenizedPreference:
    pair_id: str
    chosen: TokenizedExample
    rejected: TokenizedExample


def preference_records(record: PreferenceRecord) -> tuple[SFTRecord, SFTRecord]:
    """Represent both answers with the same prompt through the SFT tokenizer contract."""
    command = first_command(record.chosen)
    if command is None:
        raise ValueError(f'{record.pair_id}: chosen document has no first command')
    common = {
        'corpus': record.corpus,
        'source': record.source,
        'license': record.license,
        'provenance': record.provenance,
        'command': command,
        'platform': record.platform,
        'instruction': record.instruction,
        'context': record.context,
    }
    chosen = SFTRecord(
        record_id=f'{record.pair_id}:chosen',
        response=json.dumps(record.chosen, separators=(',', ':')),
        **common,
    )
    rejected = SFTRecord(
        record_id=f'{record.pair_id}:rejected',
        response=json.dumps(record.rejected, separators=(',', ':')),
        **common,
    )
    return chosen, rejected


def tokenize_preference(record: PreferenceRecord, tokenizer, *, max_length: int) -> TokenizedPreference:
    chosen, rejected = preference_records(record)
    return TokenizedPreference(
        pair_id=record.pair_id,
        chosen=tokenize_record(
            chosen,
            tokenizer,
            max_length=max_length,
            prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
        ),
        rejected=tokenize_record(
            rejected,
            tokenizer,
            max_length=max_length,
            prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
        ),
    )


def split_preference_records(
    records: list[PreferenceRecord], *, seed: int, validation_fraction: float = 0.25
) -> tuple[list[PreferenceRecord], list[PreferenceRecord]]:
    """Split by chosen command so closely related pairs cannot cross the boundary."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    groups: dict[str, list[PreferenceRecord]] = {}
    for record in records:
        command = first_command(record.chosen)
        if command is None:
            raise ValueError(f'{record.pair_id}: chosen document has no first command')
        groups.setdefault(command, []).append(record)
    ordered_commands = sorted(groups, key=lambda command: hashlib.sha256(f'{seed}\0{command}'.encode()).digest())
    validation_group_count = max(1, round(len(ordered_commands) * validation_fraction))
    validation_commands = set(ordered_commands[:validation_group_count])
    train = [record for command in ordered_commands if command not in validation_commands for record in groups[command]]
    validation = [record for command in ordered_commands if command in validation_commands for record in groups[command]]
    if not train or not validation:
        raise ValueError('preference split produced an empty partition')
    return train, validation


def completion_log_probabilities(logits: jax.Array, labels: jax.Array, attention_mask: jax.Array) -> jax.Array:
    """Sum completion-token log probabilities for each batch row."""
    if logits.shape[:-1] != labels.shape or labels.shape != attention_mask.shape:
        raise ValueError('logits, labels, and attention mask must share batch dimensions')
    shifted_labels = labels[:, 1:]
    mask = (shifted_labels != IGNORE_INDEX) & attention_mask[:, 1:].astype(jnp.bool_)
    safe_labels = jnp.where(mask, shifted_labels, 0)
    token_logps = jnp.take_along_axis(
        jax.nn.log_softmax(logits[:, :-1, :].astype(jnp.float32)), safe_labels[..., None], axis=-1
    ).squeeze(-1)
    return jnp.sum(jnp.where(mask, token_logps, 0.0), axis=-1)


def model_completion_log_probabilities(model, batch: CausalLMBatch) -> jax.Array:
    logits = model(batch['input_ids'], batch['attention_mask'])
    return completion_log_probabilities(logits, batch['labels'], batch['attention_mask'])


@nnx.jit
def preference_log_probability_step(
    model, chosen_batch: CausalLMBatch, rejected_batch: CausalLMBatch
) -> tuple[jax.Array, jax.Array]:
    """Score a pair batch without retaining a second reference model."""
    return (
        model_completion_log_probabilities(model, chosen_batch),
        model_completion_log_probabilities(model, rejected_batch),
    )


def dpo_loss(
    policy_chosen_logps: jax.Array,
    policy_rejected_logps: jax.Array,
    reference_chosen_logps: jax.Array,
    reference_rejected_logps: jax.Array,
    *,
    beta: float,
) -> jax.Array:
    """Direct-preference loss using cached frozen-reference log probabilities."""
    if beta <= 0:
        raise ValueError('beta must be positive')
    shapes = {
        policy_chosen_logps.shape,
        policy_rejected_logps.shape,
        reference_chosen_logps.shape,
        reference_rejected_logps.shape,
    }
    if len(shapes) != 1:
        raise ValueError('all log-probability vectors must share shape')
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    reference_ratio = reference_chosen_logps - reference_rejected_logps
    return -jnp.mean(jax.nn.log_sigmoid(beta * (policy_ratio - reference_ratio)))


def _preference_objective(
    model,
    chosen_batch: CausalLMBatch,
    rejected_batch: CausalLMBatch,
    reference_chosen_logps: jax.Array,
    reference_rejected_logps: jax.Array,
    replay_batch: CausalLMBatch,
    beta: jax.Array,
    replay_weight: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    chosen_logps = model_completion_log_probabilities(model, chosen_batch)
    rejected_logps = model_completion_log_probabilities(model, rejected_batch)
    policy_ratio = chosen_logps - rejected_logps
    reference_ratio = reference_chosen_logps - reference_rejected_logps
    preference_loss = -jnp.mean(jax.nn.log_sigmoid(beta * (policy_ratio - reference_ratio)))
    replay_loss = causal_lm_loss(model, replay_batch)
    loss = preference_loss + replay_weight * replay_loss
    preference_accuracy = jnp.mean(policy_ratio > reference_ratio)
    return loss, (preference_loss, replay_loss, preference_accuracy)


@nnx.jit
def preference_train_step(
    model,
    optimizer: nnx.Optimizer,
    chosen_batch: CausalLMBatch,
    rejected_batch: CausalLMBatch,
    reference_chosen_logps: jax.Array,
    reference_rejected_logps: jax.Array,
    replay_batch: CausalLMBatch,
    beta: jax.Array,
    replay_weight: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update LoRA parameters with cached-reference DPO plus supervised replay."""
    (loss, metrics), gradients = nnx.value_and_grad(
        _preference_objective,
        argnums=nnx.DiffState(0, nnx.LoRAParam),
        has_aux=True,
    )(
        model,
        chosen_batch,
        rejected_batch,
        reference_chosen_logps,
        reference_rejected_logps,
        replay_batch,
        beta,
        replay_weight,
    )
    optimizer.update(model, gradients)
    preference_loss, replay_loss, preference_accuracy = metrics
    return loss, preference_loss, replay_loss, preference_accuracy


def _sft_control_objective(
    model,
    chosen_batch: CausalLMBatch,
    replay_batch: CausalLMBatch,
    replay_weight: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    chosen_loss = causal_lm_loss(model, chosen_batch)
    replay_loss = causal_lm_loss(model, replay_batch)
    return chosen_loss + replay_weight * replay_loss, (chosen_loss, replay_loss)


@nnx.jit
def sft_control_train_step(
    model,
    optimizer: nnx.Optimizer,
    chosen_batch: CausalLMBatch,
    replay_batch: CausalLMBatch,
    replay_weight: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Matched chosen-answer SFT control with the same replay stream."""
    (loss, metrics), gradients = nnx.value_and_grad(
        _sft_control_objective,
        argnums=nnx.DiffState(0, nnx.LoRAParam),
        has_aux=True,
    )(model, chosen_batch, replay_batch, replay_weight)
    optimizer.update(model, gradients)
    chosen_loss, replay_loss = metrics
    return loss, chosen_loss, replay_loss
