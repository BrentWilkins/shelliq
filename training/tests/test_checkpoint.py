import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from shelliq_training.checkpoint import CheckpointError, restore_checkpoint, save_checkpoint
from shelliq_training.config import Qwen2Config
from shelliq_training.data import Corpus
from shelliq_training.lora import inject_lora
from shelliq_training.model import Qwen2ForCausalLM
from shelliq_training.training import create_lora_optimizer, train_step


def training_pair():
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        ),
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    inject_lora(model, rank=4, alpha=8, rngs=nnx.Rngs(1))
    return model, create_lora_optimizer(model, learning_rate=1e-2)


def batch():
    input_ids = jnp.array([[1, 2, 3, 4]])
    return {
        'input_ids': input_ids,
        'attention_mask': jnp.ones_like(input_ids),
        'labels': input_ids,
    }


def assert_trees_equal(left, right):
    comparisons = jax.tree.map(jnp.array_equal, left, right)
    assert all(jax.tree.leaves(comparisons))


def test_checkpoint_restores_adapters_optimizer_and_next_step(tmp_path):
    model, optimizer = training_pair()
    train_step(model, optimizer, batch())
    checkpoint_path = tmp_path / 'step-1'

    metadata = save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        model_id='tiny-qwen',
        corpus=Corpus.DISTRIBUTABLE,
    )

    resumed_model, resumed_optimizer = training_pair()
    restored = restore_checkpoint(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        model_id='tiny-qwen',
        corpus=Corpus.DISTRIBUTABLE,
    )
    assert restored == metadata
    assert int(resumed_optimizer.step[...]) == 1
    assert_trees_equal(nnx.state(model, nnx.LoRAParam), nnx.state(resumed_model, nnx.LoRAParam))
    assert_trees_equal(nnx.state(optimizer), nnx.state(resumed_optimizer))

    loss = train_step(model, optimizer, batch())
    resumed_loss = train_step(resumed_model, resumed_optimizer, batch())
    assert jnp.array_equal(loss, resumed_loss)
    assert_trees_equal(nnx.state(model, nnx.LoRAParam), nnx.state(resumed_model, nnx.LoRAParam))


def test_checkpoint_refuses_pipeline_mismatch(tmp_path):
    model, optimizer = training_pair()
    checkpoint_path = tmp_path / 'step-0'
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        model_id='tiny-qwen',
        corpus=Corpus.DISTRIBUTABLE,
    )

    with pytest.raises(CheckpointError, match="corpus 'distributable' != 'personal'"):
        restore_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            model_id='tiny-qwen',
            corpus=Corpus.PERSONAL,
        )


def test_checkpoint_never_overwrites_existing_directory(tmp_path):
    model, optimizer = training_pair()
    checkpoint_path = tmp_path / 'step-0'
    checkpoint_path.mkdir()

    with pytest.raises(FileExistsError, match='already exists'):
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            model_id='tiny-qwen',
            corpus=Corpus.DISTRIBUTABLE,
        )
