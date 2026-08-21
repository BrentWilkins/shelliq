import json

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from shelliq_training import checkpoint as checkpoint_module
from shelliq_training.checkpoint import (
    CheckpointError,
    CheckpointMetadata,
    TargetFormat,
    restore_adapter_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from shelliq_training.config import Qwen2Config
from shelliq_training.data import Corpus
from shelliq_training.lora import inject_lora
from shelliq_training.model import Qwen2ForCausalLM
from shelliq_training.prompt import PromptContract
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
        target_format=TargetFormat.RAW_SHELL,
        prompt_contract=PromptContract.LEGACY_USER_V1,
    )

    resumed_model, resumed_optimizer = training_pair()
    restored = restore_checkpoint(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        model_id='tiny-qwen',
        corpus=Corpus.DISTRIBUTABLE,
        target_format=TargetFormat.RAW_SHELL,
        prompt_contract=PromptContract.LEGACY_USER_V1,
    )
    assert restored == metadata
    assert int(resumed_optimizer.step[...]) == 1
    assert_trees_equal(nnx.state(model, nnx.LoRAParam), nnx.state(resumed_model, nnx.LoRAParam))
    assert_trees_equal(nnx.state(optimizer), nnx.state(resumed_optimizer))

    loss = train_step(model, optimizer, batch())
    resumed_loss = train_step(resumed_model, resumed_optimizer, batch())
    assert jnp.array_equal(loss, resumed_loss)
    assert_trees_equal(nnx.state(model, nnx.LoRAParam), nnx.state(resumed_model, nnx.LoRAParam))


def test_checkpoint_can_restore_adapters_without_optimizer_state(tmp_path, monkeypatch):
    resumed_model, _ = training_pair()
    restored_adapters = nnx.to_pure_dict(
        jax.tree.map(
            lambda value: value + 1,
            nnx.state(resumed_model, nnx.LoRAParam),
        )
    )
    checkpoint_path = tmp_path / 'scheduled-step-7'
    (checkpoint_path / 'metadata').mkdir(parents=True)
    (checkpoint_path / 'state').mkdir()
    metadata = CheckpointMetadata(
        step=7,
        model_id='tiny-qwen',
        corpus=Corpus.DISTRIBUTABLE,
        target_format=TargetFormat.SEMANTIC_DOCUMENT_V2,
        prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
        rank=4,
        alpha=8.0,
        targets=checkpoint_module.DEFAULT_TARGETS,
    )
    (checkpoint_path / 'metadata' / 'metadata').write_text(json.dumps(metadata.to_dict()))

    class FakeCheckpointer:
        def __init__(self, handler):
            self.handler = handler

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def restore(self, path, *, args):
            assert path == checkpoint_path / 'state'
            assert args.partial_restore is True
            return {'adapters': restored_adapters}

    monkeypatch.setattr(checkpoint_module.ocp, 'Checkpointer', FakeCheckpointer)
    metadata = restore_adapter_checkpoint(
        checkpoint_path,
        resumed_model,
        model_id='tiny-qwen',
        corpus=Corpus.DISTRIBUTABLE,
        target_format=TargetFormat.SEMANTIC_DOCUMENT_V2,
        prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
    )

    assert metadata.step == 7
    assert_trees_equal(
        restored_adapters,
        nnx.to_pure_dict(nnx.state(resumed_model, nnx.LoRAParam)),
    )


def test_checkpoint_refuses_pipeline_mismatch(tmp_path):
    model, optimizer = training_pair()
    checkpoint_path = tmp_path / 'step-0'
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        model_id='tiny-qwen',
        corpus=Corpus.DISTRIBUTABLE,
        target_format=TargetFormat.RAW_SHELL,
        prompt_contract=PromptContract.LEGACY_USER_V1,
    )

    with pytest.raises(CheckpointError, match="corpus 'distributable' != 'personal'"):
        restore_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            model_id='tiny-qwen',
            corpus=Corpus.PERSONAL,
            target_format=TargetFormat.RAW_SHELL,
            prompt_contract=PromptContract.LEGACY_USER_V1,
        )


def test_checkpoint_refuses_target_format_mismatch(tmp_path):
    model, optimizer = training_pair()
    checkpoint_path = tmp_path / 'semantic-step-0'
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        model_id='tiny-qwen',
        corpus=Corpus.DISTRIBUTABLE,
        target_format=TargetFormat.SEMANTIC_DOCUMENT_V2,
        prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
    )

    with pytest.raises(CheckpointError, match='target format'):
        restore_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            model_id='tiny-qwen',
            corpus=Corpus.DISTRIBUTABLE,
            target_format=TargetFormat.RAW_SHELL,
            prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
        )


def test_checkpoint_refuses_prompt_contract_mismatch(tmp_path):
    model, optimizer = training_pair()
    checkpoint_path = tmp_path / 'semantic-step-0'
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        model_id='tiny-qwen',
        corpus=Corpus.DISTRIBUTABLE,
        target_format=TargetFormat.SEMANTIC_DOCUMENT_V2,
        prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
    )

    with pytest.raises(CheckpointError, match='prompt contract'):
        restore_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            model_id='tiny-qwen',
            corpus=Corpus.DISTRIBUTABLE,
            target_format=TargetFormat.SEMANTIC_DOCUMENT_V2,
            prompt_contract=PromptContract.LEGACY_USER_V1,
        )


def test_schema_v1_metadata_is_read_as_raw_shell():
    metadata = CheckpointMetadata.from_dict(
        {
            'schema_version': 1,
            'step': 4,
            'model_id': 'tiny-qwen',
            'corpus': 'distributable',
            'rank': 4,
            'alpha': 8.0,
            'targets': ['q_proj'],
        }
    )

    assert metadata.target_format is TargetFormat.RAW_SHELL
    assert metadata.prompt_contract is PromptContract.LEGACY_USER_V1
    assert metadata.to_dict()['schema_version'] == 3


def test_schema_v2_metadata_is_read_as_legacy_prompt():
    metadata = CheckpointMetadata.from_dict(
        {
            'schema_version': 2,
            'step': 4,
            'model_id': 'tiny-qwen',
            'corpus': 'distributable',
            'target_format': 'semantic-document-v2-json',
            'rank': 4,
            'alpha': 8.0,
            'targets': ['q_proj'],
        }
    )

    assert metadata.target_format is TargetFormat.SEMANTIC_DOCUMENT_V2
    assert metadata.prompt_contract is PromptContract.LEGACY_USER_V1


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
            target_format=TargetFormat.RAW_SHELL,
            prompt_contract=PromptContract.LEGACY_USER_V1,
        )
