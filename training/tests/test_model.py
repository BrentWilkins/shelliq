import jax.numpy as jnp
from flax import nnx

from shelliq_training.config import Qwen2Config
from shelliq_training.model import Qwen2ForCausalLM


def tiny_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_theta=1_000_000.0,
    )


def test_forward_shape_and_finite():
    config = tiny_config()
    model = Qwen2ForCausalLM(config, param_dtype=jnp.float32, rngs=nnx.Rngs(0))
    input_ids = jnp.array([[1, 2, 3, 4, 5]])

    logits = model(input_ids)

    assert logits.shape == (1, 5, config.vocab_size)
    assert bool(jnp.all(jnp.isfinite(logits)))


def test_causal_mask_blocks_future_tokens():
    config = tiny_config()
    model = Qwen2ForCausalLM(config, param_dtype=jnp.float32, rngs=nnx.Rngs(0))
    prefix = jnp.array([[1, 2, 3]])
    longer = jnp.array([[1, 2, 3, 4]])

    prefix_logits = model(prefix)
    longer_logits = model(longer)

    assert jnp.allclose(prefix_logits, longer_logits[:, :3], atol=1e-5)


def test_left_padding_matches_unpadded_tokens():
    config = tiny_config()
    model = Qwen2ForCausalLM(config, param_dtype=jnp.float32, rngs=nnx.Rngs(0))
    unpadded = jnp.array([[1, 2, 3]])
    padded = jnp.array([[0, 0, 1, 2, 3]])
    attention_mask = jnp.array([[0, 0, 1, 1, 1]])

    expected = model(unpadded)
    actual = model(padded, attention_mask)[:, 2:]

    assert bool(jnp.all(jnp.isfinite(actual)))
    assert jnp.allclose(expected, actual, atol=1e-5)
