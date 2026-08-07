import jax
import jax.numpy as jnp
from flax import nnx

from shelliq_training.config import Qwen2Config
from shelliq_training.lora import LoRALinear, inject_lora, parameter_count
from shelliq_training.model import Qwen2ForCausalLM
from shelliq_training.training import create_lora_optimizer, next_token_loss, train_step


def tiny_model() -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return Qwen2ForCausalLM(config, param_dtype=jnp.float32, rngs=nnx.Rngs(0))


def test_next_token_loss_ignores_prompt_and_padding():
    logits = jnp.zeros((1, 4, 3))
    logits = logits.at[0, 1, 2].set(10.0)
    labels = jnp.array([[-100, -100, 2, 1]])
    attention_mask = jnp.array([[1, 1, 1, 0]])

    loss = next_token_loss(logits, labels, attention_mask)

    assert float(loss) < 1e-3


def test_lora_starts_as_identity_and_only_adapters_update():
    model = tiny_model()
    input_ids = jnp.array([[1, 2, 3, 4]])
    before_logits = model(input_ids)
    base_kernel = model.model.layers[0].self_attn.q_proj.kernel[...].copy()

    inject_lora(model, rank=4, alpha=8, rngs=nnx.Rngs(1))

    assert isinstance(model.model.layers[0].self_attn.q_proj, LoRALinear)
    assert parameter_count(model, nnx.LoRAParam) > 0
    assert jnp.array_equal(before_logits, model(input_ids))

    adapter_before = jax.tree.map(lambda value: value.copy(), nnx.state(model, nnx.LoRAParam))
    optimizer = create_lora_optimizer(model, learning_rate=1e-2)
    batch = {
        'input_ids': input_ids,
        'attention_mask': jnp.ones_like(input_ids),
        'labels': input_ids,
    }
    loss = train_step(model, optimizer, batch)
    adapter_after = nnx.state(model, nnx.LoRAParam)

    assert bool(jnp.isfinite(loss))
    assert jnp.array_equal(
        base_kernel,
        model.model.layers[0].self_attn.q_proj.base_module.kernel[...],
    )
    changed = [
        not jnp.array_equal(old, new)
        for old, new in zip(
            jax.tree.leaves(adapter_before),
            jax.tree.leaves(adapter_after),
            strict=True,
        )
    ]
    assert any(changed)
