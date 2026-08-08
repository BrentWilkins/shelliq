from collections.abc import Iterable

import jax
import jax.numpy as jnp
from flax import nnx

from shelliq_training.model import Qwen2ForCausalLM

DEFAULT_TARGETS = (
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
)


class LoRALinear(nnx.Module):
    """A frozen base linear plus a scaled, zero-initialized LoRA adapter."""

    def __init__(
        self,
        base_module: nnx.Linear,
        *,
        rank: int,
        alpha: float,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        if rank <= 0:
            raise ValueError('LoRA rank must be positive')
        self.base_module = base_module
        self.adapter = nnx.LoRA(
            base_module.in_features,
            rank,
            base_module.out_features,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.scale = alpha / rank

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.base_module(x) + self.scale * self.adapter(x)


def inject_lora(
    model: Qwen2ForCausalLM,
    *,
    rank: int = 16,
    alpha: float = 32.0,
    targets: Iterable[str] = DEFAULT_TARGETS,
    param_dtype: jnp.dtype = jnp.float32,
    rngs: nnx.Rngs,
) -> None:
    """Inject LoRA adapters in-place after loading base model weights."""
    target_set = frozenset(targets)
    unknown = target_set.difference(DEFAULT_TARGETS)
    if unknown:
        names = ', '.join(sorted(unknown))
        raise ValueError(f'unknown LoRA target modules: {names}')

    for layer in model.model.layers:
        modules = {
            'q_proj': layer.self_attn,
            'k_proj': layer.self_attn,
            'v_proj': layer.self_attn,
            'o_proj': layer.self_attn,
            'gate_proj': layer.mlp,
            'up_proj': layer.mlp,
            'down_proj': layer.mlp,
        }
        # Keep RNG consumption stable across Python hash seeds.
        for name in DEFAULT_TARGETS:
            if name not in target_set:
                continue
            parent = modules[name]
            base_module = getattr(parent, name)
            if isinstance(base_module, LoRALinear):
                raise ValueError(f'LoRA already injected into {name}')
            setattr(
                parent,
                name,
                LoRALinear(
                    base_module,
                    rank=rank,
                    alpha=alpha,
                    param_dtype=param_dtype,
                    rngs=rngs,
                ),
            )


def parameter_count(model: nnx.Module, variable_type: type[nnx.Variable]) -> int:
    state = nnx.state(model, variable_type)
    return sum(int(value.size) for value in jax.tree.leaves(state))
