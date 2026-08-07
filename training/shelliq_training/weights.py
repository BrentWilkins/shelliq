import jax.numpy as jnp
from flax import nnx

from shelliq_training.model import Qwen2ForCausalLM


def load_hf_state_dict(model: Qwen2ForCausalLM, state_dict: dict[str, jnp.ndarray], param_dtype: jnp.dtype) -> None:
    """Copies a Qwen2ForCausalLM HF safetensors state dict into an nnx model in place.

    HF `nn.Linear` weights are `[out, in]`; nnx `Linear` kernels are `[in, out]`, so every
    linear weight is transposed on the way in.
    """

    def linear(module: nnx.Linear, weight_key: str, bias_key: str | None = None) -> None:
        module.kernel.set_value(state_dict[weight_key].astype(param_dtype).T)
        if bias_key is not None:
            module.bias.set_value(state_dict[bias_key].astype(param_dtype))

    def norm(module: nnx.Module, weight_key: str) -> None:
        module.weight.set_value(state_dict[weight_key].astype(param_dtype))

    model.model.embed_tokens.embedding.set_value(state_dict['model.embed_tokens.weight'].astype(param_dtype))
    norm(model.model.norm, 'model.norm.weight')

    for i, layer in enumerate(model.model.layers):
        prefix = f'model.layers.{i}.'
        attn = layer.self_attn
        linear(attn.q_proj, prefix + 'self_attn.q_proj.weight', prefix + 'self_attn.q_proj.bias')
        linear(attn.k_proj, prefix + 'self_attn.k_proj.weight', prefix + 'self_attn.k_proj.bias')
        linear(attn.v_proj, prefix + 'self_attn.v_proj.weight', prefix + 'self_attn.v_proj.bias')
        linear(attn.o_proj, prefix + 'self_attn.o_proj.weight')

        mlp = layer.mlp
        linear(mlp.gate_proj, prefix + 'mlp.gate_proj.weight')
        linear(mlp.up_proj, prefix + 'mlp.up_proj.weight')
        linear(mlp.down_proj, prefix + 'mlp.down_proj.weight')

        norm(layer.input_layernorm, prefix + 'input_layernorm.weight')
        norm(layer.post_attention_layernorm, prefix + 'post_attention_layernorm.weight')

    if not model.config.tie_word_embeddings:
        linear(model.lm_head, 'lm_head.weight')
