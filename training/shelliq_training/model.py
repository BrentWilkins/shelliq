import jax
import jax.numpy as jnp
from flax import nnx

from shelliq_training.config import Qwen2Config


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float, *, param_dtype: jnp.dtype):
        self.weight = nnx.Param(jnp.ones((dim,), dtype=param_dtype))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        input_dtype = x.dtype
        normalized = x.astype(jnp.float32)
        variance = jnp.mean(normalized * normalized, axis=-1, keepdims=True)
        normalized *= jax.lax.rsqrt(variance + self.eps)
        return self.weight[...] * normalized.astype(input_dtype)


def rotate_half(x: jax.Array) -> jax.Array:
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def compute_rope(head_dim: int, theta: float, position_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return RoPE cos/sin for position ids shaped ``[batch, sequence]``."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / float(head_dim)))
    frequencies = position_ids.astype(jnp.float32)[..., None] * inv_freq
    embeddings = jnp.concatenate((frequencies, frequencies), axis=-1)
    return jnp.cos(embeddings), jnp.sin(embeddings)


def apply_rope(x: jax.Array, cosine: jax.Array, sine: jax.Array) -> jax.Array:
    cosine = cosine[:, :, None, :].astype(x.dtype)
    sine = sine[:, :, None, :].astype(x.dtype)
    return (x * cosine) + (rotate_half(x) * sine)


class Attention(nnx.Module):
    def __init__(
        self,
        config: Qwen2Config,
        *,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        self.q_proj = nnx.Linear(
            hidden,
            self.num_heads * self.head_dim,
            use_bias=True,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            hidden,
            self.num_kv_heads * self.head_dim,
            use_bias=True,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            hidden,
            self.num_kv_heads * self.head_dim,
            use_bias=True,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            self.num_heads * self.head_dim,
            hidden,
            use_bias=False,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        cosine: jax.Array,
        sine: jax.Array,
        attention_mask: jax.Array,
    ) -> jax.Array:
        batch, sequence, _ = x.shape
        query = self.q_proj(x).reshape(batch, sequence, self.num_heads, self.head_dim)
        key = self.k_proj(x).reshape(batch, sequence, self.num_kv_heads, self.head_dim)
        value = self.v_proj(x).reshape(batch, sequence, self.num_kv_heads, self.head_dim)
        query = apply_rope(query, cosine, sine)
        key = apply_rope(key, cosine, sine)

        repeats = self.num_heads // self.num_kv_heads
        key = jnp.repeat(key, repeats, axis=2)
        value = jnp.repeat(value, repeats, axis=2)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        scores = jnp.einsum('bhqd,bhkd->bhqk', query, key).astype(jnp.float32)
        scores *= self.head_dim**-0.5
        causal = jnp.tril(jnp.ones((sequence, sequence), dtype=jnp.bool_))
        visible_keys = attention_mask[:, None, None, :].astype(jnp.bool_)
        allowed = causal[None, None, :, :] & visible_keys

        # A left-padding query can otherwise have no visible key and produce
        # NaNs in softmax. Its output is ignored, but keeping it finite avoids
        # contaminating gradients in a padded batch.
        valid_queries = attention_mask[:, None, :, None].astype(jnp.bool_)
        diagonal = jnp.eye(sequence, dtype=jnp.bool_)[None, None, :, :]
        allowed |= (~valid_queries) & diagonal
        scores = jnp.where(allowed, scores, jnp.finfo(jnp.float32).min)

        weights = jax.nn.softmax(scores, axis=-1).astype(value.dtype)
        output = jnp.einsum('bhqk,bhkd->bhqd', weights, value)
        output = output.transpose(0, 2, 1, 3).reshape(batch, sequence, self.num_heads * self.head_dim)
        return self.o_proj(output)


class SwiGLU(nnx.Module):
    def __init__(
        self,
        config: Qwen2Config,
        *,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        self.gate_proj = nnx.Linear(
            hidden,
            intermediate,
            use_bias=False,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            hidden,
            intermediate,
            use_bias=False,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            intermediate,
            hidden,
            use_bias=False,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: Qwen2Config,
        *,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, param_dtype=param_dtype)
        self.self_attn = Attention(config, param_dtype=param_dtype, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, param_dtype=param_dtype)
        self.mlp = SwiGLU(config, param_dtype=param_dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        cosine: jax.Array,
        sine: jax.Array,
        attention_mask: jax.Array,
    ) -> jax.Array:
        x += self.self_attn(self.input_layernorm(x), cosine, sine, attention_mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Model(nnx.Module):
    def __init__(
        self,
        config: Qwen2Config,
        *,
        param_dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = nnx.List(
            [DecoderLayer(config, param_dtype=param_dtype, rngs=rngs) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, param_dtype=param_dtype)

    def __call__(self, input_ids: jax.Array, attention_mask: jax.Array | None = None) -> jax.Array:
        if input_ids.ndim != 2:
            raise ValueError('input_ids must have shape [batch, sequence]')
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
        elif attention_mask.shape != input_ids.shape:
            raise ValueError('attention_mask must have the same shape as input_ids')
        attention_mask = attention_mask.astype(jnp.bool_)

        # Cumulative positions make both left and right padding match the HF
        # convention while preserving ordinary 0..sequence-1 positions.
        position_ids = jnp.maximum(jnp.cumsum(attention_mask, axis=-1) - 1, 0)
        cosine, sine = compute_rope(self.config.head_dim, self.config.rope_theta, position_ids)
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, cosine, sine, attention_mask)
        return self.norm(hidden)


class Qwen2ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: Qwen2Config,
        *,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.model = Qwen2Model(config, param_dtype=param_dtype, rngs=rngs)
        if not config.tie_word_embeddings:
            self.lm_head = nnx.Linear(
                config.hidden_size,
                config.vocab_size,
                use_bias=False,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def __call__(self, input_ids: jax.Array, attention_mask: jax.Array | None = None) -> jax.Array:
        hidden = self.model(input_ids, attention_mask)
        if self.config.tie_word_embeddings:
            return hidden @ self.model.embed_tokens.embedding[...].T
        return self.lm_head(hidden)
