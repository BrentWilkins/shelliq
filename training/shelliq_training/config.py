import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Qwen2Config:
    """Matches Qwen/Qwen2.5-Coder-0.5B-Instruct's config.json."""

    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    @classmethod
    def from_json(cls, path: Path) -> Qwen2Config:
        """Load the architecture fields used by the local Qwen2 implementation."""
        document = json.loads(path.read_text())
        names = cls.__dataclass_fields__
        return cls(**{name: document[name] for name in names})

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError('hidden_size must be divisible by num_attention_heads')
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError('num_attention_heads must be divisible by num_key_value_heads')
        if self.head_dim % 2 != 0:
            raise ValueError('head_dim must be even for rotary embeddings')

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
