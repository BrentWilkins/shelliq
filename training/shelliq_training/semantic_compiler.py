"""A small encoder-decoder Transformer for semantic shell compilation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from shelliq_training.compiler_data import LABEL_IGNORE_INDEX, CompilerBatch


@dataclass(frozen=True, slots=True)
class CompilerConfig:
    vocab_size: int
    pad_token_id: int
    decoder_start_token_id: int
    eos_token_id: int
    d_model: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    feedforward_size: int = 1024
    dropout: float = 0.1
    max_source_length: int = 512
    max_target_length: int = 256

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError('vocab_size must be positive')
        if self.d_model <= 0 or self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError('d_model must be positive and divisible by num_heads')
        if self.num_encoder_layers <= 0 or self.num_decoder_layers <= 0:
            raise ValueError('encoder and decoder must each have at least one layer')
        if self.feedforward_size <= 0:
            raise ValueError('feedforward_size must be positive')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if self.max_source_length < 2 or self.max_target_length < 2:
            raise ValueError('maximum sequence lengths must be at least 2')
        for name in ('pad_token_id', 'decoder_start_token_id', 'eos_token_id'):
            value = getattr(self, name)
            if not 0 <= value < self.vocab_size:
                raise ValueError(f'{name} must be in the vocabulary')

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class SemanticCompiler(nn.Module):
    """Project-owned seq2seq model with no pretrained neural weights."""

    def __init__(self, config: CompilerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.source_positions = nn.Embedding(config.max_source_length, config.d_model)
        self.target_positions = nn.Embedding(config.max_target_length, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.num_heads,
            config.feedforward_size,
            config.dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            config.d_model,
            config.num_heads,
            config.feedforward_size,
            config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_encoder_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_decoder_layers)
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(self, source_ids: torch.Tensor, source_attention_mask: torch.Tensor) -> torch.Tensor:
        self._check_length(source_ids, self.config.max_source_length, 'source')
        hidden = self._embed(source_ids, self.source_positions)
        memory = self.encoder(hidden, src_key_padding_mask=~source_attention_mask.bool())
        return self.encoder_norm(memory)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        memory: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._check_length(decoder_input_ids, self.config.max_target_length, 'target')
        hidden = self._embed(decoder_input_ids, self.target_positions)
        target_size = decoder_input_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones((target_size, target_size), dtype=torch.bool, device=decoder_input_ids.device),
            diagonal=1,
        )
        decoded = self.decoder(
            hidden,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~decoder_attention_mask.bool(),
            memory_key_padding_mask=~source_attention_mask.bool(),
        )
        return self.output_projection(self.decoder_norm(decoded))

    def forward(
        self,
        source_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(source_ids, source_attention_mask)
        return self.decode(decoder_input_ids, decoder_attention_mask, memory, source_attention_mask)

    def loss(self, batch: CompilerBatch) -> torch.Tensor:
        logits = self(
            batch.source_ids,
            batch.source_attention_mask,
            batch.decoder_input_ids,
            batch.decoder_attention_mask,
        )
        return functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch.labels.reshape(-1),
            ignore_index=LABEL_IGNORE_INDEX,
        )

    @torch.no_grad()
    def generate(
        self,
        source_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        *,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding for deterministic evaluation."""
        limit = max_new_tokens or self.config.max_target_length
        if not 1 <= limit <= self.config.max_target_length:
            raise ValueError('max_new_tokens must fit max_target_length')
        memory = self.encode(source_ids, source_attention_mask)
        generated = torch.full(
            (source_ids.shape[0], 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=source_ids.device,
        )
        finished = torch.zeros(source_ids.shape[0], dtype=torch.bool, device=source_ids.device)
        for _ in range(limit):
            decoder_mask = torch.ones_like(generated, dtype=torch.bool)
            logits = self.decode(generated, decoder_mask, memory, source_attention_mask)
            next_token = logits[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, self.config.eos_token_id, next_token)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            finished |= next_token == self.config.eos_token_id
            if bool(finished.all()):
                break
        return generated[:, 1:]

    def _embed(self, token_ids: torch.Tensor, positions: nn.Embedding) -> torch.Tensor:
        indices = torch.arange(token_ids.shape[1], device=token_ids.device)
        return self.token_embedding(token_ids) * math.sqrt(self.config.d_model) + positions(indices)[None, :, :]

    @staticmethod
    def _check_length(token_ids: torch.Tensor, maximum: int, name: str) -> None:
        if token_ids.ndim != 2:
            raise ValueError(f'{name}_ids must have shape [batch, sequence]')
        if token_ids.shape[1] > maximum:
            raise ValueError(f'{name} sequence exceeds configured maximum {maximum}')
