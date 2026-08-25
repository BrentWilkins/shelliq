"""Pretrained semantic encoder with a compact project-owned action decoder."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional

from shelliq_training.semantic_actions import ActionGrammar

LABEL_IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class ActionDecoderConfig:
    vocab_size: int = 320
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 4
    feedforward_size: int = 2048
    dropout: float = 0.1
    max_target_length: int = 192
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.d_model <= 0 or self.d_model % self.num_heads:
            raise ValueError('invalid action decoder dimensions')
        if self.num_layers <= 0 or self.feedforward_size <= 0:
            raise ValueError('decoder layers and feedforward size must be positive')
        if not 0 <= self.dropout < 1 or self.max_target_length < 2:
            raise ValueError('invalid dropout or target length')
        for token in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
            if not 0 <= token < self.vocab_size:
                raise ValueError('special action token is outside vocabulary')

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActionBatch:
    source_ids: torch.Tensor
    source_attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> ActionBatch:
        return ActionBatch(
            self.source_ids.to(device),
            self.source_attention_mask.to(device),
            self.decoder_input_ids.to(device),
            self.decoder_attention_mask.to(device),
            self.labels.to(device),
        )


def collate_actions(examples, *, source_length: int, target_length: int, source_pad_token_id: int) -> ActionBatch:
    if not examples:
        raise ValueError('cannot collate an empty action batch')
    if any(example.corpus is not examples[0].corpus for example in examples):
        raise ValueError('cannot mix corpus privacy classes')
    source_rows = []
    source_masks = []
    decoder_rows = []
    decoder_masks = []
    labels = []
    for example in examples:
        if len(example.source_ids) > source_length or len(example.action_ids) > target_length:
            raise ValueError(f'{example.record_id}: sequence exceeds fixed batch shape')
        if len(example.action_ids) < 2 or example.action_ids[0] != 1 or example.action_ids[-1] != 2:
            raise ValueError(f'{example.record_id}: action target lacks BOS/EOS')
        source_padding = source_length - len(example.source_ids)
        shifted = example.action_ids[:-1]
        expected = example.action_ids[1:]
        target_padding = target_length - len(expected)
        source_rows.append(example.source_ids + (source_pad_token_id,) * source_padding)
        source_masks.append((1,) * len(example.source_ids) + (0,) * source_padding)
        decoder_rows.append(shifted + (0,) * target_padding)
        decoder_masks.append((1,) * len(shifted) + (0,) * target_padding)
        labels.append(expected + (LABEL_IGNORE_INDEX,) * target_padding)
    return ActionBatch(
        torch.tensor(source_rows, dtype=torch.long),
        torch.tensor(source_masks, dtype=torch.bool),
        torch.tensor(decoder_rows, dtype=torch.long),
        torch.tensor(decoder_masks, dtype=torch.bool),
        torch.tensor(labels, dtype=torch.long),
    )


class SemanticActionModel(nn.Module):
    """CodeT5-compatible encoder paired with a small Transformer decoder."""

    def __init__(self, encoder: nn.Module, config: ActionDecoderConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.action_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.target_positions = nn.Embedding(config.max_target_length, config.d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(layer, config.num_layers)
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_projection.weight = self.action_embedding.weight
        self._reset_decoder()

    def _reset_decoder(self) -> None:
        for name, parameter in self.named_parameters():
            if not name.startswith('encoder.') and parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(self, source_ids: torch.Tensor, source_attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=source_ids, attention_mask=source_attention_mask, return_dict=True)
        memory = output.last_hidden_state
        if memory.shape[-1] != self.config.d_model:
            raise ValueError(f'encoder width {memory.shape[-1]} does not match decoder width {self.config.d_model}')
        return memory

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        memory: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        length = decoder_input_ids.shape[1]
        if length > self.config.max_target_length:
            raise ValueError('decoder input exceeds configured target length')
        positions = torch.arange(length, device=decoder_input_ids.device).unsqueeze(0)
        hidden = self.action_embedding(decoder_input_ids) * math.sqrt(self.config.d_model)
        hidden = hidden + self.target_positions(positions)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=hidden.device), diagonal=1)
        decoded = self.decoder(
            hidden,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=~decoder_attention_mask.bool(),
            memory_key_padding_mask=~source_attention_mask.bool(),
        )
        return self.output_projection(self.decoder_norm(decoded))

    def forward(self, batch: ActionBatch) -> torch.Tensor:
        memory = self.encode(batch.source_ids, batch.source_attention_mask)
        return self.decode(batch.decoder_input_ids, batch.decoder_attention_mask, memory, batch.source_attention_mask)

    def loss(self, batch: ActionBatch) -> torch.Tensor:
        logits = self(batch)
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
        grammar: ActionGrammar,
        *,
        max_new_tokens: int | None = None,
    ) -> list[tuple[int, ...]]:
        limit = max_new_tokens or self.config.max_target_length - 1
        if not 1 <= limit < self.config.max_target_length:
            raise ValueError('max_new_tokens must leave room for BOS')
        memory = self.encode(source_ids, source_attention_mask)
        generated = torch.full((source_ids.shape[0], 1), self.config.bos_token_id, dtype=torch.long, device=source_ids.device)
        cursors = [grammar.cursor() for _ in range(source_ids.shape[0])]
        for cursor in cursors:
            cursor.advance(self.config.bos_token_id)
        for _ in range(limit):
            if all(cursor.complete for cursor in cursors):
                break
            mask = torch.ones_like(generated, dtype=torch.bool)
            logits = self.decode(generated, mask, memory, source_attention_mask)[:, -1]
            constrained = torch.full_like(logits, -torch.inf)
            for row, cursor in enumerate(cursors):
                allowed = (self.config.pad_token_id,) if cursor.complete else cursor.allowed()
                if not allowed:
                    raise RuntimeError('grammar reached a non-complete state with no allowed actions')
                constrained[row, list(allowed)] = logits[row, list(allowed)]
            next_tokens = constrained.argmax(dim=-1)
            generated = torch.cat((generated, next_tokens[:, None]), dim=1)
            for cursor, value in zip(cursors, next_tokens.tolist(), strict=True):
                if not cursor.complete:
                    cursor.advance(value)
        outputs: list[tuple[int, ...]] = []
        for row, cursor in zip(generated.tolist(), cursors, strict=True):
            if cursor.complete:
                eos = row.index(self.config.eos_token_id)
                outputs.append(tuple(row[: eos + 1]))
            else:
                outputs.append(tuple(row))
        return outputs
