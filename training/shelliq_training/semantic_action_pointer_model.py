"""Transferred CodeT5 decoder with source-byte pointer/generator actions."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from shelliq_training.semantic_action_model import LABEL_IGNORE_INDEX, ActionDecoderConfig
from shelliq_training.semantic_actions import ActionGrammar

COPY_IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class PointerActionBatch:
    source_ids: torch.Tensor
    source_attention_mask: torch.Tensor
    source_bytes: torch.Tensor
    source_byte_mask: torch.Tensor
    source_byte_alignment: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    labels: torch.Tensor
    copy_labels: torch.Tensor

    def to(self, device: torch.device) -> PointerActionBatch:
        return PointerActionBatch(
            self.source_ids.to(device),
            self.source_attention_mask.to(device),
            self.source_bytes.to(device),
            self.source_byte_mask.to(device),
            self.source_byte_alignment.to(device),
            self.decoder_input_ids.to(device),
            self.decoder_attention_mask.to(device),
            self.labels.to(device),
            self.copy_labels.to(device),
        )


@dataclass(frozen=True, slots=True)
class PointerActionOutput:
    log_probabilities: torch.Tensor
    pointer_logits: torch.Tensor
    gate_logits: torch.Tensor


class SemanticActionPointerModel(nn.Module):
    """Four transferred T5 decoder blocks plus a supervised byte pointer."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        config: ActionDecoderConfig,
        *,
        maximum_source_bytes: int,
        initial_action_embedding: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if maximum_source_bytes <= 0:
            raise ValueError('maximum_source_bytes must be positive')
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.maximum_source_bytes = maximum_source_bytes
        self.action_embedding = nn.Embedding(config.vocab_size, config.d_model)
        if initial_action_embedding is None:
            nn.init.normal_(self.action_embedding.weight, mean=0, std=config.d_model**-0.5)
        else:
            if initial_action_embedding.shape != self.action_embedding.weight.shape:
                raise ValueError('initial action embedding has the wrong shape')
            self.action_embedding.weight.data.copy_(initial_action_embedding)
        self.decoder.embed_tokens = self.action_embedding
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_projection.weight = self.action_embedding.weight
        self.source_byte_embedding = nn.Embedding(256, config.d_model)
        self.source_byte_positions = nn.Embedding(maximum_source_bytes, config.d_model)
        self.pointer_key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.pointer_query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.pointer_norm = nn.LayerNorm(config.d_model)
        self.copy_gate = nn.Linear(config.d_model, 1)
        for module in (
            self.source_byte_embedding,
            self.source_byte_positions,
            self.pointer_key,
            self.pointer_query,
        ):
            for parameter in module.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

    @classmethod
    def from_codet5(
        cls,
        pretrained,
        tokenizer,
        *,
        maximum_source_bytes: int,
        dropout: float = 0.1,
    ) -> SemanticActionPointerModel:
        config = ActionDecoderConfig(dropout=dropout)
        encoder = pretrained.get_encoder()
        decoder = pretrained.get_decoder()
        decoder.config = copy.deepcopy(decoder.config)
        decoder.block = nn.ModuleList(list(decoder.block[: config.num_layers]))
        decoder.config.num_layers = config.num_layers
        for module in decoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
        initial = _action_embedding_initialization(pretrained.shared.weight.detach(), tokenizer, config)
        return cls(
            encoder,
            decoder,
            config,
            maximum_source_bytes=maximum_source_bytes,
            initial_action_embedding=initial,
        )

    def encode(self, batch: PointerActionBatch) -> torch.Tensor:
        return self.encoder(
            input_ids=batch.source_ids,
            attention_mask=batch.source_attention_mask,
            return_dict=True,
        ).last_hidden_state

    def decode_hidden(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        memory: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if decoder_input_ids.shape[1] > self.config.max_target_length:
            raise ValueError('decoder input exceeds configured target length')
        return self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=source_attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    def distributions(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        source_bytes: torch.Tensor,
        source_byte_mask: torch.Tensor,
        source_byte_alignment: torch.Tensor,
    ) -> PointerActionOutput:
        aligned_memory = torch.bmm(source_byte_alignment.to(memory.dtype), memory)
        positions = torch.arange(source_bytes.shape[1], device=source_bytes.device).unsqueeze(0)
        byte_keys = self.pointer_norm(
            aligned_memory + self.source_byte_embedding(source_bytes) + self.source_byte_positions(positions)
        )
        pointer_logits = torch.einsum('btd,bsd->bts', self.pointer_query(hidden), self.pointer_key(byte_keys)) / math.sqrt(
            self.config.d_model
        )
        pointer_logits = pointer_logits.masked_fill(~source_byte_mask[:, None, :], -torch.inf)
        pointer_probabilities = pointer_logits.softmax(dim=-1)
        copied = torch.zeros((*hidden.shape[:2], self.config.vocab_size), dtype=hidden.dtype, device=hidden.device)
        byte_actions = source_bytes[:, None, :].expand(hidden.shape[0], hidden.shape[1], -1) + 64
        copied.scatter_add_(2, byte_actions, pointer_probabilities)
        # T5's tied LM head applies this scale before projection. Retaining it is
        # essential when reusing pretrained decoder blocks with a tied action head.
        generated = self.output_projection(hidden * self.config.d_model**-0.5).softmax(dim=-1)
        gate_logits = self.copy_gate(hidden).squeeze(-1)
        gate = gate_logits.sigmoid().unsqueeze(-1)
        probabilities = generated * (1 - gate) + copied * gate
        return PointerActionOutput(probabilities.clamp_min(1e-9).log(), pointer_logits, gate_logits)

    def forward(self, batch: PointerActionBatch) -> PointerActionOutput:
        memory = self.encode(batch)
        hidden = self.decode_hidden(
            batch.decoder_input_ids,
            batch.decoder_attention_mask,
            memory,
            batch.source_attention_mask,
        )
        return self.distributions(
            hidden,
            memory,
            batch.source_bytes,
            batch.source_byte_mask,
            batch.source_byte_alignment,
        )

    def loss(self, batch: PointerActionBatch) -> torch.Tensor:
        return self.loss_components(batch)['total']

    def loss_components(self, batch: PointerActionBatch) -> dict[str, torch.Tensor]:
        output = self(batch)
        action_loss = functional.nll_loss(
            output.log_probabilities.reshape(-1, self.config.vocab_size),
            batch.labels.reshape(-1),
            ignore_index=LABEL_IGNORE_INDEX,
        )
        copyable = batch.copy_labels != COPY_IGNORE_INDEX
        if bool(copyable.any()):
            pointer_loss = functional.cross_entropy(output.pointer_logits[copyable], batch.copy_labels[copyable])
        else:
            pointer_loss = action_loss.new_zeros(())
        active = batch.labels != LABEL_IGNORE_INDEX
        gate_targets = copyable.to(output.gate_logits.dtype)
        gate_loss = functional.binary_cross_entropy_with_logits(output.gate_logits[active], gate_targets[active])
        return {
            'action': action_loss,
            'pointer': pointer_loss,
            'gate': gate_loss,
            'total': action_loss + 0.25 * pointer_loss + 0.1 * gate_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        batch: PointerActionBatch,
        grammar: ActionGrammar,
        *,
        max_new_tokens: int | None = None,
    ) -> list[tuple[int, ...]]:
        limit = max_new_tokens or self.config.max_target_length - 1
        if not 1 <= limit < self.config.max_target_length:
            raise ValueError('max_new_tokens must leave room for BOS')
        memory = self.encode(batch)
        generated = torch.full(
            (batch.source_ids.shape[0], 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=batch.source_ids.device,
        )
        cursors = [grammar.cursor() for _ in range(batch.source_ids.shape[0])]
        for cursor in cursors:
            cursor.advance(self.config.bos_token_id)
        for _ in range(limit):
            if all(cursor.complete for cursor in cursors):
                break
            decoder_mask = torch.ones_like(generated, dtype=torch.bool)
            hidden = self.decode_hidden(generated, decoder_mask, memory, batch.source_attention_mask)
            output = self.distributions(
                hidden[:, -1:],
                memory,
                batch.source_bytes,
                batch.source_byte_mask,
                batch.source_byte_alignment,
            )
            logits = output.log_probabilities[:, -1]
            constrained = torch.full_like(logits, -torch.inf)
            for row, cursor in enumerate(cursors):
                allowed = (self.config.pad_token_id,) if cursor.complete else cursor.allowed()
                constrained[row, list(allowed)] = logits[row, list(allowed)]
            next_tokens = constrained.argmax(dim=-1)
            generated = torch.cat((generated, next_tokens[:, None]), dim=1)
            for cursor, value in zip(cursors, next_tokens.tolist(), strict=True):
                if not cursor.complete:
                    cursor.advance(value)
        outputs = []
        for row, cursor in zip(generated.tolist(), cursors, strict=True):
            if cursor.complete:
                outputs.append(tuple(row[: row.index(self.config.eos_token_id) + 1]))
            else:
                outputs.append(tuple(row))
        return outputs


def _action_embedding_initialization(shared_embedding: torch.Tensor, tokenizer, config: ActionDecoderConfig) -> torch.Tensor:
    generator = torch.Generator(device=shared_embedding.device).manual_seed(20260827)
    initialized = torch.empty(
        config.vocab_size,
        config.d_model,
        dtype=shared_embedding.dtype,
        device=shared_embedding.device,
    )
    initialized.normal_(mean=0, std=float(shared_embedding.std()), generator=generator)
    for action_id, source_id in ((config.pad_token_id, 0), (config.bos_token_id, 1), (config.eos_token_id, 2)):
        initialized[action_id] = shared_embedding[source_id]
    for byte in range(128):
        token_ids = tokenizer.encode(bytes([byte]).decode('ascii'), add_special_tokens=False)
        if token_ids:
            initialized[64 + byte] = shared_embedding[token_ids].mean(dim=0)
    return initialized
