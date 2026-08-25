"""Transferred CodeT5 decoder with atomic lexical-candidate selection."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from shelliq_training.semantic_action_model import LABEL_IGNORE_INDEX, ActionDecoderConfig
from shelliq_training.semantic_action_pointer_model import _action_embedding_initialization
from shelliq_training.semantic_actions import ActionGrammar, GrammarCursor

CANDIDATE_IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class CandidateActionBatch:
    source_ids: torch.Tensor
    source_attention_mask: torch.Tensor
    source_bytes: torch.Tensor
    source_byte_mask: torch.Tensor
    source_byte_alignment: torch.Tensor
    candidate_starts: torch.Tensor
    candidate_ends: torch.Tensor
    candidate_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    labels: torch.Tensor
    candidate_labels: torch.Tensor
    word_starts: torch.Tensor

    def to(self, device: torch.device) -> CandidateActionBatch:
        return CandidateActionBatch(
            self.source_ids.to(device),
            self.source_attention_mask.to(device),
            self.source_bytes.to(device),
            self.source_byte_mask.to(device),
            self.source_byte_alignment.to(device),
            self.candidate_starts.to(device),
            self.candidate_ends.to(device),
            self.candidate_mask.to(device),
            self.decoder_input_ids.to(device),
            self.decoder_attention_mask.to(device),
            self.labels.to(device),
            self.candidate_labels.to(device),
            self.word_starts.to(device),
        )


@dataclass(frozen=True, slots=True)
class CandidateActionOutput:
    action_logits: torch.Tensor
    candidate_logits: torch.Tensor
    gate_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class _BeamHypothesis:
    actions: tuple[int, ...]
    cursor: GrammarCursor
    score: float


class SemanticActionCandidateModel(nn.Module):
    """Select complete lexer-owned spans or fall back to byte generation."""

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
        nn.init.normal_(self.action_embedding.weight, mean=0, std=config.d_model**-0.5)
        if initial_action_embedding is not None:
            if initial_action_embedding.shape != self.action_embedding.weight.shape:
                raise ValueError('initial action embedding has wrong shape')
            self.action_embedding.weight.data.copy_(initial_action_embedding)
        self.decoder.embed_tokens = self.action_embedding
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_projection.weight = self.action_embedding.weight
        self.source_byte_embedding = nn.Embedding(256, config.d_model)
        self.source_byte_positions = nn.Embedding(maximum_source_bytes, config.d_model)
        self.byte_norm = nn.LayerNorm(config.d_model)
        self.candidate_key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.candidate_query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.copy_gate = nn.Linear(config.d_model, 1)
        for module in (self.source_byte_embedding, self.source_byte_positions, self.candidate_key, self.candidate_query):
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
    ) -> SemanticActionCandidateModel:
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

    def encode(self, batch: CandidateActionBatch) -> torch.Tensor:
        return self.encoder(
            input_ids=batch.source_ids,
            attention_mask=batch.source_attention_mask,
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
        ).last_hidden_state

    def _byte_keys(self, batch: CandidateActionBatch, memory: torch.Tensor) -> torch.Tensor:
        aligned = torch.bmm(batch.source_byte_alignment.to(memory.dtype), memory)
        positions = torch.arange(batch.source_bytes.shape[1], device=batch.source_bytes.device).unsqueeze(0)
        return self.byte_norm(aligned + self.source_byte_embedding(batch.source_bytes) + self.source_byte_positions(positions))

    def _candidate_keys(self, batch: CandidateActionBatch, byte_keys: torch.Tensor) -> torch.Tensor:
        prefix = functional.pad(byte_keys.cumsum(dim=1), (0, 0, 1, 0))
        width = byte_keys.shape[-1]
        starts = batch.candidate_starts.unsqueeze(-1).expand(-1, -1, width)
        ends = batch.candidate_ends.unsqueeze(-1).expand(-1, -1, width)
        sums = prefix.gather(1, ends) - prefix.gather(1, starts)
        lengths = (batch.candidate_ends - batch.candidate_starts).clamp_min(1).unsqueeze(-1)
        return sums / lengths.to(sums.dtype)

    def distributions(
        self,
        batch: CandidateActionBatch,
        hidden: torch.Tensor,
        memory: torch.Tensor,
    ) -> CandidateActionOutput:
        candidates = self._candidate_keys(batch, self._byte_keys(batch, memory))
        candidate_logits = torch.einsum(
            'btd,bcd->btc',
            self.candidate_query(hidden),
            self.candidate_key(candidates),
        ) / math.sqrt(self.config.d_model)
        candidate_logits = candidate_logits.masked_fill(~batch.candidate_mask[:, None, :], -torch.inf)
        action_logits = self.output_projection(hidden * self.config.d_model**-0.5)
        return CandidateActionOutput(action_logits, candidate_logits, self.copy_gate(hidden).squeeze(-1))

    def forward(self, batch: CandidateActionBatch) -> CandidateActionOutput:
        memory = self.encode(batch)
        hidden = self.decode_hidden(
            batch.decoder_input_ids,
            batch.decoder_attention_mask,
            memory,
            batch.source_attention_mask,
        )
        return self.distributions(batch, hidden, memory)

    def loss_components(self, batch: CandidateActionBatch) -> dict[str, torch.Tensor]:
        output = self(batch)
        action_loss = functional.cross_entropy(
            output.action_logits.reshape(-1, self.config.vocab_size),
            batch.labels.reshape(-1),
            ignore_index=LABEL_IGNORE_INDEX,
        )
        copyable = batch.candidate_labels != CANDIDATE_IGNORE_INDEX
        candidate_loss = (
            functional.cross_entropy(output.candidate_logits[copyable], batch.candidate_labels[copyable])
            if bool(copyable.any())
            else action_loss.new_zeros(())
        )
        word_starts = batch.word_starts & (batch.labels != LABEL_IGNORE_INDEX)
        gate_loss = (
            functional.binary_cross_entropy_with_logits(
                output.gate_logits[word_starts],
                copyable[word_starts].to(output.gate_logits.dtype),
            )
            if bool(word_starts.any())
            else action_loss.new_zeros(())
        )
        return {
            'action': action_loss,
            'candidate': candidate_loss,
            'gate': gate_loss,
            'total': action_loss + 0.25 * candidate_loss + 0.1 * gate_loss,
        }

    def loss(self, batch: CandidateActionBatch) -> torch.Tensor:
        return self.loss_components(batch)['total']

    @torch.no_grad()
    def generate(
        self,
        batch: CandidateActionBatch,
        grammar: ActionGrammar,
        *,
        max_new_tokens: int | None = None,
        copy_threshold: float = 0.5,
    ) -> list[tuple[int, ...]]:
        limit = max_new_tokens or self.config.max_target_length - 1
        if not 1 <= limit < self.config.max_target_length:
            raise ValueError('max_new_tokens leaves no room for BOS')
        if batch.source_ids.shape[0] != 1:
            raise ValueError('atomic candidate generation requires batch size one')
        memory = self.encode(batch)
        generated = torch.full(
            (1, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=batch.source_ids.device,
        )
        cursor = grammar.cursor()
        cursor.advance(self.config.bos_token_id)
        maximum_length = min(self.config.max_target_length, limit + 1)
        while not cursor.complete and generated.shape[1] < maximum_length:
            hidden = self.decode_hidden(
                generated,
                torch.ones_like(generated, dtype=torch.bool),
                memory,
                batch.source_attention_mask,
            )
            output = self.distributions(batch, hidden[:, -1:], memory)
            payload = grammar.states[cursor.state].byte_payload
            remaining = maximum_length - generated.shape[1]
            if payload is not None and not cursor.payload and float(output.gate_logits[0, -1].sigmoid()) >= copy_threshold:
                choice = int(output.candidate_logits[0, -1].argmax())
                start = int(batch.candidate_starts[0, choice])
                end = int(batch.candidate_ends[0, choice])
                raw = bytes(batch.source_bytes[0, start:end].tolist())
                if raw and len(raw) + 1 <= remaining:
                    expanded = [payload.byte_offset + byte for byte in raw] + [payload.end_token]
                    try:
                        for value in expanded:
                            cursor.advance(value)
                    except ValueError:
                        pass
                    else:
                        generated = torch.cat(
                            (generated, torch.tensor([expanded], dtype=torch.long, device=generated.device)),
                            dim=1,
                        )
                        continue
            logits = output.action_logits[0, -1]
            constrained = torch.full_like(logits, -torch.inf)
            constrained[list(cursor.allowed())] = logits[list(cursor.allowed())]
            value = int(constrained.argmax())
            cursor.advance(value)
            generated = torch.cat(
                (generated, torch.tensor([[value]], dtype=torch.long, device=generated.device)),
                dim=1,
            )
        row = generated[0].tolist()
        if self.config.eos_token_id in row:
            row = row[: row.index(self.config.eos_token_id) + 1]
        return [tuple(row)]

    @torch.no_grad()
    def generate_beam(
        self,
        batch: CandidateActionBatch,
        grammar: ActionGrammar,
        *,
        beam_width: int = 8,
        max_new_tokens: int | None = None,
    ) -> list[tuple[int, ...]]:
        """Decode fixed lexical candidates and structure with a bounded beam."""
        if beam_width <= 0:
            raise ValueError('beam_width must be positive')
        limit = max_new_tokens or self.config.max_target_length - 1
        if not 1 <= limit < self.config.max_target_length:
            raise ValueError('max_new_tokens leaves no room for BOS')
        if batch.source_ids.shape[0] != 1:
            raise ValueError('candidate beam generation requires batch size one')
        memory = self.encode(batch)
        cursor = grammar.cursor()
        cursor.advance(self.config.bos_token_id)
        initial = _BeamHypothesis((self.config.bos_token_id,), cursor, 0.0)
        active = [initial]
        completed: list[_BeamHypothesis] = []
        bounded: list[_BeamHypothesis] = []
        maximum_length = min(self.config.max_target_length, limit + 1)

        while active:
            if completed and completed[0].score >= active[0].score:
                break
            expanded: dict[tuple[int, ...], _BeamHypothesis] = {}
            for hypothesis in active:
                produced = False
                hidden = self.decode_hidden(
                    torch.tensor([hypothesis.actions], dtype=torch.long, device=batch.source_ids.device),
                    torch.ones((1, len(hypothesis.actions)), dtype=torch.bool, device=batch.source_ids.device),
                    memory,
                    batch.source_attention_mask,
                )
                output = self.distributions(batch, hidden[:, -1:], memory)
                payload = grammar.states[hypothesis.cursor.state].byte_payload
                if payload is not None:
                    candidate_scores = output.candidate_logits[0, -1].log_softmax(dim=-1)
                    choices = candidate_scores.topk(min(beam_width, int(batch.candidate_mask[0].sum()))).indices
                    for choice_tensor in choices:
                        choice = int(choice_tensor)
                        start = int(batch.candidate_starts[0, choice])
                        end = int(batch.candidate_ends[0, choice])
                        raw = bytes(batch.source_bytes[0, start:end].tolist())
                        addition = tuple(payload.byte_offset + byte for byte in raw) + (payload.end_token,)
                        if not raw or len(hypothesis.actions) + len(addition) > maximum_length:
                            continue
                        next_cursor = _clone_cursor(hypothesis.cursor)
                        for value in addition:
                            next_cursor.advance(value)
                        candidate = _BeamHypothesis(
                            hypothesis.actions + addition,
                            next_cursor,
                            hypothesis.score + float(candidate_scores[choice]),
                        )
                        _retain_best(expanded, candidate)
                        produced = True
                else:
                    allowed = hypothesis.cursor.allowed()
                    allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=batch.source_ids.device)
                    scores = output.action_logits[0, -1, allowed_tensor].log_softmax(dim=-1)
                    for index, value in enumerate(allowed):
                        if len(hypothesis.actions) + 1 > maximum_length:
                            continue
                        next_cursor = _clone_cursor(hypothesis.cursor)
                        next_cursor.advance(value)
                        candidate = _BeamHypothesis(
                            hypothesis.actions + (value,),
                            next_cursor,
                            hypothesis.score + float(scores[index]),
                        )
                        _retain_best(expanded, candidate)
                        produced = True
                if not produced:
                    bounded.append(hypothesis)

            active = []
            for candidate in expanded.values():
                if candidate.cursor.complete:
                    completed.append(candidate)
                else:
                    active.append(candidate)
            active.sort(key=lambda item: (-item.score, item.actions))
            active = active[:beam_width]
            completed.sort(key=lambda item: (-item.score, item.actions))
            completed = completed[:beam_width]

        bounded.sort(key=lambda item: (-item.score, item.actions))
        winner = completed[0] if completed else (bounded[0] if bounded else initial)
        return [winner.actions]


def _clone_cursor(cursor: GrammarCursor) -> GrammarCursor:
    return GrammarCursor(cursor.grammar, cursor.state, cursor.payload)


def _retain_best(target: dict[tuple[int, ...], _BeamHypothesis], candidate: _BeamHypothesis) -> None:
    previous = target.get(candidate.actions)
    if previous is None or candidate.score > previous.score:
        target[candidate.actions] = candidate
