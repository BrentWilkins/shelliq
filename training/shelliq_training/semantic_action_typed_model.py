"""Typed candidate and argument-count semantic-action model."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from shelliq_training.semantic_action_candidate_model import (
    CANDIDATE_IGNORE_INDEX,
    SemanticActionCandidateModel,
    _clone_cursor,
)
from shelliq_training.semantic_action_model import LABEL_IGNORE_INDEX, ActionDecoderConfig
from shelliq_training.semantic_action_pointer_model import _action_embedding_initialization
from shelliq_training.semantic_actions import ActionGrammar, GrammarCursor

COUNT_IGNORE_INDEX = -100
MAXIMUM_ARGUMENTS = 12


@dataclass(frozen=True, slots=True)
class TypedActionBatch:
    source_ids: torch.Tensor
    source_attention_mask: torch.Tensor
    source_bytes: torch.Tensor
    source_byte_mask: torch.Tensor
    source_byte_alignment: torch.Tensor
    candidate_bytes: torch.Tensor
    candidate_byte_mask: torch.Tensor
    candidate_byte_histogram: torch.Tensor
    candidate_bigram_histogram: torch.Tensor
    candidate_token_ids: torch.Tensor
    candidate_token_mask: torch.Tensor
    candidate_starts: torch.Tensor
    candidate_ends: torch.Tensor
    candidate_features: torch.Tensor
    candidate_role_mask: torch.Tensor
    candidate_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    labels: torch.Tensor
    candidate_labels: torch.Tensor
    argument_count_labels: torch.Tensor
    word_role_labels: torch.Tensor

    def to(self, device: torch.device) -> TypedActionBatch:
        return TypedActionBatch(*(getattr(self, field).to(device) for field in self.__dataclass_fields__))


@dataclass(frozen=True, slots=True)
class TypedActionOutput:
    action_logits: torch.Tensor
    candidate_logits: torch.Tensor
    argument_count_logits: torch.Tensor
    role_candidate_logits: torch.Tensor | None = None
    content_role_candidate_logits: torch.Tensor | None = None
    grounding_logits: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _TypedBeam:
    actions: tuple[int, ...]
    cursor: GrammarCursor
    score: float
    desired_arguments: int | None
    emitted_arguments: int
    command_index: int
    word_index: int


class SemanticActionTypedModel(SemanticActionCandidateModel):
    """One role-masked word inventory plus explicit command argument counts."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        config: ActionDecoderConfig,
        *,
        maximum_source_bytes: int,
        initial_action_embedding: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            encoder,
            decoder,
            config,
            maximum_source_bytes=maximum_source_bytes,
            initial_action_embedding=initial_action_embedding,
        )
        self.candidate_feature_embedding = nn.Parameter(torch.empty(3, config.d_model))
        nn.init.normal_(self.candidate_feature_embedding, mean=0, std=config.d_model**-0.5)
        self.candidate_norm = nn.LayerNorm(config.d_model)
        self.argument_count_head = nn.Linear(config.d_model, MAXIMUM_ARGUMENTS + 1)
        nn.init.xavier_uniform_(self.argument_count_head.weight)
        nn.init.zeros_(self.argument_count_head.bias)

    @classmethod
    def from_codet5(
        cls,
        pretrained,
        tokenizer,
        *,
        maximum_source_bytes: int,
        dropout: float = 0.1,
    ) -> SemanticActionTypedModel:
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

    def _typed_candidate_keys(self, batch: TypedActionBatch, memory: torch.Tensor) -> torch.Tensor:
        content = torch.matmul(
            batch.candidate_byte_histogram.to(self.source_byte_embedding.weight.dtype),
            self.source_byte_embedding.weight,
        )
        context = self._candidate_keys(batch, self._byte_keys(batch, memory))
        context = context * batch.candidate_features[:, :, :1]
        features = torch.einsum('bcf,fd->bcd', batch.candidate_features, self.candidate_feature_embedding)
        return self.candidate_norm(content + context + features)

    def distributions(
        self,
        batch: TypedActionBatch,
        hidden: torch.Tensor,
        memory: torch.Tensor,
    ) -> TypedActionOutput:
        candidates = self._typed_candidate_keys(batch, memory)
        candidate_logits = torch.einsum(
            'btd,bcd->btc',
            self.candidate_query(hidden),
            self.candidate_key(candidates),
        ) / math.sqrt(self.config.d_model)
        candidate_logits = candidate_logits.masked_fill(~batch.candidate_mask[:, None, :], -torch.inf)
        action_logits = self.output_projection(hidden * self.config.d_model**-0.5)
        return TypedActionOutput(action_logits, candidate_logits, self.argument_count_head(hidden))

    def forward(self, batch: TypedActionBatch) -> TypedActionOutput:
        memory = self.encode(batch)
        hidden = self.decode_hidden(
            batch.decoder_input_ids,
            batch.decoder_attention_mask,
            memory,
            batch.source_attention_mask,
        )
        return self.distributions(batch, hidden, memory)

    def loss_components(self, batch: TypedActionBatch) -> dict[str, torch.Tensor]:
        output = self(batch)
        action_loss = functional.cross_entropy(
            output.action_logits.reshape(-1, self.config.vocab_size),
            batch.labels.reshape(-1),
            ignore_index=LABEL_IGNORE_INDEX,
        )
        copyable = batch.candidate_labels != CANDIDATE_IGNORE_INDEX
        if output.role_candidate_logits is None:
            candidate_logits = output.candidate_logits
        else:
            role_indices_4d = (
                batch.word_role_labels.clamp_min(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(*batch.word_role_labels.shape, 1, output.candidate_logits.shape[-1])
            )
            candidate_logits = output.role_candidate_logits.gather(2, role_indices_4d).squeeze(2)
        role_indices = batch.word_role_labels.clamp_min(0).unsqueeze(-1).expand_as(candidate_logits)
        role_mask = batch.candidate_role_mask.gather(1, role_indices)
        typed_logits = candidate_logits.masked_fill(~role_mask, -torch.inf)
        candidate_loss = (
            functional.cross_entropy(typed_logits[copyable], batch.candidate_labels[copyable])
            if bool(copyable.any())
            else action_loss.new_zeros(())
        )
        counted = batch.argument_count_labels != COUNT_IGNORE_INDEX
        count_loss = (
            functional.cross_entropy(
                output.argument_count_logits[counted],
                batch.argument_count_labels[counted],
            )
            if bool(counted.any())
            else action_loss.new_zeros(())
        )
        return {
            'action': action_loss,
            'candidate': candidate_loss,
            'argument_count': count_loss,
            'total': action_loss + 0.25 * candidate_loss + 0.25 * count_loss,
        }

    def loss(self, batch: TypedActionBatch) -> torch.Tensor:
        return self.loss_components(batch)['total']

    def candidate_bytes(self, batch: TypedActionBatch, choice: int) -> bytes:
        length = int(batch.candidate_byte_mask[0, choice].sum())
        return bytes(batch.candidate_bytes[0, choice, :length].tolist())

    @torch.no_grad()
    def generate_typed_beam(
        self,
        batch: TypedActionBatch,
        grammar: ActionGrammar,
        role_names: tuple[str, ...],
        *,
        beam_width: int = 8,
        max_new_tokens: int | None = None,
        forced_argument_counts: tuple[int, ...] | None = None,
        forced_words: tuple[bytes, ...] | None = None,
        argument_count_top_k: int = 4,
        ground_arguments: bool = False,
        post_command_counts: bool = False,
    ) -> list[tuple[int, ...]]:
        if beam_width <= 0:
            raise ValueError('beam_width must be positive')
        if not 1 <= argument_count_top_k <= MAXIMUM_ARGUMENTS + 1:
            raise ValueError('argument_count_top_k is outside the count vocabulary')
        limit = max_new_tokens or self.config.max_target_length - 1
        if not 1 <= limit < self.config.max_target_length:
            raise ValueError('max_new_tokens leaves no room for BOS')
        if batch.source_ids.shape[0] != 1:
            raise ValueError('typed beam generation requires batch size one')
        role_ids = {name: index for index, name in enumerate(role_names)}
        memory = self.encode(batch)
        cursor = grammar.cursor()
        cursor.advance(self.config.bos_token_id)
        initial = _TypedBeam((self.config.bos_token_id,), cursor, 0.0, None, 0, 0, 0)
        active = [initial]
        completed: list[_TypedBeam] = []
        bounded: list[_TypedBeam] = []
        maximum_length = min(self.config.max_target_length, limit + 1)
        argument_start = grammar.tokens['ARGUMENT_START']
        command_end = grammar.tokens['COMMAND_END']
        redirect_start = grammar.tokens['REDIRECT_START']

        while active:
            if completed and completed[0].score >= active[0].score:
                break
            expanded: dict[tuple[tuple[int, ...], int | None, int, int, int], _TypedBeam] = {}
            for hypothesis in active:
                hidden = self.decode_hidden(
                    torch.tensor([hypothesis.actions], dtype=torch.long, device=batch.source_ids.device),
                    torch.ones((1, len(hypothesis.actions)), dtype=torch.bool, device=batch.source_ids.device),
                    memory,
                    batch.source_attention_mask,
                )
                output = self.distributions(batch, hidden[:, -1:], memory)
                state = grammar.states[hypothesis.cursor.state]
                produced = False
                if post_command_counts and state.name == 'command_body' and hypothesis.desired_arguments is None:
                    count_scores = output.argument_count_logits[0, -1].log_softmax(dim=-1)
                    if forced_argument_counts is not None:
                        if hypothesis.command_index == 0 or hypothesis.command_index > len(forced_argument_counts):
                            count_choices = torch.empty(0, dtype=torch.long, device=batch.source_ids.device)
                        else:
                            count_choices = torch.tensor(
                                [forced_argument_counts[hypothesis.command_index - 1]],
                                dtype=torch.long,
                                device=batch.source_ids.device,
                            )
                    else:
                        count_choices = count_scores.topk(argument_count_top_k).indices
                    for count_tensor in count_choices:
                        count = int(count_tensor)
                        score = hypothesis.score
                        if forced_argument_counts is None:
                            score += float(count_scores[count])
                        _retain_typed(
                            expanded,
                            _TypedBeam(
                                hypothesis.actions,
                                _clone_cursor(hypothesis.cursor),
                                score,
                                count,
                                hypothesis.emitted_arguments,
                                hypothesis.command_index,
                                hypothesis.word_index,
                            ),
                        )
                        produced = True
                    continue
                if state.byte_payload is not None:
                    role = role_ids[state.name]
                    mask = batch.candidate_role_mask[0, role]
                    if ground_arguments and state.name == 'argument_bytes':
                        grounded = batch.candidate_features[0, :, :2].bool().any(dim=-1)
                        grounded_mask = mask & grounded
                        if bool(grounded_mask.any()):
                            mask = grounded_mask
                    candidate_logits = (
                        output.candidate_logits[0, -1]
                        if output.role_candidate_logits is None
                        else output.role_candidate_logits[0, -1, role]
                    )
                    scores = candidate_logits.masked_fill(~mask, -torch.inf).log_softmax(dim=-1)
                    if forced_words is not None:
                        if hypothesis.word_index >= len(forced_words):
                            word_choices = torch.empty(0, dtype=torch.long, device=batch.source_ids.device)
                        else:
                            expected = forced_words[hypothesis.word_index]
                            matches = [
                                index
                                for index in mask.nonzero().flatten().tolist()
                                if self.candidate_bytes(batch, index) == expected
                            ]
                            word_choices = torch.tensor(matches[:1], dtype=torch.long, device=batch.source_ids.device)
                    else:
                        word_choices = scores.topk(min(4 if state.name == 'command_name_bytes' else 8, int(mask.sum()))).indices
                    if state.name == 'command_name_bytes' and not post_command_counts:
                        count_scores = output.argument_count_logits[0, -1].log_softmax(dim=-1)
                        if forced_argument_counts is not None:
                            if hypothesis.command_index >= len(forced_argument_counts):
                                count_choices = torch.empty(0, dtype=torch.long, device=batch.source_ids.device)
                            else:
                                count_choices = torch.tensor(
                                    [forced_argument_counts[hypothesis.command_index]],
                                    dtype=torch.long,
                                    device=batch.source_ids.device,
                                )
                        else:
                            count_choices = count_scores.topk(argument_count_top_k).indices
                    else:
                        count_scores = None
                        count_choices = torch.tensor([-1], device=batch.source_ids.device)
                    for word_tensor in word_choices:
                        choice = int(word_tensor)
                        raw = self.candidate_bytes(batch, choice)
                        addition = tuple(state.byte_payload.byte_offset + byte for byte in raw) + (state.byte_payload.end_token,)
                        if not raw or len(hypothesis.actions) + len(addition) > maximum_length:
                            continue
                        for count_tensor in count_choices:
                            count = int(count_tensor)
                            next_cursor = _clone_cursor(hypothesis.cursor)
                            for value in addition:
                                next_cursor.advance(value)
                            desired = (
                                count
                                if state.name == 'command_name_bytes' and not post_command_counts
                                else hypothesis.desired_arguments
                            )
                            emitted = hypothesis.emitted_arguments + int(state.name == 'argument_bytes')
                            score = hypothesis.score
                            if forced_words is None:
                                score += float(scores[choice])
                            if count_scores is not None and forced_argument_counts is None:
                                score += float(count_scores[count])
                            item = _TypedBeam(
                                hypothesis.actions + addition,
                                next_cursor,
                                score,
                                desired,
                                emitted,
                                hypothesis.command_index + int(state.name == 'command_name_bytes'),
                                hypothesis.word_index + 1,
                            )
                            _retain_typed(expanded, item)
                            produced = True
                else:
                    allowed = list(hypothesis.cursor.allowed())
                    if state.name == 'command_body' and hypothesis.desired_arguments is not None:
                        if hypothesis.emitted_arguments < hypothesis.desired_arguments:
                            allowed = [value for value in allowed if value == argument_start]
                        else:
                            allowed = [value for value in allowed if value != argument_start]
                    if hypothesis.emitted_arguments < (hypothesis.desired_arguments or 0):
                        allowed = [value for value in allowed if value not in (command_end, redirect_start)]
                    if allowed and len(hypothesis.actions) + 1 <= maximum_length:
                        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=batch.source_ids.device)
                        scores = output.action_logits[0, -1, allowed_tensor].log_softmax(dim=-1)
                        for index, value in enumerate(allowed):
                            next_cursor = _clone_cursor(hypothesis.cursor)
                            next_cursor.advance(value)
                            desired = None if value == command_end else hypothesis.desired_arguments
                            emitted = 0 if value == command_end else hypothesis.emitted_arguments
                            item = _TypedBeam(
                                hypothesis.actions + (value,),
                                next_cursor,
                                hypothesis.score + float(scores[index]),
                                desired,
                                emitted,
                                hypothesis.command_index,
                                hypothesis.word_index,
                            )
                            _retain_typed(expanded, item)
                            produced = True
                if not produced:
                    bounded.append(hypothesis)
            active = []
            for item in expanded.values():
                if item.cursor.complete:
                    completed.append(item)
                else:
                    active.append(item)
            active.sort(
                key=lambda item: (
                    -item.score,
                    item.actions,
                    item.desired_arguments or -1,
                    item.command_index,
                    item.word_index,
                )
            )
            active = active[:beam_width]
            completed.sort(key=lambda item: (-item.score, item.actions))
            completed = completed[:beam_width]
        bounded.sort(key=lambda item: (-item.score, item.actions))
        winner = completed[0] if completed else (bounded[0] if bounded else initial)
        return [winner.actions]


def _retain_typed(
    target: dict[tuple[tuple[int, ...], int | None, int, int, int], _TypedBeam],
    candidate: _TypedBeam,
) -> None:
    key = (
        candidate.actions,
        candidate.desired_arguments,
        candidate.emitted_arguments,
        candidate.command_index,
        candidate.word_index,
    )
    previous = target.get(key)
    if previous is None or candidate.score > previous.score:
        target[key] = candidate
