"""Rust-owned semantic actions exposed to Python without duplicating the grammar."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from shelliq_training.data import Corpus, SequenceTooLongError, SFTRecord, format_user_message
from shelliq_training.prompt import PromptContract


@dataclass(frozen=True, slots=True)
class ActionExample:
    record_id: str
    corpus: Corpus
    source_ids: tuple[int, ...]
    action_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ActionDecode:
    valid: bool
    document: dict[str, object] | None
    rendered: str | None
    error: str | None


class SemanticActionClient:
    """Batch interface to the authoritative Rust codec."""

    def __init__(self, executable: str | Path) -> None:
        self.executable = Path(executable)

    def manifest(self) -> dict[str, object]:
        completed = subprocess.run(
            [str(self.executable), 'manifest'],
            text=True,
            capture_output=True,
            check=True,
        )
        raw = json.loads(completed.stdout)
        if not isinstance(raw, dict):
            raise ValueError('semantic action manifest is not an object')
        return raw

    def encode(self, documents: Sequence[Mapping[str, object]]) -> list[tuple[int, ...]]:
        rows = self._run('encode', [{'document': document} for document in documents])
        encoded: list[tuple[int, ...]] = []
        for index, row in enumerate(rows, start=1):
            if not row.get('valid'):
                raise ValueError(f'action encode failed at row {index}: {row.get("error")}')
            tokens = row.get('tokens')
            if not isinstance(tokens, list) or any(not isinstance(token, int) for token in tokens):
                raise ValueError(f'invalid action tokens at row {index}')
            encoded.append(tuple(tokens))
        return encoded

    def decode(self, sequences: Sequence[Sequence[int]]) -> list[ActionDecode]:
        rows = self._run('decode', [{'tokens': list(sequence)} for sequence in sequences])
        decoded: list[ActionDecode] = []
        for index, row in enumerate(rows, start=1):
            document = row.get('document')
            rendered = row.get('rendered')
            error = row.get('error')
            if document is not None and not isinstance(document, dict):
                raise ValueError(f'invalid decoded document at row {index}')
            if rendered is not None and not isinstance(rendered, str):
                raise ValueError(f'invalid decoded render at row {index}')
            if error is not None and not isinstance(error, str):
                raise ValueError(f'invalid decoded error at row {index}')
            decoded.append(ActionDecode(bool(row.get('valid')), document, rendered, error))
        return decoded

    def _run(self, mode: str, requests: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        if not requests:
            return []
        completed = subprocess.run(
            [str(self.executable), mode],
            input=''.join(json.dumps(request, separators=(',', ':')) + '\n' for request in requests),
            text=True,
            capture_output=True,
            check=True,
        )
        rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
        if len(rows) != len(requests):
            raise ValueError(f'semantic action codec returned {len(rows)} rows for {len(requests)} requests')
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or row.get('line') != index or not isinstance(row.get('valid'), bool):
                raise ValueError(f'invalid semantic action response at row {index}')
        return rows


@dataclass(slots=True)
class GrammarCursor:
    """One incremental cursor driven entirely by a Rust-emitted manifest."""

    grammar: ActionGrammar
    state: int
    payload: bytes = b''

    def allowed(self) -> tuple[int, ...]:
        spec = self.grammar.states[self.state]
        allowed = list(spec.fixed)
        if spec.byte_payload is not None:
            payload = spec.byte_payload
            for byte in range(payload.byte_count):
                candidate = self.payload + bytes([byte])
                if _valid_utf8_prefix(candidate):
                    allowed.append(payload.byte_offset + byte)
            if self.payload and _complete_utf8(self.payload):
                allowed.append(payload.end_token)
        return tuple(allowed)

    def advance(self, token: int) -> None:
        spec = self.grammar.states[self.state]
        next_state = spec.fixed.get(token)
        if next_state is not None:
            self.state = next_state
            self.payload = b''
            return
        payload = spec.byte_payload
        if payload is not None:
            if payload.byte_offset <= token < payload.byte_offset + payload.byte_count:
                candidate = self.payload + bytes([token - payload.byte_offset])
                if _valid_utf8_prefix(candidate):
                    self.payload = candidate
                    return
            if token == payload.end_token and self.payload and _complete_utf8(self.payload):
                self.state = payload.next_state
                self.payload = b''
                return
        raise ValueError(f'token {token} is not allowed in action state {spec.name}')

    @property
    def complete(self) -> bool:
        return self.state == self.grammar.complete_state


@dataclass(frozen=True, slots=True)
class _BytePayload:
    byte_offset: int
    byte_count: int
    end_token: int
    next_state: int


@dataclass(frozen=True, slots=True)
class _State:
    name: str
    fixed: dict[int, int]
    byte_payload: _BytePayload | None


class ActionGrammar:
    """Generic finite-state/UTF-8 interpreter for the versioned Rust manifest."""

    def __init__(self, manifest: Mapping[str, object]) -> None:
        if manifest.get('schema_version') != 1:
            raise ValueError('unsupported semantic action manifest version')
        self.vocab_size = _integer(manifest, 'vocab_size')
        self.start_state = _integer(manifest, 'start_state')
        self.complete_state = _integer(manifest, 'complete_state')
        raw_tokens = manifest.get('tokens')
        raw_states = manifest.get('states')
        if not isinstance(raw_tokens, list) or not isinstance(raw_states, list):
            raise ValueError('semantic action manifest is missing tokens or states')
        self.tokens: dict[str, int] = {}
        for item in raw_tokens:
            if not isinstance(item, Mapping) or not isinstance(item.get('name'), str):
                raise ValueError('invalid named action token')
            self.tokens[item['name']] = _integer(item, 'id')
        states: dict[int, _State] = {}
        for item in raw_states:
            if not isinstance(item, Mapping) or not isinstance(item.get('name'), str):
                raise ValueError('invalid action state')
            state_id = _integer(item, 'id')
            raw_fixed = item.get('fixed')
            if not isinstance(raw_fixed, list):
                raise ValueError('invalid fixed action transitions')
            fixed: dict[int, int] = {}
            for edge in raw_fixed:
                if not isinstance(edge, Mapping):
                    raise ValueError('invalid action transition')
                fixed[_integer(edge, 'token')] = _integer(edge, 'next_state')
            raw_payload = item.get('byte_payload')
            payload = None
            if raw_payload is not None:
                if not isinstance(raw_payload, Mapping):
                    raise ValueError('invalid byte payload state')
                if raw_payload.get('require_nonempty') is not True or raw_payload.get('require_valid_utf8') is not True:
                    raise ValueError('unsupported byte payload policy')
                payload = _BytePayload(
                    _integer(raw_payload, 'byte_offset'),
                    _integer(raw_payload, 'byte_count'),
                    _integer(raw_payload, 'end_token'),
                    _integer(raw_payload, 'next_state'),
                )
            states[state_id] = _State(item['name'], fixed, payload)
        if set(states) != set(range(len(states))):
            raise ValueError('action state IDs must be contiguous')
        self.states = states
        for state in states.values():
            for token, next_state in state.fixed.items():
                if not 0 <= token < self.vocab_size or next_state not in states:
                    raise ValueError('action transition is outside manifest bounds')

    def cursor(self) -> GrammarCursor:
        return GrammarCursor(self, self.start_state)


def action_examples(
    records: Sequence[SFTRecord],
    actions: Sequence[Sequence[int]],
    tokenizer,
    *,
    source_length: int,
    target_length: int,
    prompt_contract: PromptContract,
) -> list[ActionExample]:
    """Build aligned examples while rejecting all implicit truncation."""
    if len(records) != len(actions):
        raise ValueError('records and semantic actions are not aligned')
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('source tokenizer requires pad and EOS tokens')
    examples: list[ActionExample] = []
    for record, target in zip(records, actions, strict=True):
        source = tuple(
            int(token)
            for token in tokenizer.encode(
                format_user_message(record, prompt_contract=prompt_contract),
                add_special_tokens=False,
            )
        ) + (int(tokenizer.eos_token_id),)
        action_ids = tuple(int(token) for token in target)
        if len(source) > source_length:
            raise SequenceTooLongError(
                f'{record.record_id}: source has {len(source)} tokens, exceeding source_length={source_length}'
            )
        if len(action_ids) > target_length:
            raise SequenceTooLongError(
                f'{record.record_id}: target has {len(action_ids)} actions, exceeding target_length={target_length}'
            )
        examples.append(ActionExample(record.record_id, record.corpus, source, action_ids))
    return examples


def _integer(mapping: Mapping[str, object], field: str) -> int:
    value = mapping.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f'{field} must be an integer')
    return value


def _complete_utf8(value: bytes) -> bool:
    try:
        value.decode('utf-8')
    except UnicodeDecodeError:
        return False
    return True


def _valid_utf8_prefix(value: bytes) -> bool:
    try:
        value.decode('utf-8')
    except UnicodeDecodeError as error:
        return error.reason == 'unexpected end of data' and error.end == len(value)
    return True
