"""Source-copy supervision for semantic actions, derived from the Rust grammar."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from shelliq_training.semantic_actions import ActionGrammar

COPY_IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class CopyAlignment:
    labels: tuple[int, ...]
    roles: tuple[str | None, ...]
    word_counts: dict[str, tuple[int, int]]
    byte_counts: dict[str, tuple[int, int]]


def align_action_bytes_to_source(source: str, actions: Sequence[int], grammar: ActionGrammar) -> CopyAlignment:
    """Align complete action words to their first exact UTF-8 source span."""
    source_bytes = source.encode('utf-8')
    labels = [COPY_IGNORE_INDEX] * len(actions)
    roles: list[str | None] = [None] * len(actions)
    word_total: Counter[str] = Counter()
    word_copyable: Counter[str] = Counter()
    byte_total: Counter[str] = Counter()
    byte_copyable: Counter[str] = Counter()
    cursor = grammar.cursor()
    position = 0
    while position < len(actions):
        state = grammar.states[cursor.state]
        payload = state.byte_payload
        if payload is None:
            cursor.advance(actions[position])
            position += 1
            continue
        begin = position
        raw = bytearray()
        while position < len(actions) and actions[position] != payload.end_token:
            token = actions[position]
            if not payload.byte_offset <= token < payload.byte_offset + payload.byte_count:
                raise ValueError(f'non-byte action {token} inside {state.name}')
            raw.append(token - payload.byte_offset)
            roles[position] = state.name
            cursor.advance(token)
            position += 1
        if position == len(actions):
            raise ValueError(f'incomplete action word in {state.name}')
        raw.decode('utf-8')
        word_total[state.name] += 1
        byte_total[state.name] += len(raw)
        source_position = source_bytes.find(raw)
        if source_position >= 0:
            word_copyable[state.name] += 1
            byte_copyable[state.name] += len(raw)
            for offset in range(len(raw)):
                labels[begin + offset] = source_position + offset
        cursor.advance(actions[position])
        position += 1
    if not cursor.complete:
        raise ValueError('action sequence did not complete grammar during copy alignment')
    role_names = sorted(word_total)
    return CopyAlignment(
        tuple(labels),
        tuple(roles),
        {role: (word_copyable[role], word_total[role]) for role in role_names},
        {role: (byte_copyable[role], byte_total[role]) for role in role_names},
    )


def merge_copy_counts(alignments: Sequence[CopyAlignment], field: str) -> dict[str, dict[str, int | float]]:
    """Aggregate role counts into stable report rows."""
    copied: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for alignment in alignments:
        values = alignment.word_counts if field == 'words' else alignment.byte_counts
        for role, (role_copied, role_total) in values.items():
            copied[role] += role_copied
            total[role] += role_total
    copied['all'] = sum(value for role, value in copied.items() if role != 'all')
    total['all'] = sum(value for role, value in total.items() if role != 'all')
    return {
        role: {
            'copyable': copied[role],
            'total': total[role],
            'rate': copied[role] / total[role] if total[role] else 0.0,
        }
        for role in sorted(total)
    }
