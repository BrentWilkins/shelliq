"""Generic retrieval and binding of Rust-typed documentation templates."""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from shelliq_training.data import Platform, SFTRecord
from shelliq_training.semantic_equivalence import canonicalize_semantic_document

_TOKEN = re.compile(r"--?[a-z0-9][a-z0-9-]*|[a-z0-9_./:+@='%{}$~-]+")
_QUOTED = re.compile(r"(['\"])(.+?)\1")
_REMOTE_PATH = re.compile(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+:[^\s,;]+')
_URL = re.compile(r'(?:https?|ssh|rsync)://[^\s,;]+')
_PATH = re.compile(r'(?:\.{0,2}/|/|~\/)[^\s,;]+|[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8}')
_INTEGER = re.compile(r'(?<![A-Za-z0-9_.-])\d+(?![A-Za-z0-9_-])')
_PLACEHOLDER_PART = re.compile(
    r'(?:^|[_/.-])(?:'
    r'archive|command|count|date|directory|domain|extension|field|file|filename|'
    r'group|host|hostname|id|key|line|name|number|package|path|pattern|pid|port|'
    r'query|regex|repository|scope|server|string|team|text|time|url|user|username|'
    r'value|width'
    r')(?:$|[_/.-])',
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DocumentationTemplate:
    record_id: str
    command: str
    platform: Platform
    instruction: str
    context: str
    document: dict[str, object]


@dataclass(frozen=True, slots=True)
class RetrievedTemplate:
    template: DocumentationTemplate
    score: float


@dataclass(frozen=True, slots=True)
class BoundTemplate:
    template: DocumentationTemplate
    score: float
    document: dict[str, object]
    bindings: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ComposedTemplate:
    source_record_ids: tuple[str, ...]
    score: float
    document: dict[str, object]
    bindings: tuple[tuple[str, str], ...]


def documentation_templates(
    records: Iterable[SFTRecord], *, allowed_sources: frozenset[str] = frozenset({'tldr-pages'})
) -> list[DocumentationTemplate]:
    """Select independently authored, already-converted semantic documents."""
    templates: list[DocumentationTemplate] = []
    for record in records:
        if record.source not in allowed_sources:
            continue
        document = json.loads(record.response)
        if not isinstance(document, dict):
            raise ValueError(f'{record.record_id}: semantic target is not an object')
        templates.append(
            DocumentationTemplate(
                record_id=record.record_id,
                command=record.command,
                platform=record.platform,
                instruction=record.instruction,
                context=record.context,
                document=document,
            )
        )
    return templates


class DocumentationTemplateIndex:
    """Deterministic TF-IDF retrieval scoped to command and platform."""

    def __init__(self, templates: Sequence[DocumentationTemplate]) -> None:
        self.templates = tuple(templates)
        groups: dict[tuple[str, Platform], list[int]] = defaultdict(list)
        token_rows: list[Counter[str]] = []
        document_frequency: Counter[str] = Counter()
        for index, template in enumerate(self.templates):
            groups[(template.command, template.platform)].append(index)
            tokens = _tokenize(f'{template.instruction}\n{template.context}')
            token_rows.append(tokens)
            document_frequency.update(tokens)
        self._groups = {key: tuple(value) for key, value in groups.items()}
        count = len(self.templates)
        self._idf = {token: math.log((1 + count) / (1 + frequency)) + 1.0 for token, frequency in document_frequency.items()}
        self._vectors = tuple(_vectorize(tokens, self._idf) for tokens in token_rows)

    def rank(
        self,
        *,
        command: str,
        platform: Platform,
        instruction: str,
        context: str,
        limit: int = 8,
    ) -> list[RetrievedTemplate]:
        if limit <= 0:
            raise ValueError('retrieval limit must be positive')
        query = _vectorize(_tokenize(f'{instruction}\n{context}'), self._idf)
        ranked = [
            RetrievedTemplate(self.templates[index], _cosine(query, self._vectors[index]))
            for index in self._groups.get((command, platform), ())
        ]
        ranked.sort(key=lambda item: (-item.score, item.template.record_id))
        return ranked[:limit]


def bind_template(retrieved: RetrievedTemplate, instruction: str) -> BoundTemplate:
    """Bind generic TLDR slots to compatible literals explicitly in a request."""
    document, bindings = _bind_document(retrieved.template.document, instruction)
    return BoundTemplate(
        template=retrieved.template,
        score=retrieved.score,
        document=document,
        bindings=bindings,
    )


def compose_template_candidates(
    retrieved: Sequence[RetrievedTemplate],
    instruction: str,
    *,
    limit: int = 64,
    maximum_sources: int = 3,
) -> list[ComposedTemplate]:
    """Compose bounded typed candidates from multiple same-command examples."""
    if limit <= 0:
        raise ValueError('composition limit must be positive')
    if maximum_sources <= 0:
        raise ValueError('maximum_sources must be positive')
    states: dict[str, tuple[dict[str, object], tuple[str, ...], float]] = {}
    frontier: list[tuple[dict[str, object], tuple[str, ...], float]] = []
    for item in retrieved:
        document = copy.deepcopy(item.template.document)
        source_ids = (item.template.record_id,)
        state = (document, source_ids, item.score)
        key = _document_key(document)
        if key not in states or item.score > states[key][2]:
            states[key] = state
            frontier.append(state)

    for _ in range(1, maximum_sources):
        expanded: list[tuple[dict[str, object], tuple[str, ...], float]] = []
        for document, source_ids, score in frontier:
            for item in retrieved:
                if item.template.record_id in source_ids:
                    continue
                merged = _merge_simple_command_documents(document, item.template.document)
                if merged is None:
                    continue
                merged_ids = (*source_ids, item.template.record_id)
                merged_score = (score * len(source_ids) + item.score) / len(merged_ids)
                key = _document_key(merged)
                previous = states.get(key)
                if previous is None or merged_score > previous[2]:
                    state = (merged, merged_ids, merged_score)
                    states[key] = state
                    expanded.append(state)
        frontier = expanded
        if not frontier:
            break

    candidates: list[ComposedTemplate] = []
    for document, source_ids, score in states.values():
        bound, bindings = _bind_document(document, instruction)
        candidates.append(
            ComposedTemplate(
                source_record_ids=source_ids,
                score=score,
                document=bound,
                bindings=bindings,
            )
        )
    candidates.sort(
        key=lambda item: (
            -item.score,
            len(item.source_record_ids),
            item.source_record_ids,
            _document_key(item.document),
        )
    )
    return candidates[:limit]


def _bind_document(document: Mapping[str, object], instruction: str) -> tuple[dict[str, object], tuple[tuple[str, str], ...]]:
    bound = copy.deepcopy(dict(document))
    available = _request_literals(instruction)
    assigned: dict[str, str] = {}
    used: set[int] = set()

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        if set(value) == {'s'} and isinstance(value['s'], str):
            placeholder = value['s']
            if not is_placeholder(placeholder):
                return
            if placeholder not in assigned:
                kind = placeholder_kind(placeholder)
                for index, (candidate_kind, candidate) in enumerate(available):
                    if index not in used and _compatible(kind, candidate_kind):
                        assigned[placeholder] = candidate
                        used.add(index)
                        break
            if placeholder in assigned:
                value['s'] = assigned[placeholder]
            return
        for item in value.values():
            visit(item)

    visit(bound)
    return bound, tuple(sorted(assigned.items()))


def is_placeholder(value: str) -> bool:
    lowered = value.strip('\'"').lower()
    if not lowered or lowered.startswith('-'):
        return False
    if '{{' in lowered or '}}' in lowered or 'path/to/' in lowered:
        return True
    if lowered in {'command', 'file', 'host', 'name', 'number', 'path', 'pattern', 'string', 'text'}:
        return True
    return bool(_PLACEHOLDER_PART.search(lowered))


def placeholder_kind(value: str) -> str:
    lowered = value.strip('\'"').lower()
    if '://' in lowered or 'url' in lowered:
        return 'url'
    if '@' in lowered or 'host' in lowered or 'server' in lowered or 'domain' in lowered:
        return 'remote'
    if any(part in lowered for part in ('path', 'file', 'directory', 'archive', 'repository')):
        return 'path'
    if any(part in lowered for part in ('number', 'count', 'width', 'port', 'pid', 'id')):
        return 'integer'
    return 'text'


def template_matches_document(template: Mapping[str, object], document: Mapping[str, object]) -> bool:
    """Compare typed structure while treating documented placeholders as slots."""
    expected = canonicalize_semantic_document(dict(template))
    actual = canonicalize_semantic_document(dict(document))

    def matches(left: object, right: object) -> bool:
        if isinstance(left, dict) and set(left) == {'s'} and isinstance(left['s'], str) and is_placeholder(left['s']):
            return isinstance(right, dict) and set(right) == {'s'} and isinstance(right['s'], str) and bool(right['s'])
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(matches(left[key], right[key]) for key in left)
        if isinstance(left, list):
            return len(left) == len(right) and all(matches(a, b) for a, b in zip(left, right, strict=True))
        return left == right

    return matches(expected, actual)


def _tokenize(value: str) -> Counter[str]:
    return Counter(_TOKEN.findall(value.lower()))


def _vectorize(tokens: Counter[str], inverse_document_frequency: Mapping[str, float]) -> dict[str, float]:
    values = {
        token: (1.0 + math.log(count)) * inverse_document_frequency[token]
        for token, count in tokens.items()
        if token in inverse_document_frequency
    }
    norm = math.sqrt(sum(value * value for value in values.values()))
    return {token: value / norm for token, value in values.items()} if norm else {}


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def _request_literals(instruction: str) -> list[tuple[str, str]]:
    found: list[tuple[int, str, str]] = []
    occupied: list[tuple[int, int]] = []

    def add(pattern: re.Pattern[str], kind: str, group: int = 0) -> None:
        for match in pattern.finditer(instruction):
            start, end = match.span(group)
            if any(start < previous_end and end > previous_start for previous_start, previous_end in occupied):
                continue
            value = match.group(group).rstrip('.')
            found.append((start, kind, value))
            occupied.append((start, end))

    add(_QUOTED, 'text', 2)
    add(_URL, 'url')
    add(_REMOTE_PATH, 'remote')
    add(_PATH, 'path')
    add(_INTEGER, 'integer')
    found.sort(key=lambda item: item[0])
    return [(kind, value) for _, kind, value in found]


def _compatible(slot_kind: str, candidate_kind: str) -> bool:
    if slot_kind == candidate_kind:
        return True
    return slot_kind == 'text' or (slot_kind == 'path' and candidate_kind == 'remote')


def _merge_simple_command_documents(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, object] | None:
    left_command = _simple_command(left)
    right_command = _simple_command(right)
    if left_command is None or right_command is None:
        return None
    left_name, left_arguments = left_command
    right_name, right_arguments = right_command
    if left_name != right_name:
        return None
    merged = copy.deepcopy(dict(left))
    command = merged['s'][0]['c'][0]
    arguments = _shortest_common_supersequence(left_arguments, right_arguments)
    if arguments:
        command['a'] = [{'s': value} for value in arguments]
    else:
        command.pop('a', None)
    return merged


def _simple_command(document: Mapping[str, object]) -> tuple[str, tuple[str, ...]] | None:
    statements = document.get('s')
    if not isinstance(statements, list) or len(statements) != 1 or not isinstance(statements[0], dict):
        return None
    statement = statements[0]
    commands = statement.get('c')
    if statement.get('t') != 'p' or not isinstance(commands, list) or len(commands) != 1:
        return None
    command = commands[0]
    if not isinstance(command, dict) or set(command) - {'n', 'a'}:
        return None
    name = command.get('n')
    if not isinstance(name, dict) or set(name) != {'s'} or not isinstance(name['s'], str):
        return None
    raw_arguments = command.get('a', [])
    if not isinstance(raw_arguments, list):
        return None
    arguments: list[str] = []
    for argument in raw_arguments:
        if not isinstance(argument, dict) or set(argument) != {'s'} or not isinstance(argument['s'], str):
            return None
        arguments.append(argument['s'])
    return name['s'], tuple(arguments)


def _shortest_common_supersequence(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    lengths = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for left_index, left_value in enumerate(left, start=1):
        for right_index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                lengths[left_index][right_index] = lengths[left_index - 1][right_index - 1] + 1
            else:
                lengths[left_index][right_index] = max(lengths[left_index - 1][right_index], lengths[left_index][right_index - 1])
    common: list[str] = []
    left_index, right_index = len(left), len(right)
    while left_index and right_index:
        if left[left_index - 1] == right[right_index - 1]:
            common.append(left[left_index - 1])
            left_index -= 1
            right_index -= 1
        elif lengths[left_index - 1][right_index] >= lengths[left_index][right_index - 1]:
            left_index -= 1
        else:
            right_index -= 1
    common.reverse()

    merged: list[str] = []
    left_index = right_index = 0
    for shared in common:
        while left[left_index] != shared:
            merged.append(left[left_index])
            left_index += 1
        while right[right_index] != shared:
            merged.append(right[right_index])
            right_index += 1
        merged.append(shared)
        left_index += 1
        right_index += 1
    merged.extend(left[left_index:])
    merged.extend(right[right_index:])
    return tuple(merged)


def _document_key(document: Mapping[str, object]) -> str:
    return json.dumps(document, separators=(',', ':'), sort_keys=True)
