"""Generic retrieval and binding of Rust-typed documentation templates."""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from shelliq_training.data import Platform, SFTRecord
from shelliq_training.semantic_equivalence import canonicalize_semantic_document

_TOKEN = re.compile(r"--?[a-z0-9][a-z0-9-]*|[a-z0-9_./:+@='%{}$~-]+")
_QUOTED = re.compile(r"(['\"])(.+?)\1")
_REMOTE_PATH = re.compile(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+:[^\s,;]+')
_URL = re.compile(r'(?:https?|ssh|rsync)://[^\s,;]+')
_PATH = re.compile(r'(?:\.{0,2}/|/|~\/)[^\s,;]+|[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8}')
_INTEGER = re.compile(r'(?<![A-Za-z0-9_.-])\d+(?![A-Za-z0-9_-])')
_OPTION = re.compile(r'(?<![A-Za-z0-9_])--?[A-Za-z0-9][A-Za-z0-9-]*(?:=[A-Za-z0-9_./:@%+,-]+)?')
_VERSION_SUFFIX = re.compile(r'\d+(?:\.\d+)*$')
_PLACEHOLDER_PART = re.compile(
    r'(?:^|[_/.-])(?:'
    r'archive|command|count|date|directory|domain|extension|field|file|filename|'
    r'group|host|hostname|id|key|line|name|number|package|path|pattern|pid|port|'
    r'query|regex|repository|scope|server|string|team|text|time|url|user|username|'
    r'value|width'
    r')(?:$|[_/.-])',
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    {
        'a',
        'an',
        'and',
        'as',
        'at',
        'by',
        'for',
        'from',
        'in',
        'is',
        'it',
        'of',
        'on',
        'or',
        'the',
        'to',
        'with',
    }
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


@dataclass(frozen=True, slots=True)
class WordProvenance:
    word: str
    source: str


@dataclass(frozen=True, slots=True)
class TemplateCompilation:
    status: str
    document: dict[str, object] | None
    source_record_ids: tuple[str, ...]
    selected_options: tuple[str, ...]
    bindings: tuple[tuple[str, str], ...]
    unresolved_slots: tuple[str, ...]
    word_provenance: tuple[WordProvenance, ...] = ()


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
        if not _document_contains_command(document, record.command):
            continue
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
        instruction_rows: list[Counter[str]] = []
        document_frequency: Counter[str] = Counter()
        for index, template in enumerate(self.templates):
            groups[(template.command, template.platform)].append(index)
            tokens = _tokenize(f'{template.instruction}\n{template.context}')
            token_rows.append(tokens)
            instruction_rows.append(_tokenize(template.instruction))
            document_frequency.update(tokens)
        self._groups = {key: tuple(value) for key, value in groups.items()}
        count = len(self.templates)
        self._idf = {token: math.log((1 + count) / (1 + frequency)) + 1.0 for token, frequency in document_frequency.items()}
        self._vectors = tuple(_vectorize(tokens, self._idf) for tokens in token_rows)
        self._instruction_vectors = tuple(_vectorize(tokens, self._idf) for tokens in instruction_rows)
        self._template_indices = {template.record_id: index for index, template in enumerate(self.templates)}

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
        indices = self._groups.get((command, platform), ())
        alias = False
        if not indices:
            normalized = _VERSION_SUFFIX.sub('', command)
            if normalized and normalized != command:
                indices = self._groups.get((normalized, platform), ())
                alias = bool(indices)
        ranked = [
            RetrievedTemplate(
                _instantiate_command(self.templates[index], command) if alias else self.templates[index],
                _cosine(query, self._vectors[index]),
            )
            for index in indices
        ]
        ranked.sort(key=lambda item: (-item.score, item.template.record_id))
        return ranked[:limit]

    def instruction_similarity(self, instruction: str, record_id: str) -> float:
        index = self._template_indices.get(record_id)
        if index is None:
            return 0.0
        query = _vectorize(_tokenize(instruction), self._idf)
        documented = self.templates[index].instruction.strip().lower().rstrip('.')
        requested = instruction.strip().lower()
        prefix_bonus = 1.0 if documented and requested.startswith(documented) else 0.0
        return prefix_bonus + _cosine(query, self._instruction_vectors[index])


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


def compose_contextual_candidates(
    retrieved: Sequence[RetrievedTemplate],
    instruction: str,
    context: str,
    *,
    limit: int = 256,
    maximum_context_options: int = 8,
) -> list[ComposedTemplate]:
    """Overlay bounded prompt-documented option subsets onto typed templates."""
    if not retrieved:
        return []
    options = documented_context_options(context)[:maximum_context_options]
    option_sequences = [
        tuple(options[index] for index in indices)
        for size in range(1, min(len(options), maximum_context_options) + 1)
        for indices in combinations(range(len(options)), size)
    ]
    states: dict[str, tuple[dict[str, object], tuple[str, ...], float, tuple[tuple[str, str], ...]]] = {}

    def add(
        document: dict[str, object],
        source_ids: tuple[str, ...],
        score: float,
        bindings: tuple[tuple[str, str], ...],
    ) -> None:
        key = _document_key(document)
        previous = states.get(key)
        if previous is None or score > previous[2]:
            states[key] = (document, source_ids, score, bindings)

    for base in compose_template_candidates(retrieved, instruction, limit=64, maximum_sources=3):
        add(base.document, base.source_record_ids, base.score, base.bindings)
    for retrieved_item in retrieved:
        base = retrieved_item.template.document
        command = _argument_command(base)
        if command is None:
            continue
        _, base_arguments = command
        for option_sequence in option_sequences:
            merged_arguments = _overlay_context_options(base_arguments, option_sequence)
            merged = _replace_arguments(base, merged_arguments)
            source_ids = (retrieved_item.template.record_id, f'context:{"|".join(option_sequence)}')
            bound, bindings = _bind_document(merged, instruction)
            add(bound, source_ids, retrieved_item.score, bindings)

    candidates = [
        ComposedTemplate(source_ids, score, document, bindings) for document, source_ids, score, bindings in states.values()
    ]
    candidates.sort(
        key=lambda item: (
            -item.score,
            len(item.source_record_ids),
            item.source_record_ids,
            _document_key(item.document),
        )
    )
    return candidates[:limit]


def documented_context_options(context: str) -> tuple[str, ...]:
    """Extract stable option spellings from authoritative documentation text."""
    return tuple(dict.fromkeys(match.group() for match in _OPTION.finditer(context)))


def compile_documented_command(
    index: DocumentationTemplateIndex,
    *,
    command: str,
    platform: Platform,
    instruction: str,
    context: str,
) -> TemplateCompilation:
    """Select one target-blind template or abstain when literal slots remain."""
    retrieved = index.rank(
        command=command,
        platform=platform,
        instruction=instruction,
        context=context,
        limit=8,
    )
    if not retrieved:
        return TemplateCompilation('no_documentation', None, (), (), (), ())
    selected_options = select_documented_options(instruction, context)
    base = retrieved[0]
    command_shape = _argument_command(base.template.document)
    document = copy.deepcopy(base.template.document)
    if command_shape is not None and selected_options:
        _, arguments = command_shape
        document = _replace_arguments(document, _overlay_context_options(arguments, selected_options))
    bound, bindings = _bind_document(document, instruction)
    unresolved = unresolved_placeholders(bound)
    return TemplateCompilation(
        status='needs_input' if unresolved else 'ready',
        document=bound,
        source_record_ids=(base.template.record_id,),
        selected_options=selected_options,
        bindings=bindings,
        unresolved_slots=unresolved,
    )


def compile_documented_command_scored(
    index: DocumentationTemplateIndex,
    *,
    command: str,
    platform: Platform,
    instruction: str,
    context: str,
) -> TemplateCompilation:
    """Score whole candidates and conservatively audit every emitted operand."""
    retrieved = index.rank(
        command=command,
        platform=platform,
        instruction=instruction,
        context=context,
        limit=8,
    )
    if not retrieved:
        return TemplateCompilation('no_documentation', None, (), (), (), ())
    selected_options = select_documented_options(instruction, context)
    required_literals = tuple(value for _, value in _request_literals(instruction))
    candidates = compose_contextual_candidates(
        retrieved,
        instruction,
        context,
        limit=512,
        maximum_context_options=6,
    )
    if not candidates:
        return TemplateCompilation('no_documentation', None, (), selected_options, (), ())

    ranked: list[tuple[tuple[object, ...], ComposedTemplate, tuple[str, ...]]] = []
    for candidate in candidates:
        unsupported = _unsupported_operands(candidate, instruction)
        arguments = _argument_command(candidate.document)
        candidate_options = tuple(value for value in (arguments[1] if arguments else ()) if value.startswith('-'))
        candidate_words = {node['s'].strip('"\'') for node in _word_nodes(candidate.document)}
        missing_options = sum(option.startswith('-') and option not in candidate_options for option in selected_options)
        missing_literals = tuple(value for value in required_literals if value.strip('"\'') not in candidate_words)
        extra_options = sum(option not in selected_options for option in candidate_options) if selected_options else 0
        source_similarity = max(
            (
                index.instruction_similarity(instruction, source)
                for source in candidate.source_record_ids
                if not source.startswith('context:')
            ),
            default=0.0,
        )
        rank_key: tuple[object, ...] = (
            missing_options,
            len(missing_literals),
            len(candidate.source_record_ids),
            -source_similarity,
            len(unsupported),
            extra_options,
            -candidate.score,
            _document_key(candidate.document),
        )
        ranked.append((rank_key, candidate, (*missing_literals, *unsupported)))
    ranked.sort(key=lambda item: item[0])
    _, selected, unsupported = ranked[0]
    unresolved = tuple(dict.fromkeys((*unresolved_placeholders(selected.document), *unsupported)))
    return TemplateCompilation(
        status='needs_input' if unresolved else 'ready',
        document=selected.document,
        source_record_ids=selected.source_record_ids,
        selected_options=selected_options,
        bindings=selected.bindings,
        unresolved_slots=unresolved,
        word_provenance=_candidate_word_provenance(selected, instruction, selected_options),
    )


def select_documented_options(instruction: str, context: str) -> tuple[str, ...]:
    """Select option fragments whose local documentation overlaps the request."""
    query = _content_tokens(instruction)
    selected: list[str] = []
    clauses = re.split(r'(?<=[.;])\s+|,\s*|\s+and\s+(?=--?)', context)
    for clause in clauses:
        options = list(_OPTION.finditer(clause))
        if not options:
            continue
        description = _content_tokens(_OPTION.sub(' ', clause))
        explicit = any(match.group() in instruction for match in options)
        if not explicit and not query.intersection(description):
            continue
        for match in options:
            selected.append(match.group())
            remainder = clause[match.end() :]
            value = re.match(r"\s+(\d+|'[^']*'|\"[^\"]*\")", remainder)
            if value:
                selected.append(value.group(1))
    return tuple(dict.fromkeys(selected))


def unresolved_placeholders(document: Mapping[str, object]) -> tuple[str, ...]:
    unresolved: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if set(value) == {'s'} and isinstance(value['s'], str) and is_placeholder(value['s']):
                unresolved.append(value['s'])
            else:
                for item in value.values():
                    visit(item)

    visit(document)
    return tuple(dict.fromkeys(unresolved))


def _unsupported_operands(candidate: ComposedTemplate, instruction: str) -> tuple[str, ...]:
    command = _argument_command(candidate.document)
    if command is None:
        return tuple(
            node['s']
            for node in _word_nodes(candidate.document)[1:]
            if not node['s'].startswith('-') and node['s'] not in instruction
        )
    _, arguments = command
    prefix_end = 0
    for argument in arguments:
        if argument.startswith('-') or is_placeholder(argument):
            break
        prefix_end += 1
    prefix_end = min(prefix_end, 1)
    bound_values = {value for _, value in candidate.bindings}
    context_values = {
        value
        for source in candidate.source_record_ids
        if source.startswith('context:')
        for value in source.removeprefix('context:').split('|')
    }
    unsupported: list[str] = []
    for index, argument in enumerate(arguments):
        if index < prefix_end or argument.startswith('-'):
            continue
        normalized = argument.strip('"\'')
        if argument in bound_values or argument in context_values or argument in instruction or normalized in instruction:
            continue
        unsupported.append(argument)
    argument_ids = {id(item) for item in _argument_word_nodes(candidate.document)}
    for node in _word_nodes(candidate.document):
        if id(node) in argument_ids:
            continue
        word = node['s']
        if word.startswith('-') or word in instruction or word.strip('"\'') in instruction:
            continue
        if word == command[0]:
            continue
        unsupported.append(word)
    return tuple(dict.fromkeys(unsupported))


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


def _content_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN.findall(value.lower()):
        if token.startswith('-') or token in _STOPWORDS:
            continue
        for suffix in ('ing', 'ed', 'es', 's'):
            if len(token) > len(suffix) + 2 and token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        tokens.add(token)
    return tokens


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
    command = _argument_command(document)
    if command is None:
        return None
    statements = document.get('s')
    raw_command = statements[0]['c'][0]
    if set(raw_command) - {'n', 'a'}:
        return None
    return command


def _argument_command(document: Mapping[str, object]) -> tuple[str, tuple[str, ...]] | None:
    statements = document.get('s')
    if not isinstance(statements, list) or len(statements) != 1 or not isinstance(statements[0], dict):
        return None
    statement = statements[0]
    commands = statement.get('c')
    if statement.get('t') != 'p' or not isinstance(commands, list) or len(commands) != 1:
        return None
    command = commands[0]
    if not isinstance(command, dict):
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


def _replace_arguments(document: Mapping[str, object], arguments: Sequence[str]) -> dict[str, object]:
    replaced = copy.deepcopy(dict(document))
    command = replaced['s'][0]['c'][0]
    if arguments:
        command['a'] = [{'s': value} for value in arguments]
    else:
        command.pop('a', None)
    return replaced


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


def _overlay_context_options(base_arguments: Sequence[str], options: Sequence[str]) -> tuple[str, ...]:
    prefix_end = 0
    for argument in base_arguments:
        if argument.startswith('-') or is_placeholder(argument):
            break
        prefix_end += 1
    prefix = tuple(base_arguments[:prefix_end])
    remainder = tuple(base_arguments[prefix_end:])
    return (*prefix, *_shortest_common_supersequence(options, remainder))


def _instantiate_command(template: DocumentationTemplate, command: str) -> DocumentationTemplate:
    document = copy.deepcopy(template.document)
    statements = document.get('s')
    if isinstance(statements, list):
        for statement in statements:
            if not isinstance(statement, dict) or not isinstance(statement.get('c'), list):
                continue
            for item in statement['c']:
                if isinstance(item, dict) and isinstance(item.get('n'), dict) and item['n'].get('s') == template.command:
                    item['n']['s'] = command
    return DocumentationTemplate(
        record_id=template.record_id,
        command=command,
        platform=template.platform,
        instruction=template.instruction,
        context=template.context,
        document=document,
    )


def _document_contains_command(document: Mapping[str, object], command: str) -> bool:
    normalized = _VERSION_SUFFIX.sub('', command)
    return any(
        name == command or (_VERSION_SUFFIX.sub('', name) == normalized and bool(normalized)) for name in _command_names(document)
    )


def _command_names(document: Mapping[str, object]) -> tuple[str, ...]:
    names: list[str] = []
    statements = document.get('s')
    if not isinstance(statements, list):
        return ()
    for statement in statements:
        if not isinstance(statement, dict) or not isinstance(statement.get('c'), list):
            continue
        for command in statement['c']:
            if isinstance(command, dict) and isinstance(command.get('n'), dict) and isinstance(command['n'].get('s'), str):
                names.append(command['n']['s'])
    return tuple(names)


def _candidate_word_provenance(
    candidate: ComposedTemplate, instruction: str, selected_options: Sequence[str]
) -> tuple[WordProvenance, ...]:
    bindings = {value: placeholder for placeholder, value in candidate.bindings}
    context_values = {
        value
        for source in candidate.source_record_ids
        if source.startswith('context:')
        for value in source.removeprefix('context:').split('|')
    }
    command = _argument_command(candidate.document)
    argument_nodes = _argument_word_nodes(candidate.document)
    static_node = argument_nodes[0] if argument_nodes and not argument_nodes[0]['s'].startswith('-') else None
    evidence: list[WordProvenance] = []
    command_names = set(_command_names(candidate.document))
    for node in _word_nodes(candidate.document):
        word = node['s']
        if word in command_names:
            source = 'command'
        elif word in bindings:
            source = f'request-binding:{bindings[word]}'
        elif word in instruction or word.strip('"\'') in instruction:
            source = 'request-literal'
        elif word in context_values or word in selected_options:
            source = 'context-fragment'
        elif word.startswith('-'):
            source = 'documentation-option'
        elif node is static_node and command is not None:
            source = 'static-subcommand'
        elif is_placeholder(word):
            source = 'unresolved-placeholder'
        else:
            source = 'unsupported-documentation-operand'
        evidence.append(WordProvenance(word, source))
    return tuple(evidence)


def _document_key(document: Mapping[str, object]) -> str:
    return json.dumps(document, separators=(',', ':'), sort_keys=True)


def _word_nodes(document: Mapping[str, object]) -> list[dict[str, str]]:
    nodes: list[dict[str, str]] = []

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if set(value) == {'s'} and isinstance(value['s'], str):
                nodes.append(value)
            else:
                for item in value.values():
                    visit(item)

    visit(document)
    return nodes


def _argument_word_nodes(document: Mapping[str, object]) -> list[dict[str, str]]:
    statements = document.get('s')
    if not isinstance(statements, list) or len(statements) != 1 or not isinstance(statements[0], dict):
        return []
    commands = statements[0].get('c')
    if not isinstance(commands, list) or len(commands) != 1 or not isinstance(commands[0], dict):
        return []
    arguments = commands[0].get('a', [])
    if not isinstance(arguments, list):
        return []
    return [
        argument
        for argument in arguments
        if isinstance(argument, dict) and set(argument) == {'s'} and isinstance(argument['s'], str)
    ]
