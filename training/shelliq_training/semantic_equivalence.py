"""Conservative equivalence rules for reviewed SemanticDocumentV2 pairs."""

from __future__ import annotations

import re
from copy import deepcopy

_SHORT_FLAG_CLUSTER = re.compile(r'-[A-Za-z]{1,}')
_UNQUOTED_SAFE_WORD = re.compile(r'[A-Za-z0-9_./:@%+=,-]+')


def _canonical_word(source: str) -> str:
    if len(source) >= 2 and source[0] == source[-1] and source[0] in '\'"':
        inner = source[1:-1]
        if _UNQUOTED_SAFE_WORD.fullmatch(inner):
            return inner
    return source


def _canonical_arguments(arguments: object) -> object:
    if not isinstance(arguments, list):
        return arguments
    canonical: list[object] = []
    for argument in arguments:
        if not isinstance(argument, dict) or set(argument) != {'s'} or not isinstance(argument['s'], str):
            canonical.append(argument)
            continue
        source = _canonical_word(argument['s'])
        if _SHORT_FLAG_CLUSTER.fullmatch(source) and len(source) > 2:
            canonical.extend({'s': f'-{flag}'} for flag in source[1:])
        else:
            canonical.append({'s': source})
    return canonical


def canonicalize_semantic_document(document: dict[str, object]) -> dict[str, object]:
    """Normalize only shell-preserving lexical forms we can justify statically."""
    canonical = deepcopy(document)
    statements = canonical.get('s')
    if not isinstance(statements, list):
        return canonical
    for statement in statements:
        if not isinstance(statement, dict):
            continue
        commands = statement.get('c')
        if not isinstance(commands, list):
            continue
        for command in commands:
            if not isinstance(command, dict):
                continue
            if 'n' in command and isinstance(command['n'], dict) and isinstance(command['n'].get('s'), str):
                command['n']['s'] = _canonical_word(command['n']['s'])
            if 'a' in command:
                command['a'] = _canonical_arguments(command['a'])
            redirects = command.get('r')
            if isinstance(redirects, list):
                for redirect in redirects:
                    if not isinstance(redirect, dict):
                        continue
                    target = redirect.get('t')
                    if isinstance(target, dict) and isinstance(target.get('s'), str):
                        target['s'] = _canonical_word(target['s'])
    return canonical


def semantic_documents_equivalent(expected: dict[str, object], actual: dict[str, object]) -> bool:
    """Compare semantic documents after conservative lexical normalization."""
    return canonicalize_semantic_document(expected) == canonicalize_semantic_document(actual)
