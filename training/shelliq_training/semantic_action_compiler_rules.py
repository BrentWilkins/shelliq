"""Small context-grounded compiler rules for stable command families."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_SHORT_FLAG = re.compile(r'(?<![\w-])-([A-Za-z])(?![A-Za-z])')
_DECIMAL = re.compile(r'(?<!\d)\d+(?!\d)')


@dataclass(frozen=True, slots=True)
class RuleCompilation:
    rule_id: str
    document: dict[str, Any]


def compile_semantic_rule(
    *,
    command: str,
    platform: str,
    instruction: str,
    context: str,
) -> RuleCompilation | None:
    """Compile a supported rule using only prompt-visible fields."""
    del platform
    if command == 'ss':
        return _compile_ss(instruction, context)
    if command == 'pmap':
        return _compile_pmap(instruction, context)
    return None


def _compile_ss(instruction: str, context: str) -> RuleCompilation | None:
    lowered_instruction = instruction.lower()
    lowered_context = context.lower()
    if not (re.search(r'(?:^|\n)(?:iproute2\s+)?ss:', lowered_context) or re.search(r'\bss\s+-s\b', lowered_context)):
        return None
    if (
        re.search(r'\b(summary|summarize|totals?)\b', lowered_instruction)
        or ('count' in lowered_instruction and 'state' in lowered_instruction)
    ) and re.search(r'\bss\s+-s\b', lowered_context):
        return RuleCompilation('ss-summary', _document('ss', ('-s',)))

    if 'specific port' in lowered_instruction or _DECIMAL.search(instruction):
        return None
    requested: set[str] = set()
    listening = bool(re.search(r'\blisten(?:ing)?\b', lowered_instruction))
    if 'tcp' in lowered_instruction:
        requested.add('t')
    if 'udp' in lowered_instruction:
        requested.add('u')
    if listening:
        requested.add('l')
    if re.search(r'\b(all|every)\b', lowered_instruction) and not listening:
        requested.add('a')
    if re.search(r'\b(process|processes|owning|owner)\b', lowered_instruction):
        requested.add('p')
    established = 'established' in lowered_instruction
    if established:
        requested.update(('t', 'a'))
    if '-n' in context and requested:
        requested.add('n')

    documented = tuple(dict.fromkeys(match.group(1) for match in _SHORT_FLAG.finditer(context)))
    if not requested or not requested.issubset(documented):
        return None
    combined = '-' + ''.join(flag for flag in documented if flag in requested)
    arguments = [combined]
    if established:
        arguments.extend(('state', 'established'))
    return RuleCompilation('ss-listing', _document('ss', tuple(arguments)))


def _compile_pmap(instruction: str, context: str) -> RuleCompilation | None:
    lowered_instruction = instruction.lower()
    if not re.search(r'(?:^|\n)procps\s+pmap:', context.lower()):
        return None
    if 'kernel' in lowered_instruction and '-XX' in context:
        mode = '-XX'
        rule_id = 'pmap-kernel-details'
    elif re.search(r'\bextended\b', lowered_instruction) and '-x' in context:
        mode = '-x'
        rule_id = 'pmap-extended'
    else:
        return None
    process_ids = tuple(dict.fromkeys(_DECIMAL.findall(instruction)))
    if len(process_ids) != 1:
        return None
    return RuleCompilation(rule_id, _document('pmap', (mode, process_ids[0])))


def _document(command: str, arguments: tuple[str, ...]) -> dict[str, Any]:
    return {
        'v': 2,
        'd': 'zsh',
        's': [
            {
                't': 'p',
                'c': [
                    {
                        'n': {'s': command},
                        'a': [{'s': argument} for argument in arguments],
                    }
                ],
            }
        ],
    }
