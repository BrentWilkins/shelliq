#!/usr/bin/env python3
"""Build a frozen teacher-selection benchmark from reviewed semantic records.

This benchmark selects training-shaped tasks to compare external teacher models;
it is never a student release, retention, or shadow gate and must never be added
to a training corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

V2_INSTRUCTION_OVERRIDES = {
    'curated:dev-toolchains:linux:uv-sync-frozen': 'Install locked production dependencies in CI without re-resolving.',
    'curated:files-and-processes:linux:pgrep-full': 'Find PID and command for processes matching python.*server.py.',
    'curated:safety:linux:git-clean-preview': 'Preview deleting untracked files, directories, and ignored files.',
    'curated:zsh-native:linux:zmv-dry-run': 'Preview renaming every *.jpeg file to the same basename with .jpg.',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark-version', type=int, choices=(1, 2), default=1)
    parser.add_argument(
        '--source',
        action='append',
        default=None,
        help=(
            'source field eligible for selection; repeat to allow multiple sources '
            '(default: shelliq-curated; TLDR is intentionally excluded)'
        ),
    )
    return parser.parse_args()


def category(row: dict[str, object]) -> str:
    shell = row['shell_response']
    instruction = row['instruction']
    assert isinstance(shell, str) and isinstance(instruction, str)
    if '|' in shell or '>' in shell or '<' in shell:
        return 'pipeline'
    constraint_markers = sum(shell.count(marker) for marker in (' -', ' --', '! ', '='))
    if constraint_markers >= 4 or len(instruction) >= 110:
        return 'compositional'
    if constraint_markers >= 2 or len(instruction) >= 70:
        return 'multi-constraint'
    return 'precise'


def stable_key(row: dict[str, object]) -> str:
    return hashlib.sha256(str(row['record_id']).encode()).hexdigest()


def semantic_words(value: object) -> list[str]:
    """Return every compact semantic word literal in document order."""
    if isinstance(value, dict):
        if set(value) == {'s'} and isinstance(value['s'], str):
            return [value['s']]
        words: list[str] = []
        for child in value.values():
            words.extend(semantic_words(child))
        return words
    if isinstance(value, list):
        words = []
        for child in value:
            words.extend(semantic_words(child))
        return words
    return []


def word_is_visible(word: str, prompt: str) -> bool:
    """Conservatively require an expected lexical word to be prompt-supported."""
    visible = prompt.casefold()
    literal = word.strip('\'"').casefold()
    if literal in visible:
        return True
    quoted_expression = (
        len(word) >= 2 and word[0] == word[-1] and word[0] in '\'"' and any(character in literal for character in '$(){}[]*|^\\')
    )
    if quoted_expression:
        return True
    if re.fullmatch(r'-[a-z]{2,}', literal):
        return all(f'-{flag}' in visible for flag in literal[1:])
    if '=' in literal:
        option, value = literal.split('=', 1)
        return option in visible and value in visible
    return False


def prompt_determines_expected(row: dict[str, object]) -> bool:
    """Reject references containing lexical choices hidden from the candidate."""
    expected = row.get('semantic_target')
    instruction = row.get('instruction')
    context = row.get('context')
    if not isinstance(expected, dict) or not isinstance(instruction, str) or not isinstance(context, str):
        return False
    prompt = f'{instruction}\n{context}'
    return all(word_is_visible(word, prompt) for word in semantic_words(expected))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')
    if not args.output.parent.is_dir():
        raise SystemExit(f'output parent does not exist: {args.output.parent}')
    rows = [json.loads(line) for line in args.semantic_dataset.read_text().splitlines() if line.strip()]
    if args.benchmark_version == 2:
        for row in rows:
            override = V2_INSTRUCTION_OVERRIDES.get(str(row.get('record_id')))
            if override is not None:
                row['instruction'] = override
    allowed_sources = set(args.source or ['shelliq-curated'])
    if 'tldr-pages' in allowed_sources:
        raise SystemExit(
            'tldr-pages is not allowed in the teacher-selection benchmark: widely trained examples would bias the comparison'
        )
    eligible = [
        row
        for row in rows
        if row.get('conversion_schema_version') == 1
        and row.get('source') in allowed_sources
        and row.get('platform') in {'linux', 'darwin'}
        and isinstance(row.get('semantic_target'), dict)
        and (args.benchmark_version == 1 or prompt_determines_expected(row))
    ]
    chosen: list[dict[str, object]] = []
    seen_commands: set[tuple[str, str]] = set()
    # The curated source has no Darwin compositional rows. V1 preserves the
    # original 48-case matrix. V2 is deliberately smaller: strict prompt
    # determination leaves too few trustworthy complex rows for a balanced 48,
    # and padding it would recreate the hidden-reference defect found in v1.
    quotas = (
        {
            ('linux', 'precise'): 9,
            ('darwin', 'precise'): 3,
            ('linux', 'multi-constraint'): 9,
            ('darwin', 'multi-constraint'): 3,
            ('linux', 'pipeline'): 10,
            ('darwin', 'pipeline'): 2,
            ('linux', 'compositional'): 12,
        }
        if args.benchmark_version == 1
        else {
            ('linux', 'precise'): 6,
            ('darwin', 'precise'): 2,
            ('linux', 'multi-constraint'): 6,
            ('darwin', 'multi-constraint'): 2,
            ('linux', 'pipeline'): 2,
            ('linux', 'compositional'): 2,
        }
    )
    target_size = sum(quotas.values())
    for row in sorted(eligible, key=stable_key):
        platform = str(row['platform'])
        group = category(row)
        command_key = (platform, str(row['command']))
        quota_key = (platform, group)
        if quotas.get(quota_key, 0) <= 0 or command_key in seen_commands:
            continue
        chosen.append(row)
        seen_commands.add(command_key)
        quotas[quota_key] -= 1
        if len(chosen) == target_size:
            break
    if len(chosen) != target_size or any(quotas.values()):
        raise SystemExit(f'dataset cannot satisfy benchmark quotas: {quotas}')
    output_rows = []
    for index, row in enumerate(chosen, start=1):
        output_rows.append(
            {
                'schema_version': 1,
                'challenge_id': f'teacher-selection:v{args.benchmark_version}:{row["platform"]}:{index:03d}',
                'category': category(row),
                'platform': row['platform'],
                'command': row['command'],
                'instruction': row['instruction'],
                'context': row['context'],
                'expected': row['semantic_target'],
                'grounding_paths': [],
                'constraints': ['satisfies every instruction constraint', 'uses only authoritative context literals'],
                'risk': 'static-only',
                'functional_eligible': False,
            }
        )
    args.output.write_text(
        ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':'), sort_keys=True) + '\n' for row in output_rows)
    )
    print(f'wrote {len(output_rows)} challenges from {", ".join(sorted(allowed_sources))}: {args.output}')


if __name__ == '__main__':
    main()
