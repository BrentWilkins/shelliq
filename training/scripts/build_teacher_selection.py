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
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
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


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')
    if not args.output.parent.is_dir():
        raise SystemExit(f'output parent does not exist: {args.output.parent}')
    rows = [json.loads(line) for line in args.semantic_dataset.read_text().splitlines() if line.strip()]
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
    ]
    chosen: list[dict[str, object]] = []
    seen_commands: set[tuple[str, str]] = set()
    # The curated source has no Darwin compositional rows. Keep both the 40/8
    # platform balance and 12/category balance with an explicit feasible matrix.
    quotas = {
        ('linux', 'precise'): 9,
        ('darwin', 'precise'): 3,
        ('linux', 'multi-constraint'): 9,
        ('darwin', 'multi-constraint'): 3,
        ('linux', 'pipeline'): 10,
        ('darwin', 'pipeline'): 2,
        ('linux', 'compositional'): 12,
    }
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
        if len(chosen) == 48:
            break
    if len(chosen) != 48 or any(quotas.values()):
        raise SystemExit(f'dataset cannot satisfy benchmark quotas: {quotas}')
    output_rows = []
    for index, row in enumerate(chosen, start=1):
        output_rows.append(
            {
                'schema_version': 1,
                'challenge_id': f'teacher-selection:v1:{row["platform"]}:{index:03d}',
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
