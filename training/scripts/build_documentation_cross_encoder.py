#!/usr/bin/env python3
"""Build frozen command-disjoint cross-encoder train/development/test cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_documentation_compiler_benchmark as benchmark

from shelliq_training.documentation_cross_encoder import (
    EXPERIMENT,
    RankCase,
    load_templates,
    paraphrase,
    split_commands,
)
from shelliq_training.documentation_templates import RetrievedTemplate, bind_template, unresolved_placeholders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--train-output', type=Path, required=True)
    parser.add_argument('--development-output', type=Path, required=True)
    parser.add_argument('--test-output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (args.train_output, args.development_output, args.test_output, args.manifest)
    if any(path.exists() for path in outputs):
        raise FileExistsError('cross-encoder outputs must not already exist')
    typed_by_id, _, grouped = load_templates(args.documentation_index)
    splits = split_commands(grouped)
    command_sets = {name: set(commands) for name, commands in splits.items()}
    if any(
        command_sets[left] & command_sets[right]
        for left, right in [('train', 'development'), ('train', 'test'), ('development', 'test')]
    ):
        raise AssertionError('command split overlap')

    rows: dict[str, list[RankCase]] = {}
    global_index = 0
    for split, commands in splits.items():
        cases: list[RankCase] = []
        limit = 4 if split == 'train' else 1
        for command in commands:
            selected = 0
            for record_id in grouped[command]:
                template = typed_by_id[record_id]
                query = _complete_query(template, global_index)
                if query is None:
                    global_index += 1
                    continue
                cases.append(RankCase(command, record_id, paraphrase(query)))
                selected += 1
                global_index += 1
                if selected == limit:
                    break
            if selected == 0:
                raise ValueError(f'{split} command has no bindable template: {command}')
        rows[split] = cases

    _write_jsonl(args.train_output, rows['train'])
    _write_jsonl(args.development_output, rows['development'])
    _write_jsonl(args.test_output, rows['test'])
    manifest = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'documentation_index': str(args.documentation_index),
        'documentation_index_sha256': _sha256(args.documentation_index),
        'split_seed': 20260901,
        'command_disjoint': True,
        'splits': {
            split: {
                'commands': list(splits[split]),
                'command_count': len(splits[split]),
                'records': len(rows[split]),
                'sha256': _sha256(path),
            }
            for split, path in [
                ('train', args.train_output),
                ('development', args.development_output),
                ('test', args.test_output),
            ]
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _write_jsonl(path: Path, cases: list[RankCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        for case in cases:
            output.write(
                json.dumps(
                    {
                        'schema_version': 1,
                        'command': case.command,
                        'source_record_id': case.source_record_id,
                        'query': case.query,
                    },
                    sort_keys=True,
                )
                + '\n'
            )


def _complete_query(template, record_index: int) -> str | None:
    placeholders = unresolved_placeholders(template.document)
    values = [
        benchmark._slot_value(benchmark.placeholder_kind(placeholder), record_index, slot_index)
        for slot_index, placeholder in enumerate(placeholders)
    ]
    rendered = [
        benchmark._formatted(value, benchmark.placeholder_kind(placeholder))
        for placeholder, value in zip(placeholders, values, strict=True)
    ]
    instruction = template.instruction
    if rendered:
        instruction = f'{instruction}. Use {" and ".join(rendered)}.'
    bound = bind_template(RetrievedTemplate(template, 1.0), instruction)
    return instruction if set(placeholders).issubset(dict(bound.bindings)) else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    main()
