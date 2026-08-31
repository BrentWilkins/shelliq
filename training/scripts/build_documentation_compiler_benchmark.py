#!/usr/bin/env python3
"""Build deterministic input-complete requests for unseen command families."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.data import load_semantic_jsonl
from shelliq_training.documentation_templates import (
    RetrievedTemplate,
    bind_template,
    documentation_templates,
    is_placeholder,
    placeholder_kind,
    unresolved_placeholders,
)

EXPERIMENT = 'documentation-compiler-input-complete-v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit != 100:
        raise ValueError('v1 freezes the benchmark limit at 100')
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError('benchmark outputs must not already exist')
    semantic_records = load_semantic_jsonl(args.semantic_dataset)
    semantic_by_id = {record.record_id: record for record in semantic_records}
    split_manifest = json.loads(args.split_manifest.read_text())
    train_commands = {semantic_by_id[record_id].command for record_id in split_manifest['outer_splits']['train']['record_ids']}
    templates = sorted(
        documentation_templates(load_semantic_jsonl(args.documentation_index)),
        key=lambda item: item.record_id,
    )

    rows: list[dict[str, object]] = []
    selected_commands: set[str] = set()
    for template in templates:
        if template.command in train_commands or template.command in selected_commands:
            continue
        placeholders = _eligible_placeholders(template.document)
        if not placeholders:
            continue
        kinds = [placeholder_kind(placeholder) for placeholder in placeholders]
        values = [_slot_value(kind, len(rows), index) for index, kind in enumerate(kinds)]
        additions = ', '.join(
            f'{_formatted(value, kind)} for {placeholder}'
            for placeholder, value, kind in zip(placeholders, values, kinds, strict=True)
        )
        instruction = f'{template.instruction}. Use {additions}.'
        bound = bind_template(RetrievedTemplate(template, 1.0), instruction)
        if unresolved_placeholders(bound.document):
            continue
        rows.append(
            {
                'schema_version': 1,
                'record_id': f'input-complete:{template.record_id}',
                'command': template.command,
                'platform': template.platform.value,
                'instruction': instruction,
                'context': template.context,
                'source_record_id': template.record_id,
                'expected_document': bound.document,
            }
        )
        selected_commands.add(template.command)
        if len(rows) == args.limit:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        for row in rows:
            output.write(json.dumps(row, separators=(',', ':'), sort_keys=True) + '\n')
    manifest = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'records': len(rows),
        'distinct_commands': len(selected_commands),
        'excluded_neural_train_commands': len(train_commands),
        'documentation_index_sha256': hashlib.sha256(args.documentation_index.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'selection': 'first eligible record per unseen command in stable record-id order',
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _eligible_placeholders(document: dict[str, object]) -> tuple[str, ...]:
    statements = document.get('s')
    if not isinstance(statements, list) or len(statements) != 1 or not isinstance(statements[0], dict):
        return ()
    statement = statements[0]
    commands = statement.get('c')
    if statement.get('t') != 'p' or set(statement) != {'t', 'c'} or not isinstance(commands, list) or len(commands) != 1:
        return ()
    command = commands[0]
    if not isinstance(command, dict) or set(command) - {'n', 'a'}:
        return ()
    arguments = command.get('a')
    if not isinstance(arguments, list):
        return ()
    placeholders: list[str] = []
    leading_static = 0
    for argument in arguments:
        if not isinstance(argument, dict) or set(argument) != {'s'} or not isinstance(argument['s'], str):
            return ()
        value = argument['s']
        if value.startswith('-'):
            continue
        if is_placeholder(value):
            placeholders.append(value)
            continue
        if not placeholders and leading_static == 0:
            leading_static += 1
            continue
        return ()
    return tuple(dict.fromkeys(placeholders))


def _slot_value(kind: str, record_index: int, slot_index: int) -> str:
    suffix = record_index * 10 + slot_index + 1
    if kind == 'path':
        return f'/tmp/shelliq-input-{suffix}.dat'
    if kind == 'integer':
        return str(4000 + suffix)
    if kind == 'remote':
        return f'user{suffix}@example.com:/tmp/data'
    if kind == 'url':
        return f'https://example.com/item/{suffix}'
    return f'value_{suffix}'


def _formatted(value: str, kind: str) -> str:
    return f'"{value}"' if kind == 'text' else value


if __name__ == '__main__':
    main()
