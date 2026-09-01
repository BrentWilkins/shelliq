#!/usr/bin/env python3
"""Build frozen command-disjoint clarification-flow development and test sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_documentation_compiler_benchmark as benchmark
import build_documentation_fallback_e2e as fallback

from shelliq_training.data import load_semantic_jsonl
from shelliq_training.documentation_templates import (
    RetrievedTemplate,
    _request_literals,
    bind_template,
    documentation_templates,
    unresolved_placeholders,
)

EXPERIMENT = 'documentation-clarification-flow-v1'
SAFE_COMMAND = re.compile(r'^[A-Za-z0-9][A-Za-z0-9+._-]*$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--prior-benchmark', type=Path, required=True)
    parser.add_argument('--exclude-dataset', type=Path, action='append', default=[])
    parser.add_argument('--development-output', type=Path, required=True)
    parser.add_argument('--test-output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (args.development_output, args.test_output, args.manifest)
    if any(path.exists() for path in outputs):
        raise FileExistsError('frozen clarification output already exists')

    semantic_records = load_semantic_jsonl(args.semantic_dataset)
    semantic_by_id = {record.record_id: record for record in semantic_records}
    split_manifest = json.loads(args.split_manifest.read_text())
    train_ids = split_manifest['outer_splits']['train']['record_ids']
    train_commands = {semantic_by_id[record_id].command for record_id in train_ids}
    prior_rows = [json.loads(line) for line in args.prior_benchmark.read_text().splitlines()]
    excluded_commands = set(train_commands)
    excluded_commands.update(str(row['command']) for row in prior_rows)
    for path in args.exclude_dataset:
        excluded_commands.update(str(json.loads(line)['command']) for line in path.read_text().splitlines())

    selected = []
    selected_commands: set[str] = set()
    templates = sorted(
        documentation_templates(load_semantic_jsonl(args.documentation_index)),
        key=lambda item: item.record_id,
    )
    for template in templates:
        if template.platform.value != 'linux':
            continue
        if template.command in excluded_commands or template.command in selected_commands:
            continue
        if not SAFE_COMMAND.fullmatch(template.command):
            continue
        placeholders = benchmark._eligible_placeholders(template.document)
        if len(placeholders) != 1 or _request_literals(template.instruction):
            continue
        words = fallback._simple_words(template.document)
        if words is None:
            continue
        placeholder = placeholders[0]
        kind = benchmark.placeholder_kind(placeholder)
        value = benchmark._slot_value(kind, len(selected), 0)
        answer = benchmark._formatted(value, kind)
        completed_instruction = f'{template.instruction}. Use {answer}.'
        bound = bind_template(RetrievedTemplate(template, 1.0), completed_instruction)
        if unresolved_placeholders(bound.document):
            continue
        expected_words = fallback._simple_words(bound.document)
        if expected_words is None:
            continue
        selected.append(
            {
                'record_id': f'{EXPERIMENT}:{template.record_id}',
                'template_record_id': template.record_id,
                'command': template.command,
                'platform': template.platform.value,
                'instruction': template.instruction,
                'expected_slot_kind': kind,
                'expected_slot_label': placeholder,
                'answer': answer,
                'completed_instruction': completed_instruction,
                'expected_command': ' '.join(expected_words),
                'documented_options': sorted({word for word in expected_words[1:] if word.startswith('-')}),
            }
        )
        selected_commands.add(template.command)
        if len(selected) == 64:
            break
    if len(selected) != 64:
        raise ValueError(f'expected 64 eligible command families, found {len(selected)}')

    development = selected[:16]
    test = selected[16:]
    _write_jsonl(args.development_output, development)
    _write_jsonl(args.test_output, test)
    manifest = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'command_disjoint': True,
        'selection': (
            'stable record-id order; Linux literal-free simple single-command TLDR '
            'recipes with exactly one bindable placeholder; one recipe per command'
        ),
        'distinct_commands': len(selected_commands),
        'excluded_neural_train_commands': len(train_commands),
        'excluded_commands_total': len(excluded_commands),
        'documentation_index_sha256': _sha256(args.documentation_index),
        'development': {
            'records': len(development),
            'sha256': _sha256(args.development_output),
        },
        'test': {'records': len(test), 'sha256': _sha256(args.test_output)},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + '\n')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    main()
