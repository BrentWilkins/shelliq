#!/usr/bin/env python3
"""Build frozen development and test sets for the shipped documentation fallback."""

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

from shelliq_training.data import load_semantic_jsonl
from shelliq_training.documentation_templates import (
    DocumentationTemplate,
    RetrievedTemplate,
    bind_template,
    documentation_templates,
    unresolved_placeholders,
)

DEFAULT_EXPERIMENT = 'documentation-fallback-e2e-v1'
SAFE_COMMAND = re.compile(r'^[A-Za-z0-9][A-Za-z0-9+._-]*$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--prior-benchmark', type=Path, required=True)
    parser.add_argument('--exclude-dataset', action='append', type=Path, default=[])
    parser.add_argument('--experiment', default=DEFAULT_EXPERIMENT)
    parser.add_argument('--development-output', type=Path, required=True)
    parser.add_argument('--test-output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (args.development_output, args.test_output, args.manifest)
    if any(path.exists() for path in outputs):
        raise FileExistsError('frozen outputs must not already exist')

    semantic_records = load_semantic_jsonl(args.semantic_dataset)
    semantic_by_id = {record.record_id: record for record in semantic_records}
    split_manifest = json.loads(args.split_manifest.read_text())
    train_ids = split_manifest['outer_splits']['train']['record_ids']
    train_commands = {semantic_by_id[record_id].command for record_id in train_ids}
    excluded_commands = set(train_commands)
    prior_rows = [json.loads(line) for line in args.prior_benchmark.read_text().splitlines()]
    prior_commands = {row['command'] for row in prior_rows}
    excluded_commands.update(prior_commands)
    additional_commands = {
        row['command'] for path in args.exclude_dataset for row in (json.loads(line) for line in path.read_text().splitlines())
    }
    excluded_commands.update(additional_commands)

    templates = sorted(
        documentation_templates(load_semantic_jsonl(args.documentation_index)),
        key=lambda item: item.record_id,
    )
    selected: list[DocumentationTemplate] = []
    selected_commands: set[str] = set()
    for template in templates:
        if template.platform.value != 'linux':
            continue
        if template.command in excluded_commands or template.command in selected_commands:
            continue
        if not SAFE_COMMAND.fullmatch(template.command):
            continue
        if not benchmark._eligible_placeholders(template.document):
            continue
        if _simple_words(template.document) is None:
            continue
        if not _can_build_complete_request(template, len(selected)):
            continue
        selected.append(template)
        selected_commands.add(template.command)
        if len(selected) == 96:
            break
    if len(selected) != 96:
        raise ValueError(f'expected 96 eligible command families, found {len(selected)}')

    development = _rows(selected[:24], ready=True, offset=0)
    development.extend(_rows(selected[24:32], ready=False, offset=24))
    test = _rows(selected[32:80], ready=True, offset=32)
    test.extend(_rows(selected[80:96], ready=False, offset=80))

    _write_jsonl(args.development_output, development)
    _write_jsonl(args.test_output, test)
    manifest = {
        'schema_version': 1,
        'experiment': args.experiment,
        'selection': (
            'stable record-id order; Linux simple single-command typed TLDR recipes with '
            'at least one placeholder; one recipe per command'
        ),
        'development': {
            'records': len(development),
            'ready': 24,
            'needs_input': 8,
            'sha256': _sha256(args.development_output),
        },
        'test': {
            'records': len(test),
            'ready': 48,
            'needs_input': 16,
            'sha256': _sha256(args.test_output),
        },
        'distinct_commands': len(selected_commands),
        'excluded_neural_train_commands': len(train_commands),
        'excluded_prior_benchmark_commands': len(prior_commands),
        'excluded_additional_commands': len(additional_commands),
        'documentation_index_sha256': _sha256(args.documentation_index),
        'command_disjoint': True,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _rows(templates: list[DocumentationTemplate], *, ready: bool, offset: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for local_index, template in enumerate(templates):
        record_index = offset + local_index
        placeholders = benchmark._eligible_placeholders(template.document)
        instruction = template.instruction
        expected_document = None
        if ready:
            values = [
                benchmark._slot_value(benchmark.placeholder_kind(placeholder), record_index, slot_index)
                for slot_index, placeholder in enumerate(placeholders)
            ]
            rendered_values = [
                benchmark._formatted(value, benchmark.placeholder_kind(placeholder))
                for placeholder, value in zip(placeholders, values, strict=True)
            ]
            instruction = f'{instruction}. Use {" and ".join(rendered_values)}.'
            bound = bind_template(RetrievedTemplate(template, 1.0), instruction)
            if unresolved_placeholders(bound.document):
                raise AssertionError(f'{template.record_id}: generated request is not complete')
            expected_document = bound.document
        words = _simple_words(expected_document or template.document)
        if words is None:
            raise AssertionError(f'{template.record_id}: expected a simple command')
        rows.append(
            {
                'schema_version': 1,
                'record_id': f'e2e:{"ready" if ready else "needs-input"}:{template.record_id}',
                'source_record_id': template.record_id,
                'command': template.command,
                'platform': template.platform.value,
                'instruction': instruction,
                'expected_status': 'ready' if ready else 'needs_input',
                'expected_command': ' '.join(words) if ready else None,
                'documented_options': sorted({word for word in words[1:] if word.startswith('-')}),
            }
        )
    return rows


def _can_build_complete_request(template: DocumentationTemplate, record_index: int) -> bool:
    placeholders = benchmark._eligible_placeholders(template.document)
    values = [
        benchmark._slot_value(benchmark.placeholder_kind(placeholder), record_index, slot_index)
        for slot_index, placeholder in enumerate(placeholders)
    ]
    rendered_values = [
        benchmark._formatted(value, benchmark.placeholder_kind(placeholder))
        for placeholder, value in zip(placeholders, values, strict=True)
    ]
    instruction = f'{template.instruction}. Use {" and ".join(rendered_values)}.'
    bound = bind_template(RetrievedTemplate(template, 1.0), instruction)
    return not unresolved_placeholders(bound.document)


def _simple_words(document: object) -> list[str] | None:
    if not isinstance(document, dict) or set(document) != {'v', 'd', 's'}:
        return None
    statements = document.get('s')
    if not isinstance(statements, list) or len(statements) != 1:
        return None
    statement = statements[0]
    if not isinstance(statement, dict) or set(statement) != {'t', 'c'} or statement.get('t') != 'p':
        return None
    commands = statement.get('c')
    if not isinstance(commands, list) or len(commands) != 1:
        return None
    command = commands[0]
    if not isinstance(command, dict) or set(command) - {'n', 'a'}:
        return None
    nodes = [command.get('n'), *command.get('a', [])]
    if any(not isinstance(node, dict) or set(node) != {'s'} for node in nodes):
        return None
    words = [node['s'] for node in nodes]
    return words if all(isinstance(word, str) for word in words) else None


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        for row in rows:
            output.write(json.dumps(row, separators=(',', ':'), sort_keys=True) + '\n')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    main()
