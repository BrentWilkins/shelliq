#!/usr/bin/env python3
"""Evaluate the target-blind documentation compiler and abstention contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.data import load_semantic_jsonl
from shelliq_training.documentation_templates import (
    DocumentationTemplateIndex,
    compile_documented_command,
    documentation_templates,
)
from shelliq_training.semantic_actions import SemanticActionClient
from shelliq_training.teacher_verification import RustValidation, verify_against_reference

EXPERIMENT = 'semantic-action-documentation-compiler-v1'
MINIMUM_ACCEPTED = 5
MINIMUM_PRECISION = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = DocumentationTemplateIndex(documentation_templates(load_semantic_jsonl(args.documentation_index)))
    records = load_semantic_jsonl(args.semantic_dataset)
    by_id = {record.record_id: record for record in records}
    manifest = json.loads(args.manifest.read_text())
    evaluation = [by_id[item] for item in manifest['inner_splits']['validation']['record_ids']]

    compilations = [
        compile_documented_command(
            index,
            command=record.command,
            platform=record.platform,
            instruction=record.instruction,
            context=record.context,
        )
        for record in evaluation
    ]
    ready_indices = [index for index, item in enumerate(compilations) if item.status == 'ready']
    ready_documents = [compilations[index].document for index in ready_indices]
    if any(document is None for document in ready_documents):
        raise AssertionError('ready compilation has no document')
    client = SemanticActionClient(args.actions)
    decoded = client.decode(client.encode(ready_documents))

    statuses = Counter(item.status for item in compilations)
    accepted = 0
    first_command = 0
    rows: list[dict[str, object]] = []
    decoded_by_index = dict(zip(ready_indices, decoded, strict=True))
    for index_value, (record, compilation) in enumerate(zip(evaluation, compilations, strict=True)):
        row: dict[str, object] = {
            'record_id': record.record_id,
            'command': record.command,
            'status': compilation.status,
            'sources': list(compilation.source_record_ids),
            'selected_options': list(compilation.selected_options),
            'bindings': list(compilation.bindings),
            'unresolved_slots': list(compilation.unresolved_slots),
        }
        if compilation.status == 'ready':
            result = decoded_by_index[index_value]
            verification = verify_against_reference(
                json.dumps(compilation.document, separators=(',', ':')),
                json.loads(record.response),
                RustValidation(result.valid, result.rendered, result.error),
            )
            accepted += verification.accepted
            first_command += verification.first_command_match
            row.update(
                {
                    'rust_valid': verification.rust_round_trip,
                    'first_command_match': verification.first_command_match,
                    'reference_accepted': verification.accepted,
                    'rendered': verification.rendered,
                }
            )
        rows.append(row)

    ready = len(ready_indices)
    rust_valid = sum(item.valid for item in decoded)
    precision = accepted / ready if ready else 0.0
    metrics = {
        'examples': len(evaluation),
        'ready': ready,
        'needs_input': statuses['needs_input'],
        'no_documentation': statuses['no_documentation'],
        'ready_rust_valid': rust_valid,
        'ready_first_command_match': first_command,
        'reference_accepted': accepted,
        'ready_reference_precision': precision,
    }
    gate_passed = rust_valid == ready and accepted >= MINIMUM_ACCEPTED and precision >= MINIMUM_PRECISION
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'inner-target-blind',
        'gate_passed': gate_passed,
        'metrics': metrics,
        'floors': {
            'require_all_ready_rust_valid': True,
            'minimum_reference_accepted': MINIMUM_ACCEPTED,
            'minimum_ready_reference_precision': MINIMUM_PRECISION,
        },
        'outcomes': rows,
        'split_status': 'open inner-development only; outer result not consulted and test sealed',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({**report, 'outcomes': f'{len(rows)} rows'}, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit('target-blind documentation compiler gate failed')


if __name__ == '__main__':
    main()
