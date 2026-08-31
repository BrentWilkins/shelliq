#!/usr/bin/env python3
"""Evaluate exact typed compilation on input-complete unseen command families."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.data import Platform, load_semantic_jsonl
from shelliq_training.documentation_templates import (
    DocumentationTemplateIndex,
    compile_documented_command_scored,
    documentation_templates,
)
from shelliq_training.semantic_actions import SemanticActionClient
from shelliq_training.semantic_equivalence import semantic_documents_equivalent

EXPERIMENT = 'documentation-compiler-input-complete-v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--benchmark-manifest', type=Path, required=True)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.benchmark_manifest.read_text())
    if manifest.get('experiment') != EXPERIMENT:
        raise ValueError('unexpected input-complete benchmark manifest')
    rows = [json.loads(line) for line in args.benchmark.read_text().splitlines()]
    index = DocumentationTemplateIndex(documentation_templates(load_semantic_jsonl(args.documentation_index)))
    compilations = [
        compile_documented_command_scored(
            index,
            command=row['command'],
            platform=Platform(row['platform']),
            instruction=row['instruction'],
            context=row['context'],
        )
        for row in rows
    ]
    ready_indices = [index for index, compilation in enumerate(compilations) if compilation.status == 'ready']
    ready_documents = [compilations[index].document for index in ready_indices]
    if any(document is None for document in ready_documents):
        raise AssertionError('ready compilation has no document')
    client = SemanticActionClient(args.actions)
    decoded = client.decode(client.encode(ready_documents))
    decoded_by_index = dict(zip(ready_indices, decoded, strict=True))

    delivered_exact = 0
    template_exact = 0
    outcomes: list[dict[str, object]] = []
    for index_value, (row, compilation) in enumerate(zip(rows, compilations, strict=True)):
        is_template_exact = compilation.document is not None and semantic_documents_equivalent(
            row['expected_document'], compilation.document
        )
        is_delivered_exact = compilation.status == 'ready' and is_template_exact
        template_exact += is_template_exact
        delivered_exact += is_delivered_exact
        outcomes.append(
            {
                'record_id': row['record_id'],
                'command': row['command'],
                'status': compilation.status,
                'template_exact': is_template_exact,
                'delivered_exact': is_delivered_exact,
                'rust_valid': decoded_by_index[index_value].valid if index_value in decoded_by_index else None,
                'sources': list(compilation.source_record_ids),
                'unresolved_slots': list(compilation.unresolved_slots),
                'word_provenance': [{'word': item.word, 'source': item.source} for item in compilation.word_provenance],
            }
        )
    ready = len(ready_indices)
    valid = sum(item.valid for item in decoded)
    precision = delivered_exact / ready if ready else 0.0
    coverage = delivered_exact / len(rows) if rows else 0.0
    metrics = {
        'examples': len(rows),
        'distinct_commands': len({row['command'] for row in rows}),
        'ready': ready,
        'needs_input': sum(item.status == 'needs_input' for item in compilations),
        'no_documentation': sum(item.status == 'no_documentation' for item in compilations),
        'ready_rust_valid': valid,
        'template_exact': template_exact,
        'delivered_exact': delivered_exact,
        'exact_coverage': coverage,
        'ready_precision': precision,
    }
    gate_passed = metrics['distinct_commands'] >= 50 and valid == ready and coverage >= 0.80 and precision >= 0.95
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'gate_passed': gate_passed,
        'metrics': metrics,
        'floors': {
            'minimum_distinct_commands': 50,
            'minimum_exact_coverage': 0.80,
            'minimum_ready_precision': 0.95,
            'require_all_ready_rust_valid': True,
        },
        'outcomes': outcomes,
        'split_status': 'documentation-derived capability benchmark; outer validation and test sealed',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({**report, 'outcomes': f'{len(rows)} rows'}, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit('input-complete documentation compiler gate failed')


if __name__ == '__main__':
    main()
