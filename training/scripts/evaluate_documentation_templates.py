#!/usr/bin/env python3
"""Evaluate generic typed documentation retrieval on open inner development."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.data import load_semantic_jsonl
from shelliq_training.documentation_templates import (
    DocumentationTemplateIndex,
    bind_template,
    documentation_templates,
    template_matches_document,
)
from shelliq_training.semantic_actions import SemanticActionClient
from shelliq_training.teacher_verification import RustValidation, verify_against_reference

EXPERIMENT = 'semantic-action-documentation-templates-v1'
INDEX_EXPERIMENT = 'documentation-template-index-v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--index-manifest', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit != 8:
        raise ValueError('v1 preregistration freezes retrieval limit at eight')
    index_manifest = json.loads(args.index_manifest.read_text())
    if index_manifest.get('experiment') != INDEX_EXPERIMENT:
        raise ValueError('unexpected documentation index manifest')
    if index_manifest.get('output_sha256') != hashlib.sha256(args.documentation_index.read_bytes()).hexdigest():
        raise ValueError('documentation index hash mismatch')

    docs = load_semantic_jsonl(args.documentation_index)
    templates = documentation_templates(docs)
    if len(templates) != index_manifest.get('records') or len(templates) != len(docs):
        raise ValueError('documentation index contains non-TLDR or missing records')
    index = DocumentationTemplateIndex(templates)

    records = load_semantic_jsonl(args.semantic_dataset)
    by_id = {record.record_id: record for record in records}
    experiment_manifest = json.loads(args.manifest.read_text())
    evaluation_ids = experiment_manifest['inner_splits']['validation']['record_ids']
    evaluation_records = [by_id[item] for item in evaluation_ids]

    rows: list[dict[str, object]] = []
    generated_documents: list[dict[str, object]] = []
    top8_commands: set[str] = set()
    for record in evaluation_records:
        reference = json.loads(record.response)
        ranked = index.rank(
            command=record.command,
            platform=record.platform,
            instruction=record.instruction,
            context=record.context,
            limit=args.limit,
        )
        top1_skeleton = bool(ranked) and template_matches_document(ranked[0].template.document, reference)
        matching = [item for item in ranked if template_matches_document(item.template.document, reference)]
        top8_skeleton = bool(matching)
        if top8_skeleton:
            top8_commands.add(record.command)
        bound = bind_template(ranked[0], record.instruction) if ranked else None
        generated_documents.append(bound.document if bound else _command_only_document(record.command))
        rows.append(
            {
                'record_id': record.record_id,
                'command': record.command,
                'documentation_candidates': len(ranked),
                'top1_template': ranked[0].template.record_id if ranked else None,
                'top1_score': ranked[0].score if ranked else None,
                'top1_skeleton_match': top1_skeleton,
                'top8_skeleton_match': top8_skeleton,
                'matching_template': matching[0].template.record_id if matching else None,
                'bindings': list(bound.bindings) if bound else [],
            }
        )

    client = SemanticActionClient(args.actions)
    encoded = client.encode(generated_documents)
    decoded = client.decode(encoded)
    metrics = {
        'examples': len(evaluation_records),
        'documentation_covered': sum(bool(row['documentation_candidates']) for row in rows),
        'top1_skeleton_match': sum(bool(row['top1_skeleton_match']) for row in rows),
        'top8_skeleton_match': sum(bool(row['top8_skeleton_match']) for row in rows),
        'top8_distinct_commands': len(top8_commands),
        'rust_valid': sum(item.valid for item in decoded),
        'first_command_match': 0,
        'reference_accepted': 0,
    }
    for record, document, result, row in zip(evaluation_records, generated_documents, decoded, rows, strict=True):
        verification = verify_against_reference(
            json.dumps(document, separators=(',', ':')),
            json.loads(record.response),
            RustValidation(result.valid, result.rendered, result.error),
        )
        metrics['first_command_match'] += verification.first_command_match
        metrics['reference_accepted'] += verification.accepted
        row['rust_valid'] = verification.rust_round_trip
        row['first_command_match'] = verification.first_command_match
        row['reference_accepted'] = verification.accepted

    gate_passed = (
        metrics['rust_valid'] == metrics['examples']
        and metrics['top8_skeleton_match'] >= 19
        and metrics['top8_distinct_commands'] >= 10
        and metrics['top8_skeleton_match'] > metrics['top1_skeleton_match']
    )
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'inner-retrieval-feasibility',
        'gate_passed': gate_passed,
        'retrieval_limit': args.limit,
        'documentation_index_sha256': index_manifest['output_sha256'],
        'documentation_records': len(templates),
        'metrics': metrics,
        'floors': {
            'minimum_rust_valid': len(evaluation_records),
            'minimum_top8_skeleton_match': 19,
            'minimum_top8_distinct_commands': 10,
            'require_top8_exceeds_top1': True,
        },
        'outcomes': rows,
        'split_status': 'open inner-development only; outer result not consulted and test sealed',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({**report, 'outcomes': f'{len(rows)} rows'}, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit('documentation-template retrieval feasibility gate failed')


def _command_only_document(command: str) -> dict[str, object]:
    return {'v': 2, 'd': 'zsh', 's': [{'t': 'p', 'c': [{'n': {'s': command}}]}]}


if __name__ == '__main__':
    main()
