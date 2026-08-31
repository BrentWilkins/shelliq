#!/usr/bin/env python3
"""Evaluate bounded generic composition of typed documentation fragments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.data import load_semantic_jsonl
from shelliq_training.documentation_templates import (
    DocumentationTemplateIndex,
    compose_template_candidates,
    documentation_templates,
    template_matches_document,
)
from shelliq_training.semantic_actions import SemanticActionClient
from shelliq_training.semantic_equivalence import semantic_documents_equivalent

EXPERIMENT = 'semantic-action-documentation-fragments-v1'


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

    rows: list[dict[str, object]] = []
    all_documents: list[dict[str, object]] = []
    covered_commands: set[str] = set()
    for record in evaluation:
        reference = json.loads(record.response)
        retrieved = index.rank(
            command=record.command,
            platform=record.platform,
            instruction=record.instruction,
            context=record.context,
            limit=8,
        )
        candidates = compose_template_candidates(retrieved, record.instruction, limit=64, maximum_sources=3)
        skeleton_index = next(
            (index for index, candidate in enumerate(candidates) if template_matches_document(candidate.document, reference)),
            None,
        )
        exact_index = next(
            (index for index, candidate in enumerate(candidates) if semantic_documents_equivalent(reference, candidate.document)),
            None,
        )
        if skeleton_index is not None:
            covered_commands.add(record.command)
        offset = len(all_documents)
        all_documents.extend(candidate.document for candidate in candidates)
        rows.append(
            {
                'record_id': record.record_id,
                'command': record.command,
                'retrieved_templates': len(retrieved),
                'composed_candidates': len(candidates),
                'document_offset': offset,
                'skeleton_candidate_index': skeleton_index,
                'exact_candidate_index': exact_index,
                'skeleton_sources': (list(candidates[skeleton_index].source_record_ids) if skeleton_index is not None else []),
            }
        )

    client = SemanticActionClient(args.actions)
    decoded = client.decode(client.encode(all_documents))
    invalid = [index for index, item in enumerate(decoded) if not item.valid]
    metrics = {
        'examples': len(evaluation),
        'documentation_covered': sum(bool(row['retrieved_templates']) for row in rows),
        'candidate_documents': len(all_documents),
        'rust_valid_candidates': len(decoded) - len(invalid),
        'skeleton_oracle': sum(row['skeleton_candidate_index'] is not None for row in rows),
        'skeleton_oracle_distinct_commands': len(covered_commands),
        'exact_oracle': sum(row['exact_candidate_index'] is not None for row in rows),
    }
    gate_passed = not invalid and metrics['skeleton_oracle'] >= 19 and metrics['skeleton_oracle_distinct_commands'] >= 10
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'inner-composition-feasibility',
        'gate_passed': gate_passed,
        'retrieval_limit': 8,
        'composition_limit': 64,
        'maximum_sources': 3,
        'metrics': metrics,
        'floors': {
            'require_all_candidates_rust_valid': True,
            'minimum_skeleton_oracle': 19,
            'minimum_skeleton_oracle_distinct_commands': 10,
        },
        'outcomes': rows,
        'split_status': 'open inner-development only; outer result not consulted and test sealed',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({**report, 'outcomes': f'{len(rows)} rows'}, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit('documentation-fragment composition feasibility gate failed')


if __name__ == '__main__':
    main()
