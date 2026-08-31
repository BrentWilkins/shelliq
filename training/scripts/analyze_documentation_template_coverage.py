#!/usr/bin/env python3
"""Attribute documentation-template misses on the open inner split."""

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
    documentation_templates,
    template_matches_document,
)

EXPERIMENT = 'semantic-action-documentation-template-coverage-v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--retrieval-report', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval = json.loads(args.retrieval_report.read_text())
    if retrieval.get('experiment') != 'semantic-action-documentation-templates-v1':
        raise ValueError('unexpected retrieval report')

    index_records = load_semantic_jsonl(args.documentation_index)
    index = DocumentationTemplateIndex(documentation_templates(index_records))
    records = load_semantic_jsonl(args.semantic_dataset)
    by_id = {record.record_id: record for record in records}
    manifest = json.loads(args.manifest.read_text())
    evaluation = [by_id[item] for item in manifest['inner_splits']['validation']['record_ids']]

    counts: Counter[str] = Counter()
    commands: dict[str, set[str]] = {
        'missing_documentation': set(),
        'missing_recipe': set(),
        'retrieval_rank_miss': set(),
        'top8_recipe': set(),
    }
    rows: list[dict[str, object]] = []
    for record in evaluation:
        reference = json.loads(record.response)
        ranked = index.rank(
            command=record.command,
            platform=record.platform,
            instruction=record.instruction,
            context=record.context,
            limit=100_000,
        )
        rank = next(
            (
                position
                for position, item in enumerate(ranked, start=1)
                if template_matches_document(item.template.document, reference)
            ),
            None,
        )
        if not ranked:
            category = 'missing_documentation'
        elif rank is None:
            category = 'missing_recipe'
        elif rank > 8:
            category = 'retrieval_rank_miss'
        else:
            category = 'top8_recipe'
        counts[category] += 1
        commands[category].add(record.command)
        rows.append(
            {
                'record_id': record.record_id,
                'command': record.command,
                'category': category,
                'documentation_templates': len(ranked),
                'matching_rank': rank,
                'matching_template': ranked[rank - 1].template.record_id if rank is not None else None,
            }
        )

    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'evaluation_records': len(evaluation),
        'categories': {name: {'records': counts[name], 'distinct_commands': len(values)} for name, values in commands.items()},
        'outcomes': rows,
        'split_status': 'read-only attribution on open inner-development split',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({**report, 'outcomes': f'{len(rows)} rows'}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
