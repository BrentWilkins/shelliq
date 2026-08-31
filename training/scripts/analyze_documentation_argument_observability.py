#!/usr/bin/env python3
"""Measure whether inner reference words are observable without target leakage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.data import load_semantic_jsonl
from shelliq_training.documentation_templates import documentation_templates, is_placeholder
from shelliq_training.semantic_equivalence import canonicalize_semantic_document

EXPERIMENT = 'semantic-action-documentation-observability-v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documentation = documentation_templates(load_semantic_jsonl(args.documentation_index))
    document_words: dict[tuple[str, object], set[str]] = defaultdict(set)
    placeholder_commands: set[tuple[str, object]] = set()
    for template in documentation:
        key = (template.command, template.platform)
        for word in _document_words(template.document)[1:]:
            document_words[key].add(word)
            if is_placeholder(word):
                placeholder_commands.add(key)

    records = load_semantic_jsonl(args.semantic_dataset)
    by_id = {record.record_id: record for record in records}
    manifest = json.loads(args.manifest.read_text())
    evaluation = [by_id[item] for item in manifest['inner_splits']['validation']['record_ids']]
    counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    fully_exact = 0
    fully_templated = 0
    for record in evaluation:
        local = f'{record.instruction}\n{record.context}'
        key = (record.command, record.platform)
        words = _document_words(json.loads(record.response))[1:]
        decisions: list[dict[str, object]] = []
        exact_record = True
        templated_record = True
        for word in words:
            if word in local:
                source = 'request'
            elif word in document_words[key]:
                source = 'documentation'
            elif not word.startswith('-') and key in placeholder_commands:
                source = 'placeholder-only'
                exact_record = False
            else:
                source = 'unobservable'
                exact_record = False
                templated_record = False
            counts[source] += 1
            decisions.append({'word': word, 'source': source})
        fully_exact += exact_record
        fully_templated += templated_record
        rows.append(
            {
                'record_id': record.record_id,
                'command': record.command,
                'fully_exact_observable': exact_record,
                'fully_template_observable': templated_record,
                'words': decisions,
            }
        )

    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'evaluation_records': len(evaluation),
        'reference_words': sum(counts.values()),
        'word_sources': dict(sorted(counts.items())),
        'fully_exact_observable_records': fully_exact,
        'fully_template_observable_records': fully_templated,
        'outcomes': rows,
        'split_status': 'read-only attribution on open inner-development split',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({**report, 'outcomes': f'{len(rows)} rows'}, indent=2, sort_keys=True))


def _document_words(document: dict[str, object]) -> list[str]:
    canonical = canonicalize_semantic_document(document)
    words: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if set(value) == {'s'} and isinstance(value['s'], str):
                words.append(value['s'])
            else:
                for item in value.values():
                    visit(item)

    visit(canonical)
    return words


if __name__ == '__main__':
    main()
