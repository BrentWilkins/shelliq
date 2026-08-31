#!/usr/bin/env python3
"""Compile one request from typed documentation, or report why it must abstain."""

from __future__ import annotations

import argparse
import hashlib
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

INDEX_EXPERIMENT = 'documentation-template-index-v2'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--index-manifest', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--command', required=True)
    parser.add_argument('--platform', choices=tuple(item.value for item in Platform), required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--context', default='')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.index_manifest.read_text())
    if manifest.get('experiment') != INDEX_EXPERIMENT:
        raise ValueError('unexpected documentation index manifest')
    if manifest.get('output_sha256') != hashlib.sha256(args.documentation_index.read_bytes()).hexdigest():
        raise ValueError('documentation index hash mismatch')
    records = load_semantic_jsonl(args.documentation_index)
    templates = documentation_templates(records)
    if len(templates) != manifest.get('records') or len(templates) != len(records):
        raise ValueError('documentation index integrity mismatch')
    compilation = compile_documented_command_scored(
        DocumentationTemplateIndex(templates),
        command=args.command,
        platform=Platform(args.platform),
        instruction=args.instruction,
        context=args.context,
    )

    rendered = None
    rust_valid = None
    if compilation.document is not None:
        client = SemanticActionClient(args.actions)
        decoded = client.decode(client.encode([compilation.document]))[0]
        rust_valid = decoded.valid
        if not decoded.valid:
            raise ValueError(f'compiled document failed Rust round trip: {decoded.error}')
        if compilation.status == 'ready':
            rendered = decoded.rendered
    result = {
        'schema_version': 1,
        'status': compilation.status,
        'document': compilation.document,
        'rendered': rendered,
        'rust_valid': rust_valid,
        'source_record_ids': list(compilation.source_record_ids),
        'selected_options': list(compilation.selected_options),
        'bindings': list(compilation.bindings),
        'unresolved_slots': list(compilation.unresolved_slots),
        'word_provenance': [{'word': item.word, 'source': item.source} for item in compilation.word_provenance],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
