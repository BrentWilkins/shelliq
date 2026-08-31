#!/usr/bin/env python3
"""Build a versioned TLDR-only index of Rust-typed command templates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.data import load_semantic_jsonl
from shelliq_training.documentation_templates import documentation_templates
from shelliq_training.semantic_actions import SemanticActionClient

EXPERIMENT = 'documentation-template-index-v1'
SOURCE = 'tldr-pages'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-corpus', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError('documentation index outputs must not already exist')
    raw_by_id: dict[str, dict[str, object]] = {}
    with args.semantic_corpus.open() as source:
        for line_number, line in enumerate(source, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict) or not isinstance(raw.get('record_id'), str):
                raise ValueError(f'{args.semantic_corpus}:{line_number}: invalid semantic record')
            raw_by_id[raw['record_id']] = raw

    records = load_semantic_jsonl(args.semantic_corpus)
    templates = documentation_templates(records)
    documents = [template.document for template in templates]
    client = SemanticActionClient(args.actions)
    encoded = client.encode(documents)
    decoded = client.decode(encoded)
    if not all(item.valid for item in decoded):
        raise ValueError('documentation template failed Rust action round trip')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        for template in templates:
            output.write(json.dumps(raw_by_id[template.record_id], separators=(',', ':'), sort_keys=True) + '\n')
    output_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    revisions = sorted({raw_by_id[template.record_id]['provenance'].split('@', 1)[1].split(':', 1)[0] for template in templates})
    manifest = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'source': SOURCE,
        'source_corpus': str(args.semantic_corpus),
        'source_sha256': hashlib.sha256(args.semantic_corpus.read_bytes()).hexdigest(),
        'output_sha256': output_sha256,
        'records': len(templates),
        'commands': len({template.command for template in templates}),
        'platforms': dict(sorted(Counter(template.platform.value for template in templates).items())),
        'revisions': revisions,
        'rust_action_round_trips': len(decoded),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
