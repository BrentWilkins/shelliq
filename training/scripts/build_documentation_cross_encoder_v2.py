#!/usr/bin/env python3
"""Build the fresh v2 test split and zero-logit derived checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_documentation_cross_encoder as v1_builder
import torch

from shelliq_training.documentation_cross_encoder import (
    EXPERIMENT_V2,
    RankCase,
    load_templates,
    ordered_commands,
    paraphrase,
)

EXPECTED_V1_CHECKPOINT = '138b6c79ed36458308fff038672e195a6ccc54c3fd4f0c584200b3e533f15cd4'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--v1-manifest', type=Path, required=True)
    parser.add_argument('--v1-checkpoint', type=Path, required=True)
    parser.add_argument('--test-output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (args.test_output, args.manifest, args.checkpoint)
    if any(path.exists() for path in outputs):
        raise FileExistsError('v2 outputs must not already exist')
    if _sha256(args.v1_checkpoint) != EXPECTED_V1_CHECKPOINT:
        raise ValueError('unexpected v1 checkpoint hash')
    v1_manifest = json.loads(args.v1_manifest.read_text())
    typed_by_id, _, grouped = load_templates(args.documentation_index)
    ordered = ordered_commands(grouped)
    commands = ordered[704:832]
    if len(commands) != 128:
        raise ValueError('v2 requires exactly 128 fresh commands')
    opened = set()
    for split in ('train', 'development', 'test'):
        opened.update(v1_manifest['splits'][split]['commands'])
    if opened & set(commands):
        raise ValueError('v2 command overlaps a v1 split')

    cases: list[RankCase] = []
    for record_index, command in enumerate(commands, start=100_000):
        for record_id in grouped[command]:
            query = v1_builder._complete_query(typed_by_id[record_id], record_index)
            if query is not None:
                cases.append(RankCase(command, record_id, paraphrase(query)))
                break
        else:
            raise ValueError(f'v2 command has no bindable template: {command}')
    v1_builder._write_jsonl(args.test_output, cases)

    manifest = {
        'schema_version': 1,
        'experiment': EXPERIMENT_V2,
        'documentation_index': str(args.documentation_index),
        'documentation_index_sha256': _sha256(args.documentation_index),
        'selection': 'v1 eligible hash order positions 704 through 831',
        'v1_manifest_sha256': _sha256(args.v1_manifest),
        'splits': {
            'test': {
                'commands': list(commands),
                'command_count': len(commands),
                'records': len(cases),
                'sha256': _sha256(args.test_output),
            }
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    checkpoint = torch.load(args.v1_checkpoint, map_location='cpu', weights_only=True)
    checkpoint['experiment'] = EXPERIMENT_V2
    checkpoint['threshold'] = 0.0
    checkpoint['manifest_sha256'] = _sha256(args.manifest)
    checkpoint['source_checkpoint_sha256'] = EXPECTED_V1_CHECKPOINT
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    print(
        json.dumps(
            {
                **manifest,
                'checkpoint_sha256': _sha256(args.checkpoint),
                'threshold': checkpoint['threshold'],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    main()
