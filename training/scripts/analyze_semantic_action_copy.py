#!/usr/bin/env python3
"""Freeze source-copy oracle evidence for the semantic-action pointer experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.data import format_user_message, load_semantic_jsonl
from shelliq_training.prompt import PromptContract
from shelliq_training.semantic_action_copy import align_action_bytes_to_source, merge_copy_counts
from shelliq_training.semantic_actions import ActionGrammar, SemanticActionClient

EXPERIMENT = 'semantic-action-pointer-v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = load_semantic_jsonl(args.semantic_dataset)
    by_id = {record.record_id: record for record in records}
    client = SemanticActionClient(args.actions)
    grammar = ActionGrammar(client.manifest())
    report_splits = {}
    for split in ('train', 'validation'):
        selected = [by_id[record_id] for record_id in manifest['inner_splits'][split]['record_ids']]
        actions = client.encode([json.loads(record.response) for record in selected])
        alignments = [
            align_action_bytes_to_source(
                format_user_message(record, prompt_contract=PromptContract.CONTEXT_AUTHORITATIVE_V1),
                target,
                grammar,
            )
            for record, target in zip(selected, actions, strict=True)
        ]
        report_splits[split] = {
            'records': len(selected),
            'word_coverage': merge_copy_counts(alignments, 'words'),
            'byte_coverage': merge_copy_counts(alignments, 'bytes'),
        }
    validation = report_splits['validation']['word_coverage']
    command_rate = validation['command_name_bytes']['rate']
    all_rate = validation['all']['rate']
    passed = command_rate >= 0.95 and all_rate >= 0.50
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'phase': 'copy-oracle',
        'gate': {'minimum_command_word_coverage': 0.95, 'minimum_all_word_coverage': 0.50},
        'gate_passed': passed,
        'splits': report_splits,
        'decision': 'implement-pretrained-decoder-pointer-mixture' if passed else 'stop',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'gate_passed': passed, 'command_word_coverage': command_rate, 'all_word_coverage': all_rate}))
    if not passed:
        raise SystemExit('copy oracle gate failed')


if __name__ == '__main__':
    main()
