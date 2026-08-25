#!/usr/bin/env python3
"""Evaluate preregistered candidate-only semantic-action beam decoding."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_semantic_action_candidate as candidate
import run_semantic_action_decoder as v1
import torch

EXPERIMENT = 'semantic-action-candidate-beam-v1'
SOURCE_EXPERIMENT = 'semantic-action-candidate-v1'
BEAM_WIDTH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--outer-manifest', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--copy-oracle', type=Path, required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, _, examples, tokenizer, client, grammar, manifest = candidate.load_inputs(args)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('experiment') != SOURCE_EXPERIMENT:
        raise ValueError('candidate checkpoint belongs to another experiment')
    metadata = checkpoint.get('metadata', {})
    if metadata.get('best_epoch') != 17:
        raise ValueError('candidate checkpoint is not the frozen epoch-17 inner selection')
    device = v1._device(args.device)
    model = candidate.build_model(tokenizer).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    by_record = {record.record_id: record for record in records}
    by_example = {example.record_id: example for group in examples.values() for example in group}
    ids = manifest['inner_splits']['validation']['record_ids']
    evaluation_records = [by_record[item] for item in ids]
    evaluation_examples = [by_example[item] for item in ids]
    started = time.monotonic()
    generated = []
    with torch.no_grad():
        for index, example in enumerate(evaluation_examples, start=1):
            batch = candidate.make_batch([example], tokenizer).to(device)
            generated.extend(model.generate_beam(batch, grammar, beam_width=BEAM_WIDTH))
            if index % 10 == 0:
                print(f'decoded={index}/{len(evaluation_examples)}', flush=True)
    metrics, outcomes = v1._evaluate(evaluation_records, evaluation_examples, generated, client)
    count = len(evaluation_examples)
    passed = (
        metrics['rust_valid'] / count >= 0.90
        and metrics['first_command_match'] / count >= 0.80
        and metrics['reference_accepted'] / count >= 0.05
    )
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
        'source_experiment': SOURCE_EXPERIMENT,
        'phase': 'inner',
        'gate_passed': passed,
        'beam_width': BEAM_WIDTH,
        'candidate_only': True,
        'score': 'sum-normalized-decision-log-probability',
        'evaluation_records': count,
        'elapsed_seconds': time.monotonic() - started,
        'metrics': metrics,
        'outcomes': outcomes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'gate_passed': passed, **metrics}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit('candidate beam inner gate failed')


if __name__ == '__main__':
    main()
