#!/usr/bin/env python3
"""Select comparable checkpoints and summarize the unique-data scaling curve."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = TRAINING_ROOT / 'artifacts'
CHECKPOINT_STEPS = (2000, 4000, 8000, 12000, 14124)


@dataclass(frozen=True, slots=True)
class Point:
    label: str
    unique_curated: int
    root: Path
    curated_fraction: float = 0.0
    unique_train: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=ARTIFACTS / 'unique-data-scaling-v1')
    parser.add_argument(
        '--full-exposure-root',
        type=Path,
        default=ARTIFACTS / 'sft-v3-qwen1p5b-rank8-full-exposure-v1',
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--loss-tolerance', type=float, default=0.005)
    parser.add_argument('--material-cases', type=int, default=5)
    return parser.parse_args()


def _load_checkpoint(root: Path, step: int) -> dict[str, object]:
    prefix = root / 'evaluations' / f'step-{step:08d}'
    development = json.loads(prefix.with_suffix('.curated-development.json').read_text())['trained']['overall']
    retention = json.loads(prefix.with_suffix('.retention-analysis.json').read_text())
    loss = json.loads(prefix.with_suffix('.validation-loss.json').read_text())['mean_loss']
    total = int(development['total_examples'])
    return {
        'step': step,
        'validation_loss': float(loss),
        'first_command': round(float(development['first_command_accuracy']) * total),
        'flags': round(float(development['command_flag_sequence_exact_match']) * total),
        'grounded': round(float(development['grounded_document_exact_match']) * total),
        'json': round(float(development['json_parse_rate']) * total),
        'envelope': round(float(development['document_envelope_rate']) * total),
        'retention': retention['candidate']['metrics'],
        'retention_gate': retention['gate'],
    }


def summarize_point(point: Point, *, loss_tolerance: float) -> dict[str, object]:
    checkpoints = [_load_checkpoint(point.root, step) for step in CHECKPOINT_STEPS]
    minimum_loss = min(float(item['validation_loss']) for item in checkpoints)
    eligible = [
        item
        for item in checkpoints
        if float(item['validation_loss']) <= minimum_loss + loss_tolerance
        and item['json'] == 104
        and item['envelope'] == 104
        and item['retention_gate']['passed']  # type: ignore[index]
    ]
    selected = None
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                int(item['grounded']),
                int(item['flags']),
                int(item['first_command']),
                -float(item['validation_loss']),
            ),
        )
    return {
        'label': point.label,
        'root': str(point.root),
        'curated_fraction': point.curated_fraction,
        'unique_curated': point.unique_curated,
        'unique_train': point.unique_train,
        'minimum_validation_loss': minimum_loss,
        'loss_tolerance': loss_tolerance,
        'selected': selected,
        'checkpoints': checkpoints,
    }


def build_summary(
    study_root: Path,
    full_exposure_root: Path,
    *,
    loss_tolerance: float,
    material_cases: int,
) -> dict[str, object]:
    points = (
        Point('25%', 159, study_root / 'fraction-025', 0.25, 23312),
        Point('50%', 319, study_root / 'fraction-050', 0.5, 23472),
        Point('100%', 637, full_exposure_root, 1.0, 23790),
    )
    summaries = [summarize_point(point, loss_tolerance=loss_tolerance) for point in points]
    selected_50 = summaries[1]['selected']
    selected_100 = summaries[2]['selected']
    if selected_50 is None or selected_100 is None:
        decision = {
            'add_200_percent': False,
            'reason': 'at least one comparison point has no checkpoint passing contract, loss, and retention gates',
        }
    else:
        grounded_gain = int(selected_100['grounded']) - int(selected_50['grounded'])  # type: ignore[index]
        decision = {
            'add_200_percent': grounded_gain >= material_cases,
            'grounded_gain_50_to_100': grounded_gain,
            'material_case_threshold': material_cases,
            'adjudicated_correctness_pending': True,
            'reason': (
                'strict grounded slope is materially positive'
                if grounded_gain >= material_cases
                else 'strict grounded slope is below the preregistered material threshold'
            ),
        }
    capacity_recipe = None
    if not decision['add_200_percent']:
        eligible = [summary for summary in summaries if summary['selected'] is not None]
        if eligible:
            winner = max(
                eligible,
                key=lambda summary: (
                    int(summary['selected']['grounded']),  # type: ignore[index]
                    int(summary['selected']['flags']),  # type: ignore[index]
                    int(summary['selected']['first_command']),  # type: ignore[index]
                    -float(summary['selected']['validation_loss']),  # type: ignore[index]
                ),
            )
            capacity_recipe = {
                'label': winner['label'],
                'root': winner['root'],
                'curated_fraction': winner['curated_fraction'],
                'unique_curated': winner['unique_curated'],
                'unique_train': winner['unique_train'],
                'selection_basis': 'strict grounded, flags, first command, then validation loss among gated points',
            }
    return {
        'schema_version': 1,
        'selection_policy': {
            'loss_tolerance': loss_tolerance,
            'hard_contract': '104/104 JSON and envelopes',
            'retention': 'tiered-absolute-v1 pass',
            'behavior_order': ['grounded', 'flags', 'first_command', 'validation_loss'],
        },
        'points': summaries,
        'decision': decision,
        'capacity_recipe': capacity_recipe,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')
    if not args.output.parent.is_dir():
        raise SystemExit(f'output parent does not exist: {args.output.parent}')
    if args.loss_tolerance < 0 or args.material_cases <= 0:
        raise SystemExit('loss tolerance must be non-negative and material cases positive')
    summary = build_summary(
        args.study_root,
        args.full_exposure_root,
        loss_tolerance=args.loss_tolerance,
        material_cases=args.material_cases,
    )
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary['decision'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
