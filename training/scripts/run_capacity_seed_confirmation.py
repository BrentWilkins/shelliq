#!/usr/bin/env python3
"""Run seeds 2027 and 2028 for at most two capacity-matrix leaders."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_capacity_matrix import (
    ARTIFACTS,
    CHECKPOINT_STEPS,
    CURATED_PRESENTATIONS,
    EFFECTIVE_PRESENTATIONS,
    LARGE_MODEL,
    SMALL_MODEL,
    Configuration,
    Stage,
    _evaluation_stages,
    _prepare_manifest,
    _python,
    _run,
)

CONFIGURATIONS = {
    'qwen0p5b-rank8': Configuration('qwen0p5b-rank8', SMALL_MODEL, 8),
    'qwen0p5b-rank32': Configuration('qwen0p5b-rank32', SMALL_MODEL, 32),
    'qwen1p5b-rank8': Configuration('qwen1p5b-rank8', LARGE_MODEL, 8),
    'qwen1p5b-rank32': Configuration('qwen1p5b-rank32', LARGE_MODEL, 32),
}
CONFIRMATION_SEEDS = (2027, 2028)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--matrix-root', type=Path, default=ARTIFACTS / 'capacity-matrix-v1')
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=ARTIFACTS / 'capacity-seed-confirmation-v1')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def load_design(analysis_path: Path, matrix_root: Path) -> dict[str, object]:
    analysis = json.loads(analysis_path.read_text())
    leaders = analysis.get('seed_confirmation_leaders')
    if not isinstance(leaders, list) or not 1 <= len(leaders) <= 2 or len(set(leaders)) != len(leaders):
        raise ValueError('capacity analysis must name one or two unique seed-confirmation leaders')
    if any(label not in CONFIGURATIONS for label in leaders):
        raise ValueError('capacity analysis names an unknown configuration')
    matrix = json.loads((matrix_root / 'manifest.json').read_text())
    recipe = matrix['data_recipe']
    return {
        'leaders': leaders,
        'dataset': matrix['dataset'],
        'data_recipe': recipe,
        'split_seed': matrix['fixed']['split_seed'],
        'source_analysis': str(analysis_path),
    }


def build_stages(
    output_root: Path,
    *,
    design: dict[str, object],
    matrix_root: Path,
    reference_root: Path,
) -> tuple[Stage, ...]:
    recipe = design['data_recipe']
    assert isinstance(recipe, dict)
    dataset = Path(str(design['dataset']['path']))  # type: ignore[index]
    split_seed = int(design['split_seed'])
    stages: list[Stage] = []
    for label in design['leaders']:  # type: ignore[union-attr]
        configuration = CONFIGURATIONS[str(label)]
        for seed in CONFIRMATION_SEEDS:
            run_root = output_root / configuration.label / f'seed-{seed}'
            checkpoints = run_root / 'checkpoints'
            report = run_root / 'training-report.json'
            stages.append(
                Stage(
                    f'{configuration.label}/seed-{seed}: controlled training',
                    _python(
                        'smoke_finetune.py',
                        '--semantic-dataset',
                        str(dataset),
                        '--model-id',
                        configuration.model_id,
                        '--sequence-length',
                        '384',
                        '--all-train-examples',
                        '--train-loss-eval-examples',
                        '256',
                        '--eval-examples',
                        '256',
                        '--batch-size',
                        '2',
                        '--steps',
                        '14124',
                        '--eval-interval',
                        '2000',
                        '--gradient-diagnostics',
                        '--shuffle-each-epoch',
                        '--lr-schedule',
                        'warmup-cosine',
                        '--warmup-steps',
                        '1412',
                        '--learning-rate',
                        '0.0001',
                        '--end-learning-rate',
                        '0.00001',
                        '--seed',
                        str(seed),
                        '--split-seed',
                        str(split_seed),
                        '--rank',
                        str(configuration.rank),
                        '--alpha',
                        str(configuration.alpha),
                        '--curated-data-fraction',
                        str(recipe['curated_fraction']),
                        '--curated-presentations',
                        str(recipe['curated_presentations']),
                        '--expected-unique-train-examples',
                        str(recipe['unique_train']),
                        '--expected-effective-train-examples',
                        str(recipe['effective_presentations']),
                        '--expected-unique-curated-examples',
                        str(recipe['unique_curated']),
                        '--checkpoint-interval-root',
                        str(checkpoints),
                        '--retain-checkpoint-steps',
                        ','.join(map(str, CHECKPOINT_STEPS)),
                        '--report',
                        str(report),
                    ),
                    (report, *(checkpoints / f'step-{step:08d}' for step in CHECKPOINT_STEPS)),
                )
            )
            stages.extend(
                _evaluation_stages(
                    run_root,
                    configuration,
                    dataset=dataset,
                    reference_root=reference_root,
                    small_baseline_root=matrix_root / 'qwen0p5b-rank8' / 'evaluations',
                )
            )
    return tuple(stages)


def build_manifest(design: dict[str, object], stages: tuple[Stage, ...]) -> dict[str, object]:
    return {
        'schema_version': 1,
        'purpose': 'confirm at most two capacity-matrix leaders across three total seeds',
        'leaders': design['leaders'],
        'training_seeds': [2026, *CONFIRMATION_SEEDS],
        'new_training_seeds': list(CONFIRMATION_SEEDS),
        'split_seed': design['split_seed'],
        'dataset': design['dataset'],
        'data_recipe': design['data_recipe'],
        'source_analysis': design['source_analysis'],
        'hard_gates_apply_per_seed': True,
        'release_evaluation_forbidden': True,
        'stages': [
            {'name': stage.name, 'command': list(stage.command), 'outputs': list(map(str, stage.outputs))} for stage in stages
        ],
    }


def main() -> None:
    args = parse_args()
    try:
        design = load_design(args.analysis, args.matrix_root)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    recipe = design['data_recipe']
    assert isinstance(recipe, dict)
    if (
        int(recipe['curated_presentations']) != CURATED_PRESENTATIONS
        or int(recipe['effective_presentations']) != EFFECTIVE_PRESENTATIONS
    ):
        raise SystemExit('matrix recipe does not preserve preregistered presentation counts')
    stages = build_stages(
        args.output_root.resolve(),
        design=design,
        matrix_root=args.matrix_root.resolve(),
        reference_root=args.reference_root.resolve(),
    )
    manifest = build_manifest(design, stages)
    if not args.execute:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        for stage in stages:
            print(shlex.join(stage.command))
        return
    _prepare_manifest(args.output_root.resolve(), manifest, resume=args.resume)
    for stage in stages:
        _run(stage, resume=args.resume)


if __name__ == '__main__':
    main()
