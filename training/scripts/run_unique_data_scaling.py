#!/usr/bin/env python3
"""Run the preregistered 25%/50%/100% curated unique-data scaling curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = TRAINING_ROOT / 'artifacts'
MODEL_ID = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
DATASET = ARTIFACTS / 'distributable-semantic-v3.jsonl'
CURATED_DEVELOPMENT = ARTIFACTS / 'curated-development-v1.jsonl'
RETENTION = ARTIFACTS / 'semantic-retention-v1.jsonl'
FULL_EXPOSURE = ARTIFACTS / 'sft-v3-qwen1p5b-rank8-full-exposure-v1'
CHECKPOINT_STEPS = (2000, 4000, 8000, 12000, 14124)
CURATED_PRESENTATIONS = 5096
EFFECTIVE_PRESENTATIONS = 28248


@dataclass(frozen=True, slots=True)
class ScalingPoint:
    label: str
    fraction: float
    unique_curated: int

    @property
    def unique_train(self) -> int:
        return 23153 + self.unique_curated


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]


SCALING_POINTS = (
    ScalingPoint('fraction-025', 0.25, 159),
    ScalingPoint('fraction-050', 0.50, 319),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ARTIFACTS / 'unique-data-scaling-v1')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def _python(script: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, str(TRAINING_ROOT / 'scripts' / script), *arguments)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_stages(run_root: Path, point: ScalingPoint) -> list[Stage]:
    stages = []
    for step in CHECKPOINT_STEPS:
        checkpoint = run_root / 'checkpoints' / f'step-{step:08d}'
        prefix = run_root / 'evaluations' / f'step-{step:08d}'
        curated = prefix.with_suffix('.curated-development.json')
        analysis = prefix.with_suffix('.curated-development-analysis.json')
        retention = prefix.with_suffix('.retention.json')
        retention_analysis = prefix.with_suffix('.retention-analysis.json')
        loss = prefix.with_suffix('.validation-loss.json')
        first_prefix = run_root / 'evaluations' / f'step-{CHECKPOINT_STEPS[0]:08d}'
        baseline_args = (
            ()
            if step == CHECKPOINT_STEPS[0]
            else ('--baseline-report', str(first_prefix.with_suffix('.curated-development.json')))
        )
        retention_baseline_args = (
            () if step == CHECKPOINT_STEPS[0] else ('--baseline-report', str(first_prefix.with_suffix('.retention.json')))
        )
        stages.extend(
            [
                Stage(
                    f'{point.label}: curated development at {step}',
                    _python(
                        'evaluate_semantic_checkpoint.py',
                        '--semantic-dataset',
                        str(CURATED_DEVELOPMENT),
                        '--model-id',
                        MODEL_ID,
                        '--checkpoint',
                        str(checkpoint),
                        '--output',
                        str(curated),
                        '--examples',
                        '104',
                        '--selection',
                        'all',
                        '--sequence-length',
                        '448',
                        '--rank',
                        '8',
                        '--alpha',
                        '4',
                        '--grounding-audit',
                        str(TRAINING_ROOT / 'evaluation' / 'curated-development-grounding-v1.json'),
                        *baseline_args,
                    ),
                    (curated,),
                ),
                Stage(
                    f'{point.label}: paired analysis at {step}',
                    _python(
                        'analyze_semantic_errors.py',
                        '--baseline',
                        str(ARTIFACTS / 'curated-development-v1-evaluation' / 'incumbent.eval.json'),
                        '--candidate',
                        str(curated),
                        '--grounding-audit',
                        str(TRAINING_ROOT / 'evaluation' / 'curated-development-grounding-v1.json'),
                        '--output',
                        str(analysis),
                    ),
                    (analysis,),
                ),
                Stage(
                    f'{point.label}: retention evaluation at {step}',
                    _python(
                        'evaluate_semantic_checkpoint.py',
                        '--semantic-dataset',
                        str(RETENTION),
                        '--model-id',
                        MODEL_ID,
                        '--checkpoint',
                        str(checkpoint),
                        '--output',
                        str(retention),
                        '--examples',
                        '20',
                        '--selection',
                        'all',
                        '--sequence-length',
                        '448',
                        '--rank',
                        '8',
                        '--alpha',
                        '4',
                        '--priority-source',
                        'shelliq-retention',
                        '--grounding-audit',
                        str(TRAINING_ROOT / 'evaluation' / 'semantic-retention-grounding-v1.json'),
                        *retention_baseline_args,
                    ),
                    (retention,),
                ),
                Stage(
                    f'{point.label}: tiered retention analysis at {step}',
                    _python(
                        'analyze_semantic_retention.py',
                        '--baseline',
                        str(ARTIFACTS / 'semantic-retention-incumbent-v1.eval.json'),
                        '--candidate',
                        str(retention),
                        '--grounding-audit',
                        str(TRAINING_ROOT / 'evaluation' / 'semantic-retention-grounding-v1.json'),
                        '--output',
                        str(retention_analysis),
                        '--tiered',
                    ),
                    (retention_analysis,),
                ),
                Stage(
                    f'{point.label}: full validation loss at {step}',
                    _python(
                        'evaluate_corpus_loss.py',
                        '--semantic-dataset',
                        str(DATASET),
                        '--model-id',
                        MODEL_ID,
                        '--checkpoint',
                        str(checkpoint),
                        '--output',
                        str(loss),
                        '--split',
                        'validation',
                        '--sequence-length',
                        '384',
                        '--batch-size',
                        '2',
                        '--seed',
                        '2026',
                        '--split-seed',
                        '2026',
                        '--rank',
                        '8',
                        '--alpha',
                        '4',
                    ),
                    (loss,),
                ),
            ]
        )
    return stages


def build_stages(output_root: Path) -> tuple[Stage, ...]:
    stages = []
    for point in SCALING_POINTS:
        run_root = output_root / point.label
        checkpoints = run_root / 'checkpoints'
        report = run_root / 'training-report.json'
        stages.append(
            Stage(
                f'{point.label}: equal-presentation training',
                _python(
                    'smoke_finetune.py',
                    '--semantic-dataset',
                    str(DATASET),
                    '--model-id',
                    MODEL_ID,
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
                    '2026',
                    '--split-seed',
                    '2026',
                    '--rank',
                    '8',
                    '--alpha',
                    '4',
                    '--curated-data-fraction',
                    str(point.fraction),
                    '--curated-presentations',
                    str(CURATED_PRESENTATIONS),
                    '--expected-unique-train-examples',
                    str(point.unique_train),
                    '--expected-effective-train-examples',
                    str(EFFECTIVE_PRESENTATIONS),
                    '--expected-unique-curated-examples',
                    str(point.unique_curated),
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
        stages.extend(_evaluation_stages(run_root, point))
    return tuple(stages)


def build_manifest(stages: tuple[Stage, ...]) -> dict[str, object]:
    required = (DATASET, CURATED_DEVELOPMENT, RETENTION, FULL_EXPOSURE / 'training-report.json')
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('missing scaling inputs: ' + ', '.join(missing))
    return {
        'schema_version': 1,
        'purpose': 'measure command-family-stratified unique curated data scaling',
        'controlled_recipe': {
            'model_id': MODEL_ID,
            'rank': 8,
            'alpha': 4.0,
            'alpha_over_rank': 0.5,
            'curated_presentations': CURATED_PRESENTATIONS,
            'total_presentations': EFFECTIVE_PRESENTATIONS,
            'steps': 14124,
            'batch_size': 2,
            'seed': 2026,
            'split_seed': 2026,
            'checkpoint_steps': list(CHECKPOINT_STEPS),
        },
        'dataset': {'path': str(DATASET), 'sha256': _sha256(DATASET)},
        'points': [
            {
                'label': point.label,
                'fraction': point.fraction,
                'unique_curated': point.unique_curated,
                'unique_train': point.unique_train,
            }
            for point in SCALING_POINTS
        ]
        + [{'label': 'fraction-100', 'fraction': 1.0, 'unique_curated': 637, 'reuse': str(FULL_EXPOSURE)}],
        'release_evaluation_forbidden': True,
        'stages': [
            {'name': stage.name, 'command': list(stage.command), 'outputs': list(map(str, stage.outputs))} for stage in stages
        ],
    }


def _prepare_manifest(output_root: Path, manifest: dict[str, object], *, resume: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / 'manifest.json'
    if path.exists():
        if not resume:
            raise FileExistsError(f'manifest already exists: {path}; use --resume')
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError('existing manifest does not match requested unique-data study')
        return
    if any(output_root.iterdir()):
        raise FileExistsError(f'output root is not empty: {output_root}')
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


def _run(stage: Stage, *, resume: bool) -> None:
    present = tuple(path.exists() for path in stage.outputs)
    if all(present):
        if not resume:
            raise FileExistsError(f'stage output exists: {stage.name}')
        print(f'skip: {stage.name}')
        return
    if any(present):
        raise RuntimeError(f'partial stage outputs: {stage.name}')
    directory_outputs = [path for path in stage.outputs if not path.suffix]
    for parent in {path.parent for path in directory_outputs}:
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    for path in stage.outputs:
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
    print(f'run: {stage.name}', flush=True)
    subprocess.run(stage.command, cwd=TRAINING_ROOT, check=True)


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    stages = build_stages(root)
    manifest = build_manifest(stages)
    if not args.execute:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        for stage in stages:
            print(shlex.join(stage.command))
        return
    _prepare_manifest(root, manifest, resume=args.resume)
    for stage in stages:
        _run(stage, resume=args.resume)


if __name__ == '__main__':
    main()
