#!/usr/bin/env python3
"""Run the preregistered base-model by LoRA-rank capacity matrix."""

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
CURATED_DEVELOPMENT = ARTIFACTS / 'curated-development-v1.jsonl'
RETENTION = ARTIFACTS / 'semantic-retention-v1.jsonl'
CHECKPOINT_STEPS = (2000, 4000, 8000, 12000, 14124)
CURATED_PRESENTATIONS = 5096
EFFECTIVE_PRESENTATIONS = 28248
SMALL_MODEL = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
LARGE_MODEL = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'


@dataclass(frozen=True, slots=True)
class Configuration:
    label: str
    model_id: str
    rank: int

    @property
    def alpha(self) -> int:
        return self.rank // 2


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]


TRAINED_CONFIGURATIONS = (
    Configuration('qwen0p5b-rank8', SMALL_MODEL, 8),
    Configuration('qwen0p5b-rank32', SMALL_MODEL, 32),
    Configuration('qwen1p5b-rank32', LARGE_MODEL, 32),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, default=ARTIFACTS / 'distributable-semantic-v3.jsonl')
    parser.add_argument('--reference-root', type=Path, required=True, help='Completed matching 1.5B/rank-8 run.')
    parser.add_argument('--curated-data-fraction', type=float, required=True)
    parser.add_argument('--unique-curated', type=int, required=True)
    parser.add_argument('--unique-train', type=int, required=True)
    parser.add_argument('--output-root', type=Path, default=ARTIFACTS / 'capacity-matrix-v1')
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


def _baseline_report(reference_root: Path, configuration: Configuration, *, retention: bool) -> Path:
    suffix = '.retention.json' if retention else '.curated-development.json'
    if configuration.model_id == LARGE_MODEL:
        return (reference_root / 'evaluations' / 'step-00002000').with_suffix(suffix)
    return (reference_root.parent / 'capacity-matrix-small-baseline-placeholder' / 'step-00002000').with_suffix(suffix)


def _evaluation_stages(
    run_root: Path,
    configuration: Configuration,
    *,
    dataset: Path,
    reference_root: Path,
    small_baseline_root: Path,
) -> list[Stage]:
    stages = []
    for step in CHECKPOINT_STEPS:
        checkpoint = run_root / 'checkpoints' / f'step-{step:08d}'
        prefix = run_root / 'evaluations' / f'step-{step:08d}'
        curated = prefix.with_suffix('.curated-development.json')
        retention = prefix.with_suffix('.retention.json')
        retention_analysis = prefix.with_suffix('.retention-analysis.json')
        loss = prefix.with_suffix('.validation-loss.json')
        if configuration.model_id == LARGE_MODEL:
            base_curated = (reference_root / 'evaluations' / 'step-00002000').with_suffix('.curated-development.json')
            base_retention = (reference_root / 'evaluations' / 'step-00002000').with_suffix('.retention.json')
        else:
            base_curated = (small_baseline_root / 'step-00002000').with_suffix('.curated-development.json')
            base_retention = (small_baseline_root / 'step-00002000').with_suffix('.retention.json')
        curated_baseline_args = () if curated == base_curated else ('--baseline-report', str(base_curated))
        retention_baseline_args = () if retention == base_retention else ('--baseline-report', str(base_retention))
        stages.extend(
            [
                Stage(
                    f'{configuration.label}: curated development at {step}',
                    _python(
                        'evaluate_semantic_checkpoint.py',
                        '--semantic-dataset',
                        str(CURATED_DEVELOPMENT),
                        '--model-id',
                        configuration.model_id,
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
                        str(configuration.rank),
                        '--alpha',
                        str(configuration.alpha),
                        '--grounding-audit',
                        str(TRAINING_ROOT / 'evaluation' / 'curated-development-grounding-v1.json'),
                        *curated_baseline_args,
                    ),
                    (curated,),
                ),
                Stage(
                    f'{configuration.label}: retention evaluation at {step}',
                    _python(
                        'evaluate_semantic_checkpoint.py',
                        '--semantic-dataset',
                        str(RETENTION),
                        '--model-id',
                        configuration.model_id,
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
                        str(configuration.rank),
                        '--alpha',
                        str(configuration.alpha),
                        '--priority-source',
                        'shelliq-retention',
                        '--grounding-audit',
                        str(TRAINING_ROOT / 'evaluation' / 'semantic-retention-grounding-v1.json'),
                        *retention_baseline_args,
                    ),
                    (retention,),
                ),
                Stage(
                    f'{configuration.label}: tiered retention analysis at {step}',
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
                    f'{configuration.label}: full validation loss at {step}',
                    _python(
                        'evaluate_corpus_loss.py',
                        '--semantic-dataset',
                        str(dataset),
                        '--model-id',
                        configuration.model_id,
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
                        str(configuration.rank),
                        '--alpha',
                        str(configuration.alpha),
                    ),
                    (loss,),
                ),
            ]
        )
    return stages


def build_stages(
    output_root: Path,
    *,
    dataset: Path,
    reference_root: Path,
    fraction: float,
    unique_curated: int,
    unique_train: int,
) -> tuple[Stage, ...]:
    stages = []
    small_baseline_root = output_root / TRAINED_CONFIGURATIONS[0].label / 'evaluations'
    for configuration in TRAINED_CONFIGURATIONS:
        run_root = output_root / configuration.label
        checkpoints = run_root / 'checkpoints'
        report = run_root / 'training-report.json'
        stages.append(
            Stage(
                f'{configuration.label}: controlled training',
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
                    '2026',
                    '--split-seed',
                    '2026',
                    '--rank',
                    str(configuration.rank),
                    '--alpha',
                    str(configuration.alpha),
                    '--curated-data-fraction',
                    str(fraction),
                    '--curated-presentations',
                    str(CURATED_PRESENTATIONS),
                    '--expected-unique-train-examples',
                    str(unique_train),
                    '--expected-effective-train-examples',
                    str(EFFECTIVE_PRESENTATIONS),
                    '--expected-unique-curated-examples',
                    str(unique_curated),
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
                small_baseline_root=small_baseline_root,
            )
        )
    return tuple(stages)


def _validate_reference(reference_root: Path, *, dataset: Path, unique_curated: int, unique_train: int) -> dict[str, object]:
    path = reference_root / 'training-report.json'
    report = json.loads(path.read_text())
    if report['model_id'] != LARGE_MODEL or report['adapter'] != {'alpha': 4.0, 'rank': 8, 'scale': 0.5}:
        raise ValueError('reference is not the required 1.5B/rank-8/alpha-4 configuration')
    if report['dataset']['sha256'] != _sha256(dataset):
        raise ValueError('reference dataset does not match matrix dataset')
    selection = report['selection']
    if selection['unique_curated_examples'] != unique_curated:
        raise ValueError('reference unique curated count does not match matrix recipe')
    if selection['train_examples'] != unique_train:
        raise ValueError('reference unique training count does not match matrix recipe')
    if selection['effective_train_examples'] != EFFECTIVE_PRESENTATIONS:
        raise ValueError('reference effective presentation count does not match matrix recipe')
    if selection['effective_train_source_counts']['shelliq-curated'] != CURATED_PRESENTATIONS:
        raise ValueError('reference curated presentations do not match matrix recipe')
    if selection['seed'] != 2026 or selection['split_seed'] != 2026:
        raise ValueError('reference data or split seed does not match matrix recipe')
    training = report['training']
    fixed_training = {
        'steps': 14124,
        'batch_size': 2,
        'lr_schedule': 'warmup-cosine',
        'warmup_steps': 1412,
        'shuffle_each_epoch': True,
    }
    if any(training.get(key) != value for key, value in fixed_training.items()):
        raise ValueError('reference training schedule does not match matrix recipe')
    return {
        'path': str(reference_root),
        'training_report_sha256': _sha256(path),
        'train_record_ids_sha256': selection['train_record_ids_sha256'],
    }


def build_manifest(args: argparse.Namespace, stages: tuple[Stage, ...]) -> dict[str, object]:
    reference = _validate_reference(
        args.reference_root,
        dataset=args.semantic_dataset,
        unique_curated=args.unique_curated,
        unique_train=args.unique_train,
    )
    return {
        'schema_version': 1,
        'purpose': 'separate base-model capacity from LoRA capacity',
        'dataset': {'path': str(args.semantic_dataset), 'sha256': _sha256(args.semantic_dataset)},
        'data_recipe': {
            'curated_fraction': args.curated_data_fraction,
            'unique_curated': args.unique_curated,
            'unique_train': args.unique_train,
            'curated_presentations': CURATED_PRESENTATIONS,
            'effective_presentations': EFFECTIVE_PRESENTATIONS,
        },
        'fixed': {
            'alpha_over_rank': 0.5,
            'steps': 14124,
            'seed': 2026,
            'split_seed': 2026,
            'checkpoint_steps': list(CHECKPOINT_STEPS),
        },
        'matrix': [
            {'label': 'qwen0p5b-rank8', 'model_id': SMALL_MODEL, 'rank': 8, 'alpha': 4, 'trained': True},
            {'label': 'qwen0p5b-rank32', 'model_id': SMALL_MODEL, 'rank': 32, 'alpha': 16, 'trained': True},
            {'label': 'qwen1p5b-rank8', 'model_id': LARGE_MODEL, 'rank': 8, 'alpha': 4, 'trained': False, 'reference': reference},
            {'label': 'qwen1p5b-rank32', 'model_id': LARGE_MODEL, 'rank': 32, 'alpha': 16, 'trained': True},
        ],
        'release_evaluation_forbidden': True,
        'stages': [
            {'name': stage.name, 'command': list(stage.command), 'outputs': list(map(str, stage.outputs))} for stage in stages
        ],
    }


def _prepare_manifest(root: Path, manifest: dict[str, object], *, resume: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'manifest.json'
    if path.exists():
        if not resume:
            raise FileExistsError(f'manifest already exists: {path}; use --resume')
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError('existing manifest does not match requested capacity matrix')
        return
    if any(root.iterdir()):
        raise FileExistsError(f'output root is not empty: {root}')
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
    if not 0 < args.curated_data_fraction <= 1 or args.unique_curated <= 0 or args.unique_train <= 0:
        raise SystemExit('fraction and unique counts must be positive, with fraction at most one')
    root = args.output_root.resolve()
    stages = build_stages(
        root,
        dataset=args.semantic_dataset,
        reference_root=args.reference_root,
        fraction=args.curated_data_fraction,
        unique_curated=args.unique_curated,
        unique_train=args.unique_train,
    )
    manifest = build_manifest(args, stages)
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
