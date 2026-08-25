#!/usr/bin/env python3
"""Preregister and run the full-epoch 1.5B/rank-8 exposure diagnosis."""

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
CHECKPOINT_STEPS = (2000, 4000, 8000, 12000, 14124)
EFFECTIVE_TRAIN_EXAMPLES = 28248
UNIQUE_TRAIN_EXAMPLES = 23790
UNIQUE_CURATED_EXAMPLES = 637


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root',
        type=Path,
        default=ARTIFACTS / 'sft-v3-qwen1p5b-rank8-full-exposure-v1',
    )
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _python(script: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, str(TRAINING_ROOT / 'scripts' / script), *arguments)


def build_stages(output_root: Path) -> tuple[Stage, ...]:
    checkpoints = output_root / 'checkpoints'
    training_report = output_root / 'training-report.json'
    stages = [
        Stage(
            'train one complete effective epoch',
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
                '--rehearsal',
                'curated:=8',
                '--expected-unique-train-examples',
                str(UNIQUE_TRAIN_EXAMPLES),
                '--expected-effective-train-examples',
                str(EFFECTIVE_TRAIN_EXAMPLES),
                '--expected-unique-curated-examples',
                str(UNIQUE_CURATED_EXAMPLES),
                '--checkpoint-interval-root',
                str(checkpoints),
                '--retain-checkpoint-steps',
                ','.join(map(str, CHECKPOINT_STEPS)),
                '--report',
                str(training_report),
            ),
            (training_report, *(checkpoints / f'step-{step:08d}' for step in CHECKPOINT_STEPS)),
        )
    ]

    for step in CHECKPOINT_STEPS:
        checkpoint = checkpoints / f'step-{step:08d}'
        prefix = output_root / 'evaluations' / f'step-{step:08d}'
        curated_eval = prefix.with_suffix('.curated-development.json')
        curated_analysis = prefix.with_suffix('.curated-development-analysis.json')
        retention_eval = prefix.with_suffix('.retention.json')
        retention_analysis = prefix.with_suffix('.retention-analysis.json')
        validation_loss = prefix.with_suffix('.validation-loss.json')
        stages.extend(
            (
                Stage(
                    f'curated development at step {step}',
                    _python(
                        'evaluate_semantic_checkpoint.py',
                        '--semantic-dataset',
                        str(CURATED_DEVELOPMENT),
                        '--model-id',
                        MODEL_ID,
                        '--checkpoint',
                        str(checkpoint),
                        '--output',
                        str(curated_eval),
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
                        *(
                            ()
                            if step == CHECKPOINT_STEPS[0]
                            else (
                                '--baseline-report',
                                str(
                                    (output_root / 'evaluations' / f'step-{CHECKPOINT_STEPS[0]:08d}').with_suffix(
                                        '.curated-development.json'
                                    )
                                ),
                            )
                        ),
                    ),
                    (curated_eval,),
                ),
                Stage(
                    f'paired curated analysis at step {step}',
                    _python(
                        'analyze_semantic_errors.py',
                        '--baseline',
                        str(ARTIFACTS / 'curated-development-v1-evaluation' / 'incumbent.eval.json'),
                        '--candidate',
                        str(curated_eval),
                        '--grounding-audit',
                        str(TRAINING_ROOT / 'evaluation' / 'curated-development-grounding-v1.json'),
                        '--output',
                        str(curated_analysis),
                    ),
                    (curated_analysis,),
                ),
                Stage(
                    f'tiered retention evaluation at step {step}',
                    _python(
                        'evaluate_semantic_checkpoint.py',
                        '--semantic-dataset',
                        str(RETENTION),
                        '--model-id',
                        MODEL_ID,
                        '--checkpoint',
                        str(checkpoint),
                        '--output',
                        str(retention_eval),
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
                        *(
                            ()
                            if step == CHECKPOINT_STEPS[0]
                            else (
                                '--baseline-report',
                                str(
                                    (output_root / 'evaluations' / f'step-{CHECKPOINT_STEPS[0]:08d}').with_suffix(
                                        '.retention.json'
                                    )
                                ),
                            )
                        ),
                    ),
                    (retention_eval,),
                ),
                Stage(
                    f'tiered retention analysis at step {step}',
                    _python(
                        'analyze_semantic_retention.py',
                        '--baseline',
                        str(ARTIFACTS / 'semantic-retention-incumbent-v1.eval.json'),
                        '--candidate',
                        str(retention_eval),
                        '--grounding-audit',
                        str(TRAINING_ROOT / 'evaluation' / 'semantic-retention-grounding-v1.json'),
                        '--output',
                        str(retention_analysis),
                        '--tiered',
                    ),
                    (retention_analysis,),
                ),
                Stage(
                    f'full validation loss at step {step}',
                    _python(
                        'evaluate_corpus_loss.py',
                        '--semantic-dataset',
                        str(DATASET),
                        '--model-id',
                        MODEL_ID,
                        '--checkpoint',
                        str(checkpoint),
                        '--output',
                        str(validation_loss),
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
                    (validation_loss,),
                ),
            )
        )
    return tuple(stages)


def build_manifest(stages: tuple[Stage, ...]) -> dict[str, object]:
    required = (DATASET, CURATED_DEVELOPMENT, RETENTION)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('missing study inputs: ' + ', '.join(missing))
    return {
        'schema_version': 1,
        'purpose': 'distinguish premature stopping from a genuine 1.5B/rank-8 plateau',
        'model_id': MODEL_ID,
        'adapter': {'rank': 8, 'alpha': 4.0, 'alpha_over_rank': 0.5},
        'data': {
            'dataset': str(DATASET),
            'dataset_sha256': _sha256(DATASET),
            'unique_train_examples': UNIQUE_TRAIN_EXAMPLES,
            'unique_curated_examples': UNIQUE_CURATED_EXAMPLES,
            'curated_weight': 8,
            'effective_train_examples': EFFECTIVE_TRAIN_EXAMPLES,
        },
        'training': {
            'batch_size': 2,
            'steps': EFFECTIVE_TRAIN_EXAMPLES // 2,
            'presentations': EFFECTIVE_TRAIN_EXAMPLES,
            'checkpoint_steps': list(CHECKPOINT_STEPS),
            'schedule': {'name': 'warmup-cosine', 'warmup_steps': 1412, 'peak_lr': 1e-4, 'end_lr': 1e-5},
            'seed': 2026,
            'split_seed': 2026,
        },
        'selection': {
            'primary': [
                'human_adjudicated_correctness',
                'grounded_document_exact_match',
                'command_flag_sequence_exact_match',
                'first_command_accuracy',
            ],
            'retention_gate': {'json': 20, 'envelope': 20, 'first_command': 19, 'flags': 13, 'grounded': 9},
            'release_evaluation_forbidden': True,
        },
        'stages': [
            {'name': stage.name, 'command': list(stage.command), 'outputs': list(map(str, stage.outputs))} for stage in stages
        ],
    }


def _prepare_manifest(output_root: Path, manifest: dict[str, object], *, resume: bool) -> None:
    manifest_path = output_root / 'manifest.json'
    output_root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f'manifest already exists: {manifest_path}; use --resume')
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('existing manifest does not match the requested exposure study')
        return
    if any(output_root.iterdir()):
        raise FileExistsError(f'output root is not empty: {output_root}')
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


def _run_stage(stage: Stage, *, resume: bool) -> None:
    present = tuple(path.exists() for path in stage.outputs)
    if all(present):
        if not resume:
            raise FileExistsError(f'stage output already exists: {stage.name}')
        print(f'skip: {stage.name}')
        return
    if any(present):
        raise RuntimeError(f'stage has partial outputs and cannot safely resume: {stage.name}')
    _prepare_stage_output_parents(stage)
    print(f'run: {stage.name}', flush=True)
    subprocess.run(stage.command, cwd=TRAINING_ROOT, check=True)


def _prepare_stage_output_parents(stage: Stage) -> None:
    """Create file parents without pre-creating checkpoint roots.

    A prior runner version created the parent of every declared checkpoint directory,
    leaving an empty root that the trainer correctly rejected. Remove only that exact
    empty parent so ``--resume`` can recover from the orchestration failure.
    """
    directory_outputs = [output for output in stage.outputs if not output.suffix]
    for parent in {output.parent for output in directory_outputs}:
        siblings = [output for output in directory_outputs if output.parent == parent]
        if parent.is_dir() and not any(parent.iterdir()) and not any(output.exists() for output in siblings):
            parent.rmdir()
    for output in stage.outputs:
        if output.suffix:
            output.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    stages = build_stages(output_root)
    manifest = build_manifest(stages)
    if not args.execute:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        for stage in stages:
            print(shlex.join(stage.command))
        return
    _prepare_manifest(output_root, manifest, resume=args.resume)
    for stage in stages:
        _run_stage(stage, resume=args.resume)


if __name__ == '__main__':
    main()
