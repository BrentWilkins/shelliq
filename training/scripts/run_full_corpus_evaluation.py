#!/usr/bin/env python3
"""Evaluate full-corpus LoRA finalists without consulting final holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from shelliq_training.semantic_error_analysis import (  # noqa: E402
    PromotionThresholds,
    gate_metrics,
)

ARTIFACTS = TRAINING_ROOT / 'artifacts'
DATASET = ARTIFACTS / 'distributable-semantic-v2.jsonl'
TRAINING_RUN_ROOT = ARTIFACTS / 'lora-full-corpus-finalists-v1'


@dataclass(frozen=True, slots=True)
class Finalist:
    name: str
    rank: int
    alpha: float

    @property
    def checkpoint(self) -> Path:
        return TRAINING_RUN_ROOT / self.name / 'best-checkpoint' / 'step-00010000'


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]


FINALISTS = (
    Finalist('rank-8_scale-0p5_lr-2e-04_seed-2026', rank=8, alpha=4.0),
    Finalist('rank-16_scale-1_lr-5e-05_seed-2026', rank=16, alpha=16.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root',
        type=Path,
        default=ARTIFACTS / 'lora-full-corpus-finalists-v1-evaluation',
    )
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


def build_plan(finalist: Finalist, output_root: Path) -> tuple[Stage, ...]:
    root = output_root / finalist.name
    adapter = ('--rank', str(finalist.rank), '--alpha', str(finalist.alpha))
    test_loss = root / 'corpus-test-loss.json'
    release_eval = root / 'release.eval.json'
    release_analysis = root / 'release.analysis.json'
    retention_eval = root / 'retention.eval.json'
    retention_analysis = root / 'retention.analysis.json'
    return (
        Stage(
            'corpus test loss',
            _python(
                'evaluate_corpus_loss.py',
                '--semantic-dataset',
                str(DATASET),
                '--checkpoint',
                str(finalist.checkpoint),
                '--output',
                str(test_loss),
                '--split',
                'test',
                '--sequence-length',
                '384',
                '--batch-size',
                '2',
                '--seed',
                '2026',
                '--split-seed',
                '2026',
                *adapter,
            ),
            (test_loss,),
        ),
        Stage(
            'release development evaluation',
            _python(
                'evaluate_semantic_checkpoint.py',
                '--semantic-dataset',
                str(DATASET),
                '--checkpoint',
                str(finalist.checkpoint),
                '--output',
                str(release_eval),
                '--selection',
                'test',
                '--sequence-length',
                '384',
                '--priority-source',
                'shelliq-curated',
                '--grounding-audit',
                str(TRAINING_ROOT / 'evaluation' / 'curated-grounding-v1.json'),
                *adapter,
            ),
            (release_eval,),
        ),
        Stage(
            'release development analysis',
            _python(
                'analyze_semantic_errors.py',
                '--baseline',
                str(ARTIFACTS / 'semantic-authoritative-curated-v3.eval.json'),
                '--candidate',
                str(release_eval),
                '--grounding-audit',
                str(TRAINING_ROOT / 'evaluation' / 'curated-grounding-v1.json'),
                '--output',
                str(release_analysis),
            ),
            (release_analysis,),
        ),
        Stage(
            'retention evaluation',
            _python(
                'evaluate_semantic_checkpoint.py',
                '--semantic-dataset',
                str(ARTIFACTS / 'semantic-retention-v1.jsonl'),
                '--checkpoint',
                str(finalist.checkpoint),
                '--output',
                str(retention_eval),
                '--selection',
                'all',
                '--sequence-length',
                '448',
                '--priority-source',
                'shelliq-retention',
                '--grounding-audit',
                str(TRAINING_ROOT / 'evaluation' / 'semantic-retention-grounding-v1.json'),
                *adapter,
            ),
            (retention_eval,),
        ),
        Stage(
            'retention analysis',
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
            ),
            (retention_analysis,),
        ),
    )


def build_manifest(output_root: Path) -> dict[str, object]:
    inputs = (
        DATASET,
        ARTIFACTS / 'semantic-retention-v1.jsonl',
        ARTIFACTS / 'semantic-authoritative-curated-v3.eval.json',
        ARTIFACTS / 'semantic-retention-incumbent-v1.eval.json',
        TRAINING_ROOT / 'evaluation' / 'curated-grounding-v1.json',
        TRAINING_ROOT / 'evaluation' / 'semantic-retention-grounding-v1.json',
        *(finalist.checkpoint / 'metadata' / 'metadata' for finalist in FINALISTS),
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing evaluation inputs: {missing}')
    return {
        'schema_version': 1,
        'purpose': 'select between full-corpus SFT finalists',
        'selection_order': ['corpus test loss', 'release development', 'retention'],
        'forbidden_evaluations': ['pipeline', 'shadow'],
        'inputs': {str(path): _sha256(path) for path in inputs},
        'finalists': [
            {
                'name': finalist.name,
                'rank': finalist.rank,
                'alpha': finalist.alpha,
                'checkpoint': str(finalist.checkpoint),
                'stages': [
                    {
                        'name': stage.name,
                        'command': list(stage.command),
                        'outputs': [str(output) for output in stage.outputs],
                    }
                    for stage in build_plan(finalist, output_root)
                ],
            }
            for finalist in FINALISTS
        ],
    }


def _run_stage(stage: Stage, *, resume: bool, console: Console) -> None:
    present = [output.exists() for output in stage.outputs]
    if all(present):
        if not resume:
            raise FileExistsError(f'{stage.name} outputs already exist: {stage.outputs}')
        console.print(f'[dim]skip {stage.name}[/dim]')
        return
    if any(present):
        raise RuntimeError(f'{stage.name} has incomplete outputs: {stage.outputs}')
    for output in stage.outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    console.rule(stage.name)
    subprocess.run(stage.command, cwd=TRAINING_ROOT, check=True)


def _print_results(output_root: Path, console: Console) -> None:
    table = Table(title='Full-corpus finalist evaluation')
    for heading in (
        'rank',
        'test loss',
        'release flags',
        'release grounded',
        'release gate',
        'retention flags',
        'retention grounded',
        'retention gate',
    ):
        table.add_column(heading, justify='right')
    for finalist in FINALISTS:
        root = output_root / finalist.name
        needed = (
            root / 'corpus-test-loss.json',
            root / 'release.eval.json',
            root / 'retention.eval.json',
        )
        if not all(path.is_file() for path in needed):
            continue
        test_loss = json.loads(needed[0].read_text())['mean_loss']
        release = json.loads(needed[1].read_text())['trained']['overall']
        retention = json.loads(needed[2].read_text())['trained']['overall']
        release_passed = not gate_metrics(release, PromotionThresholds())
        retention_analysis = json.loads((root / 'retention.analysis.json').read_text())
        table.add_row(
            str(finalist.rank),
            f'{test_loss:.4f}',
            f'{release["command_flag_sequence_exact_match"]:.3f}',
            f'{release["grounded_document_exact_match"]:.3f}',
            'pass' if release_passed else 'FAIL',
            f'{retention["command_flag_sequence_exact_match"]:.3f}',
            f'{retention["grounded_document_exact_match"]:.3f}',
            'pass' if retention_analysis['gate']['passed'] else 'FAIL',
        )
    console.print(table)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    manifest = build_manifest(output_root)
    manifest_path = output_root / 'manifest.json'
    console = Console(stderr=True)

    if not args.execute:
        console.print_json(json.dumps(manifest))
        for finalist in FINALISTS:
            for stage in build_plan(finalist, output_root):
                print(shlex.join(stage.command))
        return

    if args.resume:
        if not manifest_path.is_file():
            raise SystemExit(f'cannot resume without manifest: {manifest_path}')
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit('existing manifest does not match requested evaluation')
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    for finalist in FINALISTS:
        console.rule(finalist.name)
        for stage in build_plan(finalist, output_root):
            _run_stage(stage, resume=args.resume, console=console)
    _print_results(output_root, console)


if __name__ == '__main__':
    main()
