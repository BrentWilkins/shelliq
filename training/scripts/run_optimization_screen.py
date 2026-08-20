#!/usr/bin/env python3
"""Screen LoRA rank, scale, and learning rate using corpus held-out loss only."""

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
ARTIFACTS = TRAINING_ROOT / 'artifacts'
DATASET = ARTIFACTS / 'distributable-semantic-v2.jsonl'
DEFAULT_RANKS = (4, 8, 16, 32, 64)
DEFAULT_SCALE_RATIOS = (1, 2)
DEFAULT_LEARNING_RATES = (1e-4, 2e-4, 4e-4)


@dataclass(frozen=True, slots=True)
class ScreenRun:
    rank: int
    scale_ratio: int
    learning_rate: float
    command: tuple[str, ...]
    report: Path

    @property
    def name(self) -> str:
        return f'rank-{self.rank}_scale-{self.scale_ratio}_lr-{self.learning_rate:.0e}'


def _positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('values must be comma-separated integers') from error
    if not values or any(item <= 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('values must be unique positive integers')
    return values


def _positive_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('values must be comma-separated numbers') from error
    if not values or any(item <= 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('values must be unique positive numbers')
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root',
        type=Path,
        default=ARTIFACTS / 'lora-optimization-screen-v1',
    )
    parser.add_argument('--ranks', type=_positive_ints, default=DEFAULT_RANKS)
    parser.add_argument(
        '--scale-ratios',
        type=_positive_ints,
        default=DEFAULT_SCALE_RATIOS,
        help='Comma-separated alpha/rank ratios.',
    )
    parser.add_argument(
        '--learning-rates',
        type=_positive_floats,
        default=DEFAULT_LEARNING_RATES,
    )
    parser.add_argument('--steps', type=int, default=750)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_runs(
    *,
    ranks: tuple[int, ...],
    scale_ratios: tuple[int, ...],
    learning_rates: tuple[float, ...],
    steps: int,
    output_root: Path,
) -> tuple[ScreenRun, ...]:
    runs = []
    for rank in ranks:
        for scale_ratio in scale_ratios:
            alpha = rank * scale_ratio
            for learning_rate in learning_rates:
                name = f'rank-{rank}_scale-{scale_ratio}_lr-{learning_rate:.0e}'
                report = output_root / name / 'report.json'
                command = (
                    sys.executable,
                    str(TRAINING_ROOT / 'scripts' / 'smoke_finetune.py'),
                    '--semantic-dataset',
                    str(DATASET),
                    '--sequence-length',
                    '384',
                    '--train-examples',
                    '4000',
                    '--eval-examples',
                    '256',
                    '--batch-size',
                    '2',
                    '--steps',
                    str(steps),
                    '--learning-rate',
                    str(learning_rate),
                    '--seed',
                    '2026',
                    '--priority-source',
                    'shelliq-curated',
                    '--rank',
                    str(rank),
                    '--alpha',
                    str(alpha),
                    '--report',
                    str(report),
                )
                runs.append(ScreenRun(rank, scale_ratio, learning_rate, command, report))
    return tuple(runs)


def build_manifest(args: argparse.Namespace, runs: tuple[ScreenRun, ...]) -> dict[str, object]:
    if not DATASET.is_file():
        raise FileNotFoundError(f'missing optimization-screen dataset: {DATASET}')
    return {
        'schema_version': 1,
        'purpose': 'short-budget optimization screen using corpus held-out loss only',
        'selection_metric': 'final_heldout_loss',
        'dataset': {'path': str(DATASET), 'sha256': _sha256(DATASET)},
        'seed': 2026,
        'steps': args.steps,
        'ranks': list(args.ranks),
        'alpha_over_rank': list(args.scale_ratios),
        'learning_rates': list(args.learning_rates),
        'forbidden_evaluations': ['release', 'retention', 'pipeline', 'shadow'],
        'runs': [
            {
                'name': run.name,
                'rank': run.rank,
                'alpha_over_rank': run.scale_ratio,
                'alpha': run.rank * run.scale_ratio,
                'learning_rate': run.learning_rate,
                'command': list(run.command),
                'report': str(run.report),
            }
            for run in runs
        ],
    }


def _prepare_manifest(
    output_root: Path,
    manifest: dict[str, object],
    *,
    execute: bool,
    resume: bool,
) -> None:
    manifest_path = output_root / 'manifest.json'
    if not execute:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise RuntimeError(f'existing manifest does not match requested screen: {manifest_path}')
        if not resume:
            raise FileExistsError(f'screen already exists; pass --resume: {output_root}')
        return
    if any(output_root.iterdir()):
        raise FileExistsError(f'output root is not empty: {output_root}')
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


def _run_one(run: ScreenRun, *, resume: bool, console: Console) -> None:
    if run.report.exists():
        if resume:
            console.print(f'[dim]skip complete: {run.name}[/dim]')
            return
        raise FileExistsError(f'report already exists: {run.report}')
    run.report.parent.mkdir(parents=True, exist_ok=True)
    console.rule(run.name)
    subprocess.run(run.command, cwd=TRAINING_ROOT, check=True)


def _print_results(runs: tuple[ScreenRun, ...], console: Console) -> None:
    complete = [run for run in runs if run.report.is_file()]
    table = Table(title=f'Optimization screen ({len(complete)}/{len(runs)} complete)')
    table.add_column('rank', justify='right')
    table.add_column('alpha/rank', justify='right')
    table.add_column('LR', justify='right')
    table.add_column('train loss', justify='right')
    table.add_column('held-out loss', justify='right')
    rows = []
    for run in complete:
        training = json.loads(run.report.read_text())['training']
        rows.append((float(training['final_heldout_loss']), run, training))
    for _, run, training in sorted(rows, key=lambda item: item[0]):
        table.add_row(
            str(run.rank),
            str(run.scale_ratio),
            f'{run.learning_rate:.0e}',
            f'{training["final_train_loss"]:.4f}',
            f'{training["final_heldout_loss"]:.4f}',
        )
    console.print(table)


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit('--steps must be positive')
    output_root = args.output_root.resolve()
    runs = build_runs(
        ranks=args.ranks,
        scale_ratios=args.scale_ratios,
        learning_rates=args.learning_rates,
        steps=args.steps,
        output_root=output_root,
    )
    manifest = build_manifest(args, runs)
    if not args.execute:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        for run in runs:
            print(shlex.join(run.command))
        return
    _prepare_manifest(output_root, manifest, execute=True, resume=args.resume)
    console = Console()
    for run in runs:
        _run_one(run, resume=args.resume, console=console)
    _print_results(runs, console)


if __name__ == '__main__':
    main()
