#!/usr/bin/env python3
"""Run a resumable, checkpointed-loss LoRA convergence study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DEFAULT_RANKS = (4, 8, 16)
DEFAULT_LEARNING_RATES = (5e-5, 1e-4)


@dataclass(frozen=True, slots=True)
class ConvergenceRun:
    rank: int
    scale_ratio: float
    learning_rate: float
    command: tuple[str, ...]
    report: Path

    @property
    def name(self) -> str:
        scale = f'{self.scale_ratio:g}'.replace('.', 'p')
        return f'rank-{self.rank}_scale-{scale}_lr-{self.learning_rate:.0e}'


def _positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('values must be comma-separated integers') from error
    if not parsed or any(not math.isfinite(item) or item <= 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError('values must be unique positive integers')
    return parsed


def _positive_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('values must be comma-separated numbers') from error
    if not parsed or any(not math.isfinite(item) or item <= 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError('values must be unique positive numbers')
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root',
        type=Path,
        default=ARTIFACTS / 'lora-convergence-study-v1',
    )
    parser.add_argument('--ranks', type=_positive_ints, default=DEFAULT_RANKS)
    parser.add_argument(
        '--learning-rates',
        type=_positive_floats,
        default=DEFAULT_LEARNING_RATES,
    )
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--eval-interval', type=int, default=250)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=0.002)
    parser.add_argument('--seed', type=int, default=2026)
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
    learning_rates: tuple[float, ...],
    steps: int,
    eval_interval: int,
    patience: int,
    min_delta: float,
    seed: int,
    output_root: Path,
) -> tuple[ConvergenceRun, ...]:
    configurations = [(rank, 1.0, learning_rate) for rank in ranks for learning_rate in learning_rates]
    if 8 in ranks and 1e-4 in learning_rates:
        configurations.extend(
            (
                (8, 0.5, 2e-4),
                (8, 2.0, 5e-5),
                (8, 2.0, 1e-4),
            )
        )

    runs = []
    for rank, scale_ratio, learning_rate in configurations:
        alpha = rank * scale_ratio
        scale = f'{scale_ratio:g}'.replace('.', 'p')
        name = f'rank-{rank}_scale-{scale}_lr-{learning_rate:.0e}'
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
            '--eval-interval',
            str(eval_interval),
            '--early-stopping-patience',
            str(patience),
            '--early-stopping-min-delta',
            str(min_delta),
            '--gradient-diagnostics',
            '--learning-rate',
            str(learning_rate),
            '--seed',
            str(seed),
            '--priority-source',
            'shelliq-curated',
            '--rank',
            str(rank),
            '--alpha',
            str(alpha),
            '--report',
            str(report),
        )
        runs.append(ConvergenceRun(rank, scale_ratio, learning_rate, command, report))
    return tuple(runs)


def build_manifest(args: argparse.Namespace, runs: tuple[ConvergenceRun, ...]) -> dict[str, object]:
    if not DATASET.is_file():
        raise FileNotFoundError(f'missing convergence-study dataset: {DATASET}')
    return {
        'schema_version': 1,
        'purpose': 'longer convergence curves with early stopping and gradient diagnostics',
        'selection_metric': 'minimum corpus held-out loss across checkpoints',
        'dataset': {'path': str(DATASET), 'sha256': _sha256(DATASET)},
        'seed': args.seed,
        'max_steps': args.steps,
        'eval_interval': args.eval_interval,
        'early_stopping': {'patience': args.patience, 'min_delta': args.min_delta},
        'ranks': list(args.ranks),
        'learning_rates': list(args.learning_rates),
        'rank_8_scale_diagnostic': [
            {
                'alpha_over_rank': run.scale_ratio,
                'learning_rate': run.learning_rate,
            }
            for run in runs
            if run.rank == 8
        ],
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
    resume: bool,
) -> None:
    manifest_path = output_root / 'manifest.json'
    output_root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f'manifest already exists: {manifest_path}; use --resume')
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('existing manifest does not match the requested study')
        return
    if resume and any(output_root.iterdir()):
        raise FileNotFoundError(f'cannot resume without manifest: {manifest_path}')
    if any(output_root.iterdir()):
        raise FileExistsError(f'output root is not empty: {output_root}')
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


def _run_one(run: ConvergenceRun, *, resume: bool, console: Console) -> None:
    if run.report.exists():
        if not resume:
            raise FileExistsError(f'report already exists: {run.report}')
        console.print(f'[dim]skip {run.name}[/dim]')
        return
    run.report.parent.mkdir(parents=True, exist_ok=True)
    console.rule(run.name)
    subprocess.run(run.command, cwd=TRAINING_ROOT, check=True)


def _print_results(runs: tuple[ConvergenceRun, ...], console: Console) -> None:
    table = Table(title='LoRA convergence study')
    for heading in ('rank', 'alpha/rank', 'LR', 'best step', 'best held-out', 'clip %'):
        table.add_column(heading, justify='right')
    rows = []
    for run in runs:
        if not run.report.is_file():
            continue
        training = json.loads(run.report.read_text())['training']
        best = training['best_observation']
        gradients = training['gradient_diagnostics']
        rows.append((float(best['heldout_loss']), run, best, gradients))
    for _, run, best, gradients in sorted(rows, key=lambda item: item[0]):
        table.add_row(
            str(run.rank),
            f'{run.scale_ratio:g}',
            f'{run.learning_rate:.0e}',
            str(best['step']),
            f'{best["heldout_loss"]:.4f}',
            f'{100 * gradients["clipped_fraction"]:.1f}',
        )
    console.print(table)


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.eval_interval <= 0 or args.patience <= 0:
        raise SystemExit('steps, eval interval, and patience must be positive')
    if args.eval_interval > args.steps:
        raise SystemExit('--eval-interval cannot exceed --steps')
    if not math.isfinite(args.min_delta) or args.min_delta < 0:
        raise SystemExit('--min-delta must be finite and non-negative')

    output_root = args.output_root.resolve()
    runs = build_runs(
        ranks=args.ranks,
        learning_rates=args.learning_rates,
        steps=args.steps,
        eval_interval=args.eval_interval,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        output_root=output_root,
    )
    manifest = build_manifest(args, runs)
    if not args.execute:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        for run in runs:
            print(shlex.join(run.command))
        return

    _prepare_manifest(output_root, manifest, resume=args.resume)
    console = Console()
    for run in runs:
        _run_one(run, resume=args.resume, console=console)
    _print_results(runs, console)


if __name__ == '__main__':
    main()
