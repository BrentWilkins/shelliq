#!/usr/bin/env python3
"""Train the LoRA finalists on every usable training-split corpus row."""

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


@dataclass(frozen=True, slots=True)
class Finalist:
    name: str
    rank: int
    alpha: float
    peak_learning_rate: float


@dataclass(frozen=True, slots=True)
class FullCorpusRun:
    finalist: Finalist
    seed: int
    command: tuple[str, ...]
    report: Path
    best_checkpoint_root: Path

    @property
    def name(self) -> str:
        return f'{self.finalist.name}_seed-{self.seed}'


DEFAULT_FINALISTS = (
    Finalist('rank-8_scale-0p5_lr-2e-04', rank=8, alpha=4.0, peak_learning_rate=2e-4),
    Finalist('rank-16_scale-1_lr-5e-05', rank=16, alpha=16.0, peak_learning_rate=5e-5),
)


def _positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('values must be comma-separated integers') from error
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError('values must be unique integers')
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root',
        type=Path,
        default=ARTIFACTS / 'lora-full-corpus-finalists-v1',
    )
    parser.add_argument('--seeds', type=_positive_ints, default=(2026,))
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--end-lr-ratio', type=float, default=0.1)
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
    finalists: tuple[Finalist, ...],
    seeds: tuple[int, ...],
    split_seed: int,
    epochs: int,
    eval_interval: int,
    warmup_steps: int,
    end_lr_ratio: float,
    output_root: Path,
) -> tuple[FullCorpusRun, ...]:
    runs = []
    for finalist in finalists:
        for seed in seeds:
            name = f'{finalist.name}_seed-{seed}'
            run_root = output_root / name
            report = run_root / 'report.json'
            best_checkpoint_root = run_root / 'best-checkpoint'
            command = (
                sys.executable,
                str(TRAINING_ROOT / 'scripts' / 'smoke_finetune.py'),
                '--semantic-dataset',
                str(DATASET),
                '--sequence-length',
                '384',
                '--all-train-examples',
                '--train-loss-eval-examples',
                '256',
                '--eval-examples',
                '256',
                '--eval-split',
                'validation',
                '--batch-size',
                '2',
                '--epochs',
                str(epochs),
                '--eval-interval',
                str(eval_interval),
                '--early-stopping-patience',
                '5',
                '--early-stopping-min-delta',
                '0.001',
                '--early-stopping-min-epochs',
                '1',
                '--gradient-diagnostics',
                '--shuffle-each-epoch',
                '--lr-schedule',
                'warmup-cosine',
                '--warmup-steps',
                str(warmup_steps),
                '--learning-rate',
                str(finalist.peak_learning_rate),
                '--end-learning-rate',
                str(finalist.peak_learning_rate * end_lr_ratio),
                '--seed',
                str(seed),
                '--split-seed',
                str(split_seed),
                '--rank',
                str(finalist.rank),
                '--alpha',
                str(finalist.alpha),
                '--best-checkpoint-root',
                str(best_checkpoint_root),
                '--report',
                str(report),
            )
            runs.append(FullCorpusRun(finalist, seed, command, report, best_checkpoint_root))
    return tuple(runs)


def build_manifest(args: argparse.Namespace, runs: tuple[FullCorpusRun, ...]) -> dict[str, object]:
    if not DATASET.is_file():
        raise FileNotFoundError(f'missing full-corpus dataset: {DATASET}')
    return {
        'schema_version': 1,
        'purpose': 'full-corpus SFT of convergence-study finalists',
        'selection_metric': 'minimum corpus validation loss across checkpoints',
        'preserved_evaluation': 'corpus test split',
        'dataset': {'path': str(DATASET), 'sha256': _sha256(DATASET)},
        'split_seed': args.split_seed,
        'training_seeds': list(args.seeds),
        'epochs': args.epochs,
        'eval_interval': args.eval_interval,
        'schedule': {
            'name': 'warmup-cosine',
            'warmup_steps': args.warmup_steps,
            'end_lr_ratio': args.end_lr_ratio,
        },
        'data_order': 'deterministically reshuffled each epoch',
        'forbidden_evaluations': ['release', 'retention', 'pipeline', 'shadow'],
        'runs': [
            {
                'name': run.name,
                'rank': run.finalist.rank,
                'alpha': run.finalist.alpha,
                'alpha_over_rank': run.finalist.alpha / run.finalist.rank,
                'peak_learning_rate': run.finalist.peak_learning_rate,
                'seed': run.seed,
                'command': list(run.command),
                'report': str(run.report),
                'best_checkpoint_root': str(run.best_checkpoint_root),
            }
            for run in runs
        ],
    }


def _prepare_manifest(output_root: Path, manifest: dict[str, object], *, resume: bool) -> None:
    manifest_path = output_root / 'manifest.json'
    output_root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f'manifest already exists: {manifest_path}; use --resume')
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('existing manifest does not match the requested full-corpus run')
        return
    if resume and any(output_root.iterdir()):
        raise FileNotFoundError(f'cannot resume without manifest: {manifest_path}')
    if any(output_root.iterdir()):
        raise FileExistsError(f'output root is not empty: {output_root}')
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


def _run_one(run: FullCorpusRun, *, resume: bool, console: Console) -> None:
    if run.report.exists():
        if not resume:
            raise FileExistsError(f'report already exists: {run.report}')
        console.print(f'[dim]skip {run.name}[/dim]')
        return
    if run.best_checkpoint_root.exists():
        raise RuntimeError(
            f'incomplete run has retained checkpoint data: {run.best_checkpoint_root}; '
            'move it aside before restarting this configuration'
        )
    run.report.parent.mkdir(parents=True, exist_ok=True)
    console.rule(run.name)
    subprocess.run(run.command, cwd=TRAINING_ROOT, check=True)


def _print_results(runs: tuple[FullCorpusRun, ...], console: Console) -> None:
    table = Table(title='Full-corpus LoRA finalists')
    for heading in ('rank', 'alpha/rank', 'peak LR', 'seed', 'best step', 'held-out', 'clip %'):
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
            str(run.finalist.rank),
            f'{run.finalist.alpha / run.finalist.rank:g}',
            f'{run.finalist.peak_learning_rate:.0e}',
            str(run.seed),
            str(best['step']),
            f'{best["heldout_loss"]:.4f}',
            f'{100 * gradients["clipped_fraction"]:.1f}',
        )
    console.print(table)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.eval_interval <= 0 or args.warmup_steps < 0:
        raise SystemExit('epochs/eval interval must be positive and warmup non-negative')
    if not 0 <= args.end_lr_ratio <= 1:
        raise SystemExit('--end-lr-ratio must be between zero and one')

    output_root = args.output_root.resolve()
    runs = build_runs(
        finalists=DEFAULT_FINALISTS,
        seeds=args.seeds,
        split_seed=args.split_seed,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        warmup_steps=args.warmup_steps,
        end_lr_ratio=args.end_lr_ratio,
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
