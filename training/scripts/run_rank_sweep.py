#!/usr/bin/env python3
"""Run a reproducible 0.5B LoRA-rank comparison without touching final holdouts."""

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
DEFAULT_RANKS = (16, 32, 64)


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ARTIFACTS / 'lora-rank-sweep-v1')
    parser.add_argument('--ranks', type=_parse_ranks, default=DEFAULT_RANKS)
    parser.add_argument('--execute', action='store_true', help='Execute instead of printing the immutable plan.')
    parser.add_argument('--resume', action='store_true', help='Skip stages whose complete outputs already exist.')
    return parser.parse_args()


def _parse_ranks(value: str) -> tuple[int, ...]:
    try:
        ranks = tuple(int(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('ranks must be comma-separated integers') from error
    if not ranks or any(rank <= 0 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise argparse.ArgumentTypeError('ranks must be unique positive integers')
    return ranks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _python(script: str, *arguments: str) -> tuple[str, ...]:
    return (sys.executable, str(TRAINING_ROOT / 'scripts' / script), *arguments)


def build_rank_plan(rank: int, output_root: Path) -> tuple[Stage, ...]:
    alpha = rank * 2
    rank_root = output_root / f'rank-{rank}'
    broad_root = rank_root / 'broad'
    final_root = rank_root / 'curated'
    release_eval = rank_root / 'release.eval.json'
    release_analysis = rank_root / 'release.analysis.json'
    retention_eval = rank_root / 'retention.eval.json'
    retention_analysis = rank_root / 'retention.analysis.json'
    adapter = ('--rank', str(rank), '--alpha', str(alpha))

    return (
        Stage(
            'broad SFT',
            _python(
                'smoke_finetune.py',
                '--semantic-dataset',
                str(ARTIFACTS / 'distributable-semantic-v2.jsonl'),
                '--sequence-length',
                '384',
                '--train-examples',
                '4000',
                '--eval-examples',
                '256',
                '--batch-size',
                '2',
                '--steps',
                '2000',
                '--learning-rate',
                '2e-4',
                '--priority-source',
                'shelliq-curated',
                '--heldout-probe',
                *adapter,
                '--checkpoint',
                str(broad_root / 'checkpoint'),
                '--report',
                str(broad_root / 'report.json'),
            ),
            (broad_root / 'checkpoint', broad_root / 'report.json'),
        ),
        Stage(
            'curated SFT',
            _python(
                'smoke_finetune.py',
                '--semantic-dataset',
                str(ARTIFACTS / 'curated-semantic-v7.jsonl'),
                '--sequence-length',
                '384',
                '--train-examples',
                '636',
                '--eval-examples',
                '36',
                '--batch-size',
                '2',
                '--steps',
                '500',
                '--learning-rate',
                '5e-5',
                '--priority-source',
                'shelliq-curated',
                '--heldout-probe',
                *adapter,
                '--resume-checkpoint',
                str(broad_root / 'checkpoint'),
                '--checkpoint',
                str(final_root / 'checkpoint'),
                '--report',
                str(final_root / 'report.json'),
            ),
            (final_root / 'checkpoint', final_root / 'report.json'),
        ),
        Stage(
            'release development evaluation',
            _python(
                'evaluate_semantic_checkpoint.py',
                '--semantic-dataset',
                str(ARTIFACTS / 'distributable-semantic-v2.jsonl'),
                '--checkpoint',
                str(final_root / 'checkpoint'),
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
                str(final_root / 'checkpoint'),
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


def build_manifest(ranks: tuple[int, ...], output_root: Path) -> dict[str, object]:
    inputs = (
        ARTIFACTS / 'distributable-semantic-v2.jsonl',
        ARTIFACTS / 'curated-semantic-v7.jsonl',
        ARTIFACTS / 'semantic-retention-v1.jsonl',
        ARTIFACTS / 'semantic-shadow-release-v1.jsonl',
        TRAINING_ROOT / 'evaluation' / 'curated-grounding-v1.json',
        TRAINING_ROOT / 'evaluation' / 'semantic-retention-grounding-v1.json',
        TRAINING_ROOT / 'evaluation' / 'semantic-shadow-release-grounding-v1.json',
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'missing rank-sweep inputs: {missing}')
    return {
        'schema_version': 1,
        'purpose': '0.5B LoRA capacity comparison; pipeline and shadow holdouts excluded',
        'ranks': list(ranks),
        'alpha_rule': 'alpha = 2 * rank; constant alpha/rank scale of 2',
        'output_root': str(output_root),
        'inputs': {str(path): _sha256(path) for path in inputs},
        'plans': {
            str(rank): [
                {
                    'name': stage.name,
                    'command': list(stage.command),
                    'outputs': [str(output) for output in stage.outputs],
                }
                for stage in build_rank_plan(rank, output_root)
            ]
            for rank in ranks
        },
    }


def _stage_state(stage: Stage) -> str:
    present = [output.exists() for output in stage.outputs]
    if all(present):
        return 'complete'
    if any(present):
        return 'partial'
    return 'pending'


def _run_stage(stage: Stage, *, resume: bool, console: Console) -> None:
    state = _stage_state(stage)
    if state == 'complete' and resume:
        console.print(f'  [dim]skip complete: {stage.name}[/dim]')
        return
    if state != 'pending':
        raise FileExistsError(f'{stage.name} outputs are {state}; refusing overwrite: {stage.outputs}')
    for output in stage.outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    console.rule(stage.name)
    subprocess.run(stage.command, cwd=TRAINING_ROOT, check=True)


def _print_results(ranks: tuple[int, ...], output_root: Path, console: Console) -> None:
    table = Table(title='0.5B LoRA rank development results')
    table.add_column('rank', justify='right')
    table.add_column('release flags', justify='right')
    table.add_column('release grounded', justify='right')
    table.add_column('retention flags', justify='right')
    table.add_column('retention grounded', justify='right')
    for rank in ranks:
        root = output_root / f'rank-{rank}'
        release = json.loads((root / 'release.eval.json').read_text())['trained']['overall']
        retention = json.loads((root / 'retention.eval.json').read_text())['trained']['overall']
        table.add_row(
            str(rank),
            f'{release["command_flag_sequence_exact_match"]:.3f}',
            f'{release["grounded_document_exact_match"]:.3f}',
            f'{retention["command_flag_sequence_exact_match"]:.3f}',
            f'{retention["grounded_document_exact_match"]:.3f}',
        )
    console.print(table)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    manifest = build_manifest(args.ranks, output_root)
    manifest_path = output_root / 'manifest.json'
    console = Console(stderr=True)

    if not args.execute:
        console.print_json(json.dumps(manifest, default=str))
        for rank in args.ranks:
            for stage in build_rank_plan(rank, output_root):
                print(shlex.join(stage.command))
        return

    if args.resume:
        if not manifest_path.is_file():
            raise SystemExit(f'cannot resume without manifest: {manifest_path}')
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit('existing rank-sweep manifest does not match requested plan')
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    for rank in args.ranks:
        console.rule(f'rank {rank} / alpha {rank * 2}')
        for stage in build_rank_plan(rank, output_root):
            _run_stage(stage, resume=args.resume, console=console)
    _print_results(args.ranks, output_root, console)


if __name__ == '__main__':
    main()
