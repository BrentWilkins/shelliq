from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_optimization_screen import build_manifest, build_runs


def test_screen_is_a_training_only_factorial(tmp_path: Path):
    runs = build_runs(
        ranks=(8, 16),
        scale_ratios=(1, 2),
        learning_rates=(1e-4, 2e-4),
        steps=750,
        output_root=tmp_path,
    )

    assert len(runs) == 8
    commands = [' '.join(run.command) for run in runs]
    assert all('smoke_finetune.py' in command for command in commands)
    assert all('--steps 750' in command for command in commands)
    assert all('evaluate_' not in command for command in commands)
    assert all('shadow' not in command for command in commands)
    assert {run.rank * run.scale_ratio for run in runs} == {8, 16, 32}


def test_manifest_records_interacting_axes(monkeypatch, tmp_path: Path):
    dataset = tmp_path / 'dataset.jsonl'
    dataset.write_text('{}\n')
    monkeypatch.setattr('scripts.run_optimization_screen.DATASET', dataset)
    runs = build_runs(
        ranks=(8,),
        scale_ratios=(1, 2),
        learning_rates=(1e-4,),
        steps=100,
        output_root=tmp_path / 'output',
    )
    args = argparse.Namespace(
        steps=100,
        ranks=(8,),
        scale_ratios=(1, 2),
        learning_rates=(1e-4,),
    )

    manifest = build_manifest(args, runs)

    assert manifest['selection_metric'] == 'final_heldout_loss'
    assert manifest['alpha_over_rank'] == [1, 2]
    assert manifest['forbidden_evaluations'] == [
        'release',
        'retention',
        'pipeline',
        'shadow',
    ]
