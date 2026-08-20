from __future__ import annotations

from argparse import Namespace

from scripts.run_convergence_study import build_manifest, build_runs


def test_default_shape_builds_deep_rank_controls_and_rank_8_scale_diagnostic(tmp_path):
    runs = build_runs(
        ranks=(4, 8, 16),
        learning_rates=(5e-5, 1e-4),
        steps=3000,
        eval_interval=250,
        patience=5,
        min_delta=0.002,
        seed=2026,
        output_root=tmp_path,
    )

    assert len(runs) == 9
    assert {(run.rank, run.scale_ratio, run.learning_rate) for run in runs} == {
        (4, 1.0, 5e-5),
        (4, 1.0, 1e-4),
        (8, 1.0, 5e-5),
        (8, 1.0, 1e-4),
        (16, 1.0, 5e-5),
        (16, 1.0, 1e-4),
        (8, 0.5, 2e-4),
        (8, 2.0, 5e-5),
        (8, 2.0, 1e-4),
    }
    commands = [' '.join(run.command) for run in runs]
    assert all('--steps 3000' in command for command in commands)
    assert all('--eval-interval 250' in command for command in commands)
    assert all('--gradient-diagnostics' in command for command in commands)


def test_manifest_forbids_product_evaluations(monkeypatch, tmp_path):
    dataset = tmp_path / 'dataset.jsonl'
    dataset.write_text('{}\n')
    monkeypatch.setattr('scripts.run_convergence_study.DATASET', dataset)
    runs = build_runs(
        ranks=(8,),
        learning_rates=(1e-4,),
        steps=1000,
        eval_interval=100,
        patience=3,
        min_delta=0.001,
        seed=7,
        output_root=tmp_path / 'output',
    )
    args = Namespace(
        seed=7,
        steps=1000,
        eval_interval=100,
        patience=3,
        min_delta=0.001,
        ranks=(8,),
        learning_rates=(1e-4,),
    )

    manifest = build_manifest(args, runs)

    assert manifest['selection_metric'] == 'minimum corpus held-out loss across checkpoints'
    assert manifest['forbidden_evaluations'] == [
        'release',
        'retention',
        'pipeline',
        'shadow',
    ]
