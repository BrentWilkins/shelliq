from __future__ import annotations

from argparse import Namespace

from scripts.run_full_corpus_finalists import DEFAULT_FINALISTS, build_manifest, build_runs


def test_full_corpus_plan_uses_both_finalists_and_retains_best_checkpoints(tmp_path):
    runs = build_runs(
        finalists=DEFAULT_FINALISTS,
        seeds=(2026,),
        split_seed=2026,
        epochs=2,
        eval_interval=1000,
        warmup_steps=500,
        end_lr_ratio=0.1,
        output_root=tmp_path,
    )

    assert {(run.finalist.rank, run.finalist.alpha) for run in runs} == {(8, 4.0), (16, 16.0)}
    commands = [' '.join(run.command) for run in runs]
    assert all('--all-train-examples' in command for command in commands)
    assert all('--eval-split validation' in command for command in commands)
    assert all('--epochs 2' in command for command in commands)
    assert all('--shuffle-each-epoch' in command for command in commands)
    assert all('--lr-schedule warmup-cosine' in command for command in commands)
    assert all('--best-checkpoint-root' in command for command in commands)


def test_full_corpus_manifest_keeps_behavioral_evaluations_forbidden(monkeypatch, tmp_path):
    dataset = tmp_path / 'dataset.jsonl'
    dataset.write_text('{}\n')
    monkeypatch.setattr('scripts.run_full_corpus_finalists.DATASET', dataset)
    runs = build_runs(
        finalists=DEFAULT_FINALISTS,
        seeds=(2026,),
        split_seed=2026,
        epochs=2,
        eval_interval=1000,
        warmup_steps=500,
        end_lr_ratio=0.1,
        output_root=tmp_path / 'output',
    )
    args = Namespace(
        split_seed=2026,
        seeds=(2026,),
        epochs=2,
        eval_interval=1000,
        warmup_steps=500,
        end_lr_ratio=0.1,
    )

    manifest = build_manifest(args, runs)

    assert manifest['data_order'] == 'deterministically reshuffled each epoch'
    assert manifest['forbidden_evaluations'] == [
        'release',
        'retention',
        'pipeline',
        'shadow',
    ]
