import json
from pathlib import Path

import pytest

from scripts.run_capacity_seed_confirmation import (
    CONFIRMATION_SEEDS,
    build_manifest,
    build_stages,
    load_design,
)


def _inputs(tmp_path: Path, leaders: list[str]) -> tuple[Path, Path]:
    analysis = tmp_path / 'analysis.json'
    analysis.write_text(json.dumps({'seed_confirmation_leaders': leaders}))
    matrix = tmp_path / 'matrix'
    matrix.mkdir()
    (matrix / 'manifest.json').write_text(
        json.dumps(
            {
                'dataset': {'path': str(tmp_path / 'data.jsonl'), 'sha256': 'abc'},
                'data_recipe': {
                    'curated_fraction': 0.5,
                    'unique_curated': 319,
                    'unique_train': 23472,
                    'curated_presentations': 5096,
                    'effective_presentations': 28248,
                },
                'fixed': {'split_seed': 2026},
            }
        )
    )
    return analysis, matrix


def test_builds_only_two_new_seeds_for_each_leader(tmp_path: Path) -> None:
    analysis, matrix = _inputs(tmp_path, ['qwen0p5b-rank32', 'qwen1p5b-rank8'])
    design = load_design(analysis, matrix)
    stages = build_stages(tmp_path / 'out', design=design, matrix_root=matrix, reference_root=tmp_path / 'reference')
    training = [stage for stage in stages if stage.name.endswith('controlled training')]

    assert len(training) == 2 * len(CONFIRMATION_SEEDS)
    assert all('--steps' in stage.command and '14124' in stage.command for stage in training)
    assert {stage.command[stage.command.index('--seed') + 1] for stage in training} == {'2027', '2028'}
    assert {stage.command[stage.command.index('--split-seed') + 1] for stage in training} == {'2026'}
    assert all('--curated-presentations' in stage.command and '5096' in stage.command for stage in training)
    manifest = build_manifest(design, stages)
    assert manifest['training_seeds'] == [2026, 2027, 2028]
    assert manifest['hard_gates_apply_per_seed'] is True
    assert manifest['release_evaluation_forbidden'] is True


def test_rejects_more_than_two_or_unknown_leaders(tmp_path: Path) -> None:
    analysis, matrix = _inputs(tmp_path, ['qwen0p5b-rank8', 'qwen0p5b-rank32', 'qwen1p5b-rank8'])
    with pytest.raises(ValueError, match='one or two'):
        load_design(analysis, matrix)

    analysis.write_text(json.dumps({'seed_confirmation_leaders': ['not-a-configuration']}))
    with pytest.raises(ValueError, match='unknown'):
        load_design(analysis, matrix)
