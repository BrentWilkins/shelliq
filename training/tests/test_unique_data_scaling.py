from scripts.run_unique_data_scaling import (
    CURATED_PRESENTATIONS,
    EFFECTIVE_PRESENTATIONS,
    SCALING_POINTS,
    build_manifest,
    build_stages,
)


def test_scaling_points_hold_presentations_and_recipe_constant(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / 'dataset.jsonl'
    development = tmp_path / 'development.jsonl'
    retention = tmp_path / 'retention.jsonl'
    full = tmp_path / 'full'
    full.mkdir()
    for path in (dataset, development, retention, full / 'training-report.json'):
        path.write_text('{}\n')
    monkeypatch.setattr('scripts.run_unique_data_scaling.DATASET', dataset)
    monkeypatch.setattr('scripts.run_unique_data_scaling.CURATED_DEVELOPMENT', development)
    monkeypatch.setattr('scripts.run_unique_data_scaling.RETENTION', retention)
    monkeypatch.setattr('scripts.run_unique_data_scaling.FULL_EXPOSURE', full)

    stages = build_stages(tmp_path / 'study')
    manifest = build_manifest(stages)
    training = [stage for stage in stages if 'training' in stage.name]

    assert [(point.fraction, point.unique_curated) for point in SCALING_POINTS] == [(0.25, 159), (0.5, 319)]
    assert manifest['controlled_recipe']['curated_presentations'] == CURATED_PRESENTATIONS == 5096
    assert manifest['controlled_recipe']['total_presentations'] == EFFECTIVE_PRESENTATIONS == 28248
    assert len(training) == 2
    assert all('--steps 14124' in ' '.join(stage.command) for stage in training)
    assert all('--curated-presentations 5096' in ' '.join(stage.command) for stage in training)
    assert all('release' not in stage.name for stage in stages)


def test_scaling_evaluates_every_checkpoint(tmp_path) -> None:
    stages = build_stages(tmp_path)
    for point in SCALING_POINTS:
        point_stages = [stage for stage in stages if stage.name.startswith(point.label)]
        assert len(point_stages) == 26
