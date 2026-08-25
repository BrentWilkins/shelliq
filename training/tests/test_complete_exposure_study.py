from scripts.run_complete_exposure_study import (
    CHECKPOINT_STEPS,
    EFFECTIVE_TRAIN_EXAMPLES,
    Stage,
    _prepare_stage_output_parents,
    build_manifest,
    build_stages,
)


def test_complete_exposure_plan_is_one_epoch_and_never_touches_release(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / 'dataset.jsonl'
    development = tmp_path / 'development.jsonl'
    retention = tmp_path / 'retention.jsonl'
    for path in (dataset, development, retention):
        path.write_text('{}\n')
    monkeypatch.setattr('scripts.run_complete_exposure_study.DATASET', dataset)
    monkeypatch.setattr('scripts.run_complete_exposure_study.CURATED_DEVELOPMENT', development)
    monkeypatch.setattr('scripts.run_complete_exposure_study.RETENTION', retention)

    stages = build_stages(tmp_path / 'study')
    manifest = build_manifest(stages)
    training_command = ' '.join(stages[0].command)

    assert manifest['training']['steps'] == EFFECTIVE_TRAIN_EXAMPLES // 2 == 14124
    assert manifest['training']['checkpoint_steps'] == list(CHECKPOINT_STEPS)
    assert '--model-id Qwen/Qwen2.5-Coder-1.5B-Instruct' in training_command
    assert '--rank 8 --alpha 4' in training_command
    assert '--rehearsal curated:=8' in training_command
    assert '--retain-checkpoint-steps 2000,4000,8000,12000,14124' in training_command
    assert all('shadow-release' not in ' '.join(stage.command) for stage in stages)
    assert all(' release' not in stage.name for stage in stages)


def test_each_retained_checkpoint_gets_development_retention_and_loss(tmp_path) -> None:
    stages = build_stages(tmp_path)
    for step in CHECKPOINT_STEPS:
        names = {stage.name for stage in stages if f'step {step}' in stage.name}
        assert names == {
            f'curated development at step {step}',
            f'paired curated analysis at step {step}',
            f'tiered retention evaluation at step {step}',
            f'tiered retention analysis at step {step}',
            f'full validation loss at step {step}',
        }
    retention_commands = [' '.join(stage.command) for stage in stages if 'retention analysis' in stage.name]
    assert all('--tiered' in command and '--gate' not in command for command in retention_commands)


def test_stage_preparation_does_not_precreate_checkpoint_root(tmp_path) -> None:
    checkpoint_root = tmp_path / 'checkpoints'
    checkpoint_root.mkdir()
    stage = Stage(
        'training',
        ('unused',),
        (tmp_path / 'training-report.json', checkpoint_root / 'step-00002000'),
    )

    _prepare_stage_output_parents(stage)

    assert tmp_path.is_dir()
    assert not checkpoint_root.exists()
