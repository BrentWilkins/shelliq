import json
from argparse import Namespace

from scripts.run_capacity_matrix import LARGE_MODEL, SMALL_MODEL, build_manifest, build_stages


def test_matrix_contains_only_registered_four_cells_and_trains_three(tmp_path) -> None:
    dataset = tmp_path / 'dataset.jsonl'
    dataset.write_text('{}\n')
    reference = tmp_path / 'reference'
    reference.mkdir()
    reference_report = {
        'model_id': LARGE_MODEL,
        'adapter': {'alpha': 4.0, 'rank': 8, 'scale': 0.5},
        'dataset': {'sha256': __import__('hashlib').sha256(dataset.read_bytes()).hexdigest()},
        'selection': {
            'unique_curated_examples': 637,
            'train_examples': 23790,
            'effective_train_examples': 28248,
            'effective_train_source_counts': {'shelliq-curated': 5096},
            'seed': 2026,
            'split_seed': 2026,
            'train_record_ids_sha256': 'selection-hash',
        },
        'training': {
            'steps': 14124,
            'batch_size': 2,
            'lr_schedule': 'warmup-cosine',
            'warmup_steps': 1412,
            'shuffle_each_epoch': True,
        },
    }
    (reference / 'training-report.json').write_text(json.dumps(reference_report))
    output = tmp_path / 'matrix'
    stages = build_stages(
        output,
        dataset=dataset,
        reference_root=reference,
        fraction=1.0,
        unique_curated=637,
        unique_train=23790,
    )
    args = Namespace(
        semantic_dataset=dataset,
        reference_root=reference,
        curated_data_fraction=1.0,
        unique_curated=637,
        unique_train=23790,
    )

    manifest = build_manifest(args, stages)

    assert {(item['model_id'], item['rank']) for item in manifest['matrix']} == {
        (SMALL_MODEL, 8),
        (SMALL_MODEL, 32),
        (LARGE_MODEL, 8),
        (LARGE_MODEL, 32),
    }
    assert sum(item['trained'] for item in manifest['matrix']) == 3
    training = [stage for stage in stages if stage.name.endswith('controlled training')]
    assert len(training) == 3
    assert all('--steps 14124' in ' '.join(stage.command) for stage in training)
    assert all('release' not in stage.name for stage in stages)


def test_alpha_over_rank_is_fixed_in_every_trained_command(tmp_path) -> None:
    stages = build_stages(
        tmp_path,
        dataset=tmp_path / 'dataset',
        reference_root=tmp_path / 'reference',
        fraction=0.5,
        unique_curated=319,
        unique_train=23472,
    )
    commands = [' '.join(stage.command) for stage in stages if stage.name.endswith('controlled training')]
    assert any('--rank 8 --alpha 4' in command for command in commands)
    assert sum('--rank 32 --alpha 16' in command for command in commands) == 2
