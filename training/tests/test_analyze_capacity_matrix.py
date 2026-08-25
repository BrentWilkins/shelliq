import json

import pytest

from scripts.analyze_capacity_matrix import audit_controls, classify_effects


def selected(small8: int, small32: int, large8: int, large32: int):
    return {
        'qwen0p5b-rank8': {'grounded': small8},
        'qwen0p5b-rank32': {'grounded': small32},
        'qwen1p5b-rank8': {'grounded': large8},
        'qwen1p5b-rank32': {'grounded': large32},
    }


def test_capacity_interpretation_can_identify_rank_and_base_signals() -> None:
    result = classify_effects(selected(12, 18, 20, 28), material_cases=5)
    assert result['signals'] == ['lora_capacity', 'base_model_capacity']
    assert result['grounded_deltas']['rank_effect_1p5b'] == 8


def test_capacity_interpretation_stops_when_neither_controlled_axis_is_material() -> None:
    result = classify_effects(selected(20, 22, 23, 24), material_cases=5)
    assert result['signals'] == ['data_generalization_or_optimization']


def test_capacity_interpretation_is_incomplete_without_all_four_cells() -> None:
    result = classify_effects({'qwen1p5b-rank8': {'grounded': 28}}, material_cases=5)
    assert result == {'status': 'incomplete', 'signals': []}


def _write_report(root, *, selection_hash='same') -> None:
    root.mkdir()
    (root / 'training-report.json').write_text(
        json.dumps(
            {
                'dataset': {'sha256': 'dataset'},
                'selection': {
                    'train_record_ids_sha256': selection_hash,
                    'effective_train_examples': 28248,
                    'effective_train_source_counts': {'shelliq-curated': 5096, 'tldr-pages': 23152},
                    'seed': 2026,
                    'split_seed': 2026,
                },
                'training': {
                    'steps': 14124,
                    'batch_size': 2,
                    'lr_schedule': 'warmup-cosine',
                    'warmup_steps': 1412,
                    'shuffle_each_epoch': True,
                },
            }
        )
    )


def test_control_audit_rejects_different_selected_data(tmp_path) -> None:
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    _write_report(first)
    _write_report(second, selection_hash='different')

    with pytest.raises(ValueError, match='controls differ'):
        audit_controls({'first': first, 'second': second})
