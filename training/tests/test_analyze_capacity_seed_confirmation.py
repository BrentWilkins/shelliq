from pathlib import Path

import scripts.analyze_capacity_seed_confirmation as module


def _summary(grounded: int) -> dict[str, object]:
    return {
        'selected': {
            'step': 12000,
            'grounded': grounded,
            'flags': grounded + 10,
            'first_command': 100,
            'validation_loss': 0.2,
        }
    }


def test_confirms_same_gated_leader_on_two_of_three_seeds(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / 'matrix-analysis.json'
    source.write_text('{"seed_confirmation_leaders":["qwen0p5b-rank32","qwen1p5b-rank8"]}')
    values = iter([_summary(25), _summary(30), _summary(29), _summary(28), _summary(35), _summary(34)])
    control_seeds = iter(module.SEEDS)
    monkeypatch.setattr(module, 'summarize_point', lambda point, loss_tolerance: next(values))
    monkeypatch.setattr(module, '_paired', lambda *args: {'improvements': [], 'regressions': []})
    monkeypatch.setattr(
        module,
        'audit_controls',
        lambda roots: {'passed': True, 'shared': {'seed': next(control_seeds), 'split_seed': 2026}},
    )

    result = module.build_analysis(
        source,
        matrix_root=tmp_path / 'matrix',
        reference_root=tmp_path / 'reference',
        confirmation_root=tmp_path / 'confirmation',
        loss_tolerance=0.005,
        grounding_audit=tmp_path / 'audit.json',
    )

    assert result['strict_winner_counts'] == {'qwen0p5b-rank32': 0, 'qwen1p5b-rank8': 3}
    assert result['material_winner_counts'] == {'qwen0p5b-rank32': 0, 'qwen1p5b-rank8': 2}
    assert result['strict_confirmed_leader'] == 'qwen1p5b-rank8'


def test_does_not_call_subthreshold_difference_material(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / 'matrix-analysis.json'
    source.write_text('{"seed_confirmation_leaders":["qwen0p5b-rank32","qwen1p5b-rank8"]}')
    values = iter([_summary(25), _summary(25), _summary(25), _summary(29), _summary(29), _summary(29)])
    control_seeds = iter(module.SEEDS)
    monkeypatch.setattr(module, 'summarize_point', lambda point, loss_tolerance: next(values))
    monkeypatch.setattr(module, '_paired', lambda *args: {'improvements': [], 'regressions': []})
    monkeypatch.setattr(
        module,
        'audit_controls',
        lambda roots: {'passed': True, 'shared': {'seed': next(control_seeds), 'split_seed': 2026}},
    )

    result = module.build_analysis(
        source,
        matrix_root=tmp_path / 'matrix',
        reference_root=tmp_path / 'reference',
        confirmation_root=tmp_path / 'confirmation',
        loss_tolerance=0.005,
        grounding_audit=tmp_path / 'audit.json',
    )

    assert result['strict_winner_counts'] == {'qwen0p5b-rank32': 0, 'qwen1p5b-rank8': 3}
    assert result['material_winner_counts'] == {'qwen0p5b-rank32': 0, 'qwen1p5b-rank8': 0}
    assert result['strict_confirmed_leader'] is None


def test_does_not_average_away_a_failed_seed_gate(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / 'matrix-analysis.json'
    source.write_text('{"seed_confirmation_leaders":["qwen0p5b-rank32","qwen1p5b-rank8"]}')
    values = iter([_summary(25), {'selected': None}, _summary(29), _summary(28), _summary(35), _summary(34)])
    control_seeds = iter(module.SEEDS)
    monkeypatch.setattr(module, 'summarize_point', lambda point, loss_tolerance: next(values))
    monkeypatch.setattr(module, '_paired', lambda *args: {'improvements': [], 'regressions': []})
    monkeypatch.setattr(
        module,
        'audit_controls',
        lambda roots: {'passed': True, 'shared': {'seed': next(control_seeds), 'split_seed': 2026}},
    )

    result = module.build_analysis(
        source,
        matrix_root=tmp_path / 'matrix',
        reference_root=tmp_path / 'reference',
        confirmation_root=tmp_path / 'confirmation',
        loss_tolerance=0.005,
        grounding_audit=tmp_path / 'audit.json',
    )

    assert result['strict_confirmed_leader'] is None
    assert result['seeds']['2027']['all_leaders_pass_gates'] is False
