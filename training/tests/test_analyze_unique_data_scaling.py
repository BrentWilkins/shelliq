import json

from scripts.analyze_unique_data_scaling import CHECKPOINT_STEPS, build_summary


def write_point(root, grounded: int, *, selected_step: int = 12000) -> None:
    for step in CHECKPOINT_STEPS:
        prefix = root / 'evaluations' / f'step-{step:08d}'
        prefix.parent.mkdir(parents=True, exist_ok=True)
        score = grounded if step == selected_step else max(grounded - 2, 0)
        metrics = {
            'total_examples': 104,
            'first_command_accuracy': 100 / 104,
            'command_flag_sequence_exact_match': 40 / 104,
            'grounded_document_exact_match': score / 104,
            'json_parse_rate': 1.0,
            'document_envelope_rate': 1.0,
        }
        prefix.with_suffix('.curated-development.json').write_text(json.dumps({'trained': {'overall': metrics}}))
        prefix.with_suffix('.retention-analysis.json').write_text(
            json.dumps({'candidate': {'metrics': {'total_examples': 20}}, 'gate': {'passed': True}})
        )
        loss = 0.2 if step == selected_step else 0.202
        prefix.with_suffix('.validation-loss.json').write_text(json.dumps({'mean_loss': loss}))


def test_summary_requests_200_percent_only_for_material_100_percent_slope(tmp_path) -> None:
    study = tmp_path / 'study'
    full = tmp_path / 'full'
    write_point(study / 'fraction-025', 18)
    write_point(study / 'fraction-050', 20)
    write_point(full, 26)

    summary = build_summary(study, full, loss_tolerance=0.005, material_cases=5)

    assert [point['selected']['step'] for point in summary['points']] == [12000, 12000, 12000]
    assert summary['decision']['grounded_gain_50_to_100'] == 6
    assert summary['decision']['add_200_percent'] is True
    assert summary['capacity_recipe'] is None


def test_summary_stops_data_expansion_below_material_slope(tmp_path) -> None:
    study = tmp_path / 'study'
    full = tmp_path / 'full'
    write_point(study / 'fraction-025', 18)
    write_point(study / 'fraction-050', 24)
    write_point(full, 28)

    summary = build_summary(study, full, loss_tolerance=0.005, material_cases=5)

    assert summary['decision']['grounded_gain_50_to_100'] == 4
    assert summary['decision']['add_200_percent'] is False
    assert summary['capacity_recipe']['label'] == '100%'
    assert summary['capacity_recipe']['unique_train'] == 23790
