import json

import pytest

from scripts.run_semantic_action_compiler_rules_outer import (
    load_inner_authorization,
    outer_gate_floors,
)


def test_outer_gate_floors_round_up_frozen_rates() -> None:
    assert outer_gate_floors(78) == {
        'minimum_rust_valid': 71,
        'minimum_first_command_match': 63,
        'minimum_reference_accepted': 8,
    }


def test_inner_authorization_freezes_source_epoch_and_default_seed(tmp_path) -> None:
    path = tmp_path / 'inner.json'
    path.write_text(
        json.dumps(
            {
                'experiment': 'semantic-action-compiler-rules-v2',
                'gate_passed': True,
                'source_best_epoch': 5,
            }
        )
    )

    _, epochs, seed = load_inner_authorization(path)

    assert epochs == 5
    assert seed == 20260826


@pytest.mark.parametrize(
    'report',
    [
        {'experiment': 'semantic-action-compiler-rules-v2', 'gate_passed': False, 'source_best_epoch': 5},
        {'experiment': 'semantic-action-compiler-rules-v1', 'gate_passed': True, 'source_best_epoch': 5},
        {'experiment': 'semantic-action-compiler-rules-v2', 'gate_passed': True, 'source_best_epoch': 0},
    ],
)
def test_inner_authorization_rejects_unapproved_reports(tmp_path, report) -> None:
    path = tmp_path / 'inner.json'
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError):
        load_inner_authorization(path)
