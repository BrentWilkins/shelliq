import json

import pytest

from scripts.run_preference_adjudicator import parse_adjudication


def test_parse_adjudication_accepts_closed_chosen_decision() -> None:
    assert parse_adjudication(
        json.dumps(
            {
                'verdict': 'chosen',
                'failure_modes': ['missing-flag'],
                'reason': 'The rejected command omits the required flag.',
            }
        )
    ) == {
        'verdict': 'chosen',
        'failure_modes': ['missing-flag'],
        'reason': 'The rejected command omits the required flag.',
    }


def test_parse_adjudication_accepts_tie_without_failure_modes() -> None:
    result = parse_adjudication('{"verdict":"tie","failure_modes":[],"reason":"Both commands satisfy the task."}')
    assert result['verdict'] == 'tie'


@pytest.mark.parametrize(
    'value',
    [
        {'verdict': 'chosen', 'failure_modes': [], 'reason': 'missing modes'},
        {'verdict': 'tie', 'failure_modes': ['wrong-flag'], 'reason': 'forbidden modes'},
        {'verdict': 'maybe', 'failure_modes': [], 'reason': 'unknown'},
    ],
)
def test_parse_adjudication_rejects_invalid_contract(value) -> None:
    with pytest.raises(ValueError):
        parse_adjudication(json.dumps(value))
