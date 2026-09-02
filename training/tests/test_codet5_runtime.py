from __future__ import annotations

import pytest

from shelliq_training.codet5_runtime import validate_chat_request


def request(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        'messages': [{'role': 'user', 'content': 'compile this'}],
        'temperature': 0,
        'max_tokens': 192,
        'stream': False,
    }
    value.update(overrides)
    return value


def test_accepts_shelliq_chat_contract() -> None:
    assert validate_chat_request(request()) == 'compile this'


@pytest.mark.parametrize(
    'overrides',
    [
        {'messages': []},
        {'messages': [{'role': 'assistant', 'content': 'compile this'}]},
        {'messages': [{'role': 'user', 'content': ''}]},
        {'temperature': 1},
        {'max_tokens': 193},
        {'max_tokens': True},
        {'stream': True},
    ],
)
def test_rejects_requests_outside_frozen_contract(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_chat_request(request(**overrides))
