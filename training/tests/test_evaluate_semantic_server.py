import json

import pytest

from scripts.evaluate_semantic_server import parse_completion_response, validate_endpoint


@pytest.mark.parametrize(
    'endpoint',
    [
        'http://127.0.0.1:8080/v1/chat/completions',
        'http://localhost:11434/v1/chat/completions',
        'http://[::1]:8080/v1/chat/completions',
    ],
)
def test_validate_endpoint_accepts_only_loopback_chat_completions(endpoint):
    assert validate_endpoint(endpoint) == endpoint


@pytest.mark.parametrize(
    'endpoint',
    [
        'https://127.0.0.1:8080/v1/chat/completions',
        'http://example.com/v1/chat/completions',
        'http://user@127.0.0.1:8080/v1/chat/completions',
        'http://127.0.0.1:8080/health',
        'http://127.0.0.1:8080/v1/chat/completions?token=x',
    ],
)
def test_validate_endpoint_rejects_expanded_trust_boundary(endpoint):
    with pytest.raises(ValueError, match='loopback'):
        validate_endpoint(endpoint)


def test_parse_completion_response():
    raw = json.dumps({'model': 'shelliq-alpha', 'choices': [{'message': {'content': '{"v":2}'}}]}).encode()
    assert parse_completion_response(raw) == ('{"v":2}', 'shelliq-alpha')


def test_parse_completion_response_rejects_empty_content():
    with pytest.raises(ValueError, match='empty'):
        parse_completion_response(b'{"choices":[{"message":{"content":""}}]}')
