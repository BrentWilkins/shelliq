import json

import pytest

from shelliq_training.teacher_verification import RustValidation, rust_validate_documents, verify_against_reference


def document(*arguments: str) -> dict[str, object]:
    return {
        'v': 2,
        'd': 'zsh',
        's': [{'t': 'p', 'c': [{'n': {'s': 'df'}, 'a': [{'s': value} for value in arguments]}]}],
    }


def test_reference_verification_accepts_conservative_equivalence():
    expected = document('-ih')
    generated = json.dumps(document('-i', '-h'), separators=(',', ':'))
    result = verify_against_reference(generated, expected, RustValidation(True, 'df -i -h', None))
    assert result.accepted
    assert result.rendered == 'df -i -h'


def test_reference_verification_rejects_unsafe_unquoting():
    expected = document("'python.*server.py'")
    generated = json.dumps(document('python.*server.py'), separators=(',', ':'))
    result = verify_against_reference(generated, expected, RustValidation(True, 'df python.*server.py', None))
    assert not result.accepted
    assert result.failures == ('reference-mismatch',)


def test_reference_verification_preserves_multiple_failures():
    result = verify_against_reference('not json', document('-i'), RustValidation(False, None, 'bad'))
    assert result.failures == (
        'invalid-json',
        'invalid-semantic-envelope',
        'semantic-round-trip-failed',
        'first-command-mismatch',
        'reference-mismatch',
    )


def test_rust_validator_wrapper_is_strict(monkeypatch):
    class Completed:
        stdout = '{"line":1,"valid":true,"rendered":"df -i"}\n'

    monkeypatch.setattr('subprocess.run', lambda *_args, **_kwargs: Completed())
    assert rust_validate_documents(['{}'], 'validator') == [RustValidation(True, 'df -i', None)]


def test_rust_validator_wrapper_frames_multiline_documents(monkeypatch):
    captured = None

    class Completed:
        stdout = '{"line":1,"valid":false,"error":"bad"}\n'

    def fake_run(*_args, **kwargs):
        nonlocal captured
        captured = kwargs['input']
        return Completed()

    monkeypatch.setattr('subprocess.run', fake_run)
    rust_validate_documents(['{\n  "v": 2\n}'], 'validator')

    assert captured is not None
    assert len(captured.splitlines()) == 1
    assert json.loads(captured)['document'] == '{\n  "v": 2\n}'


def test_rust_validator_wrapper_rejects_wrong_row_count(monkeypatch):
    class Completed:
        stdout = ''

    monkeypatch.setattr('subprocess.run', lambda *_args, **_kwargs: Completed())
    with pytest.raises(ValueError, match='returned 0 rows for 1 documents'):
        rust_validate_documents(['{}'], 'validator')
