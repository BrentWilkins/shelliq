import json

import pytest

from scripts.evaluate_checkpoint import _load_option_arities, _validate_option_arities
from shelliq_training.data import Corpus, Platform, SFTRecord


def record(response: str = 'find src -name *.py') -> SFTRecord:
    return SFTRecord(
        record_id='curated:test:linux:find-name',
        corpus=Corpus.DISTRIBUTABLE,
        source='shelliq-curated',
        license='MIT OR Apache-2.0',
        provenance='fixture:test',
        command='find',
        platform=Platform.LINUX,
        instruction='Find Python files.',
        response=response,
        context='find: fixture context.',
    )


def test_option_arity_sidecar_is_strict_and_matches_expected_flags(tmp_path):
    path = tmp_path / 'arities.json'
    path.write_text(
        json.dumps(
            {
                'schema_version': 1,
                'records': {'curated:test:linux:find-name': {'-name': 1}},
            }
        )
    )
    arities = _load_option_arities(path)
    _validate_option_arities([record()], arities)
    assert arities == {'curated:test:linux:find-name': {'-name': 1}}


def test_option_arity_sidecar_rejects_boolean_arity(tmp_path):
    path = tmp_path / 'arities.json'
    path.write_text(
        json.dumps(
            {
                'schema_version': 1,
                'records': {'curated:test:linux:find-name': {'-name': True}},
            }
        )
    )
    with pytest.raises(ValueError, match='must be 0 or 1'):
        _load_option_arities(path)


def test_option_arity_validation_rejects_missing_expected_flag():
    with pytest.raises(ValueError, match=r"missing=\['-name'\]"):
        _validate_option_arities([record()], {'curated:test:linux:find-name': {}})
