from __future__ import annotations

import json

import pytest

from shelliq_training.data import DatasetFormatError
from shelliq_training.preference_data import FailureMode, PreferenceRecord, load_preference_jsonl


def pair() -> dict[str, object]:
    return {
        'schema_version': 1,
        'pair_id': 'preference:test:tar-missing-create',
        'corpus': 'distributable',
        'source': 'shelliq-reviewed-preference',
        'license': 'MIT OR Apache-2.0',
        'provenance': 'reviewed-preference/test@2026-08-20',
        'platform': 'linux',
        'instruction': 'Create archive.tar containing dist.',
        'context': 'tar: -c creates and -f names the archive.',
        'chosen': {
            'v': 2,
            'd': 'zsh',
            's': [{'t': 'p', 'c': [{'n': {'s': 'tar'}, 'a': [{'s': '-cf'}, {'s': 'archive.tar'}, {'s': 'dist'}]}]}],
        },
        'rejected': {
            'v': 2,
            'd': 'zsh',
            's': [{'t': 'p', 'c': [{'n': {'s': 'tar'}, 'a': [{'s': '-tf'}, {'s': 'archive.tar'}, {'s': 'dist'}]}]}],
        },
        'failure_modes': ['wrong-flag'],
        'verifier_evidence': ['chosen uses create; rejected uses list'],
        'reviewer': 'human-reviewed',
        'reviewed_at': '2026-08-20',
    }


def test_loads_strict_reviewed_pair(tmp_path):
    path = tmp_path / 'pairs.jsonl'
    path.write_text(json.dumps(pair()) + '\n')

    records = load_preference_jsonl(path)

    assert len(records) == 1
    assert records[0].failure_modes == (FailureMode.WRONG_FLAG,)


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (lambda value: value.update(extra='nope'), 'unknown fields'),
        (lambda value: value.update(rejected=value['chosen']), 'must differ'),
        (lambda value: value.update(failure_modes=[]), 'non-empty list'),
        (lambda value: value.update(failure_modes=['unknown']), 'invalid failure mode'),
        (lambda value: value.update(verifier_evidence=[]), 'non-empty strings'),
        (lambda value: value.update(reviewed_at='today'), 'YYYY-MM-DD'),
    ],
)
def test_rejects_unreviewable_pairs(mutation, message):
    value = pair()
    mutation(value)

    with pytest.raises(DatasetFormatError, match=message):
        PreferenceRecord.from_dict(value)


def test_rejects_duplicate_pair_ids(tmp_path):
    path = tmp_path / 'pairs.jsonl'
    line = json.dumps(pair())
    path.write_text(f'{line}\n{line}\n')

    with pytest.raises(DatasetFormatError, match='duplicate pair_id'):
        load_preference_jsonl(path)
