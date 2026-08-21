import json

import pytest

from scripts.harvest_student_failures import harvest
from shelliq_training.teacher_verification import RustValidation


def semantic(command: str, argument: str) -> dict[str, object]:
    return {
        'v': 2,
        'd': 'zsh',
        's': [{'t': 'p', 'c': [{'n': {'s': command}, 'a': [{'s': argument}]}]}],
    }


def pool_row(identifier: str, command: str = 'tool') -> dict[str, object]:
    return {
        'record_id': identifier,
        'corpus': 'distributable',
        'source': 'shelliq-curated',
        'license': 'CC-BY-4.0',
        'provenance': 'fixture',
        'platform': 'linux',
        'command': command,
        'instruction': 'Do work.',
        'context': 'tool: -a does the expected work.',
        'shell_response': f'{command} -a',
    }


def example(identifier: str, command: str = 'tool') -> dict[str, object]:
    return {
        'record_id': identifier,
        'expected': json.dumps(semantic(command, '-a')),
        'trained': json.dumps(semantic(command, '-b')),
    }


def test_harvest_keeps_valid_correct_command_near_miss(monkeypatch):
    monkeypatch.setattr('scripts.harvest_student_failures.FAILURE_QUOTAS', {'precise': 1})
    selected = harvest(
        [pool_row('record:1')],
        [example('record:1')],
        [RustValidation(True, 'tool -b', None)],
        seed=2026,
    )
    assert len(selected) == 1
    assert selected[0]['review_state'] == 'pending'
    assert selected[0]['verification']['failures'] == ('reference-mismatch',)


def test_harvest_can_keep_all_eligible_near_misses(monkeypatch):
    monkeypatch.setattr('scripts.harvest_student_failures.FAILURE_QUOTAS', {'precise': 1})
    selected = harvest(
        [pool_row('record:1'), pool_row('record:2')],
        [example('record:1'), example('record:2')],
        [RustValidation(True, 'tool -b', None), RustValidation(True, 'tool -b', None)],
        seed=2026,
        all_eligible=True,
    )

    assert len(selected) == 2


def test_harvest_excludes_invalid_and_wrong_command(monkeypatch):
    monkeypatch.setattr('scripts.harvest_student_failures.FAILURE_QUOTAS', {'precise': 1})
    wrong_command = example('record:1', command='wrong')
    wrong_command['expected'] = json.dumps(semantic('tool', '-a'))
    with pytest.raises(ValueError, match='not enough verified near misses'):
        harvest(
            [pool_row('record:1')],
            [wrong_command],
            [RustValidation(True, 'wrong -b', None)],
            seed=2026,
        )


def test_harvest_excludes_equivalent_split_flags(monkeypatch):
    monkeypatch.setattr('scripts.harvest_student_failures.FAILURE_QUOTAS', {'precise': 1})
    expected = semantic('tool', '-ab')
    actual = {
        'v': 2,
        'd': 'zsh',
        's': [{'t': 'p', 'c': [{'n': {'s': 'tool'}, 'a': [{'s': '-a'}, {'s': '-b'}]}]}],
    }
    with pytest.raises(ValueError, match='not enough verified near misses'):
        harvest(
            [pool_row('record:1')],
            [{'record_id': 'record:1', 'expected': json.dumps(expected), 'trained': json.dumps(actual)}],
            [RustValidation(True, 'tool -a -b', None)],
            seed=2026,
        )
