import json

import pytest

from scripts.build_targeted_semantic_dataset import validate_no_holdout_leakage


def write_dataset(path, *, command='git', instruction='Show a graph', record_id='targeted:one'):
    row = {
        'schema_version': 1,
        'record_id': record_id,
        'corpus': 'distributable',
        'source': 'shelliq-curated',
        'license': 'MIT OR Apache-2.0',
        'provenance': 'test',
        'command': command,
        'platform': 'linux',
        'instruction': instruction,
        'response': f'{command} --all',
        'context': f'{command}: --all includes everything.',
    }
    path.write_text(json.dumps(row) + '\n')


def write_report(path, *, command='cp', instruction='Copy everything', record_id='heldout:one'):
    expected = json.dumps({'v': 2, 'd': 'zsh', 's': [{'t': 'p', 'c': [{'n': {'s': command}, 'a': []}]}]})
    path.write_text(json.dumps({'examples': [{'record_id': record_id, 'instruction': instruction, 'expected': expected}]}))


def test_rejects_holdout_ids_instructions_and_command_families(tmp_path):
    dataset = tmp_path / 'dataset.jsonl'
    report = tmp_path / 'report.json'
    write_report(report)

    write_dataset(dataset, record_id='heldout:one')
    with pytest.raises(ValueError, match='record_ids'):
        validate_no_holdout_leakage(dataset, report, seed=2026)
    write_dataset(dataset, instruction='Copy everything')
    with pytest.raises(ValueError, match='instructions'):
        validate_no_holdout_leakage(dataset, report, seed=2026)
    write_dataset(dataset, command='cp')
    with pytest.raises(ValueError, match='command families'):
        validate_no_holdout_leakage(dataset, report, seed=2026)


def test_requires_independent_test_split(tmp_path):
    dataset = tmp_path / 'dataset.jsonl'
    report = tmp_path / 'report.json'
    write_dataset(dataset)
    write_report(report)

    with pytest.raises(ValueError, match='needs train and test rows'):
        validate_no_holdout_leakage(dataset, report, seed=2026)
