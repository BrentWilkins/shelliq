import json

import pytest

from scripts.build_student_failure_pool import _collect_frozen, grounding_document, load_frozen_identifiers, select_pool


def row(identifier: str, *, platform: str = 'linux', instruction: str = 'Do work.') -> dict[str, object]:
    return {
        'record_id': identifier,
        'source': 'shelliq-curated',
        'platform': platform,
        'command': identifier.split(':')[-1],
        'instruction': instruction,
        'shell_response': 'command -a',
    }


def test_collect_frozen_walks_nested_reports():
    ids: set[str] = set()
    instructions: set[str] = set()
    _collect_frozen({'examples': [{'record_id': 'release:1', 'instruction': 'Do Not Train'}]}, ids, instructions)
    assert ids == {'release:1'}
    assert instructions == {'do not train'}


def test_load_frozen_identifiers_reads_json_and_jsonl(tmp_path):
    (tmp_path / 'one.json').write_text(json.dumps({'record_id': 'one', 'instruction': 'First'}))
    (tmp_path / 'two.jsonl').write_text(json.dumps({'record_id': 'two', 'instruction': 'Second'}) + '\n')
    assert load_frozen_identifiers(tmp_path) == ({'one', 'two'}, {'first', 'second'})


def test_select_pool_excludes_frozen_ids_and_instructions(monkeypatch):
    monkeypatch.setattr(
        'scripts.build_student_failure_pool.PILOT_QUOTAS',
        {('linux', 'precise'): 1},
    )
    rows = [
        row('frozen:id'),
        row('same:instruction', instruction='Frozen prompt'),
        row('selected:id', instruction='Fresh prompt'),
    ]
    selected = select_pool(
        rows,
        {str(item['record_id']) for item in rows},
        {'frozen:id'},
        {'frozen prompt'},
        source='shelliq-curated',
        seed=2026,
    )
    assert [item['record_id'] for item in selected] == ['selected:id']


def test_select_pool_rejects_unsatisfied_quota(monkeypatch):
    monkeypatch.setattr(
        'scripts.build_student_failure_pool.PILOT_QUOTAS',
        {('darwin', 'pipeline'): 1},
    )
    with pytest.raises(ValueError, match='cannot satisfy'):
        select_pool([], set(), set(), set(), source='shelliq-curated', seed=2026)


def test_grounding_document_uses_current_closed_schema():
    assert grounding_document([row('selected:id')]) == {
        'schema_version': 1,
        'records': {'selected:id': {'variable_literal_paths': []}},
    }
