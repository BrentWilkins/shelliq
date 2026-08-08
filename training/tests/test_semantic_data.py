import json

import pytest

from shelliq_training.data import DatasetFormatError, load_semantic_jsonl


def semantic_row() -> dict[str, object]:
    return {
        'conversion_schema_version': 1,
        'record_id': 'curated:test:linux:print-ok',
        'corpus': 'distributable',
        'source': 'shelliq-curated',
        'license': 'MIT OR Apache-2.0',
        'provenance': 'fixture:test',
        'command': 'print',
        'platform': 'linux',
        'instruction': 'Print okay.',
        'shell_response': 'print ok',
        'context': 'print: fixture context.',
        'semantic_target': {
            'v': 2,
            'd': 'zsh',
            's': [{'t': 'p', 'c': [{'n': {'s': 'print'}, 'a': [{'s': 'ok'}]}]}],
        },
    }


def test_load_semantic_jsonl_builds_compact_json_target(tmp_path):
    path = tmp_path / 'semantic.jsonl'
    path.write_text(json.dumps(semantic_row()) + '\n')
    record = load_semantic_jsonl(path)[0]
    assert record.response == '{"v":2,"d":"zsh","s":[{"t":"p","c":[{"n":{"s":"print"},"a":[{"s":"ok"}]}]}]}'
    assert record.context.startswith('Output contract: compact SemanticDocumentV2 JSON only.\n')


def test_load_semantic_jsonl_rejects_wrong_semantic_version(tmp_path):
    row = semantic_row()
    row['semantic_target']['v'] = 1  # type: ignore[index]
    path = tmp_path / 'semantic.jsonl'
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(DatasetFormatError, match='invalid semantic schema'):
        load_semantic_jsonl(path)
