import json
from pathlib import Path

from shelliq_training.data import Corpus, load_jsonl
from shelliq_training.semantic_evaluation import load_grounding_audit

TRAINING_ROOT = Path(__file__).resolve().parents[1]
RETENTION_PATH = TRAINING_ROOT / 'evaluation' / 'semantic-retention-v1.jsonl'
GROUNDING_PATH = TRAINING_ROOT / 'evaluation' / 'semantic-retention-grounding-v1.json'
TARGETED_PATH = TRAINING_ROOT / 'corpus' / 'targeted-finishing.jsonl'
RELEASE_COMMANDS = {
    'cargo',
    'cp',
    'getent',
    'gzip',
    'mypy',
    'netstat',
    'nice',
    'pidstat',
    'psql',
    'read',
    'readlink',
    'systemd-analyze',
}


def test_retention_suite_is_frozen_audited_and_outside_training_corpus():
    records = load_jsonl(RETENTION_PATH, corpus=Corpus.DISTRIBUTABLE)
    audit = load_grounding_audit(GROUNDING_PATH)

    assert len(records) == 20
    assert len({record.record_id for record in records}) == 20
    assert {record.record_id for record in records} == set(audit)
    assert all(record.source == 'shelliq-retention' for record in records)
    assert RETENTION_PATH.parent.name == 'evaluation'


def test_retention_commands_are_disjoint_from_focused_benchmarks():
    retention_commands = {record.command for record in load_jsonl(RETENTION_PATH, corpus=Corpus.DISTRIBUTABLE)}
    targeted_commands = {json.loads(line)['command'] for line in TARGETED_PATH.read_text().splitlines()}

    assert retention_commands.isdisjoint(RELEASE_COMMANDS)
    assert retention_commands.isdisjoint(targeted_commands)
