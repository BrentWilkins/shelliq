from collections import Counter
from pathlib import Path

from shelliq_training.data import Corpus, load_jsonl

TRAINING_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = TRAINING_ROOT / 'corpus' / 'intent-fidelity.jsonl'

# These command families are frozen quality gates. Keeping them out of this
# curriculum makes any score change evidence of transfer rather than leakage.
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
RETENTION_COMMANDS = {
    'awk',
    'chmod',
    'cut',
    'df',
    'docker',
    'du',
    'fd',
    'gh',
    'grep',
    'jq',
    'lsof',
    'ps',
    'sed',
    'sort',
    'tail',
    'tr',
    'uv',
    'wc',
    'xargs',
}


def test_intent_fidelity_is_balanced_and_gate_disjoint():
    records = load_jsonl(DATASET_PATH, corpus=Corpus.DISTRIBUTABLE)
    counts = Counter(record.command for record in records)

    assert len(records) == 48
    assert len(counts) == 12
    assert set(counts.values()) == {4}
    assert not set(counts) & RELEASE_COMMANDS
    assert not set(counts) & RETENTION_COMMANDS
    assert all(record.record_id.startswith('curated:intent-fidelity:') for record in records)


def test_intent_fidelity_covers_binding_and_structure_contrasts():
    responses = [record.response for record in load_jsonl(DATASET_PATH, corpus=Corpus.DISTRIBUTABLE)]

    assert any(' -- ' in response for response in responses)
    assert any(' > ' in response for response in responses)
    assert any(' < ' in response for response in responses)
    assert any(' -o ' in response for response in responses)
    assert any('--target ' in response for response in responses)
    assert any('--signal=' in response for response in responses)
    assert any(' state ' in response for response in responses)


def test_intent_fidelity_jsonl_is_newline_terminated():
    assert DATASET_PATH.read_bytes().endswith(b'\n')
