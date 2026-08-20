from collections import Counter
from pathlib import Path

from shelliq_training.data import Corpus, load_jsonl
from shelliq_training.semantic_evaluation import load_grounding_audit

TRAINING_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = TRAINING_ROOT / 'corpus'
SHADOW_PATH = TRAINING_ROOT / 'evaluation' / 'semantic-shadow-release-v1.jsonl'
GROUNDING_PATH = TRAINING_ROOT / 'evaluation' / 'semantic-shadow-release-grounding-v1.json'


def _all_training_records():
    return [record for path in sorted(CORPUS_ROOT.glob('*.jsonl')) for record in load_jsonl(path, corpus=Corpus.DISTRIBUTABLE)]


def test_shadow_release_is_balanced_closed_and_training_disjoint():
    shadow = load_jsonl(SHADOW_PATH, corpus=Corpus.DISTRIBUTABLE)
    training = _all_training_records()
    audit = load_grounding_audit(GROUNDING_PATH)
    counts = Counter(record.command for record in shadow)

    assert len(shadow) == 20
    assert len(counts) == 10
    assert set(counts.values()) == {2}
    assert all(record.source == 'shelliq-shadow-release' for record in shadow)
    assert not set(counts) & {record.command for record in training}
    assert set(audit) == {record.record_id for record in shadow}
    assert all(not paths for paths in audit.values())


def test_shadow_release_has_no_exact_corpus_targets_or_instructions():
    shadow = load_jsonl(SHADOW_PATH, corpus=Corpus.DISTRIBUTABLE)
    training = _all_training_records()

    assert not {record.response for record in shadow} & {record.response for record in training}
    assert not {record.instruction.casefold() for record in shadow} & {record.instruction.casefold() for record in training}


def test_shadow_release_jsonl_is_newline_terminated():
    assert SHADOW_PATH.read_bytes().endswith(b'\n')
