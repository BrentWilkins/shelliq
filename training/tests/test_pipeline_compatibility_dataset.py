import json
from pathlib import Path

from shelliq_training.data import Corpus, load_jsonl
from shelliq_training.semantic_evaluation import load_grounding_audit

TRAINING_ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = TRAINING_ROOT / 'corpus' / 'pipeline-contracts.jsonl'
EVALUATION_PATH = TRAINING_ROOT / 'evaluation' / 'pipeline-compatibility-v1.jsonl'
GROUNDING_PATH = TRAINING_ROOT / 'evaluation' / 'pipeline-compatibility-grounding-v1.json'


def test_pipeline_evaluation_is_frozen_audited_and_outside_training_corpus():
    training = load_jsonl(TRAINING_PATH, corpus=Corpus.DISTRIBUTABLE)
    evaluation = load_jsonl(EVALUATION_PATH, corpus=Corpus.DISTRIBUTABLE)
    audit = load_grounding_audit(GROUNDING_PATH)

    assert len(training) == 12
    assert len(evaluation) == 8
    assert {record.record_id for record in evaluation} == set(audit)
    assert all(record.source == 'shelliq-pipeline-eval' for record in evaluation)
    assert EVALUATION_PATH.parent.name == 'evaluation'


def test_pipeline_evaluation_has_no_exact_training_leakage():
    training = load_jsonl(TRAINING_PATH, corpus=Corpus.DISTRIBUTABLE)
    evaluation = load_jsonl(EVALUATION_PATH, corpus=Corpus.DISTRIBUTABLE)

    training_instructions = {record.instruction.strip().casefold() for record in training}
    training_responses = {record.response for record in training}
    assert not training_instructions & {record.instruction.strip().casefold() for record in evaluation}
    assert not training_responses & {record.response for record in evaluation}


def test_pipeline_suite_covers_framing_and_record_shape():
    evaluation = load_jsonl(EVALUATION_PATH, corpus=Corpus.DISTRIBUTABLE)
    responses = {record.record_id: record.response for record in evaluation}

    regression = responses['pipeline-compatibility:v1:linux:large-files-largest-first']
    assert "-printf '%s %p\\n' | sort -nr" in regression
    assert any('-print0 | sort -z' in response for response in responses.values())
    assert any("-printf '%s %p\\0' | sort -z -nr" in response for response in responses.values())
    assert all('-print0 | sort -n' not in response for response in responses.values())


def test_pipeline_evaluation_jsonl_is_newline_terminated():
    text = EVALUATION_PATH.read_text(encoding='utf-8')
    assert text.endswith('\n')
    assert all(json.loads(line) for line in text.splitlines())
