import pytest

from shelliq_training.data import Corpus, SFTRecord
from shelliq_training.evaluation import ModelPrediction
from shelliq_training.semantic_evaluation import (
    evaluate_semantic_predictions,
    first_command,
    parse_semantic_document,
)


def record(record_id: str, response: str) -> SFTRecord:
    return SFTRecord(
        record_id=record_id,
        corpus=Corpus.DISTRIBUTABLE,
        source='shelliq-curated',
        license='CC-BY-4.0',
        provenance='test',
        command='printf',
        platform='linux',
        instruction='Print text',
        response=response,
        context='',
    )


def test_semantic_metrics_distinguish_json_envelope_structure_and_command():
    expected = '{"v":2,"d":"zsh","s":[{"t":"p","c":[{"n":{"s":"printf"},"a":[{"s":"hi"}]}]}]}'
    records = [record('one', expected), record('two', expected), record('three', expected)]
    predictions = [
        ModelPrediction('one', expected, 1.0),
        ModelPrediction(
            'two',
            '{"v":2,"d":"zsh","s":[{"t":"p","c":[{"n":{"s":"printf"},"a":[{"s":"bye"}]}]}]}',
            1.0,
        ),
        ModelPrediction('three', '{"v":2', 1.0),
    ]

    metrics = evaluate_semantic_predictions(records, predictions)

    assert metrics.total_examples == 3
    assert metrics.json_parse_rate == pytest.approx(2 / 3)
    assert metrics.document_envelope_rate == pytest.approx(2 / 3)
    assert metrics.structural_exact_match == pytest.approx(1 / 3)
    assert metrics.first_command_accuracy == pytest.approx(2 / 3)
    assert metrics.command_flag_sequence_exact_match == pytest.approx(2 / 3)


def test_semantic_parser_rejects_wrong_contract():
    assert parse_semantic_document('{"v":1,"d":"zsh","s":[]}') is None
    assert parse_semantic_document('{"v":2,"d":"zsh","s":[],"x":1}') is None
    assert first_command({'v': 2, 'd': 'zsh', 's': []}) is None
