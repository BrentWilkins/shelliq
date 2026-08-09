import pytest

from shelliq_training.data import Corpus, SFTRecord
from shelliq_training.evaluation import ModelPrediction
from shelliq_training.semantic_evaluation import (
    evaluate_semantic_predictions,
    first_command,
    load_grounding_audit,
    normalize_prompt_variables,
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
    assert metrics.grounded_document_exact_match == pytest.approx(1 / 3)
    assert metrics.first_command_accuracy == pytest.approx(2 / 3)
    assert metrics.command_flag_sequence_exact_match == pytest.approx(2 / 3)


def test_semantic_parser_rejects_wrong_contract():
    assert parse_semantic_document('{"v":1,"d":"zsh","s":[]}') is None
    assert parse_semantic_document('{"v":2,"d":"zsh","s":[],"x":1}') is None
    assert first_command({'v': 2, 'd': 'zsh', 's': []}) is None


def test_grounded_exact_match_ignores_only_audited_literal_values():
    expected = '{"v":2,"d":"zsh","s":[{"t":"p","c":[{"n":{"s":"cp"},"a":[{"s":"-a"},{"s":"src/"},{"s":"dest/"}]}]}]}'
    records = [record('copy', expected)]
    predictions = [
        ModelPrediction(
            'copy',
            '{"v":2,"d":"zsh","s":[{"t":"p","c":[{"n":{"s":"cp"},"a":[{"s":"-a"},{"s":"old-tree"},{"s":"new-tree"}]}]}]}',
            1.0,
        )
    ]
    audit = {'copy': ('/s/0/c/0/a/1/s', '/s/0/c/0/a/2/s')}

    metrics = evaluate_semantic_predictions(records, predictions, audit)

    assert metrics.structural_exact_match == 0
    assert metrics.grounded_document_exact_match == 1


def test_grounded_exact_match_requires_audited_shape():
    document = parse_semantic_document('{"v":2,"d":"zsh","s":[{"t":"p","c":[{"n":{"s":"cp"},"a":[]}]}]}')
    assert document is not None
    assert normalize_prompt_variables(document, ['/s/0/c/0/a/1/s']) is None


def test_load_grounding_audit_validates_schema(tmp_path):
    audit_path = tmp_path / 'grounding.json'
    audit_path.write_text('{"schema_version":1,"records":{"one":{"variable_literal_paths":["/s/0/c/0/a/0/s"]}}}')

    assert load_grounding_audit(audit_path) == {'one': ('/s/0/c/0/a/0/s',)}
