import pytest

from shelliq_training.data import Corpus, Platform, SFTRecord, Split, split_records
from shelliq_training.evaluation import (
    EvaluationExample,
    ModelPrediction,
    evaluate_predictions,
    parse_command,
    validate_heldout_splits,
)


def record(record_id, command, response, *, source='tldr-pages', platform=Platform.LINUX):
    return SFTRecord(
        record_id=record_id,
        corpus=Corpus.DISTRIBUTABLE,
        source=source,
        license='CC-BY-4.0',
        provenance=f'{source}:{record_id}',
        command=command,
        platform=platform,
        instruction=f'Use {command}',
        response=response,
        context=f'{command}: context',
    )


def test_evaluation_separates_flags_arguments_operands_and_floors():
    examples = [
        EvaluationExample(record('find', 'find', "find src -name '*.py'"), {'-name': 1}, safe_for_execution=True),
        EvaluationExample(record('ls', 'ls', 'ls -la src'), {}, safe_for_execution=True),
    ]
    predictions = [
        ModelPrediction('find', "find src -name '*.py'", latency_ms=10),
        ModelPrediction('ls', 'ls -l src', latency_ms=30),
    ]

    metrics = evaluate_predictions(
        examples,
        predictions,
        options_known=lambda text: '--invented' not in text,
        functional_equivalent=lambda expected, predicted: expected.command == 'find',
    )

    assert metrics.examples == 2
    assert metrics.parse_rate == 1.0
    assert metrics.exact_match == 0.5
    assert metrics.command_accuracy == 1.0
    assert metrics.flag_precision == 0.5
    assert metrics.flag_recall == 0.5
    assert metrics.option_argument_accuracy == 1.0
    assert metrics.operand_exact_match == 1.0
    assert metrics.options_known_rate == 1.0
    assert metrics.functional_equivalence_rate == 0.5
    assert metrics.latency_p50_ms == 10
    assert metrics.latency_p95_ms == 30


def test_parser_rejects_compound_shell_and_tracks_long_option_argument():
    assert parse_command('find . | head', {}) is None
    assert parse_command('find . 2>/tmp/results', {}) is None
    assert parse_command("printf '%s' 'literal|operand'", {}) is not None
    parsed = parse_command('rg --glob=*.py needle src', {'--glob': 1})
    assert parsed is not None
    assert parsed.flags == ('--glob',)
    assert parsed.option_arguments == (('--glob', '*.py'),)
    assert parsed.operands == ('needle', 'src')


def test_evaluation_requires_exact_prediction_set_and_runs_canary_gate():
    examples = [EvaluationExample(record('one', 'echo', 'echo safe'), {})]
    with pytest.raises(ValueError, match='prediction IDs'):
        evaluate_predictions(examples, [])
    with pytest.raises(RuntimeError, match='reproduced'):
        evaluate_predictions(
            examples,
            [ModelPrediction('one', 'echo PLANTED-123')],
            forbidden_canaries=frozenset({'PLANTED-123'}),
        )


def test_source_holdout_forces_same_command_from_other_sources_into_test():
    records = [
        record('tldr-find', 'find', 'find .'),
        record('nl-find', 'find', 'find src', source='NL2Bash'),
        record('tldr-ls', 'ls', 'ls'),
    ]
    splits = split_records(
        records,
        corpus=Corpus.DISTRIBUTABLE,
        seed=4,
        validation_fraction=0,
        test_fraction=0,
        heldout_sources=frozenset({'NL2Bash'}),
    )

    assert {item.record_id for item in splits[Split.TEST]} == {'tldr-find', 'nl-find'}
    validate_heldout_splits(splits)


def test_split_validator_rejects_command_leakage():
    splits = {
        Split.TRAIN: [record('one', 'find', 'find .')],
        Split.VALIDATION: [],
        Split.TEST: [record('two', 'find', 'find src')],
    }
    with pytest.raises(ValueError, match='leaks'):
        validate_heldout_splits(splits)
