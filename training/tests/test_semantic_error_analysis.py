import json

from shelliq_training.semantic_error_analysis import (
    PromotionThresholds,
    analyze_examples,
    classify_example,
    gate_metrics,
    load_report_examples,
)


def semantic(command='cp', arguments=('-a', 'src', 'dest')):
    return json.dumps(
        {
            'v': 2,
            'd': 'zsh',
            's': [{'t': 'p', 'c': [{'n': {'s': command}, 'a': [{'s': value} for value in arguments]}]}],
        },
        separators=(',', ':'),
    )


def test_classifies_command_flag_and_operand_failures():
    audit = {'wrong': (), 'flags': (), 'operand': ()}

    wrong = classify_example('wrong', semantic(), semantic('mv'), audit)
    flags = classify_example('flags', semantic(), semantic(arguments=('-n', 'src', 'dest')), audit)
    operand = classify_example('operand', semantic(), semantic(arguments=('-a', 'old', 'new')), audit)

    assert wrong.categories == ('wrong_command',)
    assert flags.categories == ('missing_flags', 'extra_flags')
    assert operand.categories == ('operand_or_structure_mismatch',)


def test_audited_literals_do_not_count_as_grounded_failure():
    result = classify_example(
        'copy',
        semantic(),
        semantic(arguments=('-a', 'old', 'new')),
        {'copy': ('/s/0/c/0/a/1/s', '/s/0/c/0/a/2/s')},
    )

    assert result.categories == ('audited_literal_only',)
    assert result.grounded_exact


def test_loads_checkpoint_and_server_report_shapes(tmp_path):
    common = {'record_id': 'one', 'instruction': 'Copy it', 'expected': semantic()}
    checkpoint = tmp_path / 'checkpoint.json'
    server = tmp_path / 'server.json'
    checkpoint.write_text(json.dumps({'examples': [{**common, 'trained': semantic()}]}))
    server.write_text(json.dumps({'examples': [{**common, 'candidates': [{'generated': semantic()}]}]}))

    assert load_report_examples(checkpoint) == load_report_examples(server)


def test_metrics_and_gate_require_strict_flag_and_grounded_improvement():
    examples = {'one': ('Copy it', semantic(), semantic())}
    metrics, _ = analyze_examples(examples, {'one': ()})
    incumbent = PromotionThresholds(
        first_command_accuracy=1.0,
        command_flag_sequence_exact_match=1.0,
        grounded_document_exact_match=1.0,
    )

    failures = gate_metrics(metrics, incumbent)

    assert len(failures) == 2
    assert failures[0].startswith('command_flag_sequence_exact_match')
    assert failures[1].startswith('grounded_document_exact_match')
