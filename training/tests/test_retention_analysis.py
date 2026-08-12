from shelliq_training.retention_analysis import retention_failures


def metrics(**overrides: float | int) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        'total_examples': 20,
        'json_parse_rate': 1.0,
        'document_envelope_rate': 1.0,
        'first_command_accuracy': 0.8,
        'command_flag_sequence_exact_match': 0.5,
        'grounded_document_exact_match': 0.3,
    }
    result.update(overrides)
    return result


def test_retention_gate_accepts_equal_or_better_metrics():
    assert retention_failures(metrics(), metrics(first_command_accuracy=0.85)) == []


def test_retention_gate_reports_every_regression():
    failures = retention_failures(
        metrics(),
        metrics(first_command_accuracy=0.75, grounded_document_exact_match=0.25),
    )
    assert failures == [
        'first_command_accuracy regressed: 0.8000 -> 0.7500',
        'grounded_document_exact_match regressed: 0.3000 -> 0.2500',
    ]


def test_retention_gate_rejects_different_populations():
    assert retention_failures(metrics(), metrics(total_examples=19)) == ['total_examples differs between baseline and candidate']
