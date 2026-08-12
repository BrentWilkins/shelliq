"""Compare semantic reports with a strict no-regression retention gate."""

from __future__ import annotations

from collections.abc import Mapping

METRIC_NAMES = (
    'json_parse_rate',
    'document_envelope_rate',
    'first_command_accuracy',
    'command_flag_sequence_exact_match',
    'grounded_document_exact_match',
)


def retention_failures(
    baseline: Mapping[str, float | int],
    candidate: Mapping[str, float | int],
) -> list[str]:
    """Require candidate metrics to equal or exceed the frozen incumbent."""
    failures: list[str] = []
    if baseline.get('total_examples') != candidate.get('total_examples'):
        return ['total_examples differs between baseline and candidate']
    for name in METRIC_NAMES:
        old = float(baseline[name])
        new = float(candidate[name])
        if new < old:
            failures.append(f'{name} regressed: {old:.4f} -> {new:.4f}')
    return failures
