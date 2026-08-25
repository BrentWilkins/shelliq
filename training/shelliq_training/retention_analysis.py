"""Compare semantic reports with a strict no-regression retention gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

METRIC_NAMES = (
    'json_parse_rate',
    'document_envelope_rate',
    'first_command_accuracy',
    'command_flag_sequence_exact_match',
    'grounded_document_exact_match',
)


@dataclass(frozen=True, slots=True)
class TieredRetentionThresholds:
    """Absolute protection floors for the 20-case common-shell suite."""

    total_examples: int = 20
    json_parse_count: int = 20
    document_envelope_count: int = 20
    first_command_count: int = 19
    command_flag_sequence_count: int = 13
    grounded_document_count: int = 9


TIERED_RETENTION_THRESHOLDS = TieredRetentionThresholds()


def tiered_retention_failures(
    candidate: Mapping[str, float | int],
    thresholds: TieredRetentionThresholds = TIERED_RETENTION_THRESHOLDS,
) -> list[str]:
    """Return failures against preregistered absolute retention floors."""
    total = int(candidate.get('total_examples', -1))
    if total != thresholds.total_examples:
        return [f'total_examples={total} must equal {thresholds.total_examples}']
    requirements = (
        ('json_parse_rate', thresholds.json_parse_count),
        ('document_envelope_rate', thresholds.document_envelope_count),
        ('first_command_accuracy', thresholds.first_command_count),
        ('command_flag_sequence_exact_match', thresholds.command_flag_sequence_count),
        ('grounded_document_exact_match', thresholds.grounded_document_count),
    )
    failures = []
    for metric, required_count in requirements:
        observed_count = round(float(candidate[metric]) * total)
        if observed_count < required_count:
            failures.append(f'{metric}={observed_count}/{total} below {required_count}/{total}')
    return failures


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
