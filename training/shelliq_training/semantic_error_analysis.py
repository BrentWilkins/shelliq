"""Per-example failure analysis and promotion gates for semantic models."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from shelliq_training.semantic_evaluation import (
    GroundingAudit,
    first_command,
    first_command_flags,
    normalize_prompt_variables,
    parse_semantic_document,
)


@dataclass(frozen=True, slots=True)
class ExampleResult:
    record_id: str
    categories: tuple[str, ...]
    expected_command: str | None
    actual_command: str | None
    expected_flags: tuple[str, ...]
    actual_flags: tuple[str, ...]
    structural_exact: bool
    grounded_exact: bool


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    json_parse_rate: float = 1.0
    document_envelope_rate: float = 1.0
    first_command_accuracy: float = 31 / 35
    command_flag_sequence_exact_match: float = 13 / 35
    grounded_document_exact_match: float = 7 / 35


def load_report_examples(path: Path) -> dict[str, tuple[str, str, str]]:
    """Return record ID to (instruction, expected, generated) from either evaluator."""
    report = json.loads(path.read_text())
    examples = report.get('examples')
    if not isinstance(examples, list) or not examples:
        raise ValueError(f'{path}: report has no examples')
    loaded = {}
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError(f'{path}: invalid example')
        record_id = example.get('record_id')
        expected = example.get('expected')
        instruction = example.get('instruction')
        generated = example.get('trained')
        if generated is None:
            candidates = example.get('candidates')
            if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
                generated = candidates[0].get('generated')
        if not all(isinstance(value, str) for value in (record_id, expected, instruction, generated)):
            raise ValueError(f'{path}: malformed example fields')
        if record_id in loaded:
            raise ValueError(f'{path}: duplicate record_id {record_id!r}')
        loaded[record_id] = (instruction, expected, generated)
    return loaded


def classify_example(record_id: str, expected_text: str, actual_text: str, audit: GroundingAudit) -> ExampleResult:
    expected = parse_semantic_document(expected_text)
    if expected is None:
        raise ValueError(f'{record_id}: expected output is not SemanticDocumentV2')
    actual = parse_semantic_document(actual_text)
    categories = []
    try:
        json.loads(actual_text)
    except json.JSONDecodeError, TypeError:
        categories.append('invalid_json')
    if actual is None:
        categories.append('invalid_envelope')

    expected_pair = first_command_flags(expected)
    actual_pair = first_command_flags(actual)
    expected_command = first_command(expected)
    actual_command = first_command(actual)
    expected_flags = expected_pair[1] if expected_pair is not None else ()
    actual_flags = actual_pair[1] if actual_pair is not None else ()
    if actual_command != expected_command:
        categories.append('wrong_command')
    elif actual_flags != expected_flags:
        missing = list((Counter(expected_flags) - Counter(actual_flags)).elements())
        extra = list((Counter(actual_flags) - Counter(expected_flags)).elements())
        if missing:
            categories.append('missing_flags')
        if extra:
            categories.append('extra_flags')
        if not missing and not extra:
            categories.append('reordered_flags')

    structural_exact = actual == expected
    pointers = audit.get(record_id)
    if pointers is None:
        raise ValueError(f'{record_id}: missing grounding audit annotation')
    normalized_expected = normalize_prompt_variables(expected, pointers)
    normalized_actual = normalize_prompt_variables(actual, pointers)
    grounded_exact = normalized_expected is not None and normalized_actual == normalized_expected
    if actual is not None and actual_command == expected_command and actual_flags == expected_flags and not grounded_exact:
        categories.append('operand_or_structure_mismatch')
    if not categories and not structural_exact:
        categories.append('audited_literal_only')
    if not categories:
        categories.append('exact')
    return ExampleResult(
        record_id=record_id,
        categories=tuple(categories),
        expected_command=expected_command,
        actual_command=actual_command,
        expected_flags=expected_flags,
        actual_flags=actual_flags,
        structural_exact=structural_exact,
        grounded_exact=grounded_exact,
    )


def analyze_examples(
    examples: Mapping[str, tuple[str, str, str]], audit: GroundingAudit
) -> tuple[dict[str, float | int], list[ExampleResult]]:
    results = [
        classify_example(record_id, expected, generated, audit) for record_id, (_, expected, generated) in examples.items()
    ]
    total = len(results)
    parse_count = sum('invalid_json' not in result.categories for result in results)
    envelope_count = sum('invalid_envelope' not in result.categories for result in results)
    command_count = sum(result.actual_command == result.expected_command for result in results)
    flag_count = sum(
        result.actual_command == result.expected_command and result.actual_flags == result.expected_flags for result in results
    )
    metrics: dict[str, float | int] = {
        'total_examples': total,
        'json_parse_rate': parse_count / total,
        'document_envelope_rate': envelope_count / total,
        'structural_exact_match': sum(result.structural_exact for result in results) / total,
        'grounded_document_exact_match': sum(result.grounded_exact for result in results) / total,
        'first_command_accuracy': command_count / total,
        'command_flag_sequence_exact_match': flag_count / total,
    }
    return metrics, results


def compare_results(baseline: Sequence[ExampleResult], candidate: Sequence[ExampleResult]) -> dict[str, object]:
    baseline_by_id = {result.record_id: result for result in baseline}
    candidate_by_id = {result.record_id: result for result in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError('baseline and candidate benchmark record IDs differ')
    improvements = []
    regressions = []
    for record_id in baseline_by_id:
        old = baseline_by_id[record_id]
        new = candidate_by_id[record_id]
        old_score = (old.actual_command == old.expected_command, old.actual_flags == old.expected_flags, old.grounded_exact)
        new_score = (new.actual_command == new.expected_command, new.actual_flags == new.expected_flags, new.grounded_exact)
        if new_score > old_score:
            improvements.append(record_id)
        elif new_score < old_score:
            regressions.append(record_id)
    return {'improvements': improvements, 'regressions': regressions}


def gate_metrics(metrics: Mapping[str, float | int], thresholds: PromotionThresholds) -> list[str]:
    """Return promotion failures; flag and grounded scores must beat incumbents."""
    failures = []
    for name in ('json_parse_rate', 'document_envelope_rate', 'first_command_accuracy'):
        if float(metrics[name]) < getattr(thresholds, name):
            failures.append(f'{name}={float(metrics[name]):.4f} below {getattr(thresholds, name):.4f}')
    for name in ('command_flag_sequence_exact_match', 'grounded_document_exact_match'):
        if float(metrics[name]) <= getattr(thresholds, name):
            failures.append(f'{name}={float(metrics[name]):.4f} must exceed {getattr(thresholds, name):.4f}')
    return failures


def analysis_document(
    baseline_path: Path,
    candidate_path: Path,
    audit: GroundingAudit,
) -> dict[str, object]:
    baseline_metrics, baseline_results = analyze_examples(load_report_examples(baseline_path), audit)
    candidate_metrics, candidate_results = analyze_examples(load_report_examples(candidate_path), audit)
    counts = Counter(category for result in candidate_results for category in result.categories)
    return {
        'analysis_schema_version': 1,
        'baseline': {'path': str(baseline_path), 'metrics': baseline_metrics},
        'candidate': {
            'path': str(candidate_path),
            'metrics': candidate_metrics,
            'failure_counts': dict(sorted(counts.items())),
            'examples': [asdict(result) for result in candidate_results],
        },
        'comparison': compare_results(baseline_results, candidate_results),
    }
