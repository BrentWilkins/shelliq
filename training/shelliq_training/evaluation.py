"""Held-out NL-to-command evaluation with separately reported correctness floors."""

from __future__ import annotations

import math
import shlex
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from shelliq_training.data import SFTRecord, Split
from shelliq_training.privacy import assert_canaries_not_extracted


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    record: SFTRecord
    option_arity: Mapping[str, int]
    safe_for_execution: bool = False


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    record_id: str
    text: str
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    examples: int
    parse_rate: float
    exact_match: float
    command_accuracy: float
    flag_precision: float
    flag_recall: float
    flag_f1: float
    option_argument_accuracy: float
    operand_exact_match: float
    options_known_rate: float | None
    functional_equivalence_rate: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    command: str
    flags: tuple[str, ...]
    option_arguments: tuple[tuple[str, str], ...]
    operands: tuple[str, ...]


OptionsKnown = Callable[[str], bool]
FunctionalEquivalent = Callable[[SFTRecord, str], bool]


def evaluate_predictions(
    examples: Sequence[EvaluationExample],
    predictions: Sequence[ModelPrediction],
    *,
    options_known: OptionsKnown | None = None,
    functional_equivalent: FunctionalEquivalent | None = None,
    forbidden_canaries: frozenset[str] = frozenset(),
) -> EvaluationMetrics:
    """Score one prediction per row; never execute commands itself."""
    if not examples:
        raise ValueError('evaluation examples must not be empty')
    predictions_by_id = _prediction_map(predictions)
    expected_ids = {example.record.record_id for example in examples}
    if predictions_by_id.keys() != expected_ids:
        missing = sorted(expected_ids - predictions_by_id.keys())
        unknown = sorted(predictions_by_id.keys() - expected_ids)
        raise ValueError(f'prediction IDs do not match evaluation set: missing={missing[:3]!r}, unknown={unknown[:3]!r}')
    if forbidden_canaries:
        assert_canaries_not_extracted([prediction.text for prediction in predictions], forbidden_canaries)

    parsed_count = exact = command_matches = operand_matches = 0
    true_flags = predicted_flags = matched_flags = 0
    gold_option_arguments = matched_option_arguments = 0
    known_results: list[bool] = []
    functional_results: list[bool] = []
    latencies: list[float] = []

    for example in examples:
        prediction = predictions_by_id[example.record.record_id]
        expected_text = example.record.response.strip()
        predicted_text = prediction.text.strip()
        exact += predicted_text == expected_text
        if prediction.latency_ms is not None:
            if not math.isfinite(prediction.latency_ms) or prediction.latency_ms < 0:
                raise ValueError(f'{prediction.record_id}: latency_ms must be finite and non-negative')
            latencies.append(prediction.latency_ms)
        if options_known is not None:
            known_results.append(options_known(predicted_text))
        if functional_equivalent is not None and example.safe_for_execution:
            functional_results.append(functional_equivalent(example.record, predicted_text))

        expected = parse_command(expected_text, example.option_arity)
        predicted = parse_command(predicted_text, example.option_arity)
        if expected is None:
            raise ValueError(f'{example.record.record_id}: expected command is outside the basic evaluation grammar')
        true_flags += len(expected.flags)
        gold_option_arguments += len(expected.option_arguments)
        if predicted is None:
            continue
        parsed_count += 1
        command_matches += predicted.command == expected.command
        operand_matches += predicted.operands == expected.operands
        expected_flag_counts = Counter(expected.flags)
        predicted_flag_counts = Counter(predicted.flags)
        predicted_flags += sum(predicted_flag_counts.values())
        matched_flags += sum((expected_flag_counts & predicted_flag_counts).values())
        expected_argument_counts = Counter(expected.option_arguments)
        predicted_argument_counts = Counter(predicted.option_arguments)
        matched_option_arguments += sum((expected_argument_counts & predicted_argument_counts).values())

    flag_precision = _ratio(matched_flags, predicted_flags)
    flag_recall = _ratio(matched_flags, true_flags)
    flag_f1 = 0.0 if flag_precision + flag_recall == 0 else 2 * flag_precision * flag_recall / (flag_precision + flag_recall)
    count = len(examples)
    return EvaluationMetrics(
        examples=count,
        parse_rate=parsed_count / count,
        exact_match=exact / count,
        command_accuracy=command_matches / count,
        flag_precision=flag_precision,
        flag_recall=flag_recall,
        flag_f1=flag_f1,
        option_argument_accuracy=_ratio(matched_option_arguments, gold_option_arguments),
        operand_exact_match=operand_matches / count,
        options_known_rate=_mean(known_results),
        functional_equivalence_rate=_mean(functional_results),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
    )


def parse_command(command_line: str, option_arity: Mapping[str, int]) -> ParsedCommand | None:
    """Parse the intentionally small, non-compound grammar used for metric floors."""
    try:
        lexer = shlex.shlex(command_line, posix=True, punctuation_chars='|&;<>')
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    if not tokens or any(token and set(token) <= set('|&;<>') for token in tokens):
        return None
    command = Path(tokens[0]).name
    flags: list[str] = []
    option_arguments: list[tuple[str, str]] = []
    operands: list[str] = []
    position = 1
    while position < len(tokens):
        token = tokens[position]
        if token.startswith('--') and '=' in token:
            flag, argument = token.split('=', maxsplit=1)
            flags.append(flag)
            if option_arity.get(flag, 0):
                option_arguments.append((flag, argument))
            position += 1
            continue
        if token.startswith('-') and token != '-':
            flags.append(token)
            arity = option_arity.get(token, 0)
            if arity not in {0, 1}:
                raise ValueError(f'unsupported option arity for {token}: {arity}')
            if arity == 1:
                if position + 1 >= len(tokens):
                    position += 1
                    continue
                option_arguments.append((token, tokens[position + 1]))
                position += 2
                continue
            position += 1
            continue
        operands.append(token)
        position += 1
    return ParsedCommand(command, tuple(flags), tuple(option_arguments), tuple(operands))


def validate_heldout_splits(splits: Mapping[Split, Sequence[SFTRecord]]) -> None:
    """Hard gate: a command or record ID may occur in only one split."""
    command_owner: dict[str, Split] = {}
    record_owner: dict[str, Split] = {}
    for split in Split:
        for record in splits.get(split, ()):
            previous_command = command_owner.setdefault(record.command, split)
            if previous_command is not split:
                raise ValueError(f'command {record.command!r} leaks across {previous_command.value} and {split.value}')
            previous_record = record_owner.setdefault(record.record_id, split)
            if previous_record is not split:
                raise ValueError(f'record {record.record_id!r} occurs in multiple splits')


def _prediction_map(predictions: Sequence[ModelPrediction]) -> dict[str, ModelPrediction]:
    result: dict[str, ModelPrediction] = {}
    for prediction in predictions:
        if prediction.record_id in result:
            raise ValueError(f'duplicate prediction for {prediction.record_id}')
        result[prediction.record_id] = prediction
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _mean(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(quantile * len(ordered)) - 1
    return ordered[max(0, index)]
