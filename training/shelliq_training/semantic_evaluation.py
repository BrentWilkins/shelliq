"""Metrics for compact SemanticDocumentV2 generations."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from shelliq_training.data import SFTRecord
from shelliq_training.evaluation import ModelPrediction


@dataclass(frozen=True, slots=True)
class SemanticEvaluationMetrics:
    total_examples: int
    json_parse_rate: float
    document_envelope_rate: float
    structural_exact_match: float
    first_command_accuracy: float


def parse_semantic_document(text: str) -> dict[str, object] | None:
    """Parse only a SemanticDocumentV2-shaped JSON object."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError, TypeError:
        return None
    if not isinstance(value, dict) or set(value) != {'v', 'd', 's'}:
        return None
    if value['v'] != 2 or value['d'] != 'zsh' or not isinstance(value['s'], list):
        return None
    return value


def first_command(document: dict[str, object] | None) -> str | None:
    """Extract the first pipeline stage's literal command name when present."""
    if document is None:
        return None
    statements = document['s']
    if not isinstance(statements, list) or not statements:
        return None
    statement = statements[0]
    if not isinstance(statement, dict) or statement.get('t') != 'p':
        return None
    stages = statement.get('c')
    if not isinstance(stages, list) or not stages:
        return None
    stage = stages[0]
    if not isinstance(stage, dict):
        return None
    name = stage.get('n')
    if not isinstance(name, dict) or set(name) != {'s'}:
        return None
    source = name['s']
    return source if isinstance(source, str) else None


def evaluate_semantic_predictions(
    records: Sequence[SFTRecord], predictions: Sequence[ModelPrediction]
) -> SemanticEvaluationMetrics:
    if len(records) != len(predictions) or not records:
        raise ValueError('semantic evaluation requires equal non-empty inputs')

    parsed_json = 0
    valid_envelopes = 0
    structural_matches = 0
    command_matches = 0
    for record, prediction in zip(records, predictions, strict=True):
        try:
            json.loads(prediction.text)
        except json.JSONDecodeError, TypeError:
            pass
        else:
            parsed_json += 1

        expected = parse_semantic_document(record.response)
        actual = parse_semantic_document(prediction.text)
        if expected is None:
            raise ValueError(f'{record.record_id}: invalid expected semantic target')
        if actual is not None:
            valid_envelopes += 1
            structural_matches += actual == expected
        command_matches += first_command(actual) == first_command(expected)

    total = len(records)
    return SemanticEvaluationMetrics(
        total_examples=total,
        json_parse_rate=parsed_json / total,
        document_envelope_rate=valid_envelopes / total,
        structural_exact_match=structural_matches / total,
        first_command_accuracy=command_matches / total,
    )
