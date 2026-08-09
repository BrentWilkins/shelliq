"""Metrics for compact SemanticDocumentV2 generations."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from shelliq_training.data import SFTRecord
from shelliq_training.evaluation import ModelPrediction


@dataclass(frozen=True, slots=True)
class SemanticEvaluationMetrics:
    total_examples: int
    json_parse_rate: float
    document_envelope_rate: float
    structural_exact_match: float
    grounded_document_exact_match: float
    first_command_accuracy: float
    command_flag_sequence_exact_match: float


GroundingAudit = dict[str, tuple[str, ...]]


def load_grounding_audit(path: Path) -> GroundingAudit:
    """Load reviewed JSON-pointer paths whose literal values are prompt variables."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != {'schema_version', 'records'}:
        raise ValueError(f'{path}: expected schema_version and records')
    if value['schema_version'] != 1 or not isinstance(value['records'], dict):
        raise ValueError(f'{path}: unsupported grounding audit schema')

    audit: GroundingAudit = {}
    for record_id, annotation in value['records'].items():
        if not isinstance(record_id, str) or not isinstance(annotation, dict):
            raise ValueError(f'{path}: invalid grounding audit record')
        if set(annotation) != {'variable_literal_paths'}:
            raise ValueError(f'{path}: {record_id}: expected variable_literal_paths')
        paths = annotation['variable_literal_paths']
        if (
            not isinstance(paths, list)
            or not all(isinstance(pointer, str) and pointer.startswith('/') for pointer in paths)
            or len(paths) != len(set(paths))
        ):
            raise ValueError(f'{path}: {record_id}: invalid variable literal paths')
        audit[record_id] = tuple(paths)
    return audit


def _replace_literal_at_pointer(document: dict[str, object], pointer: str) -> bool:
    current: object = document
    parts = pointer.removeprefix('/').split('/')
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdecimal() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False
    leaf = parts[-1]
    if not isinstance(current, dict) or not isinstance(current.get(leaf), str):
        return False
    current[leaf] = '<PROMPT_VARIABLE>'
    return True


def normalize_prompt_variables(document: dict[str, object], pointers: Sequence[str]) -> dict[str, object] | None:
    """Replace audited literal values while preserving the complete AST shape."""
    normalized = deepcopy(document)
    if not all(_replace_literal_at_pointer(normalized, pointer) for pointer in pointers):
        return None
    return normalized


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


def first_command_flags(document: dict[str, object] | None) -> tuple[str, tuple[str, ...]] | None:
    """Extract the first command and its ordered flag-like arguments."""
    command = first_command(document)
    if command is None or document is None:
        return None
    statements = document['s']
    assert isinstance(statements, list)
    stage = statements[0]['c'][0]
    arguments = stage.get('a', [])
    if not isinstance(arguments, list):
        return None
    flags = []
    for argument in arguments:
        if not isinstance(argument, dict) or set(argument) != {'s'}:
            return None
        source = argument['s']
        if not isinstance(source, str):
            return None
        if source.startswith('-'):
            flags.append(source)
    return command, tuple(flags)


def evaluate_semantic_predictions(
    records: Sequence[SFTRecord],
    predictions: Sequence[ModelPrediction],
    grounding_audit: GroundingAudit | None = None,
) -> SemanticEvaluationMetrics:
    if len(records) != len(predictions) or not records:
        raise ValueError('semantic evaluation requires equal non-empty inputs')

    parsed_json = 0
    valid_envelopes = 0
    structural_matches = 0
    grounded_matches = 0
    command_matches = 0
    command_flag_matches = 0
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
        pointers = () if grounding_audit is None else grounding_audit.get(record.record_id)
        if pointers is None:
            raise ValueError(f'{record.record_id}: missing grounding audit annotation')
        normalized_expected = normalize_prompt_variables(expected, pointers)
        if normalized_expected is None:
            raise ValueError(f'{record.record_id}: grounding audit path does not name a literal')
        normalized_actual = normalize_prompt_variables(actual, pointers)
        grounded_matches += normalized_actual == normalized_expected
        command_matches += first_command(actual) == first_command(expected)
        command_flag_matches += first_command_flags(actual) == first_command_flags(expected)

    total = len(records)
    return SemanticEvaluationMetrics(
        total_examples=total,
        json_parse_rate=parsed_json / total,
        document_envelope_rate=valid_envelopes / total,
        structural_exact_match=structural_matches / total,
        grounded_document_exact_match=grounded_matches / total,
        first_command_accuracy=command_matches / total,
        command_flag_sequence_exact_match=command_flag_matches / total,
    )
