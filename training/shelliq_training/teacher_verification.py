"""Evidence-backed static verification for teacher and student candidates."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shelliq_training.semantic_equivalence import semantic_documents_equivalent
from shelliq_training.semantic_evaluation import first_command, parse_semantic_document


@dataclass(frozen=True)
class RustValidation:
    valid: bool
    rendered: str | None
    error: str | None


@dataclass(frozen=True)
class ReferenceVerification:
    valid_json: bool
    valid_envelope: bool
    rust_round_trip: bool
    first_command_match: bool
    conservative_equivalent: bool
    rendered: str | None
    failures: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.failures


def rust_validate_documents(
    documents: Sequence[str], validator: str | Path = Path('target/debug/validate-semantic-documents')
) -> list[RustValidation]:
    """Validate documents in one subprocess; no generated command is executed."""
    if not documents:
        return []
    completed = subprocess.run(
        [str(validator)],
        input=''.join(json.dumps({'document': document}) + '\n' for document in documents),
        text=True,
        capture_output=True,
        check=True,
    )
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
    if len(rows) != len(documents):
        raise ValueError(f'semantic validator returned {len(rows)} rows for {len(documents)} documents')
    validations: list[RustValidation] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or row.get('line') != index or not isinstance(row.get('valid'), bool):
            raise ValueError(f'invalid semantic validator response at row {index}')
        rendered = row.get('rendered')
        error = row.get('error')
        if rendered is not None and not isinstance(rendered, str):
            raise ValueError(f'invalid rendered command at row {index}')
        if error is not None and not isinstance(error, str):
            raise ValueError(f'invalid semantic validator error at row {index}')
        validations.append(RustValidation(row['valid'], rendered, error))
    return validations


def verify_against_reference(
    generated: str, expected: dict[str, object], rust_validation: RustValidation
) -> ReferenceVerification:
    """Require valid AST plus conservative equivalence to a reviewed target."""
    failures: list[str] = []
    try:
        json.loads(generated)
        valid_json = True
    except json.JSONDecodeError, TypeError:
        valid_json = False
        failures.append('invalid-json')
    actual = parse_semantic_document(generated)
    valid_envelope = actual is not None
    if not valid_envelope:
        failures.append('invalid-semantic-envelope')
    if not rust_validation.valid:
        failures.append('semantic-round-trip-failed')
    command_match = actual is not None and first_command(actual) == first_command(expected)
    if not command_match:
        failures.append('first-command-mismatch')
    equivalent = actual is not None and semantic_documents_equivalent(expected, actual)
    if not equivalent:
        failures.append('reference-mismatch')
    return ReferenceVerification(
        valid_json=valid_json,
        valid_envelope=valid_envelope,
        rust_round_trip=rust_validation.valid,
        first_command_match=command_match,
        conservative_equivalent=equivalent,
        rendered=rust_validation.rendered,
        failures=tuple(failures),
    )
