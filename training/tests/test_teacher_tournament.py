from __future__ import annotations

import argparse
import json
from dataclasses import replace

import pytest
from rich.console import Console

from scripts.run_teacher_tournament import export_batch, import_batch
from shelliq_training.data import DatasetFormatError
from shelliq_training.teacher_tournament import (
    CandidateResult,
    TeacherChallenge,
    load_challenges,
    score_results,
)


def semantic(command: str = 'printf') -> dict[str, object]:
    return {'v': 2, 'd': 'zsh', 's': [{'t': 'p', 'c': [{'n': {'s': command}, 'a': [{'s': 'ok'}]}]}]}


def challenge(identifier: str = 'teacher-selection:v1:linux:printf-ok') -> dict[str, object]:
    return {
        'schema_version': 1,
        'challenge_id': identifier,
        'category': 'precise',
        'platform': 'linux',
        'command': 'printf',
        'instruction': 'Print ok.',
        'context': 'printf writes its arguments.',
        'expected': semantic(),
        'grounding_paths': [],
        'constraints': ['uses printf', 'prints ok'],
        'risk': 'read-only',
        'functional_eligible': True,
    }


def result(identifier: str = 'teacher-selection:v1:linux:printf-ok') -> CandidateResult:
    return CandidateResult(
        challenge_id=identifier,
        provider='local',
        model='fixture',
        prompt_hash='a' * 64,
        sample=0,
        temperature=0.1,
        seed=2026,
        generated=json.dumps(semantic(), separators=(',', ':')),
        latency_ms=12.0,
        input_tokens=20,
        output_tokens=10,
        cost_usd=0.0,
        error=None,
    )


def test_challenge_schema_is_strict():
    value = challenge()
    assert TeacherChallenge.from_dict(value).functional_eligible
    value['surprise'] = True
    with pytest.raises(DatasetFormatError, match='unknown fields'):
        TeacherChallenge.from_dict(value)


def test_load_challenges_rejects_duplicate_ids(tmp_path):
    path = tmp_path / 'challenges.jsonl'
    line = json.dumps(challenge())
    path.write_text(f'{line}\n{line}\n')
    with pytest.raises(DatasetFormatError, match='duplicate challenge_id'):
        load_challenges(path)


def test_score_results_uses_semantic_metrics():
    score = score_results([TeacherChallenge.from_dict(challenge())], [result()])
    assert score.metrics.document_envelope_rate == 1.0
    assert score.metrics.grounded_document_exact_match == 1.0
    assert score.total_output_tokens == 10


def test_score_results_counts_failed_request_in_denominator():
    failed = replace(
        result('teacher-selection:v1:linux:failed'),
        generated=None,
        latency_ms=120.0,
        error='TimeoutError: fixture',
    )
    score = score_results(
        [
            TeacherChallenge.from_dict(challenge()),
            TeacherChallenge.from_dict(challenge('teacher-selection:v1:linux:failed')),
        ],
        [result(), failed],
    )

    assert score.completed == 1
    assert score.errors == 1
    assert score.metrics.document_envelope_rate == 0.5


def test_candidate_requires_output_xor_error():
    value = result().to_dict()
    value['error'] = 'also failed'
    with pytest.raises(DatasetFormatError, match='exactly one'):
        CandidateResult.from_dict(value)


def test_export_batch_enforces_cost_cap(tmp_path):
    challenge_path = tmp_path / 'challenges.jsonl'
    challenge_path.write_text(json.dumps(challenge()) + '\n')
    output = tmp_path / 'batch.jsonl'
    args = argparse.Namespace(
        challenges=challenge_path,
        provider='openai',
        model='gpt-fixture',
        output=output,
        limit=None,
        temperature=0.1,
        max_output_tokens=1000,
        input_price=1.0,
        output_price=10_000.0,
        max_cost_usd=0.01,
        system_prompt='system',
    )
    with pytest.raises(ValueError, match='exceeds cap'):
        export_batch(args, Console(file=None, quiet=True))
    assert not output.exists()


def test_imports_openai_batch_result(tmp_path):
    requests = tmp_path / 'requests.jsonl'
    responses = tmp_path / 'responses.jsonl'
    output = tmp_path / 'normalized.jsonl'
    requests.write_text(
        json.dumps(
            {
                'custom_id': 'one',
                'body': {},
            }
        )
        + '\n'
    )
    (tmp_path / 'requests.jsonl.manifest.json').write_text(
        json.dumps(
            {
                'schema_version': 1,
                'request_ids': {'one': 'challenge:one'},
                'prompt_hashes': {'one': 'b' * 64},
            }
        )
        + '\n'
    )
    responses.write_text(
        json.dumps(
            {
                'custom_id': 'one',
                'response': {
                    'body': {
                        'choices': [{'message': {'content': json.dumps(semantic())}}],
                        'usage': {'prompt_tokens': 4, 'completion_tokens': 5},
                    }
                },
            }
        )
        + '\n'
    )
    args = argparse.Namespace(
        provider='openai',
        model='fixture',
        requests=requests,
        responses=responses,
        output=output,
        temperature=0.1,
        sample=0,
    )
    import_batch(args, Console(file=None, quiet=True))
    value = json.loads(output.read_text())
    assert value['challenge_id'] == 'challenge:one'
    assert value['input_tokens'] == 4


def test_exports_anthropic_submission_body_with_safe_ids(tmp_path):
    challenge_path = tmp_path / 'challenges.jsonl'
    challenge_path.write_text(json.dumps(challenge('challenge:with:colons')) + '\n')
    output = tmp_path / 'batch.json'
    args = argparse.Namespace(
        challenges=challenge_path,
        provider='anthropic',
        model='claude-fixture',
        output=output,
        limit=None,
        temperature=0.1,
        max_output_tokens=1000,
        input_price=1.0,
        output_price=1.0,
        max_cost_usd=5.0,
        system_prompt='system',
    )

    export_batch(args, Console(file=None, quiet=True))

    request = json.loads(output.read_text())['requests'][0]
    assert request['custom_id'].startswith('challenge-with-colons-')
    assert len(request['custom_id']) <= 64
    manifest = json.loads((tmp_path / 'batch.json.manifest.json').read_text())
    assert manifest['request_ids'] == {request['custom_id']: 'challenge:with:colons'}
