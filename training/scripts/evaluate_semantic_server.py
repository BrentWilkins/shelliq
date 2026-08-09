#!/usr/bin/env python3
"""Evaluate a GGUF model through a loopback OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.data import SFTRecord, format_user_message, load_semantic_jsonl  # noqa: E402
from shelliq_training.evaluation import ModelPrediction  # noqa: E402
from shelliq_training.semantic_evaluation import (  # noqa: E402
    evaluate_semantic_predictions,
    load_grounding_audit,
)

MAX_RESPONSE_BYTES = 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-dataset', type=Path, required=True)
    parser.add_argument('--reference-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8080/v1/chat/completions')
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--system-prompt')
    parser.add_argument('--candidates', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument(
        '--grounding-audit',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'evaluation' / 'curated-grounding-v1.json',
    )
    return parser.parse_args()


def validate_endpoint(endpoint: str) -> str:
    """Accept only an uncredentialed loopback HTTP chat-completions URL."""
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme != 'http'
        or parsed.hostname not in {'127.0.0.1', '::1'}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != '/v1/chat/completions'
    ):
        raise ValueError('endpoint must be an uncredentialed loopback HTTP /v1/chat/completions URL')
    return endpoint


def parse_completion_response(raw: bytes) -> tuple[str, str | None]:
    """Extract one assistant message from an OpenAI-compatible response."""
    try:
        value = json.loads(raw)
        choices = value['choices']
        text = choices[0]['message']['content']
        model = value.get('model')
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise ValueError('server response is not a chat completion') from error
    if not isinstance(text, str) or not text:
        raise ValueError('server returned empty assistant content')
    if model is not None and not isinstance(model, str):
        raise ValueError('server model identifier must be a string')
    return text, model


def request_completion(
    endpoint: str,
    user_message: str,
    timeout: float,
    system_prompt: str | None = None,
    *,
    temperature: float = 0.0,
    seed: int = 2026,
) -> tuple[str, str | None]:
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': user_message})
    payload = json.dumps(
        {
            'messages': messages,
            'temperature': temperature,
            'seed': seed,
            'max_tokens': 192,
            'stream': False,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.URLError as error:
        raise RuntimeError(f'model server request failed: {error}') from error
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError('server response exceeds one MiB')
    return parse_completion_response(raw)


def selected_records(dataset: list[SFTRecord], reference_report: Path) -> list[SFTRecord]:
    report = json.loads(reference_report.read_text())
    examples = report.get('examples')
    if not isinstance(examples, list) or not examples:
        raise ValueError(f'{reference_report}: expected non-empty examples')
    by_id = {record.record_id: record for record in dataset}
    selected: list[SFTRecord] = []
    for example in examples:
        if not isinstance(example, dict) or not isinstance(example.get('record_id'), str):
            raise ValueError(f'{reference_report}: invalid example')
        record = by_id.get(example['record_id'])
        if record is None:
            raise ValueError(f'{reference_report}: unknown record {example["record_id"]}')
        if example.get('expected') != record.response:
            raise ValueError(f'{reference_report}: target drift for {record.record_id}')
        selected.append(record)
    if len({record.record_id for record in selected}) != len(selected):
        raise ValueError(f'{reference_report}: duplicate record IDs')
    return selected


def oracle_score(
    record: SFTRecord, record_audit: dict[str, tuple[str, ...]], prediction: ModelPrediction
) -> tuple[float, float, float, float]:
    """Label-aware candidate score used only as a decoding upper bound."""
    metrics = evaluate_semantic_predictions([record], [prediction], record_audit)
    return (
        metrics.grounded_document_exact_match,
        metrics.command_flag_sequence_exact_match,
        metrics.first_command_accuracy,
        metrics.document_envelope_rate,
    )


def main() -> None:
    args = parse_args()
    endpoint = validate_endpoint(args.endpoint)
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')
    if not args.output.parent.is_dir():
        raise SystemExit(f'output parent does not exist: {args.output.parent}')
    if args.timeout <= 0:
        raise SystemExit('--timeout must be positive')
    if args.candidates <= 0:
        raise SystemExit('--candidates must be positive')
    if args.temperature < 0:
        raise SystemExit('--temperature must be non-negative')
    if args.candidates > 1 and args.temperature == 0:
        raise SystemExit('--candidates greater than one requires a positive --temperature')

    dataset = load_semantic_jsonl(args.semantic_dataset)
    records = selected_records(dataset, args.reference_report)
    grounding_audit = load_grounding_audit(args.grounding_audit)
    selected_ids = {record.record_id for record in records}
    if selected_ids != set(grounding_audit):
        raise SystemExit('grounding audit record IDs do not match reference selection')

    candidate_predictions: list[list[ModelPrediction]] = []
    model_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        candidates: list[ModelPrediction] = []
        for candidate_index in range(args.candidates):
            started = time.perf_counter()
            text, model_id = request_completion(
                endpoint,
                format_user_message(record),
                args.timeout,
                args.system_prompt,
                temperature=args.temperature,
                seed=args.seed + candidate_index,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            candidates.append(ModelPrediction(record.record_id, text, latency_ms))
            if model_id is not None:
                model_ids.add(model_id)
        candidate_predictions.append(candidates)
        total_latency = sum(candidate.latency_ms for candidate in candidates)
        print(f'[{index}/{len(records)}] {record.record_id}: {total_latency:.1f} ms')

    predictions = [candidates[0] for candidates in candidate_predictions]
    oracle_predictions: list[ModelPrediction] = []
    for record, candidates in zip(records, candidate_predictions, strict=True):
        record_audit = {record.record_id: grounding_audit[record.record_id]}
        oracle_predictions.append(max(candidates, key=partial(oracle_score, record, record_audit)))

    report = {
        'evaluation_schema_version': 1,
        'endpoint': endpoint,
        'system_prompt': args.system_prompt,
        'generation': {
            'candidates': args.candidates,
            'temperature': args.temperature,
            'seed': args.seed,
        },
        'models': sorted(model_ids),
        'semantic_dataset': str(args.semantic_dataset),
        'reference_report': str(args.reference_report),
        'grounding_audit': str(args.grounding_audit),
        'metrics': asdict(evaluate_semantic_predictions(records, predictions, grounding_audit)),
        'oracle_upper_bound_metrics': asdict(evaluate_semantic_predictions(records, oracle_predictions, grounding_audit)),
        'examples': [
            {
                'record_id': record.record_id,
                'instruction': record.instruction,
                'expected': record.response,
                'candidates': [{'generated': candidate.text, 'latency_ms': candidate.latency_ms} for candidate in candidates],
            }
            for record, candidates in zip(records, candidate_predictions, strict=True)
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report['metrics'], sort_keys=True))
    print(f'report: {args.output}')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit(f'{type(error).__name__}: {error}') from error
