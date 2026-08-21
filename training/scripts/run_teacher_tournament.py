#!/usr/bin/env python3
"""Run and score provider-neutral ShellIQ teacher-model tournaments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.prompt import (  # noqa: E402
    TEACHER_SYSTEM_PROMPT_V1,
    PromptContract,
    format_user_message,
)
from shelliq_training.teacher_tournament import (  # noqa: E402
    CandidateResult,
    TeacherChallenge,
    load_challenges,
    load_results,
    prompt_hash,
    score_results,
)

MAX_RESPONSE_BYTES = 1024 * 1024
DEFAULT_SYSTEM_PROMPT = TEACHER_SYSTEM_PROMPT_V1


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):  # noqa: ANN001
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    local = subparsers.add_parser('run-local', help='query a loopback OpenAI-compatible server')
    _challenge_argument(local)
    local.add_argument('--endpoint', default='http://127.0.0.1:8080/v1/chat/completions')
    local.add_argument('--model', required=True, help='stable label recorded in the result')
    local.add_argument('--output', type=Path, required=True)
    local.add_argument('--limit', type=int)
    local.add_argument('--timeout', type=float, default=120.0)
    local.add_argument('--max-total-seconds', type=float, default=3600.0)
    local.add_argument('--temperature', type=float, default=0.1)
    local.add_argument('--seed', type=int, default=2026)
    local.add_argument('--sample', type=int, default=0)
    local.add_argument(
        '--reasoning-effort',
        choices=('none', 'low', 'medium', 'high'),
        default='none',
        help='OpenAI-compatible reasoning control; none prevents hidden reasoning from dominating local runtime',
    )
    local.add_argument('--system-prompt', default=DEFAULT_SYSTEM_PROMPT)

    export = subparsers.add_parser('export-batch', help='write a provider batch request artifact')
    _challenge_argument(export)
    export.add_argument('--provider', choices=('anthropic', 'fireworks', 'openai'), required=True)
    export.add_argument('--model', required=True)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--limit', type=int)
    export.add_argument('--temperature', type=float, default=0.1)
    export.add_argument('--max-output-tokens', type=int, default=1024)
    export.add_argument('--input-price', type=float, required=True, help='USD per million input tokens')
    export.add_argument('--output-price', type=float, required=True, help='USD per million output tokens')
    export.add_argument('--max-cost-usd', type=float, default=5.0)
    export.add_argument('--system-prompt', default=DEFAULT_SYSTEM_PROMPT)

    imported = subparsers.add_parser('import-batch', help='normalize downloaded provider batch results')
    imported.add_argument('--provider', choices=('anthropic', 'fireworks', 'openai'), required=True)
    imported.add_argument('--model', required=True)
    imported.add_argument('--requests', type=Path, required=True)
    imported.add_argument('--responses', type=Path, required=True)
    imported.add_argument('--output', type=Path, required=True)
    imported.add_argument('--temperature', type=float, default=0.1)
    imported.add_argument('--sample', type=int, default=0)

    score = subparsers.add_parser('score', help='score one normalized provider/model/sample result set')
    _challenge_argument(score)
    score.add_argument('--results', type=Path, required=True)
    score.add_argument('--output', type=Path, required=True)

    return parser.parse_args()


def _challenge_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--challenges', type=Path, required=True)


def _validate_output(path: Path) -> None:
    if path.exists():
        raise ValueError(f'output already exists: {path}')
    if not path.parent.is_dir():
        raise ValueError(f'output parent does not exist: {path.parent}')


def _validate_endpoint(value: str) -> str:
    endpoint = urllib.parse.urlsplit(value)
    if (
        endpoint.scheme != 'http'
        or endpoint.hostname not in {'127.0.0.1', '::1'}
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.query
        or endpoint.fragment
        or endpoint.path != '/v1/chat/completions'
    ):
        raise ValueError('endpoint must be an uncredentialed loopback HTTP /v1/chat/completions URL')
    return value


def _user_message(challenge: TeacherChallenge) -> str:
    return format_user_message(
        platform=challenge.platform.value,
        context=challenge.context,
        instruction=challenge.instruction,
        contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
    )


def _openai_request(
    endpoint: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    seed: int,
    timeout: float,
    reasoning_effort: str,
) -> tuple[str, str, int | None, int | None]:
    body = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message},
        ],
        'temperature': temperature,
        'seed': seed,
        'max_tokens': 1024,
        'reasoning_effort': reasoning_effort,
        'stream': False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
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
    value = json.loads(raw)
    content = value['choices'][0]['message']['content']
    actual_model = value.get('model', model)
    usage = value.get('usage', {})
    if not isinstance(content, str) or not isinstance(actual_model, str):
        raise ValueError('invalid chat completion response')
    input_tokens = usage.get('prompt_tokens') if isinstance(usage, dict) else None
    output_tokens = usage.get('completion_tokens') if isinstance(usage, dict) else None
    return content, actual_model, input_tokens, output_tokens


def run_local(args: argparse.Namespace, console: Console) -> None:
    _validate_output(args.output)
    endpoint = _validate_endpoint(args.endpoint)
    challenges = load_challenges(args.challenges)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('--limit must be positive')
        challenges = challenges[: args.limit]
    if args.max_total_seconds <= 0 or args.timeout <= 0:
        raise ValueError('timeouts must be positive')
    results: list[CandidateResult] = []
    started = time.monotonic()
    with Progress(console=console) as progress:
        task = progress.add_task(f'[cyan]{args.model}', total=len(challenges))
        for index, challenge in enumerate(challenges):
            elapsed = time.monotonic() - started
            if index and elapsed / index * len(challenges) > args.max_total_seconds:
                raise RuntimeError(f'projected runtime exceeds {args.max_total_seconds:.0f}s after {index} challenges')
            message = _user_message(challenge)
            before = time.monotonic()
            try:
                generated, actual_model, input_tokens, output_tokens = _openai_request(
                    endpoint,
                    args.model,
                    args.system_prompt,
                    message,
                    args.temperature,
                    args.seed + index,
                    args.timeout,
                    args.reasoning_effort,
                )
                result = CandidateResult(
                    challenge_id=challenge.challenge_id,
                    provider='local',
                    model=actual_model,
                    prompt_hash=prompt_hash(args.system_prompt, message),
                    sample=args.sample,
                    temperature=args.temperature,
                    seed=args.seed + index,
                    generated=generated,
                    latency_ms=(time.monotonic() - before) * 1000,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=0.0,
                    error=None,
                )
            except Exception as error:  # preserve row-level failure for comparison
                result = CandidateResult(
                    challenge_id=challenge.challenge_id,
                    provider='local',
                    model=args.model,
                    prompt_hash=prompt_hash(args.system_prompt, message),
                    sample=args.sample,
                    temperature=args.temperature,
                    seed=args.seed + index,
                    generated=None,
                    latency_ms=(time.monotonic() - before) * 1000,
                    input_tokens=None,
                    output_tokens=None,
                    cost_usd=0.0,
                    error=f'{type(error).__name__}: {error}',
                )
            results.append(result)
            progress.advance(task)
    args.output.write_text(''.join(json.dumps(result.to_dict(), sort_keys=True) + '\n' for result in results))
    console.print(f'results: {args.output}')


def _batch_row(args: argparse.Namespace, challenge: TeacherChallenge) -> dict[str, object]:
    message = _user_message(challenge)
    safe_prefix = re.sub(r'[^A-Za-z0-9_-]', '-', challenge.challenge_id)[:50]
    custom_id = f'{safe_prefix}-{hashlib.sha256(challenge.challenge_id.encode()).hexdigest()[:12]}'
    messages = [{'role': 'user', 'content': message}]
    if args.provider == 'anthropic':
        return {
            'custom_id': custom_id,
            'params': {
                'model': args.model,
                'system': args.system_prompt,
                'messages': messages,
                'temperature': args.temperature,
                'max_tokens': args.max_output_tokens,
            },
        }
    body = {
        'model': args.model,
        'messages': [{'role': 'system', 'content': args.system_prompt}, *messages],
        'temperature': args.temperature,
        'max_tokens': args.max_output_tokens,
    }
    if args.provider == 'openai':
        return {
            'custom_id': custom_id,
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': body,
        }
    return {'custom_id': custom_id, 'body': body}


def export_batch(args: argparse.Namespace, console: Console) -> None:
    _validate_output(args.output)
    manifest_path = _manifest_path(args.output)
    _validate_output(manifest_path)
    challenges = load_challenges(args.challenges)
    if args.limit is not None:
        challenges = challenges[: args.limit]
    if not challenges or args.max_output_tokens <= 0:
        raise ValueError('batch must contain challenges and positive output limit')
    rows = [_batch_row(args, challenge) for challenge in challenges]
    input_estimate = sum(len(json.dumps(row, ensure_ascii=False)) for row in rows) / 4
    maximum_cost = input_estimate / 1_000_000 * args.input_price + (
        len(rows) * args.max_output_tokens / 1_000_000 * args.output_price
    )
    if maximum_cost > args.max_cost_usd:
        raise ValueError(f'worst-case estimated cost ${maximum_cost:.4f} exceeds cap ${args.max_cost_usd:.2f}')
    if args.provider == 'anthropic':
        args.output.write_text(json.dumps({'requests': rows}, ensure_ascii=False, separators=(',', ':')) + '\n')
    else:
        args.output.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows))
    request_ids = {str(row['custom_id']): challenge.challenge_id for row, challenge in zip(rows, challenges, strict=True)}
    manifest = {
        'schema_version': 1,
        'provider': args.provider,
        'model': args.model,
        'temperature': args.temperature,
        'request_ids': request_ids,
        'prompt_hashes': {
            str(row['custom_id']): prompt_hash(args.system_prompt, _user_message(challenge))
            for row, challenge in zip(rows, challenges, strict=True)
        },
        'maximum_estimated_cost_usd': maximum_cost,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    console.print(f'requests: {args.output}')
    console.print(f'manifest: {manifest_path}')
    console.print(f'worst-case token estimate: ${maximum_cost:.4f}')


def _request_metadata(path: Path) -> dict[str, tuple[str, str]]:
    value = json.loads(_manifest_path(path).read_text())
    hashes = value.get('prompt_hashes')
    request_ids = value.get('request_ids')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('request manifest has no prompt hashes')
    if not isinstance(request_ids, dict) or request_ids.keys() != hashes.keys():
        raise ValueError('request manifest has invalid request ID mapping')
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in hashes.items()):
        raise ValueError('request manifest contains invalid prompt hashes')
    if not all(isinstance(item, str) for item in request_ids.values()):
        raise ValueError('request manifest contains invalid challenge IDs')
    return {key: (request_ids[key], hashes[key]) for key in hashes}


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.name + '.manifest.json')


def _provider_response(provider: str, row: dict[str, object]) -> tuple[str, str, int | None, int | None, float | None]:
    custom_id = row.get('custom_id')
    if not isinstance(custom_id, str):
        raise ValueError('provider row is missing custom_id')
    if provider == 'anthropic':
        result = row['result']
        if not isinstance(result, dict) or result.get('type') != 'succeeded':
            raise ValueError(f'{custom_id}: unsuccessful Anthropic result')
        message = result['message']
        assert isinstance(message, dict)
        content = message['content']
        usage = message.get('usage', {})
        text = content[0]['text']  # type: ignore[index]
        return custom_id, text, usage.get('input_tokens'), usage.get('output_tokens'), None  # type: ignore[union-attr]
    response = row.get('response')
    if not isinstance(response, dict):
        raise ValueError(f'{custom_id}: missing response')
    body = response.get('body', response)
    if not isinstance(body, dict):
        raise ValueError(f'{custom_id}: invalid response body')
    text = body['choices'][0]['message']['content']  # type: ignore[index]
    usage = body.get('usage', {})
    cost = usage.get('cost_usd') if isinstance(usage, dict) else None
    return custom_id, text, usage.get('prompt_tokens'), usage.get('completion_tokens'), cost  # type: ignore[union-attr]


def import_batch(args: argparse.Namespace, console: Console) -> None:
    _validate_output(args.output)
    metadata = _request_metadata(args.requests)
    results = []
    seen_custom_ids: set[str] = set()
    with args.responses.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id, generated, input_tokens, output_tokens, cost = _provider_response(args.provider, row)
            if custom_id not in metadata:
                raise ValueError(f'response contains unknown custom_id {custom_id!r}')
            if custom_id in seen_custom_ids:
                raise ValueError(f'provider response contains duplicate custom_id {custom_id!r}')
            seen_custom_ids.add(custom_id)
            challenge_id, request_prompt_hash = metadata[custom_id]
            results.append(
                CandidateResult(
                    challenge_id=challenge_id,
                    provider=args.provider,
                    model=args.model,
                    prompt_hash=request_prompt_hash,
                    sample=args.sample,
                    temperature=args.temperature,
                    seed=None,
                    generated=generated,
                    latency_ms=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    error=None,
                )
            )
    if seen_custom_ids != set(metadata):
        raise ValueError('provider response set is incomplete')
    args.output.write_text(''.join(json.dumps(result.to_dict(), sort_keys=True) + '\n' for result in results))
    console.print(f'results: {args.output}')


def score(args: argparse.Namespace, console: Console) -> None:
    _validate_output(args.output)
    value = score_results(load_challenges(args.challenges), load_results(args.results))
    document = {**asdict(value), 'metrics': asdict(value.metrics)}
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
    table = Table(title='Teacher model tournament')
    for name in ('model', 'envelope', 'flags', 'grounded', 'errors', 'seconds', 'cost'):
        table.add_column(name)
    table.add_row(
        value.model,
        f'{value.metrics.document_envelope_rate:.3f}',
        f'{value.metrics.command_flag_sequence_exact_match:.3f}',
        f'{value.metrics.grounded_document_exact_match:.3f}',
        str(value.errors),
        f'{value.total_latency_ms / 1000:.1f}',
        f'${value.total_cost_usd:.4f}',
    )
    console.print(table)
    console.print(f'report: {args.output}')


def main() -> None:
    args = parse_args()
    console = Console()
    if args.command == 'run-local':
        run_local(args, console)
    elif args.command == 'export-batch':
        export_batch(args, console)
    elif args.command == 'import-batch':
        import_batch(args, console)
    elif args.command == 'score':
        score(args, console)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit(f'{type(error).__name__}: {error}') from error
