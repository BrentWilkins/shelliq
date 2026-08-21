#!/usr/bin/env python3
"""Run a resumable verifier-directed local teacher correction cascade."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_teacher_tournament import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    _openai_request,
    _validate_endpoint,
)
from shelliq_training.prompt import PromptContract, format_user_message  # noqa: E402
from shelliq_training.teacher_tournament import prompt_hash  # noqa: E402
from shelliq_training.teacher_verification import (  # noqa: E402
    rust_validate_documents,
    verify_against_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:11434/v1/chat/completions')
    parser.add_argument('--model', required=True)
    parser.add_argument('--validator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def _attempt(
    candidate: dict[str, object],
    *,
    endpoint: str,
    model: str,
    validator: Path,
    reasoning_effort: str,
    seed: int,
    timeout: float,
) -> dict[str, object]:
    message = format_user_message(
        platform=str(candidate['platform']),
        context=str(candidate['context']),
        instruction=str(candidate['instruction']),
        contract=PromptContract.CONTEXT_AUTHORITATIVE_V1,
    )
    started = time.monotonic()
    try:
        generated, actual_model, input_tokens, output_tokens = _openai_request(
            endpoint,
            model,
            DEFAULT_SYSTEM_PROMPT,
            message,
            0.0,
            seed,
            timeout,
            reasoning_effort,
        )
        validation = rust_validate_documents([generated], validator)[0]
        chosen = candidate['chosen']
        if not isinstance(chosen, dict):
            raise ValueError('chosen SemanticDocumentV2 must be an object')
        verification = verify_against_reference(generated, chosen, validation)
        return {
            'reasoning_effort': reasoning_effort,
            'model': actual_model,
            'prompt_hash': prompt_hash(DEFAULT_SYSTEM_PROMPT, message),
            'seed': seed,
            'generated': generated,
            'latency_ms': (time.monotonic() - started) * 1000,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'verification': asdict(verification),
            'error': None,
        }
    except Exception as error:  # retain row-level failures so the run can continue
        return {
            'reasoning_effort': reasoning_effort,
            'model': model,
            'prompt_hash': prompt_hash(DEFAULT_SYSTEM_PROMPT, message),
            'seed': seed,
            'generated': None,
            'latency_ms': (time.monotonic() - started) * 1000,
            'input_tokens': None,
            'output_tokens': None,
            'verification': None,
            'error': f'{type(error).__name__}: {error}',
        }


def attempt_accepted(attempt: dict[str, object]) -> bool:
    verification = attempt.get('verification')
    return isinstance(verification, dict) and verification.get('failures') in ([], ())


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    console = Console()
    endpoint = _validate_endpoint(args.endpoint)
    if not args.queue.is_file() or not args.validator.is_file():
        raise FileNotFoundError('queue and validator must exist')
    if args.timeout <= 0:
        raise ValueError('--timeout must be positive')
    if args.summary.exists():
        raise FileExistsError(f'summary already exists: {args.summary}')
    if not args.output.parent.is_dir() or not args.summary.parent.is_dir():
        raise FileNotFoundError('output parents must exist')

    candidates = _load_jsonl(args.queue)
    completed: dict[str, dict[str, object]] = {}
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f'output already exists: {args.output}')
        completed = {str(row['candidate_id']): row for row in _load_jsonl(args.output)}

    mode = 'a' if completed else 'w'
    with args.output.open(mode, encoding='utf-8') as stream, Progress(console=console) as progress:
        task = progress.add_task('[cyan]teacher cascade', total=len(candidates), completed=len(completed))
        for index, candidate in enumerate(candidates):
            candidate_id = str(candidate['candidate_id'])
            if candidate_id in completed:
                continue
            attempts = [
                _attempt(
                    candidate,
                    endpoint=endpoint,
                    model=args.model,
                    validator=args.validator,
                    reasoning_effort='none',
                    seed=args.seed + index * 2,
                    timeout=args.timeout,
                )
            ]
            if not attempt_accepted(attempts[0]):
                attempts.append(
                    _attempt(
                        candidate,
                        endpoint=endpoint,
                        model=args.model,
                        validator=args.validator,
                        reasoning_effort='medium',
                        seed=args.seed + index * 2 + 1,
                        timeout=args.timeout,
                    )
                )
            accepted_attempt = next((str(attempt['reasoning_effort']) for attempt in attempts if attempt_accepted(attempt)), None)
            row: dict[str, object] = {
                'schema_version': 1,
                'candidate_id': candidate_id,
                'record_id': candidate['record_id'],
                'category': candidate['category'],
                'attempts': attempts,
                'accepted_attempt': accepted_attempt,
                'review_state': 'teacher-verified' if accepted_attempt is not None else 'needs-review',
            }
            stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
            stream.flush()
            completed[candidate_id] = row
            progress.advance(task)

    rows = list(completed.values())
    summary = {
        'schema_version': 1,
        'queue': str(args.queue),
        'model': args.model,
        'records': len(rows),
        'accepted_none': sum(row['accepted_attempt'] == 'none' for row in rows),
        'accepted_medium': sum(row['accepted_attempt'] == 'medium' for row in rows),
        'needs_review': sum(row['accepted_attempt'] is None for row in rows),
        'attempts': sum(len(row['attempts']) for row in rows),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    console.print_json(data=summary)
    console.print(f'results: {args.output}')
    console.print(f'summary: {args.summary}')


if __name__ == '__main__':
    main()
