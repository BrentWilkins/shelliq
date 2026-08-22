#!/usr/bin/env python3
"""Run a resumable local-model prescreen over chosen/rejected preference pairs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_teacher_tournament import _openai_request, _validate_endpoint  # noqa: E402
from shelliq_training.preference_data import FailureMode  # noqa: E402

SYSTEM_PROMPT = """You audit shell-command preference pairs. Never execute either command.
Judge only from the supplied instruction and authoritative context.

Verdicts:
- chosen: chosen clearly satisfies the task and rejected materially fails.
- tie: both satisfy; differences are cosmetic, equivalent, or arbitrary example operands.
- ambiguous: the prompt/context does not justify preferring one answer.
- rejected: the answer labelled rejected is actually better or chosen is wrong.

When verdict is chosen, provide one or more failure_modes from the supplied allowed list.
For every other verdict, failure_modes must be empty.
Return exactly one JSON object: {"verdict":"...","failure_modes":[...],"reason":"..."}.
Be conservative: do not prefer arbitrary filenames, hosts, versions, counts, or option order when both work.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--prior-decisions', type=Path)
    parser.add_argument('--endpoint', default='http://127.0.0.1:11434/v1/chat/completions')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def user_message(candidate: dict[str, object]) -> str:
    allowed = ', '.join(mode.value for mode in FailureMode)
    return (
        f'Platform: {candidate["platform"]}\n'
        f'Instruction: {candidate["instruction"]}\n'
        f'Authoritative context: {candidate["context"]}\n'
        f'Chosen command: {candidate["chosen_rendered"]}\n'
        f'Rejected command: {candidate["student_rendered"]}\n'
        f'Allowed failure_modes: {allowed}'
    )


def parse_adjudication(generated: str) -> dict[str, object]:
    value = json.loads(generated)
    if not isinstance(value, dict) or set(value) != {'verdict', 'failure_modes', 'reason'}:
        raise ValueError('adjudication must contain exactly verdict, failure_modes, and reason')
    verdict = value['verdict']
    modes = value['failure_modes']
    reason = value['reason']
    if verdict not in {'chosen', 'tie', 'ambiguous', 'rejected'}:
        raise ValueError('invalid verdict')
    if not isinstance(modes, list) or not all(isinstance(mode, str) for mode in modes):
        raise ValueError('failure_modes must be strings')
    allowed = {mode.value for mode in FailureMode}
    if len(modes) != len(set(modes)) or not set(modes) <= allowed:
        raise ValueError('invalid or duplicate failure mode')
    if (verdict == 'chosen') != bool(modes):
        raise ValueError('chosen verdict requires modes; other verdicts forbid them')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('reason must be non-empty')
    return {'verdict': verdict, 'failure_modes': modes, 'reason': reason.strip()}


def main() -> None:
    args = parse_args()
    endpoint = _validate_endpoint(args.endpoint)
    if not args.queue.is_file() or (args.prior_decisions and not args.prior_decisions.is_file()):
        raise FileNotFoundError('queue and prior decisions must exist')
    if args.timeout <= 0:
        raise ValueError('--timeout must be positive')
    if args.summary.exists():
        raise FileExistsError(args.summary)
    if not args.output.parent.is_dir() or not args.summary.parent.is_dir():
        raise FileNotFoundError('output parents must exist')
    prior_ids = {str(row['record_id']) for row in _load_jsonl(args.prior_decisions)} if args.prior_decisions else set()
    candidates = [row for row in _load_jsonl(args.queue) if str(row['record_id']) not in prior_ids]
    completed: dict[str, dict[str, object]] = {}
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        completed = {str(row['record_id']): row for row in _load_jsonl(args.output)}

    console = Console()
    mode = 'a' if completed else 'w'
    with args.output.open(mode, encoding='utf-8') as stream, Progress(console=console) as progress:
        task = progress.add_task('[cyan]preference prescreen', total=len(candidates), completed=len(completed))
        for index, candidate in enumerate(candidates):
            record_id = str(candidate['record_id'])
            if record_id in completed:
                continue
            message = user_message(candidate)
            started = time.monotonic()
            try:
                generated, actual_model, input_tokens, output_tokens = _openai_request(
                    endpoint,
                    args.model,
                    SYSTEM_PROMPT,
                    message,
                    0.0,
                    args.seed + index,
                    args.timeout,
                    'none',
                )
                adjudication = parse_adjudication(generated)
                error = None
            except Exception as caught:  # preserve row-level errors so the run remains resumable
                generated = None
                actual_model = args.model
                input_tokens = None
                output_tokens = None
                adjudication = None
                error = f'{type(caught).__name__}: {caught}'
            row = {
                'schema_version': 1,
                'record_id': record_id,
                'model': actual_model,
                'adjudication': adjudication,
                'generated': generated,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'latency_ms': (time.monotonic() - started) * 1000,
                'error': error,
                'review_state': 'prescreen-only',
            }
            stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
            stream.flush()
            completed[record_id] = row
            progress.advance(task)

    rows = list(completed.values())
    verdicts = {verdict: 0 for verdict in ('chosen', 'tie', 'ambiguous', 'rejected')}
    for row in rows:
        adjudication = row['adjudication']
        if isinstance(adjudication, dict):
            verdicts[str(adjudication['verdict'])] += 1
    summary = {
        'schema_version': 1,
        'queue_records': len(_load_jsonl(args.queue)),
        'prior_decisions': len(prior_ids),
        'prescreened': len(rows),
        'verdicts': verdicts,
        'errors': sum(row['error'] is not None for row in rows),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    console.print_json(data=summary)


if __name__ == '__main__':
    main()
