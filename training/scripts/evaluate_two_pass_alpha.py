"""Evaluate ShellIQ's complete context-free two-pass runtime against a live server."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CASES = (
    ('copy-tree', 'cp', 'copy a tree while preserving timestamps ownership and symlinks'),
    ('find-large-files', 'find', 'find files larger than one gigabyte under the current directory'),
    ('sort-unique', 'sort', 'sort a file and remove duplicate lines'),
    ('curl-redirect', 'curl', 'download a URL while following redirects'),
    ('tar-gzip', 'tar', 'archive a directory into a gzip compressed tar file'),
)
UNSUPPORTED = ('unsupported-video', 'transcode a video to av1 with opus audio')
SHORTLIST_PREFIX = 'retrieved command shortlist:'


@dataclass(frozen=True)
class Result:
    case_id: str
    expected_command: str | None
    instruction: str
    shortlist: list[str]
    shortlist_hit: bool | None
    command: str | None
    exit_code: int
    latency_ms: float
    passed: bool
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shelliq', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8080/v1/chat/completions')
    parser.add_argument('--timeout-ms', type=int, default=15_000)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def evaluate(
    shelliq: Path,
    index: Path,
    endpoint: str,
    timeout_ms: int,
    case_id: str,
    instruction: str,
    expected_command: str | None,
) -> Result:
    command = [
        str(shelliq),
        '--index',
        str(index),
        'suggest',
        '--endpoint',
        endpoint,
        '--timeout-ms',
        str(timeout_ms),
        *instruction.split(),
    ]
    environment = {**os.environ, 'SHELLIQ_DEBUG_RETRIEVAL': '1'}
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
    latency_ms = (time.perf_counter() - started) * 1000
    stderr_lines = completed.stderr.strip().splitlines()
    shortlist_line = next((line for line in stderr_lines if line.startswith(SHORTLIST_PREFIX)), '')
    shortlist_text = shortlist_line.removeprefix(SHORTLIST_PREFIX).strip()
    shortlist = [name.strip() for name in shortlist_text.split(',') if name.strip()]
    output_lines = completed.stdout.strip().splitlines()
    rendered = output_lines[-1] if output_lines else None
    actual_command = rendered.split(maxsplit=1)[0] if rendered else None

    if expected_command is None:
        passed = completed.returncode != 0 and not shortlist and 'no installed command documentation matched' in completed.stderr
        shortlist_hit = None
    else:
        shortlist_hit = expected_command in shortlist
        passed = completed.returncode == 0 and shortlist_hit and actual_command == expected_command

    return Result(
        case_id=case_id,
        expected_command=expected_command,
        instruction=instruction,
        shortlist=shortlist,
        shortlist_hit=shortlist_hit,
        command=rendered,
        exit_code=completed.returncode,
        latency_ms=latency_ms,
        passed=passed,
        stderr='\n'.join(line for line in stderr_lines if not line.startswith(SHORTLIST_PREFIX)),
    )


def main() -> None:
    args = parse_args()
    cases = [
        evaluate(
            args.shelliq,
            args.index,
            args.endpoint,
            args.timeout_ms,
            case_id,
            instruction,
            expected,
        )
        for case_id, expected, instruction in CASES
    ]
    unsupported_id, unsupported_instruction = UNSUPPORTED
    cases.append(
        evaluate(
            args.shelliq,
            args.index,
            args.endpoint,
            args.timeout_ms,
            unsupported_id,
            unsupported_instruction,
            None,
        )
    )
    supported = [case for case in cases if case.expected_command is not None]
    report = {
        'schema_version': 1,
        'endpoint': args.endpoint,
        'index': str(args.index),
        'metrics': {
            'shortlist_recall': sum(case.shortlist_hit is True for case in supported) / len(supported),
            'end_to_end_pass_rate': sum(case.passed for case in supported) / len(supported),
            'unsupported_abstention': cases[-1].passed,
            'latency_ms': {
                'min': min(case.latency_ms for case in cases),
                'max': max(case.latency_ms for case in cases),
                'mean': sum(case.latency_ms for case in cases) / len(cases),
            },
        },
        'cases': [asdict(case) for case in cases],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report['metrics'], indent=2, sort_keys=True))
    if not all(case.passed for case in cases):
        raise SystemExit('two-pass alpha evaluation failed')


if __name__ == '__main__':
    main()
