#!/usr/bin/env python3
"""Evaluate a frozen P1B suite through the real ShellIQ suggest command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.p1b_runtime import Case, load_suite, score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--shelliq', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8080/v1/chat/completions')
    parser.add_argument('--timeout-ms', type=int, default=15_000)
    parser.add_argument('--runtime-label', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def run_case(args: argparse.Namespace, case: Case) -> dict[str, object]:
    command = [
        str(args.shelliq),
        '--index',
        str(args.index),
        'suggest',
        '--json',
        '--endpoint',
        args.endpoint,
        '--timeout-ms',
        str(args.timeout_ms),
    ]
    if case.context is not None:
        command.extend(('--context', case.context))
    command.extend(('--', case.instruction))
    environment = {**os.environ, 'SHELLIQ_DEBUG_RETRIEVAL': '1'}
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
            timeout=args.timeout_ms / 1000 + 5,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        result = score(case, completed.returncode, completed.stdout)
        stderr = completed.stderr
        returncode = completed.returncode
        stdout = completed.stdout
    except subprocess.TimeoutExpired as error:
        latency_ms = (time.perf_counter() - started) * 1000
        result = None
        stderr = (error.stderr or '') if isinstance(error.stderr, str) else ''
        stdout = (error.stdout or '') if isinstance(error.stdout, str) else ''
        returncode = None

    shortlist_prefix = 'retrieved command shortlist:'
    shortlist = []
    stderr_lines = stderr.strip().splitlines()
    for line in stderr_lines:
        if line.startswith(shortlist_prefix):
            shortlist = [item.strip() for item in line.removeprefix(shortlist_prefix).split(',') if item.strip()]
            break
    filtered_stderr = '\n'.join(line for line in stderr_lines if not line.startswith(shortlist_prefix))
    if result is None:
        scored = {
            'passed': False,
            'failures': ['process-timeout'],
            'status': None,
            'source': None,
            'command': None,
            'first_command': None,
            'arguments': [],
        }
    else:
        scored = asdict(result)
    return {
        'id': case.case_id,
        'partition': case.partition,
        'mode': case.mode,
        'instruction': case.instruction,
        'expected': asdict(case.expected),
        'shortlist': shortlist,
        'returncode': returncode,
        'latency_ms': latency_ms,
        **scored,
        'stdout': stdout.strip(),
        'stderr': filtered_stderr,
    }


def decision_for(metrics: dict[str, object], decision: dict[str, object], runtime_label: str) -> tuple[str, list[str]]:
    failures: list[str] = []
    partitions = metrics['partitions']
    assert isinstance(partitions, dict)
    for partition in ('supported', 'unsupported', 'prompt_injection', 'unsafe'):
        required = decision[f'minimum_{partition}_pass_rate']
        actual = partitions[partition]['pass_rate']
        if actual < required:
            failures.append(f'{partition} pass rate {actual:.3f} below {required:.3f}')
    latency_limit = decision[f'maximum_warm_{runtime_label}_p95_ms']
    latency = metrics['warm_latency_ms']['p95']
    if latency > latency_limit:
        failures.append(f'warm {runtime_label} p95 {latency:.1f} ms exceeds {latency_limit:.1f} ms')

    unsafe_failed = partitions['unsafe']['pass_rate'] < decision['minimum_unsafe_pass_rate']
    injection_failed = partitions['prompt_injection']['pass_rate'] < decision['minimum_prompt_injection_pass_rate']
    if not failures:
        return 'ship', []
    if unsafe_failed or injection_failed:
        return 'replace_or_harden_boundary', failures
    return 'tune', failures


def main() -> None:
    args = parse_args()
    for path in (args.suite, args.shelliq, args.index, args.model):
        if not path.is_file():
            raise SystemExit(f'required file does not exist: {path}')
    if args.timeout_ms <= 0:
        raise SystemExit('--timeout-ms must be positive')
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')
    if not args.output.parent.is_dir():
        raise SystemExit(f'output parent does not exist: {args.output.parent}')

    suite_id, cases, decision = load_suite(args.suite)
    warmup = next(case for case in cases if case.partition == 'supported')
    warmup_result = run_case(args, warmup)
    results = [run_case(args, case) for case in cases]

    partitions: dict[str, dict[str, object]] = {}
    for partition in sorted({case.partition for case in cases}):
        selected = [item for item in results if item['partition'] == partition]
        passed = sum(item['passed'] is True for item in selected)
        partitions[partition] = {
            'cases': len(selected),
            'passed': passed,
            'pass_rate': passed / len(selected),
        }
    latencies = [float(item['latency_ms']) for item in results]
    metrics: dict[str, object] = {
        'cases': len(results),
        'passed': sum(item['passed'] is True for item in results),
        'partitions': partitions,
        'warm_latency_ms': {
            'mean': statistics.fmean(latencies),
            'p95': percentile_95(latencies),
            'maximum': max(latencies),
        },
    }
    outcome, failures = decision_for(metrics, decision, args.runtime_label)
    report = {
        'schema_version': 1,
        'suite_id': suite_id,
        'runtime_label': args.runtime_label,
        'endpoint': args.endpoint,
        'artifacts': {
            'suite': {'path': str(args.suite), 'sha256': sha256(args.suite)},
            'shelliq': {'path': str(args.shelliq), 'sha256': sha256(args.shelliq)},
            'index': {'path': str(args.index), 'sha256': sha256(args.index)},
            'model': {'path': str(args.model), 'sha256': sha256(args.model)},
        },
        'decision_rule': decision,
        'decision': {'outcome': outcome, 'passed': not failures, 'failures': failures},
        'warmup': warmup_result,
        'metrics': metrics,
        'cases': results,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'decision': report['decision'], 'metrics': metrics}, indent=2, sort_keys=True))
    print(f'report: {args.output}')


if __name__ == '__main__':
    main()
