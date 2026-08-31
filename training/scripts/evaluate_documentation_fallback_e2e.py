#!/usr/bin/env python3
"""Exercise the real shelliq CLI documentation fallback on a frozen partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path

EXPERIMENT_PREFIX = 'documentation-fallback-e2e-v'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shelliq', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--partition', choices=('development', 'test'), required=True)
    parser.add_argument('--report', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    experiment = manifest.get('experiment')
    if not isinstance(experiment, str) or not experiment.startswith(EXPERIMENT_PREFIX):
        raise ValueError('unexpected fallback evaluation manifest')
    expected_hash = manifest[args.partition]['sha256']
    if _sha256(args.dataset) != expected_hash:
        raise ValueError('dataset hash does not match frozen manifest')
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    expected_records = manifest[args.partition]['records']
    if len(rows) != expected_records:
        raise ValueError(f'expected {expected_records} rows, found {len(rows)}')

    outcomes = [_evaluate(args.shelliq.resolve(), row) for row in rows]
    complete = [item for item in outcomes if item['expected_status'] == 'ready']
    incomplete = [item for item in outcomes if item['expected_status'] == 'needs_input']
    emitted = [item for item in complete if item['emitted']]
    exact = sum(bool(item['exact']) for item in complete)
    precision = sum(bool(item['exact']) for item in emitted) / len(emitted) if emitted else 0.0
    latencies = sorted(float(item['latency_ms']) for item in complete)
    measured = latencies[1:] if len(latencies) > 1 else latencies
    metrics = {
        'records': len(outcomes),
        'complete_records': len(complete),
        'incomplete_records': len(incomplete),
        'complete_exact': exact,
        'complete_emitted': len(emitted),
        'ready_precision': precision,
        'all_emitted_semantic_and_locally_valid': all(bool(item['semantic_and_locally_valid']) for item in emitted),
        'safe_incomplete_abstentions': sum(bool(item['safe_abstention']) for item in incomplete),
        'explicit_incomplete_abstentions': sum(bool(item['explicit_abstention']) for item in incomplete),
        'mean_latency_ms': sum(measured) / len(measured) if measured else 0.0,
        'maximum_latency_ms': max(measured, default=0.0),
    }
    required_abstentions = (
        metrics['explicit_incomplete_abstentions']
        if experiment == 'documentation-fallback-e2e-v4'
        else metrics['safe_incomplete_abstentions']
    )
    gate_passed = (
        metrics['complete_exact'] >= 39
        and metrics['ready_precision'] >= 0.95
        and metrics['all_emitted_semantic_and_locally_valid']
        and required_abstentions == 16
        and metrics['mean_latency_ms'] <= 250.0
        and metrics['maximum_latency_ms'] <= 1_000.0
    )
    if args.partition == 'development':
        gate_passed = metrics['all_emitted_semantic_and_locally_valid'] and required_abstentions == len(incomplete)
    report = {
        'schema_version': 1,
        'experiment': experiment,
        'partition': args.partition,
        'dataset_sha256': expected_hash,
        'gate_passed': gate_passed,
        'metrics': metrics,
        'outcomes': outcomes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({**report, 'outcomes': f'{len(outcomes)} rows'}, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit(f'{args.partition} documentation fallback gate failed')


def _evaluate(shelliq: Path, row: dict[str, object]) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix='shelliq-doc-fallback-') as temp_value:
        temp = Path(temp_value)
        bin_dir = temp / 'bin'
        bin_dir.mkdir()
        command = str(row['command'])
        executable = bin_dir / command
        options = [str(value) for value in row['documented_options']]
        help_lines = ['Usage: ' + command + ' [OPTIONS]', '', 'Options:']
        help_lines.extend(f'  {option}  documented option' for option in options)
        help_lines.append('  -h, --help  Print help')
        script = '#!/bin/sh\n' + "printf '%s\\n' " + ' '.join(_single_quote(line) for line in help_lines) + '\n'
        executable.write_text(script)
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        index = temp / 'index.sqlite'
        environment = os.environ.copy()
        environment['PATH'] = f'{bin_dir}:{environment.get("PATH", "/usr/bin:/bin")}'
        build = subprocess.run(
            [
                str(shelliq),
                '--index',
                str(index),
                'index',
                'build',
                command,
                '--allow-writable-path',
                str(bin_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
        )
        if build.returncode != 0:
            raise RuntimeError(f'{row["record_id"]}: fixture index failed: {build.stderr}')
        indexed = subprocess.run(
            [str(shelliq), '--index', str(index), 'flags', command, '--raw'],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=5,
        )
        started = time.perf_counter()
        result = subprocess.run(
            [
                str(shelliq),
                '--index',
                str(index),
                'suggest',
                '--endpoint',
                'http://127.0.0.1:9/v1/chat/completions',
                '--timeout-ms',
                '100',
                '--',
                str(row['instruction']),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=5,
        )
        latency_ms = (time.perf_counter() - started) * 1_000
        output = result.stdout.rstrip('\n')
        emitted = result.returncode == 0 and bool(output)
        exact = emitted and output == row['expected_command']
        verified = False
        if emitted:
            verification = subprocess.run(
                [str(shelliq), '--index', str(index), 'explain', '--', output],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=5,
            )
            verified = verification.returncode == 0
        expected_status = str(row['expected_status'])
        safe_abstention = (
            expected_status == 'needs_input' and result.returncode != 0 and not output and 'needs_input:' in result.stderr
        )
        explicit_abstention = (
            expected_status == 'needs_input'
            and result.returncode != 0
            and not output
            and ('needs_input:' in result.stderr or 'no_documentation:' in result.stderr)
        )
        return {
            'record_id': row['record_id'],
            'command': command,
            'expected_status': expected_status,
            'returncode': result.returncode,
            'emitted': emitted,
            'exact': exact,
            'semantic_and_locally_valid': verified,
            'safe_abstention': safe_abstention,
            'explicit_abstention': explicit_abstention,
            'latency_ms': latency_ms,
            'output': output or None,
            'stderr': result.stderr.strip(),
            'indexed_options': indexed.stdout.splitlines(),
        }


def _single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    main()
