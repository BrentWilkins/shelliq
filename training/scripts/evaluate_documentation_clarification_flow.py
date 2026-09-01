#!/usr/bin/env python3
"""Exercise clarification envelopes and answered recompilation through the real CLI."""

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

EXPERIMENT_PREFIX = 'documentation-clarification-flow-v'


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
        raise ValueError('unexpected clarification manifest')
    expected_partition = manifest[args.partition]
    if expected_partition['sha256'] != _sha256(args.dataset):
        raise ValueError('dataset hash differs from frozen manifest')
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    if len(rows) != expected_partition['records']:
        raise ValueError('dataset record count differs from frozen manifest')

    outcomes = [_evaluate(args.shelliq.resolve(), row) for row in rows]
    emitted = [item for item in outcomes if item['answered_emitted']]
    latencies = sorted(float(item['answered_latency_ms']) for item in outcomes)
    measured = latencies[1:] if len(latencies) > 1 else latencies
    metrics = {
        'records': len(outcomes),
        'safe_default_abstentions': sum(bool(item['safe_default_abstention']) for item in outcomes),
        'commandless_json_abstentions': sum(bool(item['commandless_json_abstention']) for item in outcomes),
        'focused_questions': sum(bool(item['focused_question']) for item in outcomes),
        'correct_slot_kinds': sum(bool(item['correct_slot_kind']) for item in outcomes),
        'correct_slot_labels': sum(bool(item['correct_slot_label']) for item in outcomes),
        'answered_emitted': len(emitted),
        'answered_exact': sum(bool(item['answered_exact']) for item in outcomes),
        'answered_ready_precision': (sum(bool(item['answered_exact']) for item in emitted) / len(emitted) if emitted else 0.0),
        'all_emitted_semantic_and_locally_valid': all(bool(item['answered_valid']) for item in emitted),
        'mean_latency_ms': sum(measured) / len(measured) if measured else 0.0,
        'maximum_latency_ms': max(measured, default=0.0),
    }
    count = len(outcomes)
    slot_identification = metrics['correct_slot_labels'] if experiment.endswith('v2') else metrics['correct_slot_kinds']
    gate_passed = (
        metrics['safe_default_abstentions'] == count
        and metrics['commandless_json_abstentions'] == count
        and metrics['focused_questions'] == count
        and slot_identification / count >= 0.95
        and metrics['answered_exact'] / count >= 0.80
        and metrics['answered_ready_precision'] == 1.0
        and metrics['all_emitted_semantic_and_locally_valid']
        and metrics['mean_latency_ms'] <= 250.0
        and metrics['maximum_latency_ms'] <= 1_000.0
    )
    report = {
        'schema_version': 1,
        'experiment': experiment,
        'partition': args.partition,
        'dataset_sha256': _sha256(args.dataset),
        'metrics': metrics,
        'gate_passed': gate_passed,
        'outcomes': outcomes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({**report, 'outcomes': f'{len(outcomes)} records'}, indent=2))
    if not gate_passed:
        raise SystemExit(1)


def _evaluate(shelliq: Path, row: dict[str, object]) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix='shelliq-clarification-') as temp_value:
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
        build = _run(
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
            environment,
            10,
        )
        if build.returncode != 0:
            raise RuntimeError(f'{row["record_id"]}: fixture index failed: {build.stderr}')

        base = [
            str(shelliq),
            '--index',
            str(index),
            'suggest',
            '--endpoint',
            'http://127.0.0.1:9/v1/chat/completions',
            '--timeout-ms',
            '100',
        ]
        instruction = str(row['instruction'])
        human = _run([*base, '--', instruction], environment, 5)
        structured = _run([*base, '--json', '--', instruction], environment, 5)
        envelope = _json_object(structured.stdout)
        clarification = envelope.get('clarification', {})
        commandless = 'command' not in envelope and 'semantic' not in envelope

        started = time.perf_counter()
        answered = _run(
            [*base, '--answer', str(row['answer']), '--json', '--', instruction],
            environment,
            5,
        )
        answered_latency_ms = (time.perf_counter() - started) * 1_000
        answered_envelope = _json_object(answered.stdout)
        output = answered_envelope.get('command')
        emitted = answered.returncode == 0 and isinstance(output, str) and bool(output)
        exact = emitted and output == row['expected_command']
        valid = False
        if emitted:
            verification = _run(
                [str(shelliq), '--index', str(index), 'explain', '--', output],
                environment,
                5,
            )
            valid = verification.returncode == 0 and isinstance(answered_envelope.get('semantic'), dict)

        question = clarification.get('question')
        return {
            'record_id': row['record_id'],
            'command': command,
            'safe_default_abstention': (human.returncode != 0 and not human.stdout and 'needs_input:' in human.stderr),
            'commandless_json_abstention': (
                structured.returncode != 0 and envelope.get('v') == 1 and envelope.get('status') == 'needs_input' and commandless
            ),
            'focused_question': isinstance(question, str) and bool(question.strip()),
            'expected_slot_kind': row['expected_slot_kind'],
            'actual_slot_kind': clarification.get('kind'),
            'correct_slot_kind': clarification.get('kind') == row['expected_slot_kind'],
            'expected_slot_label': row['expected_slot_label'],
            'actual_slot_label': clarification.get('label'),
            'correct_slot_label': clarification.get('label') == row['expected_slot_label'],
            'answered_emitted': emitted,
            'answered_exact': exact,
            'answered_valid': valid,
            'answered_output': output,
            'answered_status': answered_envelope.get('status'),
            'answered_latency_ms': answered_latency_ms,
            'human_stderr': human.stderr.strip(),
            'structured_stderr': structured.stderr.strip(),
            'answered_stderr': answered.stderr.strip(),
        }


def _run(command: list[str], environment: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=timeout,
    )


def _json_object(output: str) -> dict[str, object]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    main()
