#!/usr/bin/env python3
"""Evaluate clarification against the documentation recipe reported by the CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_documentation_fallback_e2e import _simple_words

from shelliq_training.data import load_semantic_jsonl
from shelliq_training.documentation_templates import (
    RetrievedTemplate,
    bind_template,
    documentation_templates,
    unresolved_placeholders,
)

EXPERIMENT = 'documentation-clarification-source-aligned-v1'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shelliq', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--partition', choices=('development', 'test'), required=True)
    parser.add_argument('--report', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('experiment') != EXPERIMENT:
        raise ValueError('unexpected experiment manifest')
    if manifest['documentation_index_sha256'] != _sha256(args.documentation_index):
        raise ValueError('documentation index hash differs from frozen manifest')
    partition = manifest[args.partition]
    if partition['sha256'] != _sha256(args.dataset):
        raise ValueError('dataset hash differs from frozen manifest')
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    if len(rows) != partition['records']:
        raise ValueError('dataset record count differs from frozen manifest')

    templates = documentation_templates(load_semantic_jsonl(args.documentation_index))
    by_intent: dict[tuple[str, str], list[object]] = {}
    for template in templates:
        by_intent.setdefault((template.command, _normalize(template.instruction)), []).append(template)

    outcomes = [_evaluate(args.shelliq.resolve(), row, by_intent) for row in rows]
    emitted = [item for item in outcomes if item['answered_emitted']]
    latencies = sorted(float(item['answered_latency_ms']) for item in outcomes)
    measured = latencies[1:] if len(latencies) > 1 else latencies
    metrics = {
        'records': len(outcomes),
        'safe_default_abstentions': sum(bool(item['safe_default_abstention']) for item in outcomes),
        'commandless_json_abstentions': sum(bool(item['commandless_json_abstention']) for item in outcomes),
        'focused_questions': sum(bool(item['focused_question']) for item in outcomes),
        'source_aligned_clarifications': sum(bool(item['source_aligned']) for item in outcomes),
        'answered_emitted': len(emitted),
        'answered_exact': sum(bool(item['answered_exact']) for item in outcomes),
        'answered_ready_precision': (sum(bool(item['answered_exact']) for item in emitted) / len(emitted) if emitted else 0.0),
        'all_emitted_semantic_and_locally_valid': all(bool(item['answered_valid']) for item in emitted),
        'mean_latency_ms': sum(measured) / len(measured) if measured else 0.0,
        'maximum_latency_ms': max(measured, default=0.0),
    }
    count = len(outcomes)
    gate_passed = (
        metrics['safe_default_abstentions'] == count
        and metrics['commandless_json_abstentions'] == count
        and metrics['focused_questions'] == count
        and metrics['source_aligned_clarifications'] / count >= 0.95
        and metrics['answered_exact'] / count >= 0.80
        and metrics['answered_ready_precision'] == 1.0
        and metrics['all_emitted_semantic_and_locally_valid']
        and metrics['mean_latency_ms'] <= 250.0
        and metrics['maximum_latency_ms'] <= 1_000.0
    )
    report = {
        'schema_version': 1,
        'experiment': EXPERIMENT,
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


def _evaluate(
    shelliq: Path,
    row: dict[str, object],
    by_intent: dict[tuple[str, str], list[object]],
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix='shelliq-source-aligned-') as temp_value:
        temp = Path(temp_value)
        bin_dir = temp / 'bin'
        bin_dir.mkdir()
        command = str(row['command'])
        executable = bin_dir / command
        options = [str(value) for value in row['documented_options']]
        help_lines = ['Usage: ' + command + ' [OPTIONS]', '', 'Options:']
        help_lines.extend(f'  {option}  documented option' for option in options)
        help_lines.append('  -h, --help  Print help')
        executable.write_text('#!/bin/sh\n' + "printf '%s\\n' " + ' '.join(_single_quote(line) for line in help_lines) + '\n')
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
        clarification = _object(envelope.get('clarification'))
        documentation = _object(envelope.get('documentation'))
        initial_templates = _reported_templates(documentation, by_intent)
        label = clarification.get('label')
        reported_command = documentation.get('command')
        reported_source = documentation.get('source')
        source_well_formed = (
            isinstance(reported_command, str)
            and isinstance(reported_source, str)
            and re.fullmatch(
                rf'tldr:[a-z0-9_-]+:{re.escape(reported_command)}:[1-9][0-9]*',
                reported_source,
            )
            is not None
        )
        source_aligned = source_well_formed and len(initial_templates) == 1 and _recipe_has_operand(initial_templates[0], label)

        answer = _answer(str(clarification.get('kind', 'text')))
        started = time.perf_counter()
        answered = _run([*base, '--answer', answer, '--json', '--', instruction], environment, 5)
        answered_latency_ms = (time.perf_counter() - started) * 1_000
        answered_envelope = _json_object(answered.stdout)
        output = answered_envelope.get('command')
        emitted = answered.returncode == 0 and isinstance(output, str) and bool(output)
        answered_documentation = _object(answered_envelope.get('documentation'))
        answered_templates = _reported_templates(answered_documentation, by_intent)
        expected = None
        if len(answered_templates) == 1:
            completed = f'{instruction}. Use {answer}.'
            bound = bind_template(RetrievedTemplate(answered_templates[0], 1.0), completed)
            if not unresolved_placeholders(bound.document):
                words = _simple_words(bound.document)
                expected = ' '.join(words) if words is not None else None
        exact = emitted and expected is not None and output == expected
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
                structured.returncode != 0
                and envelope.get('v') == 1
                and envelope.get('status') == 'needs_input'
                and 'command' not in envelope
                and 'semantic' not in envelope
            ),
            'focused_question': isinstance(question, str) and bool(question.strip()),
            'reported_source': documentation.get('source'),
            'reported_intent': documentation.get('intent'),
            'reported_label': label,
            'matched_initial_recipes': len(initial_templates),
            'source_aligned': source_aligned,
            'answer': answer,
            'answered_emitted': emitted,
            'answered_exact': exact,
            'answered_valid': valid,
            'answered_expected': expected,
            'answered_output': output,
            'answered_source': answered_documentation.get('source'),
            'answered_latency_ms': answered_latency_ms,
            'human_stderr': human.stderr.strip(),
            'structured_stderr': structured.stderr.strip(),
            'answered_stderr': answered.stderr.strip(),
        }


def _reported_templates(
    documentation: dict[str, object],
    by_intent: dict[tuple[str, str], list[object]],
) -> list[object]:
    command = documentation.get('command')
    intent = documentation.get('intent')
    if not isinstance(command, str) or not isinstance(intent, str):
        return []
    return by_intent.get((command, _normalize(intent)), [])


def _recipe_has_operand(template: object, label: object) -> bool:
    if not isinstance(label, str):
        return False
    words = _simple_words(template.document)
    if words is None:
        return False
    return label in (word.strip('\'"<>') for word in words[1:] if not word.startswith('-'))


def _answer(kind: str) -> str:
    return {
        'path': '/tmp/shelliq-clarification-answer.dat',
        'integer': '4242',
        'remote': 'user@example.com:/tmp/data',
        'url': 'https://example.com/item/42',
        'text': '"clarification value"',
    }.get(kind, '"clarification value"')


def _normalize(value: str) -> str:
    return ' '.join(re.sub(r'[^A-Za-z0-9]', ' ', value).lower().split())


def _object(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _json_object(output: str) -> dict[str, object]:
    try:
        return _object(json.loads(output))
    except json.JSONDecodeError:
        return {}


def _run(command: list[str], environment: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=timeout,
    )


def _single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    main()
