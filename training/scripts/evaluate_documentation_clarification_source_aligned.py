#!/usr/bin/env python3
"""Evaluate clarification against the documentation recipe reported by the CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
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

EXPERIMENTS = {
    'documentation-clarification-continuation-v1',
    'documentation-clarification-continuation-v2',
    'documentation-clarification-source-aligned-v1',
}


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
    experiment = manifest.get('experiment')
    if experiment not in EXPERIMENTS:
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
    options_by_command: dict[str, set[str]] = {}
    for template in templates:
        by_intent.setdefault((template.command, _normalize(template.instruction)), []).append(template)
        words = _simple_words(template.document)
        if words is not None:
            options_by_command.setdefault(template.command, set()).update(
                word.split('=', 1)[0] for word in words[1:] if word.startswith('-')
            )

    outcomes = [_evaluate(args.shelliq.resolve(), row, by_intent, options_by_command) for row in rows]
    emitted = [item for item in outcomes if item['answered_emitted']]
    latencies = sorted(float(item['answered_latency_ms']) for item in outcomes)
    measured = latencies[1:] if len(latencies) > 1 else latencies
    metrics = {
        'records': len(outcomes),
        'safe_default_abstentions': sum(bool(item['safe_default_abstention']) for item in outcomes),
        'commandless_json_abstentions': sum(bool(item['commandless_json_abstention']) for item in outcomes),
        'focused_questions': sum(bool(item['focused_question']) for item in outcomes),
        'source_aligned_clarifications': sum(bool(item['source_aligned']) for item in outcomes),
        'stable_intermediate_abstentions': sum(bool(item['stable_intermediate_abstentions']) for item in outcomes),
        'invalid_continuations_fail_closed': sum(bool(item['invalid_continuations_fail_closed']) for item in outcomes),
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
        and metrics['stable_intermediate_abstentions'] == count
        and metrics['invalid_continuations_fail_closed'] == count
        and metrics['answered_exact'] / count >= 0.95
        and metrics['answered_ready_precision'] == 1.0
        and metrics['all_emitted_semantic_and_locally_valid']
        and metrics['mean_latency_ms'] <= 500.0
        and metrics['maximum_latency_ms'] <= 2_000.0
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


def _evaluate(
    shelliq: Path,
    row: dict[str, object],
    by_intent: dict[tuple[str, str], list[object]],
    options_by_command: dict[str, set[str]],
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix='shelliq-source-aligned-') as temp_value:
        temp = Path(temp_value)
        bin_dir = temp / 'bin'
        bin_dir.mkdir()
        command = str(row['command'])
        executable = bin_dir / command
        options = sorted(options_by_command.get(command, set()))
        help_lines = ['Usage: ' + command + ' [OPTIONS]', '', 'Options:']
        help_lines.extend(f'  {option}  documented option' for option in options)
        help_lines.append('  -h, --help  Print help')
        executable.write_text('#!/bin/sh\n' + "printf '%s\\n' " + ' '.join(_single_quote(line) for line in help_lines) + '\n')
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        index = temp / 'index.sqlite'
        environment = os.environ.copy()
        environment['PATH'] = f'{bin_dir}:{environment.get("PATH", "/usr/bin:/bin")}'
        man_dir = temp / 'man'
        man_dir.mkdir()
        environment['MANPATH'] = str(man_dir)
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
        source_aligned = source_well_formed and any(_recipe_has_operand(template, label) for template in initial_templates)

        return _evaluate_continuation(
            shelliq,
            row,
            base,
            instruction,
            environment,
            index,
            human,
            structured,
            envelope,
            source_aligned,
            by_intent,
        )

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


def _evaluate_continuation(
    shelliq: Path,
    row: dict[str, object],
    base: list[str],
    instruction: str,
    environment: dict[str, str],
    index: Path,
    human: subprocess.CompletedProcess[str],
    structured: subprocess.CompletedProcess[str],
    envelope: dict[str, object],
    source_aligned: bool,
    by_intent: dict[tuple[str, str], list[object]],
) -> dict[str, object]:
    initial_documentation = _object(envelope.get('documentation'))
    initial_source = initial_documentation.get('source')
    answers: list[str] = []
    seen_states: set[tuple[object, object]] = set()
    steps: list[dict[str, object]] = []
    stable = source_aligned
    current_result = structured
    current_envelope = envelope
    started = time.perf_counter()

    for step_index in range(8):
        if current_envelope.get('status') != 'needs_input':
            break
        clarification = _object(current_envelope.get('clarification'))
        documentation = _object(current_envelope.get('documentation'))
        continuation = _object(current_envelope.get('continuation'))
        source = documentation.get('source')
        label = clarification.get('label')
        question = clarification.get('question')
        matched = _reported_templates(documentation, by_intent)
        commandless = current_result.returncode != 0 and 'command' not in current_envelope and 'semantic' not in current_envelope
        aligned = any(_recipe_has_operand(template, label) for template in matched)
        state = (source, label)
        step_safe = (
            commandless
            and isinstance(question, str)
            and bool(question.strip())
            and source == initial_source
            and continuation.get('v') == 1
            and continuation.get('source') == initial_source
            and state not in seen_states
            and aligned
        )
        stable = stable and step_safe
        steps.append({'source': source, 'label': label, 'safe': step_safe})
        if not step_safe or not isinstance(initial_source, str):
            break
        seen_states.add(state)
        answers.append(_step_answer(str(clarification.get('kind', 'text')), step_index))
        answer_args = [item for answer in answers for item in ('--answer', answer)]
        current_result = _run(
            [*base, '--continue-from', initial_source, *answer_args, '--json', '--', instruction],
            environment,
            5,
        )
        current_envelope = _json_object(current_result.stdout)

    latency_ms = (time.perf_counter() - started) * 1_000
    output = current_envelope.get('command')
    emitted = current_result.returncode == 0 and current_envelope.get('status') == 'ready' and isinstance(output, str)
    ready_documentation = _object(current_envelope.get('documentation'))
    answer_values = [shlex.split(answer)[0] for answer in answers]
    output_words = shlex.split(output) if emitted else []
    exact = (
        emitted and ready_documentation.get('source') == initial_source and all(value in output_words for value in answer_values)
    )
    valid = False
    if emitted:
        verification = _run(
            [str(shelliq), '--index', str(index), 'explain', '--', output],
            environment,
            5,
        )
        valid = verification.returncode == 0 and isinstance(current_envelope.get('semantic'), dict)

    command = str(row['command'])
    zero_source = f'tldr:linux:{command}:0'
    wrong_platform_source = str(initial_source).replace('tldr:linux:', 'tldr:darwin:', 1)
    invalid_results = [
        _run([*base, '--continue-from', str(initial_source), '--json', '--', instruction], environment, 5),
        _run(
            [*base, '--continue-from', zero_source, '--answer', '4242', '--json', '--', instruction],
            environment,
            5,
        ),
        _run(
            [*base, '--continue-from', wrong_platform_source, '--answer', '4242', '--json', '--', instruction],
            environment,
            5,
        ),
    ]
    invalid_fail_closed = all(result.returncode != 0 and not result.stdout for result in invalid_results)
    initial_clarification = _object(envelope.get('clarification'))
    initial_question = initial_clarification.get('question')

    return {
        'record_id': row['record_id'],
        'command': command,
        'safe_default_abstention': human.returncode != 0 and not human.stdout and 'needs_input:' in human.stderr,
        'commandless_json_abstention': (
            structured.returncode != 0
            and envelope.get('v') == 1
            and envelope.get('status') == 'needs_input'
            and 'command' not in envelope
            and 'semantic' not in envelope
        ),
        'focused_question': isinstance(initial_question, str) and bool(initial_question.strip()),
        'source_aligned': source_aligned,
        'stable_intermediate_abstentions': stable,
        'invalid_continuations_fail_closed': invalid_fail_closed,
        'continuation_steps': steps,
        'answers': answers,
        'answered_emitted': emitted,
        'answered_exact': exact,
        'answered_valid': valid,
        'answered_expected': 'pinned recipe with every accumulated typed answer',
        'answered_output': output,
        'answered_source': ready_documentation.get('source'),
        'answered_latency_ms': latency_ms,
        'human_stderr': human.stderr.strip(),
        'structured_stderr': structured.stderr.strip(),
        'answered_stderr': current_result.stderr.strip(),
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


def _step_answer(kind: str, step_index: int) -> str:
    suffix = step_index + 1
    return {
        'path': f'/tmp/shelliq-clarification-answer-{suffix}.dat',
        'integer': str(4241 + suffix),
        'remote': f'user{suffix}@example.com:/tmp/data',
        'url': f'https://example.com/item/{suffix}',
        'text': f'"clarification value {suffix}"',
    }.get(kind, f'"clarification value {suffix}"')


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
