import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from shelliq_training.data import DatasetFormatError, Platform
from shelliq_training.privacy import PrivateDataGate, ScrubPolicy
from shelliq_training.private_sources import (
    build_claude_transcript_corpus,
    build_claude_transcript_records,
    build_local_index_records,
    expand_claude_transcript_paths,
)

BUILD_DATASET_SCRIPT = Path(__file__).parents[1] / 'scripts' / 'build_dataset.py'


def write_successful_transcript(path, *, tool_use_id, command='printf ok', description='Print a marker'):
    entries = [
        {
            'message': {
                'content': [
                    {
                        'type': 'tool_use',
                        'name': 'Bash',
                        'id': tool_use_id,
                        'input': {'command': command, 'description': description},
                    }
                ]
            }
        },
        {'message': {'content': [{'type': 'tool_result', 'tool_use_id': tool_use_id, 'is_error': False}]}},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(entry) for entry in entries) + '\n', encoding='utf-8')


def transcript_entry(blocks):
    return json.dumps({'message': {'content': blocks}})


def test_claude_builder_keeps_only_successful_scrubbed_bash_calls(tmp_path):
    path = tmp_path / 'session.jsonl'
    entries = [
        transcript_entry(
            [
                {
                    'type': 'tool_use',
                    'id': 'safe',
                    'name': 'Bash',
                    'input': {
                        'command': 'rg needle /home/alice/project',
                        'description': 'Search alice project on devbox',
                    },
                },
                {
                    'type': 'tool_use',
                    'id': 'failed',
                    'name': 'Bash',
                    'input': {'command': 'false', 'description': 'This fails'},
                },
                {
                    'type': 'tool_use',
                    'id': 'secret',
                    'name': 'Bash',
                    'input': {
                        'command': 'echo AKIAIOSFODNN7EXAMPLE',
                        'description': 'Print credential',
                    },
                },
            ]
        ),
        transcript_entry(
            [
                {'type': 'tool_result', 'tool_use_id': 'safe', 'is_error': False},
                {'type': 'tool_result', 'tool_use_id': 'failed', 'is_error': True},
                {'type': 'tool_result', 'tool_use_id': 'secret', 'is_error': False},
            ]
        ),
    ]
    path.write_text('\n'.join(entries) + '\n', encoding='utf-8')
    gate = PrivateDataGate(ScrubPolicy(home_paths=('/home/alice', str(tmp_path)), usernames=('alice',), hostnames=('devbox',)))

    corpus = build_claude_transcript_records([path], gate=gate, platform=Platform.LINUX)

    assert len(corpus.records) == 1
    assert len(corpus.dropped) == 1
    assert corpus.records[0].response == 'rg needle <HOME>/project'
    assert '<USER>' in corpus.records[0].instruction
    assert '<HOST>' in corpus.records[0].instruction
    assert corpus.records[0].provenance.startswith('<HOME>/session.jsonl:')


def test_local_index_builder_reads_examples_and_scrubs_provenance(tmp_path):
    path = tmp_path / 'shelliq.sqlite'
    with sqlite3.connect(path) as database:
        database.executescript(
            """
            CREATE TABLE commands (
                id INTEGER PRIMARY KEY,
                name TEXT,
                platform TEXT,
                synopsis TEXT,
                description TEXT
            );
            CREATE TABLE examples (
                id INTEGER PRIMARY KEY,
                command_id INTEGER,
                text TEXT,
                description TEXT,
                source TEXT
            );
            INSERT INTO commands VALUES (1, 'find', 'linux', 'find [path]', 'Find files below a tree.');
            INSERT INTO examples VALUES (7, 1, 'find . -name *.py', 'Find Python files', '/home/alice/man/find');
            INSERT INTO examples VALUES (8, 1, 'find .', '', 'ignored');
            """
        )
    gate = PrivateDataGate(ScrubPolicy(home_paths=(str(tmp_path), '/home/alice')))

    corpus = build_local_index_records(path, gate=gate)

    assert len(corpus.records) == 1
    assert corpus.records[0].record_id == 'local-index:example:7'
    assert corpus.records[0].context == 'find [path] Find files below a tree.'
    assert str(tmp_path) not in corpus.records[0].provenance
    assert '/home/alice' not in corpus.records[0].provenance


def test_claude_directory_expansion_is_recursive_deduplicated_and_stable(tmp_path):
    root = tmp_path / 'sessions'
    first = root / 'a.jsonl'
    second = root / 'nested' / 'b.jsonl'
    write_successful_transcript(first, tool_use_id='call-a', command='printf first')
    write_successful_transcript(second, tool_use_id='call-b', command='printf second')
    (root / 'nested' / 'ignored.txt').write_text('not a transcript', encoding='utf-8')

    paths = expand_claude_transcript_paths([second, root])
    build = build_claude_transcript_corpus(
        [second, root],
        gate=PrivateDataGate(ScrubPolicy()),
        platform=Platform.LINUX,
    )

    assert paths == (first.resolve(), second.resolve())
    assert build.processed_paths == paths
    assert not build.skipped
    assert [record.response for record in build.scrubbed.records] == ['printf first', 'printf second']


def test_claude_partial_mode_is_explicit_and_default_remains_fail_closed(tmp_path):
    good = tmp_path / 'a-good.jsonl'
    bad = tmp_path / 'z-bad.jsonl'
    write_successful_transcript(good, tool_use_id='call-good')
    bad.write_text('{not JSON}\n', encoding='utf-8')
    gate = PrivateDataGate(ScrubPolicy())

    with pytest.raises(DatasetFormatError, match='invalid transcript JSON'):
        build_claude_transcript_corpus([tmp_path], gate=gate, platform=Platform.LINUX)

    build = build_claude_transcript_corpus(
        [tmp_path],
        gate=gate,
        platform=Platform.LINUX,
        allow_partial=True,
    )

    assert len(build.scrubbed.records) == 1
    assert build.processed_paths == (good.resolve(),)
    assert len(build.skipped) == 1
    assert build.skipped[0].path == bad.resolve()
    assert len(build.skipped[0].transcript_id) == 16
    assert 'invalid transcript JSON' in build.skipped[0].error
    assert str(tmp_path) not in build.skipped[0].error


def test_claude_partial_cli_reports_format_skips_and_privacy_drops(tmp_path):
    transcripts = tmp_path / 'sessions'
    good = transcripts / 'a-good.jsonl'
    secret = transcripts / 'b-secret.jsonl'
    malformed = transcripts / 'c-malformed.jsonl'
    write_successful_transcript(good, tool_use_id='call-good')
    write_successful_transcript(
        secret,
        tool_use_id='call-secret',
        command='printf AKIAIOSFODNN7EXAMPLE',
        description='Print a credential fixture',
    )
    malformed.write_text('{not JSON}\n', encoding='utf-8')
    output = tmp_path / 'private' / 'claude.jsonl'

    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_DATASET_SCRIPT),
            'claude',
            str(transcripts),
            '--platform',
            'linux',
            '--allow-partial',
            '--canary-count',
            '1',
            '--canary-seed',
            '7',
            '--output',
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.with_suffix('.jsonl.scrub-report.json').read_text(encoding='utf-8'))
    assert report['accepted'] == 1
    assert report['dropped'] == 1
    assert len(report['dropped_record_ids']) == 1
    assert report['finding_counts']['aws-access-key'] == 1
    assert len(report['processed_transcript_ids']) == 2
    assert all(len(transcript_id) == 16 for transcript_id in report['processed_transcript_ids'])
    assert len(report['skipped_transcripts'][0]['transcript_id']) == 16
    assert 'invalid transcript JSON' in report['skipped_transcripts'][0]['error']
    assert str(tmp_path) not in json.dumps(report)
    assert f'skipped transcript {malformed.resolve()}' in result.stderr
