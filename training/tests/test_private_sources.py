import json
import sqlite3

from shelliq_training.data import Platform
from shelliq_training.privacy import PrivateDataGate, ScrubPolicy
from shelliq_training.private_sources import (
    build_claude_transcript_records,
    build_local_index_records,
)


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
