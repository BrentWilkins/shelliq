import json
import subprocess
import sys
from pathlib import Path

from shelliq_training.corpus_audit import audit_distributable_corpus
from shelliq_training.data import Corpus, Platform, SFTRecord, write_jsonl

AUDIT_SCRIPT = Path(__file__).parents[1] / 'scripts' / 'audit_corpus.py'


def record(record_id, command, response, *, source='tldr-pages', instruction=None):
    return SFTRecord(
        record_id=record_id,
        corpus=Corpus.DISTRIBUTABLE,
        source=source,
        license='CC-BY-4.0' if source == 'tldr-pages' else 'MIT OR Apache-2.0',
        provenance=f'fixture:{record_id}',
        command=command,
        platform=Platform.LINUX,
        instruction=instruction or f'Run {command}.',
        response=response,
        context=f'{command}: fixture context.',
    )


def fixture_dataset(path):
    records = [
        record('tldr:find:one', 'find', "find . -name '*.py'", instruction='Find Python files.'),
        record('curated:find:one', 'find', 'find . -type f', source='shelliq-curated'),
        record('tldr:rg:one', 'rg', 'rg -n TODO src', instruction='Search for TODO.'),
        record('tldr:rg:two', 'rg', 'rg -n TODO src', instruction='Search for TODO!'),
        record('tldr:ls:one', 'ls', 'ls {{path}}'),
        record('tldr:grep:one', 'grep', "grep 'unterminated"),
    ]
    write_jsonl(path, records, corpus=Corpus.DISTRIBUTABLE)


def test_audit_is_deterministic_and_reports_quality_and_split_facts(tmp_path):
    dataset = tmp_path / 'candidate.jsonl'
    fixture_dataset(dataset)

    first = audit_distributable_corpus(
        dataset,
        seed=17,
        heldout_sources=frozenset({'shelliq-curated'}),
    )
    second = audit_distributable_corpus(
        dataset,
        seed=17,
        heldout_sources=frozenset({'shelliq-curated'}),
    )

    assert first == second
    assert first['records']['total'] == 6
    assert first['records']['preflight_accepted'] == 4
    assert first['records']['preflight_rejected'] == 2
    assert first['composition']['commands'] == 4
    assert first['duplicates']['response']['groups'] == 1
    assert first['duplicates']['normalized_instruction']['groups'] == 1
    assert first['lexical_options']['distinct_command_tokens'] == 3
    test_partition = first['split']['partitions']['test']
    assert test_partition['records'] >= 2
    assert test_partition['sources']['shelliq-curated'] == 1
    assert len(first['dataset']['sha256']) == 64


def test_audit_cli_writes_report_and_refuses_overwrite(tmp_path):
    dataset = tmp_path / 'candidate.jsonl'
    output = tmp_path / 'audit.json'
    fixture_dataset(dataset)
    command = [
        sys.executable,
        str(AUDIT_SCRIPT),
        '--dataset',
        str(dataset),
        '--output',
        str(output),
        '--seed',
        '17',
    ]

    result = subprocess.run(command, check=True, capture_output=True, text=True)

    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['audit_schema_version'] == 1
    assert report['records']['total'] == 6
    assert 'audited 6 records (4 accepted, 2 rejected)' in result.stdout

    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert 'output already exists' in repeated.stderr
