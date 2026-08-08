import json
import subprocess
import sys
from pathlib import Path

import pytest

from shelliq_training.corpus import merge_distributable_corpus
from shelliq_training.data import Corpus, DatasetFormatError, Platform, SFTRecord, load_jsonl, write_jsonl

TRAINING_DIRECTORY = Path(__file__).parents[1]
MERGE_SCRIPT = TRAINING_DIRECTORY / 'scripts' / 'merge_corpus.py'


def record(record_id, *, source='tldr-pages', command='find', platform=Platform.LINUX):
    return SFTRecord(
        record_id=record_id,
        corpus=Corpus.DISTRIBUTABLE,
        source=source,
        license='CC-BY-4.0' if source == 'tldr-pages' else 'MIT OR Apache-2.0',
        provenance=f'fixture:{record_id}',
        command=command,
        platform=platform,
        instruction=f'Run {command}.',
        response=f'{command} --help',
        context=f'{command}: fixture context.',
    )


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, records, corpus=Corpus.DISTRIBUTABLE)


def test_merge_is_base_first_then_curated_filename_order(tmp_path):
    tldr = tmp_path / 'tldr.jsonl'
    curated = tmp_path / 'corpus'
    write(tldr, [record('tldr:z'), record('tldr:a')])
    write(curated / 'zeta.jsonl', [record('curated:zeta', source='shelliq-curated', command='rg')])
    write(
        curated / 'alpha.jsonl',
        [record('curated:alpha', source='shelliq-curated', command='jq', platform=Platform.DARWIN)],
    )

    records, report = merge_distributable_corpus(tldr, curated)

    assert [item.record_id for item in records] == ['tldr:z', 'tldr:a', 'curated:alpha', 'curated:zeta']
    assert report.record_count == 4
    assert report.command_count == 3
    assert report.source_counts == (('shelliq-curated', 2), ('tldr-pages', 2))
    assert report.platform_counts == (('darwin', 1), ('linux', 3))
    assert report.curated_family_counts == (('alpha', 1), ('zeta', 1))
    assert [(item.kind, item.name, item.record_count) for item in report.inputs] == [
        ('tldr', 'tldr.jsonl', 2),
        ('curated', 'alpha.jsonl', 1),
        ('curated', 'zeta.jsonl', 1),
    ]
    assert all(len(item.sha256) == 64 for item in report.inputs)


def test_merge_rejects_record_id_collision_between_inputs(tmp_path):
    tldr = tmp_path / 'tldr.jsonl'
    curated = tmp_path / 'corpus'
    write(tldr, [record('shared:id')])
    write(curated / 'family.jsonl', [record('shared:id', source='shelliq-curated')])

    with pytest.raises(DatasetFormatError, match=r"duplicate record_id 'shared:id'.*tldr:tldr.jsonl.*curated:family.jsonl"):
        merge_distributable_corpus(tldr, curated)


def test_merge_rejects_missing_or_empty_curated_inputs(tmp_path):
    tldr = tmp_path / 'tldr.jsonl'
    write(tldr, [record('tldr:one')])

    with pytest.raises(DatasetFormatError, match='no curated JSONL files found'):
        merge_distributable_corpus(tldr, tmp_path / 'missing')

    curated = tmp_path / 'corpus'
    curated.mkdir()
    (curated / 'empty.jsonl').write_text('', encoding='utf-8')
    with pytest.raises(DatasetFormatError, match='corpus input contains no records'):
        merge_distributable_corpus(tldr, curated)


def test_merge_script_writes_artifact_and_manifest_without_overwriting(tmp_path):
    tldr = tmp_path / 'tldr.jsonl'
    curated = tmp_path / 'corpus'
    output = tmp_path / 'artifacts' / 'merged.jsonl'
    write(tldr, [record('tldr:one')])
    write(curated / 'family.jsonl', [record('curated:one', source='shelliq-curated', command='rg')])
    command = [
        sys.executable,
        str(MERGE_SCRIPT),
        '--tldr',
        str(tldr),
        '--curated-directory',
        str(curated),
        '--output',
        str(output),
    ]

    result = subprocess.run(command, check=True, capture_output=True, text=True)

    assert [item.record_id for item in load_jsonl(output, corpus=Corpus.DISTRIBUTABLE)] == [
        'tldr:one',
        'curated:one',
    ]
    manifest_path = output.with_suffix('.jsonl.manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['record_count'] == 2
    assert manifest['source_counts'] == {'shelliq-curated': 1, 'tldr-pages': 1}
    assert 'sources: shelliq-curated=1, tldr-pages=1' in result.stdout

    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert 'output already exists' in repeated.stderr
