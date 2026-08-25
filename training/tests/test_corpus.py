import json
import shutil
import subprocess
from pathlib import Path

import pytest

from shelliq_training.data import Corpus, Platform, load_jsonl
from shelliq_training.sources import shell_command_name

CORPUS_DIRECTORY = Path(__file__).parents[1] / 'corpus'
CURATED_SOURCE = 'shelliq-curated'
CURATED_LICENSE = 'MIT OR Apache-2.0'

# Every response must parse in Zsh. The long-term target adds POSIX sh and Bash;
# rows that do not reach that bar yet are recorded in dialect-exempt.txt.
PRIMARY_SHELL = ('zsh', ('zsh', '-f', '-n', '-c'))
PORTABLE_SHELLS = (
    ('bash', ('bash', '--noprofile', '--norc', '-n', '-c')),
    ('sh', ('sh', '-n', '-c')),
)


def corpus_files():
    files = [path for path in sorted(CORPUS_DIRECTORY.glob('*.jsonl')) if not path.name.startswith('reviewed-preference-')]
    assert files, 'curated corpus must contain at least one JSONL file'
    return files


def curated_records():
    return [(path, record) for path in corpus_files() for record in load_jsonl(path, corpus=Corpus.DISTRIBUTABLE)]


def read_dialect_exemptions():
    lines = (CORPUS_DIRECTORY / 'dialect-exempt.txt').read_text(encoding='utf-8').splitlines()
    entries = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    families = {entry for entry in entries if entry.endswith('.jsonl')}
    return families, {entry for entry in entries if not entry.endswith('.jsonl')}


def parses_in(shell, response):
    return subprocess.run([*shell, response], capture_output=True).returncode == 0


def requires(program):
    return pytest.mark.skipif(shutil.which(program) is None, reason=f'{program} is not installed')


def test_every_row_loads_as_a_distributable_record():
    assert curated_records(), 'curated corpus must contain at least one record'


def test_record_ids_are_unique_across_files():
    seen = {}
    for path, record in curated_records():
        assert record.record_id not in seen, f'{record.record_id} appears in both {seen.get(record.record_id)} and {path.name}'
        seen[record.record_id] = path.name


def test_record_id_encodes_its_file_and_platform():
    for path, record in curated_records():
        namespace, stem, platform, _ = record.record_id.split(':', 3)
        assert namespace == 'curated'
        assert stem == path.stem, f'{record.record_id} does not name its file {path.stem}'
        assert platform == record.platform.value


def test_provenance_and_licensing_are_uniform():
    for path, record in curated_records():
        assert record.source == CURATED_SOURCE
        assert record.license == CURATED_LICENSE
        assert record.provenance.startswith(f'corpus/{path.name}')
        assert record.provenance.endswith(':model-authored')


def test_command_matches_the_response():
    for _, record in curated_records():
        assert record.command == shell_command_name(record.response)


def test_responses_carry_no_placeholders():
    # smoke_finetune.automatic_preflight rejects tldr-style {{ }} placeholders.
    for _, record in curated_records():
        assert '{{' not in record.response and '}}' not in record.response


def test_context_is_not_a_restatement():
    for _, record in curated_records():
        assert record.context.strip(), f'{record.record_id} has no context'
        assert record.context.strip() != record.instruction.strip()


def test_both_platforms_are_represented():
    platforms = {record.platform for _, record in curated_records()}
    assert Platform.LINUX in platforms
    assert Platform.DARWIN in platforms


def test_files_are_newline_terminated_json_lines():
    for path in corpus_files():
        text = path.read_text(encoding='utf-8')
        assert text.endswith('\n'), f'{path.name} must end with a newline'
        for number, line in enumerate(text.splitlines(), start=1):
            assert line.strip(), f'{path.name}:{number} is blank'
            json.loads(line)


@requires('zsh')
def test_every_response_parses_in_zsh():
    name, shell = PRIMARY_SHELL
    failures = [record.record_id for _, record in curated_records() if not parses_in(shell, record.response)]
    assert not failures, f'rejected by {name} -n: {failures}'


@pytest.mark.parametrize('name,shell', PORTABLE_SHELLS)
def test_dialect_exemptions_are_a_ratchet(name, shell):
    if shutil.which(name) is None:
        pytest.skip(f'{name} is not installed')
    families, record_ids = read_dialect_exemptions()

    known = {path.name for path in corpus_files()}
    assert families <= known, f'dialect-exempt.txt names missing files: {families - known}'

    unexpected_failures = []
    for path, record in curated_records():
        exempt = path.name in families or record.record_id in record_ids
        if parses_in(shell, record.response):
            continue
        if not exempt:
            unexpected_failures.append(f'{record.record_id} :: {record.response}')

    assert not unexpected_failures, (
        f'these rows are rejected by {name} -n and are not in dialect-exempt.txt: {unexpected_failures}'
    )


def test_dialect_exemptions_are_still_needed():
    """An exemption must name a row that some target shell actually rejects."""
    available = [(name, shell) for name, shell in PORTABLE_SHELLS if shutil.which(name) is not None]
    if len(available) < len(PORTABLE_SHELLS):
        pytest.skip('every portable shell must be installed to prove an exemption is stale')

    families, record_ids = read_dialect_exemptions()
    records = curated_records()

    known_ids = {record.record_id for _, record in records}
    assert record_ids <= known_ids, f'dialect-exempt.txt names missing rows: {record_ids - known_ids}'

    portable_everywhere = {
        record.record_id for _, record in records if all(parses_in(shell, record.response) for _, shell in available)
    }

    stale_rows = record_ids & portable_everywhere
    assert not stale_rows, f'these rows now parse everywhere; delete them from dialect-exempt.txt: {sorted(stale_rows)}'

    for family in families:
        rows = [record.record_id for path, record in records if path.name == family]
        assert not set(rows) <= portable_everywhere, (
            f'every row in {family} now parses in all target shells; delete it from dialect-exempt.txt'
        )
