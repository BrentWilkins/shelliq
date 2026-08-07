import zipfile

import pytest

from shelliq_training.data import Corpus, DatasetFormatError, Platform
from shelliq_training.sources import (
    build_nl2bash_records,
    build_tldr_records,
    shell_command_name,
)


def write_tldr_archive(path):
    common = """# find

> Find files below a directory tree.
> More information: <https://manned.org/find>.

- Find files by extension:

`find {{path/to/directory}} -name '{{*.ext}}'`

- Find either files or directories:

`find . -type {{f|d}}`
"""
    osx = """# caffeinate

> Prevent macOS from sleeping.

- Prevent sleep for one hour:

`caffeinate -u -t {{3600}}`
"""
    with zipfile.ZipFile(path, 'w') as bundle:
        bundle.writestr('common/find.md', common)
        bundle.writestr('osx/caffeinate.md', osx)
        bundle.writestr(
            'linux/az-tag.md',
            '# az tag\n\n- Create a tag:\n\n`az tag create {{[-n|--name]}} {{tag_name}}`\n',
        )
        bundle.writestr('windows/dir.md', '# dir\n')


def test_tldr_builder_tracks_license_revision_platform_and_placeholders(tmp_path):
    archive = tmp_path / 'tldr.zip'
    write_tldr_archive(archive)

    records = build_tldr_records(
        archive,
        revision='v2.3',
        common_platforms=(Platform.LINUX, Platform.DARWIN),
    )

    assert len(records) == 6
    assert {record.corpus for record in records} == {Corpus.DISTRIBUTABLE}
    assert {record.license for record in records} == {'CC-BY-4.0'}
    assert records[0].provenance.endswith('@v2.3:example-1')
    linux_find = next(record for record in records if record.command == 'find' and record.platform is Platform.LINUX)
    assert linux_find.response == "find path/to/directory -name '*.ext'"
    assert 'More information' not in linux_find.context
    assert any(record.response == 'az tag create -n tag_name' for record in records)
    assert any(record.platform is Platform.DARWIN and record.command == 'caffeinate' for record in records)


def test_tldr_builder_rejects_unparseable_example(tmp_path):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('common/find.md', '# find\n\n- Find files:\n\nnot code\n')

    with pytest.raises(DatasetFormatError, match='no command line'):
        build_tldr_records(archive, revision='abc')


def test_tldr_builder_groups_subcommand_pages_by_executable(tmp_path):
    archive = tmp_path / 'git.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('common/git-commit.md', '# git commit\n\n> Commit files.\n\n- Commit all files:\n\n`git commit -a`\n')

    records = build_tldr_records(archive, revision='abc')

    assert records[0].command == 'git'


def test_nl2bash_builder_emits_only_reviewed_deduplicated_rows(tmp_path):
    instructions = tmp_path / 'all.nl'
    commands = tmp_path / 'all.cm'
    instructions.write_text('list files\nfind python files\nlist files\n', encoding='utf-8')
    commands.write_text("ls -la\nfind . -name '*.py'\nls -la\n", encoding='utf-8')

    records = build_nl2bash_records(
        instructions,
        commands,
        revision='deadbeef',
        license='verified-license-id',
        reviewed_lines={1, 2, 3},
    )

    assert [record.record_id for record in records] == ['nl2bash:1', 'nl2bash:2']
    assert [record.command for record in records] == ['ls', 'find']
    assert records[1].provenance == 'all.nl+all.cm@deadbeef:line-2'


def test_nl2bash_builder_requires_review_and_alignment(tmp_path):
    instructions = tmp_path / 'all.nl'
    commands = tmp_path / 'all.cm'
    instructions.write_text('one\ntwo\n', encoding='utf-8')
    commands.write_text('echo one\n', encoding='utf-8')

    with pytest.raises(ValueError, match='reviewed_lines'):
        build_nl2bash_records(
            instructions,
            commands,
            revision='r1',
            license='license',
            reviewed_lines=set(),
        )
    with pytest.raises(DatasetFormatError, match='not aligned'):
        build_nl2bash_records(
            instructions,
            commands,
            revision='r1',
            license='license',
            reviewed_lines={1},
        )


@pytest.mark.parametrize(
    ('command_line', 'expected'),
    [
        ('LC_ALL=C rg needle .', 'rg'),
        ('sudo -u root /usr/bin/find .', 'find'),
        ('env -i bash -lc true', 'bash'),
        ('env -i LC_ALL=C rg needle .', 'rg'),
        ('command -p /usr/bin/find .', 'find'),
        ('command -v rg', 'command'),
        ('sudo -l', 'sudo'),
        ('sudo --list /usr/bin/find', 'sudo'),
    ],
)
def test_shell_command_name(command_line, expected):
    assert shell_command_name(command_line) == expected
