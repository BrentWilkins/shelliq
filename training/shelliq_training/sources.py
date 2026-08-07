"""Source-specific builders for the distributable training corpus."""

from __future__ import annotations

import re
import shlex
import zipfile
from collections.abc import Collection, Sequence
from pathlib import Path

from shelliq_training.data import Corpus, DatasetFormatError, Platform, SFTRecord

TLDR_LICENSE = 'CC-BY-4.0'
TLDR_SOURCE = 'tldr-pages'
NL2BASH_SOURCE = 'NL2Bash'

_TLDR_DESCRIPTION = re.compile(r'^- (.+):$')
_TLDR_COMMAND = re.compile(r'^`(.+)`$')
_TLDR_PLACEHOLDER = re.compile(r'\{\{(.+?)\}\}')
_SHELL_ASSIGNMENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')


def build_tldr_records(
    archive: str | Path,
    *,
    revision: str,
    common_platforms: Sequence[Platform] = (Platform.LINUX,),
) -> list[SFTRecord]:
    """Build audited SFT rows from a tldr-pages English release archive."""
    revision = _required_metadata(revision, 'revision')
    platforms = tuple(dict.fromkeys(common_platforms))
    if not platforms:
        raise ValueError('common_platforms must not be empty')

    records: list[SFTRecord] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in sorted(bundle.namelist()):
            if member.endswith('/') or not member.endswith('.md'):
                continue
            source_platforms = _tldr_platforms(member, platforms)
            if not source_platforms:
                continue
            try:
                markdown = bundle.read(member).decode('utf-8')
            except UnicodeDecodeError as error:
                raise DatasetFormatError(f'{member}: tldr page is not UTF-8') from error
            records.extend(_tldr_page_records(member, markdown, revision, source_platforms))

    if not records:
        raise DatasetFormatError(f'{archive}: no supported tldr examples found')
    return records


def build_nl2bash_records(
    instructions_path: str | Path,
    commands_path: str | Path,
    *,
    revision: str,
    license: str,
    reviewed_lines: Collection[int],
    platform: Platform = Platform.LINUX,
) -> list[SFTRecord]:
    """Build only explicitly reviewed rows from NL2Bash's aligned text files."""
    revision = _required_metadata(revision, 'revision')
    license = _required_metadata(license, 'license')
    approved = frozenset(reviewed_lines)
    if not approved:
        raise ValueError('reviewed_lines must contain at least one 1-based line number')
    if any(line < 1 for line in approved):
        raise ValueError('reviewed_lines must use positive 1-based line numbers')

    instructions_file = Path(instructions_path)
    commands_file = Path(commands_path)
    instructions = instructions_file.read_text(encoding='utf-8').splitlines()
    commands = commands_file.read_text(encoding='utf-8').splitlines()
    if len(instructions) != len(commands):
        raise DatasetFormatError(f'NL2Bash files are not aligned: {len(instructions)} instructions != {len(commands)} commands')
    missing = sorted(approved.difference(range(1, len(instructions) + 1)))
    if missing:
        raise DatasetFormatError(f'reviewed NL2Bash line does not exist: {missing[0]}')

    records: list[SFTRecord] = []
    seen_pairs: set[tuple[str, str]] = set()
    for line_number in sorted(approved):
        instruction = instructions[line_number - 1].strip()
        response = commands[line_number - 1].strip()
        if not instruction or not response:
            raise DatasetFormatError(f'NL2Bash line {line_number}: reviewed pair is blank')
        pair = (instruction, response)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        command = shell_command_name(response)
        records.append(
            SFTRecord(
                record_id=f'nl2bash:{line_number}',
                corpus=Corpus.DISTRIBUTABLE,
                source=NL2BASH_SOURCE,
                license=license,
                provenance=(f'{instructions_file.name}+{commands_file.name}@{revision}:line-{line_number}'),
                command=command,
                platform=platform,
                instruction=instruction,
                response=response,
                context=f'{command}: reviewed NL2Bash example; no machine-local context.',
            )
        )
    return records


def shell_command_name(command_line: str) -> str:
    """Return the first executable for grouping without running a shell parser."""
    try:
        tokens = shlex.split(command_line, posix=True)
    except ValueError as error:
        raise DatasetFormatError(f'invalid shell quoting: {error}') from error
    position = 0
    while position < len(tokens) and _SHELL_ASSIGNMENT.match(tokens[position]):
        position += 1
    if position < len(tokens) and tokens[position] == 'command':
        position += 1
    elif position < len(tokens) and tokens[position] in {'env', 'sudo'}:
        wrapper = tokens[position]
        position += 1
        options_with_values = {'env': {'-u', '--unset', '-C', '--chdir'}, 'sudo': {'-u', '-g', '-h', '-p', '-C', '-R', '-T'}}[
            wrapper
        ]
        while position < len(tokens) and tokens[position].startswith('-'):
            option = tokens[position].split('=', maxsplit=1)[0]
            position += 1
            if option in options_with_values and '=' not in tokens[position - 1]:
                position += 1
        while position < len(tokens) and _SHELL_ASSIGNMENT.match(tokens[position]):
            position += 1
    if position >= len(tokens) or tokens[position] in {'|', '&&', '||', ';'}:
        raise DatasetFormatError('command line has no executable')
    return Path(tokens[position]).name


def _tldr_page_records(
    member: str,
    markdown: str,
    revision: str,
    platforms: Sequence[Platform],
) -> list[SFTRecord]:
    lines = markdown.splitlines()
    heading = next((line[2:].strip() for line in lines if line.startswith('# ')), '')
    command = heading.split(maxsplit=1)[0] if heading else Path(member).stem
    summary = ' '.join(
        _plain_tldr(line[2:]) for line in lines if line.startswith('> ') and not line.startswith('> More information:')
    ).strip()
    if not summary:
        summary = f'tldr examples for {command}.'

    examples: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        description_match = _TLDR_DESCRIPTION.fullmatch(line)
        if description_match is None:
            continue
        command_line = next((item for item in lines[index + 1 :] if item.strip()), '')
        command_match = _TLDR_COMMAND.fullmatch(command_line)
        if command_match is None:
            raise DatasetFormatError(f'{member}:{index + 1}: example has no command line')
        instruction = _plain_tldr(description_match.group(1))
        response = _render_tldr_command(command_match.group(1))
        examples.append((instruction, response))

    records: list[SFTRecord] = []
    for platform in platforms:
        for example_number, (instruction, response) in enumerate(examples, start=1):
            records.append(
                SFTRecord(
                    record_id=f'tldr:{platform.value}:{member[:-3]}:{example_number}',
                    corpus=Corpus.DISTRIBUTABLE,
                    source=TLDR_SOURCE,
                    license=TLDR_LICENSE,
                    provenance=f'{member}@{revision}:example-{example_number}',
                    command=command,
                    platform=platform,
                    instruction=instruction,
                    response=response,
                    context=f'{command}: {summary}',
                )
            )
    return records


def _tldr_platforms(member: str, common_platforms: Sequence[Platform]) -> tuple[Platform, ...]:
    directory = member.partition('/')[0]
    if directory == 'common':
        return tuple(common_platforms)
    if directory == 'linux':
        return (Platform.LINUX,)
    if directory == 'osx':
        return (Platform.DARWIN,)
    return ()


def _render_tldr_command(command: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(1)
        if value.startswith('[') and value.endswith(']'):
            value = value[1:-1]
        return value.split('|', maxsplit=1)[0]

    return _TLDR_PLACEHOLDER.sub(replace, command).strip()


def _plain_tldr(text: str) -> str:
    return text.replace('`', '').strip()


def _required_metadata(value: str, field: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f'{field} must be non-empty')
    return value
