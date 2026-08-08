"""Machine-local builders whose only output passes through ``PrivateDataGate``."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from shelliq_training.data import Corpus, DatasetFormatError, Platform, SFTRecord
from shelliq_training.privacy import PrivateDataGate, ScrubbedCorpus
from shelliq_training.sources import shell_command_name

PRIVATE_LICENSE = 'private-local-only'
CLAUDE_SOURCE = 'claude-code-transcript'
LOCAL_INDEX_SOURCE = 'shelliq-local-index'


@dataclass(frozen=True, slots=True)
class _ClaudeBashCall:
    tool_use_id: str
    line_number: int
    command: str
    description: str


@dataclass(frozen=True, slots=True)
class ClaudeTranscriptFailure:
    """One transcript omitted only when partial ingestion is explicit."""

    path: Path
    transcript_id: str
    error: str


@dataclass(frozen=True, slots=True)
class ClaudeTranscriptBuild:
    """Scrubbed rows plus a complete file-level ingestion audit."""

    scrubbed: ScrubbedCorpus
    processed_paths: tuple[Path, ...]
    skipped: tuple[ClaudeTranscriptFailure, ...]

    @property
    def processed_ids(self) -> tuple[str, ...]:
        return tuple(_claude_transcript_id(path) for path in self.processed_paths)


def build_claude_transcript_records(
    paths: Iterable[str | Path],
    *,
    gate: PrivateDataGate,
    platform: Platform,
) -> ScrubbedCorpus:
    """Extract successful calls from files or directories, failing closed."""
    return build_claude_transcript_corpus(paths, gate=gate, platform=platform).scrubbed


def build_claude_transcript_corpus(
    paths: Iterable[str | Path],
    *,
    gate: PrivateDataGate,
    platform: Platform,
    allow_partial: bool = False,
) -> ClaudeTranscriptBuild:
    """Extract and scrub rows while auditing every expanded transcript file."""
    raw_records: list[SFTRecord] = []
    processed_paths: list[Path] = []
    skipped: list[ClaudeTranscriptFailure] = []
    for path in expand_claude_transcript_paths(paths):
        try:
            calls, outcomes = _read_claude_transcript(path)
        except DatasetFormatError as error:
            if not allow_partial:
                raise
            skipped.append(
                ClaudeTranscriptFailure(
                    path=path,
                    transcript_id=_claude_transcript_id(path),
                    error=str(error).replace(str(path), '<TRANSCRIPT>'),
                )
            )
            continue
        processed_paths.append(path)
        transcript_id = _claude_transcript_id(path)
        for call in calls:
            if outcomes.get(call.tool_use_id) is not True:
                continue
            try:
                command_name = shell_command_name(call.command)
            except DatasetFormatError:
                continue
            call_id = hashlib.sha256(call.tool_use_id.encode()).hexdigest()[:12]
            raw_records.append(
                SFTRecord(
                    record_id=f'claude:{transcript_id}:{call.line_number}:{call_id}',
                    corpus=Corpus.PERSONAL,
                    source=CLAUDE_SOURCE,
                    license=PRIVATE_LICENSE,
                    provenance=f'{path.resolve()}:{call.line_number}',
                    command=command_name,
                    platform=platform,
                    instruction=call.description,
                    response=call.command,
                    context=f'{command_name}: command completed successfully in a local Claude Code session.',
                )
            )
    return ClaudeTranscriptBuild(gate.scrub(raw_records), tuple(processed_paths), tuple(skipped))


def expand_claude_transcript_paths(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    """Expand directories recursively and return unique resolved paths in stable order."""
    expanded: dict[Path, None] = {}
    for path_value in paths:
        path = Path(path_value)
        if path.is_dir():
            try:
                matches = sorted(path.rglob('*.jsonl'))
            except OSError as error:
                raise DatasetFormatError(f'cannot enumerate transcript directory {path}: {error}') from error
            if not matches:
                raise DatasetFormatError(f'{path}: transcript directory contains no JSONL files')
            candidates = matches
        else:
            candidates = [path]
        for candidate in candidates:
            expanded[candidate.resolve()] = None
    if not expanded:
        raise DatasetFormatError('no transcript paths provided')
    return tuple(sorted(expanded, key=lambda path: path.as_posix()))


def _claude_transcript_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


def build_local_index_records(
    database: str | Path,
    *,
    gate: PrivateDataGate,
) -> ScrubbedCorpus:
    """Build private examples from shelliq's local SQLite index, never history."""
    path = Path(database).resolve()
    uri = f'file:{path}?mode=ro'
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            rows = connection.execute(
                """
                SELECT e.id, c.name, c.platform, e.description, e.text,
                       c.synopsis, c.description, e.source
                FROM examples AS e
                JOIN commands AS c ON c.id = e.command_id
                WHERE trim(e.description) != '' AND trim(e.text) != ''
                ORDER BY e.id
                """
            ).fetchall()
    except sqlite3.Error as error:
        raise DatasetFormatError(f'cannot read shelliq local index {path}: {error}') from error

    raw_records: list[SFTRecord] = []
    for example_id, command, platform_text, instruction, response, synopsis, description, source in rows:
        try:
            platform = Platform(platform_text)
        except ValueError:
            continue
        context_parts = [part.strip() for part in (synopsis, description) if part and part.strip()]
        context = ' '.join(context_parts) or f'{command}: local indexed example.'
        source_note = f' source={source}' if source else ''
        raw_records.append(
            SFTRecord(
                record_id=f'local-index:example:{example_id}',
                corpus=Corpus.PERSONAL,
                source=LOCAL_INDEX_SOURCE,
                license=PRIVATE_LICENSE,
                provenance=f'{path}:examples/{example_id}{source_note}',
                command=command,
                platform=platform,
                instruction=instruction,
                response=response,
                context=context,
            )
        )
    return gate.scrub(raw_records)


def _read_claude_transcript(path: Path) -> tuple[list[_ClaudeBashCall], dict[str, bool]]:
    calls: list[_ClaudeBashCall] = []
    outcomes: dict[str, bool] = {}
    try:
        stream = path.open(encoding='utf-8')
    except OSError as error:
        raise DatasetFormatError(f'cannot read Claude transcript {path}: {error}') from error
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetFormatError(f'{path}:{line_number}: invalid transcript JSON: {error}') from error
            if not isinstance(entry, Mapping):
                raise DatasetFormatError(f'{path}:{line_number}: transcript entry must be an object')
            content = _message_content(entry)
            for block in content:
                if block.get('type') == 'tool_use' and block.get('name') == 'Bash':
                    call = _bash_call(block, line_number)
                    if call is not None:
                        calls.append(call)
                elif block.get('type') == 'tool_result':
                    tool_use_id = block.get('tool_use_id')
                    if isinstance(tool_use_id, str) and tool_use_id:
                        outcomes[tool_use_id] = block.get('is_error') is not True
    return calls, outcomes


def _message_content(entry: Mapping[str, object]) -> list[Mapping[str, object]]:
    message = entry.get('message')
    if not isinstance(message, Mapping):
        return []
    content = message.get('content')
    if not isinstance(content, list):
        return []
    return [cast(Mapping[str, object], block) for block in content if isinstance(block, Mapping)]


def _bash_call(block: Mapping[str, object], line_number: int) -> _ClaudeBashCall | None:
    tool_use_id = block.get('id')
    arguments = block.get('input')
    if not isinstance(tool_use_id, str) or not tool_use_id or not isinstance(arguments, Mapping):
        return None
    command = arguments.get('command')
    description = arguments.get('description')
    if not isinstance(command, str) or not command.strip():
        return None
    if not isinstance(description, str) or not description.strip():
        return None
    return _ClaudeBashCall(tool_use_id, line_number, command.strip(), description.strip())
