"""Build public or private shelliq JSONL corpora through their required gates."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import socket
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.data import Corpus, Platform, write_jsonl  # noqa: E402
from shelliq_training.privacy import (  # noqa: E402
    PrivateDataGate,
    ScrubbedCorpus,
    ScrubPolicy,
    plant_training_canaries,
)
from shelliq_training.private_sources import (  # noqa: E402
    ClaudeTranscriptBuild,
    build_claude_transcript_corpus,
    build_local_index_records,
)
from shelliq_training.sources import build_nl2bash_records, build_tldr_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='source', required=True)

    tldr = subparsers.add_parser('tldr')
    tldr.add_argument('--archive', type=Path, required=True)
    tldr.add_argument('--revision', required=True)
    tldr.add_argument('--common-platform', type=Platform, action='append', default=[])
    tldr.add_argument('--output', type=Path, required=True)

    nl2bash = subparsers.add_parser('nl2bash')
    nl2bash.add_argument('--instructions', type=Path, required=True)
    nl2bash.add_argument('--commands', type=Path, required=True)
    nl2bash.add_argument('--reviewed-lines', type=Path, required=True)
    nl2bash.add_argument('--revision', required=True)
    nl2bash.add_argument('--license', required=True)
    nl2bash.add_argument('--platform', type=Platform, default=Platform.LINUX)
    nl2bash.add_argument('--output', type=Path, required=True)

    claude = subparsers.add_parser('claude')
    claude.add_argument('transcripts', type=Path, nargs='+')
    claude.add_argument('--platform', type=Platform, required=True)
    claude.add_argument(
        '--allow-partial',
        action='store_true',
        help='skip malformed transcript files and record every skip in the scrub report',
    )
    _private_arguments(claude)

    index = subparsers.add_parser('local-index')
    index.add_argument('--database', type=Path, required=True)
    _private_arguments(index)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_new_output(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.source == 'tldr':
        platforms = tuple(args.common_platform) or (Platform.LINUX,)
        records = build_tldr_records(args.archive, revision=args.revision, common_platforms=platforms)
        write_jsonl(args.output, records, corpus=Corpus.DISTRIBUTABLE)
        print(f'wrote {len(records)} distributable tldr records to {args.output}')
        return
    if args.source == 'nl2bash':
        records = build_nl2bash_records(
            args.instructions,
            args.commands,
            revision=args.revision,
            license=args.license,
            reviewed_lines=_read_reviewed_lines(args.reviewed_lines),
            platform=args.platform,
        )
        write_jsonl(args.output, records, corpus=Corpus.DISTRIBUTABLE)
        print(f'wrote {len(records)} reviewed distributable NL2Bash records to {args.output}')
        return

    gate = PrivateDataGate(_policy(args))
    transcript_build = None
    if args.source == 'claude':
        transcript_build = build_claude_transcript_corpus(
            args.transcripts,
            gate=gate,
            platform=args.platform,
            allow_partial=args.allow_partial,
        )
        for failure in transcript_build.skipped:
            print(f'skipped transcript {failure.path}: {failure.error}', file=sys.stderr)
        scrubbed = transcript_build.scrubbed
    else:
        scrubbed = build_local_index_records(args.database, gate=gate)
    _write_private_outputs(
        args.output,
        scrubbed,
        args.canary_count,
        args.canary_seed,
        transcript_build=transcript_build,
    )


def _private_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--home', action='append', default=[])
    parser.add_argument('--username', action='append', default=[])
    parser.add_argument('--hostname', action='append', default=[])
    parser.add_argument('--canary-count', type=int, required=True)
    parser.add_argument('--canary-seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)


def _policy(args: argparse.Namespace) -> ScrubPolicy:
    homes = tuple(dict.fromkeys([str(Path.home()), *args.home]))
    usernames = tuple(dict.fromkeys([getpass.getuser(), *args.username]))
    host = socket.gethostname()
    hostnames = tuple(dict.fromkeys([host, host.partition('.')[0], *args.hostname]))
    return ScrubPolicy(homes, usernames, hostnames)


def _read_reviewed_lines(path: Path) -> frozenset[int]:
    values: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        value = line.partition('#')[0].strip()
        if not value:
            continue
        try:
            values.add(int(value))
        except ValueError as error:
            raise ValueError(f'{path}:{line_number}: expected a 1-based integer') from error
    return frozenset(values)


def _write_private_outputs(
    output: Path,
    scrubbed: ScrubbedCorpus,
    canary_count: int,
    canary_seed: int,
    *,
    transcript_build: ClaudeTranscriptBuild | None = None,
) -> None:
    if not scrubbed.records:
        raise SystemExit('privacy gate accepted no records; no artifact written')
    canaries = plant_training_canaries(scrubbed.records, count=canary_count, seed=canary_seed)
    probes_path = output.with_suffix(output.suffix + '.canaries.json')
    report_path = output.with_suffix(output.suffix + '.scrub-report.json')
    audit_path = output.with_suffix(output.suffix + '.audit.jsonl')
    audit_sample = sorted(
        canaries.records,
        key=lambda record: hashlib.sha256(f'{canary_seed}\0audit\0{record.record_id}'.encode()).digest(),
    )[:20]
    probes_path.write_text(
        json.dumps(
            [{'prompt': probe.prompt, 'forbidden_value': probe.forbidden_value} for probe in canaries.probes],
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )
    counts = Counter(finding.kind for finding in scrubbed.findings)
    report = {
        'accepted': len(scrubbed.records),
        'dropped': len(scrubbed.dropped),
        'dropped_record_ids': sorted({finding.record_id for result in scrubbed.dropped for finding in result.findings}),
        'finding_counts': dict(sorted(counts.items())),
        'audit_record_ids': [record.record_id for record in audit_sample],
    }
    if transcript_build is not None:
        report['processed_transcript_ids'] = list(transcript_build.processed_ids)
        report['skipped_transcripts'] = [
            {'transcript_id': failure.transcript_id, 'error': failure.error} for failure in transcript_build.skipped
        ]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    write_jsonl(audit_path, audit_sample, corpus=Corpus.PERSONAL)
    write_jsonl(output, canaries.records, corpus=Corpus.PERSONAL)
    print(
        f'wrote {len(canaries.records)} personal records ({len(scrubbed.dropped)} dropped), '
        f'{len(canaries.probes)} canary probes, scrub report, and {len(audit_sample)}-row audit sample'
    )


def _require_new_output(path: Path) -> None:
    related = (
        path,
        path.with_suffix(path.suffix + '.canaries.json'),
        path.with_suffix(path.suffix + '.scrub-report.json'),
        path.with_suffix(path.suffix + '.audit.jsonl'),
    )
    existing = next((candidate for candidate in related if candidate.exists()), None)
    if existing is not None:
        raise FileExistsError(f'output already exists: {existing}')


if __name__ == '__main__':
    main()
