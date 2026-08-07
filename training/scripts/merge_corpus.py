"""Merge pinned TLDR rows with the checked-in curated distributable corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAINING_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIRECTORY))

from shelliq_training.corpus import merge_distributable_corpus  # noqa: E402
from shelliq_training.data import Corpus, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tldr', type=Path, required=True, help='pinned generated TLDR schema-v1 JSONL')
    parser.add_argument(
        '--curated-directory',
        type=Path,
        default=TRAINING_DIRECTORY / 'corpus',
        help='directory of checked-in curated JSONL families',
    )
    parser.add_argument('--output', type=Path, required=True, help='new merged distributable JSONL artifact')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.output.with_suffix(args.output.suffix + '.manifest.json')
    _require_new_outputs(args.output, manifest_path)

    records, report = merge_distributable_corpus(args.tldr, args.curated_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, records, corpus=Corpus.DISTRIBUTABLE)
    manifest_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + '\n', encoding='utf-8')

    print(f'wrote {report.record_count} distributable records ({report.command_count} commands) to {args.output}')
    print('sources: ' + _format_counts(report.source_counts))
    print('curated families: ' + _format_counts(report.curated_family_counts))
    print(f'manifest: {manifest_path}')


def _require_new_outputs(output: Path, manifest: Path) -> None:
    existing = next((path for path in (output, manifest) if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f'output already exists: {existing}')


def _format_counts(counts: tuple[tuple[str, int], ...]) -> str:
    return ', '.join(f'{name}={count}' for name, count in counts)


if __name__ == '__main__':
    main()
