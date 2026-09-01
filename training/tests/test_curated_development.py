import json
from pathlib import Path

import pytest

from shelliq_training.corpus import supervised_corpus_paths
from shelliq_training.curated_development import (
    ChallengeAnnotation,
    CuratedDevelopmentManifest,
    audit_curated_development,
    load_manifest,
)
from shelliq_training.data import Corpus, Platform, SFTRecord, load_jsonl
from shelliq_training.semantic_evaluation import load_grounding_audit

TRAINING_ROOT = Path(__file__).resolve().parents[1]


def record(
    record_id: str = 'curated-development:v1:linux:apt-cache-search-names',
    *,
    command: str = 'apt-cache',
    source: str = 'shelliq-curated-development',
    instruction: str = 'Search package names only for editor.',
    response: str = 'apt-cache --names-only search editor',
) -> SFTRecord:
    return SFTRecord(
        record_id=record_id,
        corpus=Corpus.DISTRIBUTABLE,
        source=source,
        license='MIT OR Apache-2.0',
        provenance='evaluation/curated-development-v1.jsonl#test',
        command=command,
        platform=Platform.LINUX,
        instruction=instruction,
        response=response,
        context='apt-cache: --names-only restricts search to package names.',
    )


def manifest(*records: SFTRecord, status: str = 'draft') -> CuratedDevelopmentManifest:
    return CuratedDevelopmentManifest(
        status=status,
        records={
            item.record_id: ChallengeAnnotation(
                family=item.command,
                dimensions=frozenset({'flag-selection', 'operand-binding'}),
                constraint_count=2,
            )
            for item in records
        },
    )


def test_draft_suite_passes_closed_audit() -> None:
    challenge = record()
    report = audit_curated_development(
        [challenge],
        manifest(challenge),
        {challenge.record_id: ()},
    )
    assert report.records == 1
    assert report.families == 1
    assert report.status == 'draft'


def test_audit_rejects_curated_command_family_leakage() -> None:
    challenge = record()
    training = record(
        'curated:packages:linux:apt-cache-policy',
        source='shelliq-curated',
        instruction='Show package policy.',
        response='apt-cache policy editor',
    )
    with pytest.raises(ValueError, match='overlap curated training'):
        audit_curated_development(
            [challenge],
            manifest(challenge),
            {challenge.record_id: ()},
            training_records=[training],
        )


def test_audit_rejects_behavioral_holdout_command_overlap() -> None:
    challenge = record()
    holdout = record(
        'retention:v1:linux:apt-cache-policy',
        source='shelliq-retention',
        instruction='Show package policy.',
        response='apt-cache policy editor',
    )
    with pytest.raises(ValueError, match='overlap behavioral holdouts'):
        audit_curated_development(
            [challenge],
            manifest(challenge),
            {challenge.record_id: ()},
            behavioral_holdouts=[holdout],
        )


def test_audit_rejects_exact_instruction_leakage_across_commands() -> None:
    challenge = record()
    training = record(
        'curated:packages:linux:apt-mark-showhold',
        command='apt-mark',
        source='shelliq-curated',
        response='apt-mark showhold',
    )
    with pytest.raises(ValueError, match='exact leakage'):
        audit_curated_development(
            [challenge],
            manifest(challenge),
            {challenge.record_id: ()},
            training_records=[training],
        )


def test_frozen_suite_rejects_too_few_records() -> None:
    challenge = record()
    with pytest.raises(ValueError, match='at least 100 records'):
        audit_curated_development(
            [challenge],
            manifest(challenge, status='frozen'),
            {challenge.record_id: ()},
        )


def test_manifest_loader_is_closed(tmp_path: Path) -> None:
    challenge = record()
    path = tmp_path / 'manifest.json'
    path.write_text(
        json.dumps(
            {
                'schema_version': 1,
                'suite_id': 'curated-development-v1',
                'status': 'draft',
                'records': {
                    challenge.record_id: {
                        'family': 'apt-cache',
                        'dimensions': ['flag-selection', 'operand-binding'],
                        'constraint_count': 2,
                    }
                },
            }
        )
    )
    loaded = load_manifest(path)
    assert loaded.records[challenge.record_id].family == 'apt-cache'


def test_checked_in_frozen_suite_is_disjoint_and_closed() -> None:
    evaluation_root = TRAINING_ROOT / 'evaluation'
    suite_path = evaluation_root / 'curated-development-v1.jsonl'
    manifest_path = evaluation_root / 'curated-development-v1.manifest.json'
    grounding_path = evaluation_root / 'curated-development-grounding-v1.json'
    corpus_paths = [*supervised_corpus_paths(TRAINING_ROOT / 'corpus')]
    holdout_paths = [
        evaluation_root / 'pipeline-compatibility-v1.jsonl',
        evaluation_root / 'semantic-retention-v1.jsonl',
        evaluation_root / 'semantic-shadow-release-v1.jsonl',
    ]
    training = [item for path in corpus_paths for item in load_jsonl(path, corpus=Corpus.DISTRIBUTABLE)]
    holdouts = [item for path in holdout_paths for item in load_jsonl(path, corpus=Corpus.DISTRIBUTABLE)]
    report = audit_curated_development(
        load_jsonl(suite_path, corpus=Corpus.DISTRIBUTABLE),
        load_manifest(manifest_path),
        load_grounding_audit(grounding_path),
        training_records=training,
        behavioral_holdouts=holdouts,
    )
    assert report.status == 'frozen'
    assert report.records == 104
    assert report.families == 34
    assert report.dimensions['pipeline-semantics'] == 5
