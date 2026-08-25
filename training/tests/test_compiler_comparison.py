import hashlib

import pytest

from shelliq_training.compiler_comparison import (
    analyze_comparison_reports,
    freeze_comparison_manifest,
    load_comparison_manifest,
    paired_binary_counts,
    require_matching_dataset,
    write_comparison_manifest,
)
from shelliq_training.data import Corpus, DatasetFormatError, Platform, SFTRecord, Split


def record(index: int) -> SFTRecord:
    return SFTRecord(
        record_id=f'comparison:{index}',
        corpus=Corpus.DISTRIBUTABLE,
        source='fixture',
        license='MIT',
        provenance='fixture',
        command=f'command-{index}',
        platform=Platform.LINUX,
        instruction=f'Do task {index}.',
        response='{"v":2,"d":"zsh","s":[]}',
        context=f'command-{index}: context',
    )


def test_manifest_round_trip_freezes_disjoint_complete_split(tmp_path):
    records = [record(index) for index in range(100)]
    dataset = tmp_path / 'dataset.jsonl'
    dataset.write_text('fixture\n')
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest = freeze_comparison_manifest(
        records,
        dataset_sha256=digest,
        experiment='comparison-v1',
        split_seed=17,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    path = tmp_path / 'manifest.json'
    write_comparison_manifest(path, manifest)

    restored = load_comparison_manifest(path)
    split_records = restored.records_by_split(records)

    assert restored == manifest
    assert set(split_records) == set(Split)
    assert sum(len(rows) for rows in split_records.values()) == len(records)
    require_matching_dataset(dataset, restored)


def test_manifest_rejects_dataset_drift(tmp_path):
    dataset = tmp_path / 'dataset.jsonl'
    dataset.write_text('changed\n')
    manifest = freeze_comparison_manifest(
        [record(index) for index in range(100)],
        dataset_sha256='0' * 64,
        experiment='comparison-v1',
        split_seed=17,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    with pytest.raises(DatasetFormatError, match='SHA-256 mismatch'):
        require_matching_dataset(dataset, manifest)


def test_paired_counts_include_exact_mcnemar_probability():
    left = {'a': True, 'b': True, 'c': False, 'd': False, 'e': False}
    right = {'a': True, 'b': False, 'c': True, 'd': True, 'e': False}

    result = paired_binary_counts(left, right)

    assert result == {
        'both': 1,
        'left_only': 1,
        'right_only': 2,
        'neither': 1,
        'discordant': 3,
        'mcnemar_exact_two_sided_p': 1.0,
    }


def test_analysis_reduces_raw_reports_to_paired_outcomes():
    def train(contender):
        return {
            'experiment': 'semantic-compiler-heldout-v1',
            'contender': contender,
            'dataset_sha256': 'data',
            'manifest_sha256': 'manifest',
            'checkpoint_sha256': f'{contender}-checkpoint',
            'model': {'parameter_count': 60},
            'record_presentations': 100,
            'best_epoch': 2,
            'best_validation_loss': 0.5,
            'elapsed_seconds': 3.0,
        }

    def test(contender, outcomes):
        return {
            'experiment': 'semantic-compiler-heldout-v1',
            'contender': contender,
            'dataset_sha256': 'data',
            'manifest_sha256': 'manifest',
            'checkpoint_sha256': f'{contender}-checkpoint',
            'elapsed_seconds': 1.0,
            'metrics': {'examples': 2},
            'outcomes': outcomes,
        }

    custom_outcomes = [
        {
            'record_id': 'a',
            'exact': False,
            'rust_round_trip': False,
            'reference_accepted': False,
            'failures': ['first-command-mismatch'],
        },
        {
            'record_id': 'b',
            'exact': False,
            'rust_round_trip': False,
            'reference_accepted': False,
            'failures': ['first-command-mismatch'],
        },
    ]
    codet5_outcomes = [
        {
            'record_id': 'a',
            'exact': False,
            'rust_round_trip': True,
            'reference_accepted': True,
            'failures': [],
        },
        {
            'record_id': 'b',
            'exact': False,
            'rust_round_trip': True,
            'reference_accepted': False,
            'failures': ['reference-mismatch'],
        },
    ]

    result = analyze_comparison_reports(
        train('custom'),
        test('custom', custom_outcomes),
        train('codet5'),
        test('codet5', codet5_outcomes),
    )

    assert result['custom']['first_command_match'] == 0
    assert result['codet5']['first_command_match'] == 2
    assert result['paired']['rust_round_trip']['right_only'] == 2
    assert result['decision']['custom_generalization_gate_passed'] is False
