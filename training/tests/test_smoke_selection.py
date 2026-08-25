import argparse
from collections import Counter
from dataclasses import replace

import pytest

from scripts.smoke_finetune import (
    _positive_step_set,
    apply_rehearsal_weight,
    apply_rehearsal_weights,
    apply_source_presentation_target,
    parse_rehearsal_specs,
    select_examples,
    select_source_fraction,
    trim_to_complete_batches,
)
from shelliq_training.data import Corpus, Platform, SFTRecord


class FakeQwenTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize
        tokens = [1]
        for message in conversation:
            role_token = 2 if message['role'] == 'user' else 3
            tokens.extend((role_token, *message['content'].encode(), 4))
        if add_generation_prompt:
            tokens.append(3)
        return {'input_ids': tokens, 'attention_mask': [1] * len(tokens)}


def record(index: int, *, source: str = 'shelliq-curated') -> SFTRecord:
    return SFTRecord(
        record_id=f'curated:test:linux:row-{index}',
        corpus=Corpus.DISTRIBUTABLE,
        source=source,
        license='MIT',
        provenance=f'test:{index}',
        command=f'command-{index}',
        platform=Platform.LINUX,
        instruction=f'Run command {index}',
        response=f'command-{index}',
        context='',
    )


def test_priority_selection_stops_at_requested_count():
    records = [record(index) for index in range(40)]

    selected, examples = select_examples(
        records,
        FakeQwenTokenizer(),
        count=35,
        max_length=384,
        seed=2026,
        priority_source='shelliq-curated',
    )

    assert len(selected) == 35
    assert len(examples) == 35


def test_rehearsal_weight_repeats_only_matching_records_deterministically():
    records = [record(1), record(2), record(3)]
    records[1] = replace(records[1], record_id='curated:targeted-finishing:linux:row-2')
    _, examples = select_examples(
        records,
        FakeQwenTokenizer(),
        count=3,
        max_length=384,
        seed=2026,
    )

    weighted_records, weighted_examples = apply_rehearsal_weight(
        records,
        examples,
        record_prefix='curated:targeted-finishing:',
        weight=3,
        seed=2026,
    )

    assert len(weighted_records) == len(weighted_examples) == 5
    assert sum('targeted-finishing' in item.record_id for item in weighted_records) == 3


def test_positive_step_set_rejects_duplicates_and_non_positive_steps() -> None:
    assert _positive_step_set('2000,4000,14124') == frozenset({2000, 4000, 14124})
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_step_set('2000,2000')
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_step_set('0,2000')


def test_full_corpus_batch_alignment_drops_only_the_odd_final_row() -> None:
    records = [record(index) for index in range(5)]
    selected_records, examples = select_examples(records, FakeQwenTokenizer(), count=5, max_length=384, seed=2026)

    aligned_records, aligned_examples = trim_to_complete_batches(selected_records, examples, batch_size=2)

    assert aligned_records == selected_records[:4]
    assert aligned_examples == examples[:4]


def test_source_fraction_preserves_families_and_exact_target() -> None:
    records = [record(index) for index in range(12)]
    selected_records, examples = select_examples(records, FakeQwenTokenizer(), count=12, max_length=384, seed=2026)
    selected_records = [replace(item, command=f'family-{index % 3}') for index, item in enumerate(selected_records)]

    subset, subset_examples = select_source_fraction(
        selected_records,
        examples,
        source='shelliq-curated',
        fraction=0.25,
        seed=2026,
    )

    assert len(subset) == len(subset_examples) == 3
    assert {item.command for item in subset} == {'family-0', 'family-1', 'family-2'}


def test_source_presentations_are_exact_and_balanced() -> None:
    records = [record(index) for index in range(3)]
    selected_records, examples = select_examples(records, FakeQwenTokenizer(), count=3, max_length=384, seed=2026)

    weighted, weighted_examples = apply_source_presentation_target(
        selected_records,
        examples,
        source='shelliq-curated',
        presentations=8,
        seed=2026,
    )

    counts = Counter(item.record_id for item in weighted)
    assert len(weighted) == len(weighted_examples) == 8
    assert sorted(counts.values()) == [2, 3, 3]


def test_rehearsal_weight_rejects_missing_prefix():
    records = [record(1)]
    _, examples = select_examples(records, FakeQwenTokenizer(), count=1, max_length=384, seed=2026)
    with pytest.raises(ValueError, match='no selected training record IDs'):
        apply_rehearsal_weight(records, examples, record_prefix='missing:', weight=2, seed=2026)


def test_multiple_rehearsal_weights_repeat_each_slice():
    records = [
        replace(record(1), record_id='curated:targeted-finishing:linux:row-1'),
        replace(record(2), record_id='curated:intent-fidelity:linux:row-2'),
        record(3),
    ]
    _, examples = select_examples(records, FakeQwenTokenizer(), count=3, max_length=384, seed=2026)

    weighted_records, weighted_examples = apply_rehearsal_weights(
        records,
        examples,
        specs=[('curated:targeted-finishing:', 3), ('curated:intent-fidelity:', 2)],
        seed=2026,
    )

    assert len(weighted_records) == len(weighted_examples) == 6
    assert sum('targeted-finishing' in item.record_id for item in weighted_records) == 3
    assert sum('intent-fidelity' in item.record_id for item in weighted_records) == 2


def test_parse_rehearsal_specs_rejects_legacy_mix_and_duplicates():
    assert parse_rehearsal_specs(['curated:a:=3', 'curated:b:=2'], legacy_prefix=None, legacy_weight=1) == [
        ('curated:a:', 3),
        ('curated:b:', 2),
    ]
    with pytest.raises(ValueError, match='cannot be combined'):
        parse_rehearsal_specs(['curated:a:=3'], legacy_prefix='legacy:', legacy_weight=2)
    with pytest.raises(ValueError, match='duplicate rehearsal prefix'):
        parse_rehearsal_specs(['curated:a:=3', 'curated:a:=2'], legacy_prefix=None, legacy_weight=1)
