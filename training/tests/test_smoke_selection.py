from dataclasses import replace

import pytest

from scripts.smoke_finetune import apply_rehearsal_weight, select_examples
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


def test_rehearsal_weight_rejects_missing_prefix():
    records = [record(1)]
    _, examples = select_examples(records, FakeQwenTokenizer(), count=1, max_length=384, seed=2026)
    with pytest.raises(ValueError, match='no selected training record IDs'):
        apply_rehearsal_weight(records, examples, record_prefix='missing:', weight=2, seed=2026)
