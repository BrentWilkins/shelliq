from scripts.smoke_finetune import select_examples
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
