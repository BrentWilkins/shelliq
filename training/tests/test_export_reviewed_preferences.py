import pytest

from scripts.export_reviewed_preferences import export_records
from shelliq_training.teacher_verification import RustValidation


def document(argument: str) -> dict[str, object]:
    return {
        'v': 2,
        'd': 'zsh',
        's': [{'t': 'p', 'c': [{'n': {'s': 'tool'}, 'a': [{'s': argument}]}]}],
    }


def candidate(record_id: str) -> dict[str, object]:
    return {
        'record_id': record_id,
        'corpus': 'distributable',
        'source': 'shelliq-curated',
        'license': 'CC-BY-4.0',
        'provenance': 'fixture',
        'platform': 'linux',
        'instruction': 'Do the thing.',
        'context': 'tool: -a is correct and -b is incorrect.',
        'chosen': document('-a'),
        'rejected': document('-b'),
    }


def valid(count: int) -> list[RustValidation]:
    return [RustValidation(True, 'tool', None) for _ in range(count)]


def test_export_records_keeps_only_reviewed_includes() -> None:
    queue = [candidate('one'), candidate('two')]
    decisions = [
        {
            'record_id': 'one',
            'decision': 'include',
            'failure_modes': ['wrong-flag'],
            'evidence': 'The rejected flag contradicts the context.',
        },
        {'record_id': 'two', 'decision': 'exclude', 'reason': 'task-equivalent'},
    ]

    records, summary = export_records(queue, decisions, valid(4))

    assert [record['pair_id'] for record in records] == ['preference:v1:one']
    assert summary['included'] == 1
    assert summary['excluded'] == 1


def test_export_records_requires_complete_review_ledger() -> None:
    with pytest.raises(ValueError, match='cover queue exactly'):
        export_records([candidate('one')], [], valid(2))


def test_export_records_requires_valid_semantic_documents() -> None:
    with pytest.raises(ValueError, match='Rust semantic validation'):
        export_records(
            [candidate('one')],
            [{'record_id': 'one', 'decision': 'exclude', 'reason': 'task-equivalent'}],
            [RustValidation(False, None, 'bad'), RustValidation(True, 'tool -b', None)],
        )
