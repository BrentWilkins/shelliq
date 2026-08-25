from pathlib import Path

import pytest

from scripts.adjudicate_curated_development import apply_decisions, parse_report_specs, summarize_worksheet


def worksheet(*, human_correct=True, failure_modes=None):
    return {
        'schema_version': 1,
        'suite': 'curated-development-v1',
        'reports': [],
        'rows': [
            {
                'checkpoint': checkpoint,
                'record_id': 'case-1',
                'strict_grounded': False,
                'human_correct': human_correct,
                'failure_modes': ['semantic_alternative'] if failure_modes is None else failure_modes,
            }
            for checkpoint in ('0.5b', '1.5b')
        ],
    }


def test_report_specs_require_unique_labels() -> None:
    assert parse_report_specs(['small=a.json', 'large=b.json']) == [
        ('small', Path('a.json')),
        ('large', Path('b.json')),
    ]
    with pytest.raises(ValueError, match='duplicate'):
        parse_report_specs(['small=a.json', 'small=b.json'])


def test_summary_separates_human_correctness_from_strict_grounding() -> None:
    summary = summarize_worksheet(worksheet())
    assert summary['checkpoints']['0.5b']['human_correct'] == 1
    assert summary['checkpoints']['0.5b']['strict_grounded'] == 0
    assert summary['checkpoints']['1.5b']['failure_counts'] == {'semantic_alternative': 1}


def test_summary_requires_every_incorrect_output_to_be_classified() -> None:
    with pytest.raises(ValueError, match='requires a failure mode'):
        summarize_worksheet(worksheet(human_correct=False, failure_modes=[]))


def test_decisions_auto_accept_strict_rows_and_require_every_nonexact_row() -> None:
    document = worksheet()
    document['rows'][0]['strict_grounded'] = True
    document['rows'][0]['human_correct'] = None
    document['rows'][0]['failure_modes'] = []
    decisions = {
        'schema_version': 1,
        'suite': 'curated-development-v1',
        'reviewer': 'test',
        'method': 'manual',
        'decisions': [
            {
                'checkpoint': '1.5b',
                'record_id': 'case-1',
                'human_correct': False,
                'failure_modes': ['wrong_command'],
                'notes': 'wrong executable',
            }
        ],
    }

    completed = apply_decisions(document, decisions)
    assert completed['rows'][0]['human_correct'] is True
    assert completed['rows'][0]['notes'] == 'accepted by strict grounded match'
    assert completed['rows'][1]['human_correct'] is False
    summary = summarize_worksheet(completed)
    assert summary['checkpoints']['0.5b']['explicitly_reviewed'] == 0
    assert summary['checkpoints']['1.5b']['explicitly_reviewed'] == 1

    decisions['decisions'] = []
    with pytest.raises(ValueError, match='missing adjudication decision'):
        apply_decisions(document, decisions)


def test_grouped_decisions_expand_failure_mode_and_correctness() -> None:
    document = worksheet()
    decisions = {
        'schema_version': 1,
        'suite': 'curated-development-v1',
        'decisions': {
            'semantic_alternative': ['0.5b|case-1'],
            'wrong_command': ['1.5b|case-1'],
        },
    }

    completed = apply_decisions(document, decisions)

    assert completed['rows'][0]['human_correct'] is True
    assert completed['rows'][0]['failure_modes'] == ['semantic_alternative']
    assert completed['rows'][1]['human_correct'] is False
    assert completed['rows'][1]['failure_modes'] == ['wrong_command']
