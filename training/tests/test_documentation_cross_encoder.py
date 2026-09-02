from __future__ import annotations

from shelliq_training.documentation_cross_encoder import (
    EXPERIMENT,
    RankCase,
    candidate_pool,
    paraphrase,
    split_commands,
    threshold_for_complete_abstention,
)


def test_command_split_is_deterministic_and_disjoint() -> None:
    splits = split_commands(f'command-{index}' for index in range(800))
    assert len(splits['test']) == 128
    assert len(splits['development']) == 64
    assert len(splits['train']) == 512
    assert not (set(splits['test']) & set(splits['development']))
    assert not (set(splits['test']) & set(splits['train']))
    assert not (set(splits['development']) & set(splits['train']))
    expected = sorted(
        (f'command-{index}' for index in range(800)),
        key=lambda command: (__import__('hashlib').sha256(f'{EXPERIMENT}\0{command}'.encode()).hexdigest(), command),
    )
    assert list(splits['test']) == expected[:128]


def test_candidate_pool_removes_requested_command_when_insufficient() -> None:
    commands = tuple(f'command-{index}' for index in range(10))
    grouped = {command: (f'{command}:1', f'{command}:2') for command in commands}
    case = RankCase(commands[0], f'{commands[0]}:2', 'query')
    sufficient = candidate_pool(case, commands, grouped, sufficient=True)
    insufficient = candidate_pool(case, commands, grouped, sufficient=False)
    assert case.source_record_id in sufficient
    assert all(not record_id.startswith(f'{case.command}:') for record_id in insufficient)
    assert len(insufficient) == 7


def test_paraphrase_is_command_agnostic_and_preserves_literals() -> None:
    assert paraphrase('Download https://example.com/file') == 'Fetch https://example.com/file'
    assert paraphrase('Inspect /tmp/file') == 'Inspect /tmp/file'


def test_abstention_threshold_is_strictly_above_every_negative() -> None:
    threshold = threshold_for_complete_abstention([-1.0, 0.5, 0.25])
    assert threshold > 0.5
