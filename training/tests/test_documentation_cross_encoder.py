from __future__ import annotations

from shelliq_training.data import Platform
from shelliq_training.documentation_cross_encoder import (
    EXPERIMENT,
    RankCase,
    bind_selected,
    candidate_pool,
    context_commands,
    paraphrase,
    parse_authoritative_prompt,
    prompt_platform,
    runtime_groups,
    split_commands,
    threshold_for_complete_abstention,
)
from shelliq_training.documentation_templates import DocumentationTemplate


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


def test_runtime_parses_bounded_shelliq_prompt() -> None:
    source = (
        '# prompt-contract: context-authoritative-v1\n'
        '# platform: darwin\n'
        '<context>\nSelection pass\nCommands:\n- date\n- printf\n</context>\n'
        '<instruction>\nShow the current date\n</instruction>'
    )
    context, instruction = parse_authoritative_prompt(source)
    assert prompt_platform(source) == 'darwin'
    assert context_commands(context) == ('date', 'printf')
    assert instruction == 'Show the current date'


def test_runtime_parses_final_selected_command() -> None:
    context = 'Final pass: use exactly command `date`. Option tokens are case-sensitive.'
    assert context_commands(context) == ('date',)


def test_runtime_keeps_darwin_and_linux_recipes_separate() -> None:
    def template(record_id: str, platform: Platform) -> DocumentationTemplate:
        return DocumentationTemplate(
            record_id=record_id,
            command='date',
            platform=platform,
            instruction='Show date',
            context='date: Show date.',
            document={'v': 2, 'd': 'zsh', 's': [{'t': 'p', 'c': [{'n': {'s': 'date'}, 'a': []}]}]},
        )

    groups = runtime_groups([template('linux:date', Platform.LINUX), template('darwin:date', Platform.DARWIN)])
    assert groups == {('darwin', 'date'): ('darwin:date',), ('linux', 'date'): ('linux:date',)}


def test_unresolved_binding_returns_none_not_false() -> None:
    template = DocumentationTemplate(
        record_id='fixture:1',
        command='fixture',
        platform=Platform.LINUX,
        instruction='Process a file',
        context='fixture: Process a file.',
        document={
            'v': 2,
            'd': 'zsh',
            's': [{'t': 'p', 'c': [{'n': {'s': 'fixture'}, 'a': [{'s': 'path/to/file'}]}]}],
        },
    )

    class ActionsMustNotRun:
        def encode(self, documents):  # pragma: no cover - failure sentinel
            raise AssertionError(documents)

    result = bind_selected(
        RankCase('fixture', 'fixture:1', 'Process it'),
        'fixture:1',
        {'fixture:1': template},
        ActionsMustNotRun(),
    )
    assert result is None


def test_runtime_binding_can_defer_round_trip_to_shelliq_boundary() -> None:
    template = DocumentationTemplate(
        record_id='fixture:1',
        command='fixture',
        platform=Platform.DARWIN,
        instruction='Show status',
        context='fixture: Show status.',
        document={'v': 2, 'd': 'zsh', 's': [{'t': 'p', 'c': [{'n': {'s': 'fixture'}, 'a': []}]}]},
    )
    assert (
        bind_selected(
            RankCase('fixture', 'fixture:1', 'Show status'),
            'fixture:1',
            {'fixture:1': template},
            None,
        )
        == template.document
    )
