from __future__ import annotations

from shelliq_training.semantic_action_compiler_rules import compile_semantic_rule


def arguments(result) -> list[str]:
    assert result is not None
    return [item['s'] for item in result.document['s'][0]['c'][0]['a']]


def test_ss_rules_preserve_documented_option_order_and_exclude_other_protocols() -> None:
    first = compile_semantic_rule(
        command='ss',
        platform='linux',
        instruction='List every listening TCP socket with owning process.',
        context=('ss: -t is TCP, -l is listening only, -n skips name resolution, and -p shows owning process. -u adds UDP.'),
    )
    second = compile_semantic_rule(
        command='ss',
        platform='linux',
        instruction='List listening TCP sockets numerically and show owning processes.',
        context='ss: -l selects listening sockets, -n numeric output, -t TCP, and -p owning processes.',
    )
    udp = compile_semantic_rule(
        command='ss',
        platform='linux',
        instruction='List all UDP sockets with numeric addresses.',
        context='ss: -u selects UDP, -a all states, and -n prevents service-name resolution.',
    )

    assert arguments(first) == ['-tlnp']
    assert arguments(second) == ['-lntp']
    assert arguments(udp) == ['-uan']


def test_ss_summary_requires_documented_summary_form() -> None:
    result = compile_semantic_rule(
        command='ss',
        platform='linux',
        instruction='Get a summary count of sockets by state.',
        context='ss -s prints totals per protocol and TCP state.',
    )

    assert arguments(result) == ['-s']


def test_pmap_rule_extracts_one_pid_and_documented_mode() -> None:
    extended = compile_semantic_rule(
        command='pmap',
        platform='linux',
        instruction='Show an extended memory-map summary for process 4242.',
        context='procps pmap: -x adds per-mapping details and totals.',
    )
    kernel = compile_semantic_rule(
        command='pmap',
        platform='linux',
        instruction='Show kernel-provided memory-map details for PID 771.',
        context='procps pmap: -X includes kernel-detail columns while -x is the shorter extended view.',
    )

    assert arguments(extended) == ['-x', '4242']
    assert arguments(kernel) == ['-X', '771']


def test_rules_decline_unsupported_or_ambiguous_requests() -> None:
    assert (
        compile_semantic_rule(
            command='xargs',
            platform='darwin',
            instruction='Handle an empty input list safely on macOS.',
            context='BSD xargs rejects GNU -r.',
        )
        is None
    )
    assert (
        compile_semantic_rule(
            command='pmap',
            platform='linux',
            instruction='Compare extended maps for process 12 and 13.',
            context='procps pmap: -x adds extended details.',
        )
        is None
    )
