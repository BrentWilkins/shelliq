import json
from pathlib import Path

import pytest

from shelliq_training.data import Corpus, Platform, SFTRecord
from shelliq_training.documentation_templates import (
    DocumentationTemplate,
    DocumentationTemplateIndex,
    RetrievedTemplate,
    bind_template,
    compose_contextual_candidates,
    compose_template_candidates,
    documentation_templates,
    documented_context_options,
    is_placeholder,
    template_matches_document,
)
from shelliq_training.semantic_actions import SemanticActionClient


def document(command: str, *arguments: str) -> dict[str, object]:
    return {
        'v': 2,
        'd': 'zsh',
        's': [{'t': 'p', 'c': [{'n': {'s': command}, 'a': [{'s': item} for item in arguments]}]}],
    }


def record(*, record_id: str, source: str, instruction: str, target: dict[str, object]) -> SFTRecord:
    return SFTRecord(
        record_id=record_id,
        corpus=Corpus.DISTRIBUTABLE,
        source=source,
        license='CC-BY-4.0',
        provenance='test',
        command='base64',
        platform=Platform.LINUX,
        instruction=instruction,
        response=json.dumps(target),
        context='base64 documentation',
    )


def test_index_excludes_curated_targets_and_scopes_command_platform() -> None:
    docs = documentation_templates(
        [
            record(
                record_id='tldr:base64:encode',
                source='tldr-pages',
                instruction='Encode a file',
                target=document('base64', 'path/to/file'),
            ),
            record(
                record_id='curated:base64:encode',
                source='shelliq-curated',
                instruction='Encode secret fixture',
                target=document('base64', 'fixture.bin'),
            ),
        ]
    )
    index = DocumentationTemplateIndex(docs)

    ranked = index.rank(
        command='base64',
        platform=Platform.LINUX,
        instruction='Encode report.bin',
        context='',
    )

    assert [item.template.record_id for item in ranked] == ['tldr:base64:encode']
    assert index.rank(command='base64', platform=Platform.DARWIN, instruction='Encode file', context='') == []


def test_generic_binding_substitutes_explicit_compatible_literals() -> None:
    template = DocumentationTemplate(
        record_id='tldr:rsync:port',
        command='rsync',
        platform=Platform.LINUX,
        instruction='Transfer over SSH using a specific port',
        context='',
        document=document('rsync', '-e', 'ssh -p port', 'path/to/source', 'host:path/to/destination'),
    )

    bound = bind_template(
        RetrievedTemplate(template, 0.8),
        'Transfer ./site/ to deploy@example.com:/srv/site/ using port 2222.',
    )

    arguments = bound.document['s'][0]['c'][0]['a']
    assert [item['s'] for item in arguments] == [
        '-e',
        'ssh -p port',
        './site/',
        'deploy@example.com:/srv/site/',
    ]
    assert ('path/to/source', './site/') in bound.bindings


def test_template_skeleton_preserves_static_semantics_and_wildcards_slots() -> None:
    template = document('base64', '-w', '0', 'path/to/file')

    assert template_matches_document(template, document('base64', '-w', '0', 'logo.png'))
    assert not template_matches_document(template, document('base64', '-d', 'logo.png'))
    assert is_placeholder('path/to/input_file')
    assert not is_placeholder('status')


def test_bound_unseen_command_template_round_trips_through_rust() -> None:
    executable = Path('../target/debug/semantic-actions')
    if not executable.exists():
        pytest.skip('semantic-actions binary is not built')
    template = DocumentationTemplate(
        record_id='tldr:base64:wrap',
        command='base64',
        platform=Platform.LINUX,
        instruction='Wrap encoded output at a specific width',
        context='',
        document=document('base64', '-w', 'number', 'path/to/file'),
    )
    bound = bind_template(RetrievedTemplate(template, 1.0), 'Encode report.bin without wrapping using width 0.')

    client = SemanticActionClient(executable)
    decoded = client.decode(client.encode([bound.document]))[0]

    assert decoded.valid
    assert decoded.rendered == 'base64 -w 0 report.bin'


def test_composer_combines_typed_fragments_without_command_rules() -> None:
    last_failed = DocumentationTemplate(
        record_id='tldr:pytest:last-failed',
        command='pytest',
        platform=Platform.LINUX,
        instruction='Run tests that failed last time',
        context='',
        document=document('pytest', '--last-failed'),
    )
    exit_first = DocumentationTemplate(
        record_id='tldr:pytest:exit-first',
        command='pytest',
        platform=Platform.LINUX,
        instruction='Stop after the first failure',
        context='',
        document=document('pytest', '--exitfirst'),
    )

    candidates = compose_template_candidates(
        [RetrievedTemplate(last_failed, 0.9), RetrievedTemplate(exit_first, 0.8)],
        'Re-run failures and stop after the first failure.',
    )

    assert any(
        item.document == document('pytest', '--last-failed', '--exitfirst')
        and item.source_record_ids == ('tldr:pytest:last-failed', 'tldr:pytest:exit-first')
        for item in candidates
    )


def test_contextual_composer_overlays_documented_options_on_typed_structure() -> None:
    target = document('cpio', '-idmv')
    target['s'][0]['c'][0]['r'] = [{'o': '<', 't': {'s': 'initramfs.cpio'}}]
    base_document = document('cpio', '-i')
    base_document['s'][0]['c'][0]['r'] = [{'o': '<', 't': {'s': 'path/to/archive'}}]
    base = DocumentationTemplate(
        record_id='tldr:cpio:extract',
        command='cpio',
        platform=Platform.LINUX,
        instruction='Extract an archive from standard input',
        context='',
        document=base_document,
    )
    context = 'cpio: -i extracts, -d creates directories, -m preserves times, and -v lists members.'

    candidates = compose_contextual_candidates(
        [RetrievedTemplate(base, 0.9)],
        'Unpack an initramfs image.',
        context,
    )

    assert documented_context_options(context) == ('-i', '-d', '-m', '-v')
    assert any(template_matches_document(item.document, target) for item in candidates)


def test_context_options_follow_documented_subcommand_prefix() -> None:
    base = DocumentationTemplate(
        record_id='tldr:docker:logs',
        command='docker',
        platform=Platform.LINUX,
        instruction='Show container logs',
        context='',
        document=document('docker', 'logs', 'container_name'),
    )
    target = document('docker', 'logs', '--tail', 'container_name')

    candidates = compose_contextual_candidates(
        [RetrievedTemplate(base, 1.0)],
        'Show recent logs.',
        'docker logs: --tail limits the number of lines.',
    )

    assert any(template_matches_document(item.document, target) for item in candidates)


def test_version_suffix_fallback_reuses_typed_recipe_and_keeps_requested_command() -> None:
    python = DocumentationTemplate(
        record_id='tldr:python:http',
        command='python',
        platform=Platform.LINUX,
        instruction='Start an HTTP server',
        context='',
        document=document('python', '-m', 'http.server'),
    )
    index = DocumentationTemplateIndex([python])

    ranked = index.rank(
        command='python3',
        platform=Platform.LINUX,
        instruction='Start an HTTP server',
        context='',
    )

    assert ranked[0].template.command == 'python3'
    assert ranked[0].template.document == document('python3', '-m', 'http.server')
