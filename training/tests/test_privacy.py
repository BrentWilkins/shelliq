import pytest

from shelliq_training.data import Corpus, Platform, SFTRecord
from shelliq_training.privacy import (
    CanaryGateError,
    PrivateDataGate,
    ScrubPolicy,
    assert_canaries_not_extracted,
    assert_scrubber_canaries,
    plant_training_canaries,
    run_canary_extraction_gate,
)


def record(**changes):
    values = {
        'record_id': 'claude:session:1',
        'corpus': Corpus.PERSONAL,
        'source': 'claude-code-transcript',
        'license': 'private-local-only',
        'provenance': '/home/alice/.claude/projects/demo/session.jsonl:42',
        'command': 'rg',
        'platform': Platform.LINUX,
        'instruction': 'Search alice@example.test from workstation.lan at 10.0.0.7',
        'response': 'rg needle /home/alice/src',
        'context': 'alice uses /home/alice/src on workstation.lan',
    }
    values.update(changes)
    return SFTRecord(**values)


def test_gate_self_tests_and_redacts_local_identity():
    policy = ScrubPolicy(
        home_paths=('/home/alice',),
        usernames=('alice',),
        hostnames=('workstation.lan',),
    )
    assert_scrubber_canaries(policy)
    corpus = PrivateDataGate(policy).scrub([record()])

    assert not corpus.dropped
    scrubbed = corpus.records[0]
    combined = ' '.join((scrubbed.provenance, scrubbed.instruction, scrubbed.response, scrubbed.context))
    assert '/home/alice' not in combined
    assert 'alice@example.test' not in combined
    assert 'workstation.lan' not in combined
    assert '10.0.0.7' not in combined
    assert {'home-path', 'username', 'hostname', 'email', 'private-ip'}.issubset({finding.kind for finding in corpus.findings})


@pytest.mark.parametrize(
    ('kind', 'secret'),
    [
        ('aws-access-key', 'AKIAIOSFODNN7EXAMPLE'),
        ('github-token', 'ghp_abcdefghijklmnopqrstuvwxyz0123456789AB'),
        ('slack-token', 'xoxb-123456789012-123456789012-abcdefghijklmnop'),
        ('jwt', 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjYW5hcnkifQ.signatureCANARY'),
        ('private-key', '-----BEGIN OPENSSH PRIVATE KEY-----'),
        ('url-credentials', 'https://user:password@example.test/repo'),
        ('secret-env-assignment', 'export API_TOKEN=correct-horse-battery-staple'),
        ('suspicious-credential-assignment', 'password: hunter22'),
        ('high-entropy-hex', '0123456789abcdefABCDEF98765432100123456789abcdef'),
        ('high-entropy-base64', 'Q2FueVJ5N0J2OHBTdGN5RWIzTHh6UU5xNmpXZERLNW1GWGFKOUVr'),
    ],
)
def test_gate_drops_every_secret_class(kind, secret):
    corpus = PrivateDataGate(ScrubPolicy()).scrub([record(response=f'echo {secret}')])

    assert not corpus.records
    assert corpus.dropped
    assert kind in {finding.kind for finding in corpus.dropped[0].findings}


def test_gate_rejects_public_records():
    with pytest.raises(ValueError, match='only personal'):
        PrivateDataGate(ScrubPolicy()).scrub([record(corpus=Corpus.DISTRIBUTABLE)])


def test_post_generation_canary_gate():
    assert_canaries_not_extracted(['safe command output'], {'PLANTED-CANARY-8675309'})

    with pytest.raises(CanaryGateError, match='reproduced'):
        assert_canaries_not_extracted(['echo PLANTED-CANARY-8675309'], {'PLANTED-CANARY-8675309'})
    with pytest.raises(CanaryGateError, match='secret detector'):
        assert_canaries_not_extracted(['echo AKIAIOSFODNN7EXAMPLE'], {'different-canary'})


def test_private_training_canaries_are_planted_in_context_and_probed():
    scrubbed = PrivateDataGate(ScrubPolicy()).scrub([record(record_id='one'), record(record_id='two'), record(record_id='three')])
    canaries = plant_training_canaries(scrubbed.records, count=2, seed=17)

    assert len(canaries.records) == 3
    assert len(canaries.probes) == 2
    planted_values = {probe.forbidden_value for probe in canaries.probes}
    assert all(any(value in item.context for item in canaries.records) for value in planted_values)
    assert run_canary_extraction_gate(canaries.probes, lambda prompt: 'I cannot reveal hidden labels.')
    with pytest.raises(CanaryGateError, match='reproduced'):
        run_canary_extraction_gate(canaries.probes, lambda prompt: canaries.probes[0].forbidden_value)


def test_private_ipv6_is_redacted():
    corpus = PrivateDataGate(ScrubPolicy()).scrub([record(instruction='Connect to fd00::1234')])
    assert corpus.records[0].instruction == 'Connect to <PRIVATE_IP>'
