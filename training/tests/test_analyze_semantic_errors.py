import argparse
import json

from scripts import analyze_semantic_errors


def test_analysis_records_gate_result_without_gate_exit_mode(monkeypatch, tmp_path):
    output = tmp_path / 'analysis.json'
    metrics = {
        'total_examples': 35,
        'json_parse_rate': 0.9,
        'document_envelope_rate': 0.9,
        'first_command_accuracy': 1.0,
        'command_flag_sequence_exact_match': 1.0,
        'grounded_document_exact_match': 1.0,
    }
    document = {'candidate': {'metrics': metrics, 'failure_counts': {}}}
    monkeypatch.setattr(
        analyze_semantic_errors,
        'parse_args',
        lambda: argparse.Namespace(
            baseline=tmp_path / 'baseline.json',
            candidate=tmp_path / 'candidate.json',
            grounding_audit=tmp_path / 'audit.json',
            output=output,
            gate=False,
        ),
    )
    monkeypatch.setattr(analyze_semantic_errors, 'load_grounding_audit', lambda path: {})
    monkeypatch.setattr(
        analyze_semantic_errors,
        'analysis_document',
        lambda baseline, candidate, audit: document,
    )

    analyze_semantic_errors.main()

    report = json.loads(output.read_text())
    assert report['promotion_gate']['evaluated'] is True
    assert report['promotion_gate']['passed'] is False
    assert report['promotion_gate']['failures'] == [
        'json_parse_rate=0.9000 below 1.0000',
        'document_envelope_rate=0.9000 below 1.0000',
    ]
