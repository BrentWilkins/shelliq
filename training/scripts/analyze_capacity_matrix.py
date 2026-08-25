#!/usr/bin/env python3
"""Select capacity-matrix checkpoints and report controlled paired effects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_unique_data_scaling import Point, summarize_point  # noqa: E402
from shelliq_training.semantic_error_analysis import (  # noqa: E402
    analyze_examples,
    compare_results,
    load_report_examples,
)
from shelliq_training.semantic_evaluation import load_grounding_audit  # noqa: E402

TRAINING_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = TRAINING_ROOT / 'artifacts'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-root', type=Path, default=ARTIFACTS / 'capacity-matrix-v1')
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--loss-tolerance', type=float, default=0.005)
    parser.add_argument('--material-cases', type=int, default=5)
    parser.add_argument(
        '--grounding-audit',
        type=Path,
        default=TRAINING_ROOT / 'evaluation' / 'curated-development-grounding-v1.json',
    )
    return parser.parse_args()


def _training_fit(root: Path, selected: dict[str, object] | None) -> dict[str, float | int] | None:
    if selected is None:
        return None
    report = json.loads((root / 'training-report.json').read_text())
    step = int(selected['step'])
    observation = next(item for item in report['training']['loss_observations'] if item['step'] == step)
    return {
        'step': step,
        'train_loss': float(observation['train_loss']),
        'heldout_loss': float(observation['heldout_loss']),
    }


def audit_controls(roots: dict[str, Path]) -> dict[str, object]:
    """Prove that every matrix cell used the same data and optimization controls."""
    controls = {}
    for label, root in roots.items():
        report = json.loads((root / 'training-report.json').read_text())
        selection = report['selection']
        training = report['training']
        controls[label] = {
            'dataset_sha256': report['dataset']['sha256'],
            'train_record_ids_sha256': selection['train_record_ids_sha256'],
            'effective_train_examples': selection['effective_train_examples'],
            'effective_train_source_counts': selection['effective_train_source_counts'],
            'seed': selection['seed'],
            'split_seed': selection['split_seed'],
            'steps': training['steps'],
            'batch_size': training['batch_size'],
            'lr_schedule': training['lr_schedule'],
            'warmup_steps': training['warmup_steps'],
            'shuffle_each_epoch': training['shuffle_each_epoch'],
        }
    reference = next(iter(controls.values()))
    mismatches = [label for label, control in controls.items() if control != reference]
    if mismatches:
        raise ValueError(f'capacity matrix controls differ for: {", ".join(mismatches)}')
    return {'passed': True, 'shared': reference, 'configurations': sorted(controls)}


def _report(root: Path, selected: dict[str, object]) -> Path:
    return (root / 'evaluations' / f'step-{int(selected["step"]):08d}').with_suffix('.curated-development.json')


def _paired(
    old_root: Path, old: dict[str, object], new_root: Path, new: dict[str, object], audit_path: Path
) -> dict[str, object]:
    audit = load_grounding_audit(audit_path)
    old_examples = load_report_examples(_report(old_root, old))
    new_examples = load_report_examples(_report(new_root, new))
    _, old_results = analyze_examples(old_examples, audit)
    _, new_results = analyze_examples(new_examples, audit)
    comparison = compare_results(old_results, new_results)
    return {
        'from': str(_report(old_root, old)),
        'to': str(_report(new_root, new)),
        'grounded_delta': int(new['grounded']) - int(old['grounded']),
        'flags_delta': int(new['flags']) - int(old['flags']),
        'first_command_delta': int(new['first_command']) - int(old['first_command']),
        'improvements': comparison['improvements'],
        'regressions': comparison['regressions'],
        'improvement_count': len(comparison['improvements']),
        'regression_count': len(comparison['regressions']),
    }


def classify_effects(selected: dict[str, dict[str, object]], *, material_cases: int) -> dict[str, object]:
    required = ('qwen0p5b-rank8', 'qwen0p5b-rank32', 'qwen1p5b-rank8', 'qwen1p5b-rank32')
    if any(label not in selected for label in required):
        return {'status': 'incomplete', 'signals': []}
    rank_small = int(selected['qwen0p5b-rank32']['grounded']) - int(selected['qwen0p5b-rank8']['grounded'])
    rank_large = int(selected['qwen1p5b-rank32']['grounded']) - int(selected['qwen1p5b-rank8']['grounded'])
    base_rank8 = int(selected['qwen1p5b-rank8']['grounded']) - int(selected['qwen0p5b-rank8']['grounded'])
    base_rank32 = int(selected['qwen1p5b-rank32']['grounded']) - int(selected['qwen0p5b-rank32']['grounded'])
    signals = []
    if rank_small >= material_cases or rank_large >= material_cases:
        signals.append('lora_capacity')
    if base_rank8 >= material_cases and base_rank32 >= material_cases:
        signals.append('base_model_capacity')
    if not signals:
        signals.append('data_generalization_or_optimization')
    return {
        'status': 'preliminary_seed_2026',
        'material_case_threshold': material_cases,
        'grounded_deltas': {
            'rank_effect_0p5b': rank_small,
            'rank_effect_1p5b': rank_large,
            'base_effect_rank8': base_rank8,
            'base_effect_rank32': base_rank32,
        },
        'signals': signals,
    }


def build_analysis(
    matrix_root: Path,
    reference_root: Path,
    *,
    loss_tolerance: float,
    material_cases: int,
    grounding_audit: Path,
) -> dict[str, object]:
    roots = {
        'qwen0p5b-rank8': matrix_root / 'qwen0p5b-rank8',
        'qwen0p5b-rank32': matrix_root / 'qwen0p5b-rank32',
        'qwen1p5b-rank8': reference_root,
        'qwen1p5b-rank32': matrix_root / 'qwen1p5b-rank32',
    }
    control_audit = audit_controls(roots)
    summaries = {label: summarize_point(Point(label, 0, root), loss_tolerance=loss_tolerance) for label, root in roots.items()}
    for label, summary in summaries.items():
        summary['training_fit'] = _training_fit(roots[label], summary['selected'])
    selected = {label: summary['selected'] for label, summary in summaries.items() if summary['selected'] is not None}
    comparisons = {}
    pairs = {
        'rank_effect_0p5b': ('qwen0p5b-rank8', 'qwen0p5b-rank32'),
        'rank_effect_1p5b': ('qwen1p5b-rank8', 'qwen1p5b-rank32'),
        'base_effect_rank8': ('qwen0p5b-rank8', 'qwen1p5b-rank8'),
        'base_effect_rank32': ('qwen0p5b-rank32', 'qwen1p5b-rank32'),
    }
    for name, (old_label, new_label) in pairs.items():
        if old_label in selected and new_label in selected:
            comparisons[name] = _paired(
                roots[old_label], selected[old_label], roots[new_label], selected[new_label], grounding_audit
            )
    leaders = sorted(
        selected,
        key=lambda label: (
            int(selected[label]['grounded']),
            int(selected[label]['flags']),
            int(selected[label]['first_command']),
            -float(selected[label]['validation_loss']),
        ),
        reverse=True,
    )[:2]
    return {
        'schema_version': 1,
        'selection_policy': {
            'loss_tolerance': loss_tolerance,
            'hard_contract': '104/104 JSON and envelopes',
            'retention': 'tiered-absolute-v1 pass',
        },
        'control_audit': control_audit,
        'configurations': summaries,
        'paired_comparisons': comparisons,
        'preliminary_interpretation': classify_effects(selected, material_cases=material_cases),
        'seed_confirmation_leaders': leaders,
        'adjudicated_correctness_pending': True,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')
    analysis = build_analysis(
        args.matrix_root,
        args.reference_root,
        loss_tolerance=args.loss_tolerance,
        material_cases=args.material_cases,
        grounding_audit=args.grounding_audit,
    )
    args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + '\n')
    print(json.dumps(analysis['preliminary_interpretation'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
