#!/usr/bin/env python3
"""Select seed checkpoints and test whether a matrix leader wins on two seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_capacity_matrix import _paired, audit_controls  # noqa: E402
from scripts.analyze_unique_data_scaling import Point, summarize_point  # noqa: E402

TRAINING_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = TRAINING_ROOT / 'artifacts'
SEEDS = (2026, 2027, 2028)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-analysis', type=Path, required=True)
    parser.add_argument('--matrix-root', type=Path, default=ARTIFACTS / 'capacity-matrix-v1')
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--confirmation-root', type=Path, default=ARTIFACTS / 'capacity-seed-confirmation-v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--loss-tolerance', type=float, default=0.005)
    parser.add_argument('--material-difference', type=int, default=5)
    parser.add_argument(
        '--grounding-audit',
        type=Path,
        default=TRAINING_ROOT / 'evaluation' / 'curated-development-grounding-v1.json',
    )
    return parser.parse_args()


def _root(label: str, seed: int, *, matrix_root: Path, reference_root: Path, confirmation_root: Path) -> Path:
    if seed != 2026:
        return confirmation_root / label / f'seed-{seed}'
    if label == 'qwen1p5b-rank8':
        return reference_root
    return matrix_root / label


def _score(selected: dict[str, object]) -> tuple[int, int, int]:
    return (
        int(selected['grounded']),
        int(selected['flags']),
        int(selected['first_command']),
    )


def _material_winner(
    first: str,
    first_selected: dict[str, object],
    second: str,
    second_selected: dict[str, object],
    *,
    material_difference: int,
) -> str | None:
    """Apply the preregistered five-case threshold in primary-metric order."""
    for metric in ('grounded', 'flags', 'first_command'):
        delta = int(second_selected[metric]) - int(first_selected[metric])
        if abs(delta) >= material_difference:
            return second if delta > 0 else first
    return None


def build_analysis(
    matrix_analysis: Path,
    *,
    matrix_root: Path,
    reference_root: Path,
    confirmation_root: Path,
    loss_tolerance: float,
    grounding_audit: Path,
    material_difference: int = 5,
) -> dict[str, object]:
    if material_difference < 1:
        raise ValueError('material difference must be positive')
    source = json.loads(matrix_analysis.read_text())
    leaders = source['seed_confirmation_leaders']
    if not isinstance(leaders, list) or not 1 <= len(leaders) <= 2:
        raise ValueError('matrix analysis must contain one or two leaders')
    configurations: dict[str, dict[str, object]] = {str(label): {} for label in leaders}
    roots: dict[tuple[str, int], Path] = {}
    for label in leaders:
        for seed in SEEDS:
            run_root = _root(
                str(label),
                seed,
                matrix_root=matrix_root,
                reference_root=reference_root,
                confirmation_root=confirmation_root,
            )
            roots[(str(label), seed)] = run_root
            configurations[str(label)][str(seed)] = summarize_point(Point(str(label), 0, run_root), loss_tolerance=loss_tolerance)

    seeds: dict[str, object] = {}
    control_audits: dict[str, object] = {}
    strict_winner_counts = {str(label): 0 for label in leaders}
    material_winner_counts = {str(label): 0 for label in leaders}
    for seed in SEEDS:
        control_audit = audit_controls({str(label): roots[(str(label), seed)] for label in leaders})
        shared = control_audit['shared']
        assert isinstance(shared, dict)
        if shared['seed'] != seed or shared['split_seed'] != 2026:
            raise ValueError(f'seed-{seed} reports do not preserve the registered training and split seeds')
        control_audits[str(seed)] = control_audit
        selected = {
            str(label): configurations[str(label)][str(seed)]['selected']  # type: ignore[index]
            for label in leaders
        }
        passed = {label: item is not None for label, item in selected.items()}
        result: dict[str, object] = {'all_leaders_pass_gates': all(passed.values()), 'passed': passed}
        if len(leaders) == 2 and all(passed.values()):
            first, second = map(str, leaders)
            first_selected = selected[first]
            second_selected = selected[second]
            assert isinstance(first_selected, dict) and isinstance(second_selected, dict)
            first_score = _score(first_selected)
            second_score = _score(second_selected)
            strict_winner = first if first_score > second_score else second if second_score > first_score else None
            material_winner = _material_winner(
                first,
                first_selected,
                second,
                second_selected,
                material_difference=material_difference,
            )
            if strict_winner is not None:
                strict_winner_counts[strict_winner] += 1
            if material_winner is not None:
                material_winner_counts[material_winner] += 1
            result.update(
                {
                    'strict_winner': strict_winner,
                    'material_winner': material_winner,
                    'paired_first_to_second': _paired(
                        roots[(first, seed)],
                        first_selected,
                        roots[(second, seed)],
                        second_selected,
                        grounding_audit,
                    ),
                }
            )
        seeds[str(seed)] = result

    confirmed = None
    if len(leaders) == 2:
        eligible = [label for label, wins in material_winner_counts.items() if wins >= 2]
        if len(eligible) == 1 and all(bool(seeds[str(seed)]['all_leaders_pass_gates']) for seed in SEEDS):  # type: ignore[index]
            confirmed = eligible[0]
    return {
        'schema_version': 1,
        'leaders': leaders,
        'selection_policy': {
            'loss_tolerance': loss_tolerance,
            'hard_contract': '104/104 JSON and envelopes per configuration and seed',
            'retention': 'tiered-absolute-v1 pass per configuration and seed',
            'winner_order': ['grounded', 'flags', 'first_command'],
            'material_difference': material_difference,
        },
        'configurations': configurations,
        'control_audits': control_audits,
        'seeds': seeds,
        'strict_winner_counts': strict_winner_counts,
        'material_winner_counts': material_winner_counts,
        'strict_confirmed_leader': confirmed,
        'confirmation_rule': 'material winner on at least two of three seeds; all leaders pass every per-seed gate',
        'adjudicated_correctness_pending': True,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')
    try:
        analysis = build_analysis(
            args.matrix_analysis,
            matrix_root=args.matrix_root,
            reference_root=args.reference_root,
            confirmation_root=args.confirmation_root,
            loss_tolerance=args.loss_tolerance,
            grounding_audit=args.grounding_audit,
            material_difference=args.material_difference,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'strict_confirmed_leader': analysis['strict_confirmed_leader']}, indent=2))


if __name__ == '__main__':
    main()
