import argparse
from pathlib import Path

import pytest

from scripts.run_rank_sweep import _parse_ranks, build_rank_plan


def test_rank_plan_controls_capacity_and_excludes_final_holdouts():
    stages = build_rank_plan(32, Path('/tmp/rank-sweep-test'))
    commands = [' '.join(stage.command) for stage in stages]

    assert len(stages) == 6
    assert all('--rank 32 --alpha 64' in command for command in commands if 'evaluate_' in command or 'smoke_' in command)
    assert all('pipeline-compatibility' not in command for command in commands)
    assert all('shadow' not in command for command in commands)


def test_rank_parser_rejects_duplicates_and_nonpositive_values():
    assert _parse_ranks('16,32,64') == (16, 32, 64)
    with pytest.raises(argparse.ArgumentTypeError, match='unique positive integers'):
        _parse_ranks('16,16')
    with pytest.raises(argparse.ArgumentTypeError, match='unique positive integers'):
        _parse_ranks('0,16')
