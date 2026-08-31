#!/usr/bin/env python3
"""Evaluate filtered documentation, literal completeness, and word provenance."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_documentation_compiler as v1

from shelliq_training.documentation_templates import compile_documented_command_scored

v1.EXPERIMENT = 'semantic-action-documentation-compiler-v3'
v1.MINIMUM_ACCEPTED = 2
v1.MINIMUM_PRECISION = 0.25
v1.compile_documented_command = compile_documented_command_scored


if __name__ == '__main__':
    v1.main()
