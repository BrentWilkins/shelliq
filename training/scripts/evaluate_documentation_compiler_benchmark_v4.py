#!/usr/bin/env python3
"""Evaluate input-complete documentation compiler benchmark v4."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_documentation_compiler_benchmark as v1

v1.EXPERIMENT = 'documentation-compiler-input-complete-v4'


if __name__ == '__main__':
    v1.main()
