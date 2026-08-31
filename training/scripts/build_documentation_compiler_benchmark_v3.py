#!/usr/bin/env python3
"""Build the unchanged v2 input-complete requests under the v3 experiment ID."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_documentation_compiler_benchmark as v1

v1.EXPERIMENT = 'documentation-compiler-input-complete-v3'
v1.INCLUDE_SLOT_NAMES = False


if __name__ == '__main__':
    v1.main()
