#!/usr/bin/env python3
"""Build the v2 command-integrity-filtered typed documentation index."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_documentation_template_index as v1

v1.EXPERIMENT = 'documentation-template-index-v2'


if __name__ == '__main__':
    v1.main()
