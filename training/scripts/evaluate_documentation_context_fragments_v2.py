#!/usr/bin/env python3
"""Run preregistered v2 corrections for authoritative-context fragments."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_documentation_context_fragments as v1

v1.EXPERIMENT = 'semantic-action-context-fragments-v2'


if __name__ == '__main__':
    v1.main()
