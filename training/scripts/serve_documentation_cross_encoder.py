#!/usr/bin/env python3
"""Compatibility entry point for the ShellIQ documentation ranker."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.model_runtime import main  # noqa: E402, I001


if __name__ == '__main__':
    raise SystemExit(main(['serve', *sys.argv[1:]]))
