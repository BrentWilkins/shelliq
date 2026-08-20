"""TTY-aware Rich progress displays with quiet log-file fallback."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def interactive_progress() -> bool:
    """Return whether stderr can safely host an updating progress display."""
    return sys.stderr.isatty()


def training_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn('loss {task.fields[loss]}'),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),
        disable=not interactive_progress(),
    )


def evaluation_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn('{task.fields[record_id]}'),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),
        disable=not interactive_progress(),
    )
