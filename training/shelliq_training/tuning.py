"""Training-loss observations and deterministic early-stopping decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LossObservation:
    step: int
    examples_seen: int
    train_loss: float
    heldout_loss: float

    def __post_init__(self) -> None:
        if self.step < 0 or self.examples_seen < 0:
            raise ValueError('step and examples_seen must be non-negative')
        if not math.isfinite(self.train_loss) or not math.isfinite(self.heldout_loss):
            raise ValueError('loss observations must be finite')

    def to_dict(self) -> dict[str, int | float]:
        return {
            'step': self.step,
            'examples_seen': self.examples_seen,
            'train_loss': self.train_loss,
            'heldout_loss': self.heldout_loss,
        }


class EarlyStoppingTracker:
    """Track best held-out loss and stop after a fixed number of stale checks."""

    def __init__(self, *, patience: int, min_delta: float = 0.0) -> None:
        if patience <= 0:
            raise ValueError('patience must be positive')
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('min_delta must be finite and non-negative')
        self.patience = patience
        self.min_delta = min_delta
        self.best: LossObservation | None = None
        self.stale_checks = 0
        self._last_step = -1

    def observe(self, observation: LossObservation) -> bool:
        """Record one strictly increasing check and return whether training should stop."""
        if observation.step <= self._last_step:
            raise ValueError('observation steps must be strictly increasing')
        self._last_step = observation.step
        if self.best is None or observation.heldout_loss < self.best.heldout_loss - self.min_delta:
            self.best = observation
            self.stale_checks = 0
        else:
            self.stale_checks += 1
        return self.stale_checks >= self.patience


def best_observation(observations: list[LossObservation]) -> LossObservation:
    if not observations:
        raise ValueError('at least one loss observation is required')
    return min(observations, key=lambda item: (item.heldout_loss, item.step))
