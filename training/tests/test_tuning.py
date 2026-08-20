import math

import pytest

from shelliq_training.tuning import EarlyStoppingTracker, LossObservation, best_observation


def observation(step: int, heldout_loss: float) -> LossObservation:
    return LossObservation(
        step=step,
        examples_seen=step * 2,
        train_loss=heldout_loss / 2,
        heldout_loss=heldout_loss,
    )


def test_tracker_resets_patience_on_material_improvement():
    tracker = EarlyStoppingTracker(patience=2, min_delta=0.01)

    assert not tracker.observe(observation(0, 1.0))
    assert not tracker.observe(observation(10, 0.995))
    assert tracker.stale_checks == 1
    assert not tracker.observe(observation(20, 0.98))
    assert tracker.stale_checks == 0
    assert not tracker.observe(observation(30, 0.975))
    assert tracker.observe(observation(40, 0.974))
    assert tracker.best == observation(20, 0.98)


def test_best_observation_uses_earlier_step_to_break_tie():
    observations = [observation(20, 0.4), observation(10, 0.4), observation(30, 0.5)]

    assert best_observation(observations).step == 10


@pytest.mark.parametrize(
    'kwargs',
    [
        {'step': -1, 'examples_seen': 0, 'train_loss': 1.0, 'heldout_loss': 1.0},
        {'step': 0, 'examples_seen': -1, 'train_loss': 1.0, 'heldout_loss': 1.0},
        {'step': 0, 'examples_seen': 0, 'train_loss': math.nan, 'heldout_loss': 1.0},
        {'step': 0, 'examples_seen': 0, 'train_loss': 1.0, 'heldout_loss': math.inf},
    ],
)
def test_observation_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        LossObservation(**kwargs)


def test_tracker_rejects_non_increasing_steps():
    tracker = EarlyStoppingTracker(patience=2)
    tracker.observe(observation(10, 1.0))

    with pytest.raises(ValueError, match='strictly increasing'):
        tracker.observe(observation(10, 0.9))
