import jax.numpy as jnp
import pytest

from shelliq_training.data import IGNORE_INDEX
from shelliq_training.preference_training import completion_log_probabilities, dpo_loss


def test_completion_log_probabilities_ignore_prompt_and_padding() -> None:
    logits = jnp.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 3.0],
                [3.0, 0.0, 0.0],
            ]
        ]
    )
    labels = jnp.asarray([[IGNORE_INDEX, IGNORE_INDEX, 2, 0]])
    attention = jnp.asarray([[1, 1, 1, 0]])

    result = completion_log_probabilities(logits, labels, attention)

    expected = jnp.log(jnp.exp(3.0) / (jnp.exp(3.0) + 2.0))
    assert float(result[0]) == pytest.approx(float(expected))


def test_dpo_is_log_two_at_reference_and_rewards_preference_margin() -> None:
    reference_chosen = jnp.asarray([-3.0])
    reference_rejected = jnp.asarray([-4.0])

    baseline = dpo_loss(
        reference_chosen,
        reference_rejected,
        reference_chosen,
        reference_rejected,
        beta=0.1,
    )
    improved = dpo_loss(
        jnp.asarray([-2.0]),
        jnp.asarray([-5.0]),
        reference_chosen,
        reference_rejected,
        beta=0.1,
    )

    assert float(baseline) == pytest.approx(0.693147, rel=1e-5)
    assert float(improved) < float(baseline)


def test_dpo_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match='share shape'):
        dpo_loss(jnp.ones(2), jnp.ones(1), jnp.ones(2), jnp.ones(2), beta=0.1)
