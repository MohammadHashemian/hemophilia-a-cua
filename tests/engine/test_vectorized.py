"""Tests for the vectorized batch markov chain engine."""
import numpy as np
import pytest

from engine.vectorized import BatchMarkovChain, BatchResult


def test_batch_walk_basic():
    """8-state chain, 3 iters, 5 steps: verify sequences + absorbing work."""
    states = ["a", "b", "c", "d", "e", "f", "g", "dead"]
    n = len(states)
    n_iters = 3
    steps = 5
    # All iters start with absorbing state 'dead' (row=delta_ij)
    matrices = np.zeros((n_iters, n, n))
    for i in range(n_iters):
        matrices[i] = np.eye(n)
    matrices[:, -1, :] = 0
    matrices[:, -1, -1] = 1.0  # 'dead' absorbing

    mask = np.zeros((n_iters, n), dtype=bool)
    mask[:, -1] = True  # 'dead' absorbing

    batch = BatchMarkovChain(
        matrices=matrices,
        absorbing_mask=mask,
        death_idx=7,
        lt_bleeding_idx=-1,
    )
    rng = np.random.default_rng(0)
    result = batch.walk_batch(steps=steps, entrance_idx=0, rng=rng)

    assert isinstance(result, BatchResult)
    assert result.sequences.shape == (n_iters, steps + 1)
    # All iters start at 0 and stay there (no transitions defined)
    assert np.all(result.sequences == 0)
    # Never absorbed (state 0 is not absorbing)
    assert np.all(result.absorbed_at == steps + 1)


def test_batch_walk_with_absorption():
    """Iters in absorbing state stay there; absorbed_at recorded correctly."""
    n = 3
    n_iters = 2
    matrices = np.zeros((n_iters, n, n))
    # Iter 0: deterministic to dead in 1 step
    matrices[0, 0] = [0, 0, 1]
    matrices[0, 1] = [0, 0, 1]
    matrices[0, 2] = [0, 0, 1]
    # Iter 1: stays in s0 forever
    matrices[1, 0] = [1, 0, 0]
    matrices[1, 1] = [0, 0, 1]
    matrices[1, 2] = [0, 0, 1]

    mask = np.zeros((n_iters, n), dtype=bool)
    mask[:, 2] = True  # 'dead' absorbing

    batch = BatchMarkovChain(
        matrices=matrices, absorbing_mask=mask, death_idx=2, lt_bleeding_idx=-1
    )
    rng = np.random.default_rng(0)
    result = batch.walk_batch(steps=3, entrance_idx=0, rng=rng)

    # Iter 0: at step 0 state 0, step 1 state 2 (dead)
    assert result.sequences[0, 0] == 0
    assert result.sequences[0, 1] == 2
    assert result.sequences[0, 2] == 2
    assert result.sequences[0, 3] == 2
    assert result.absorbed_at[0] == 1  # absorbed at step 1

    # Iter 1: stays at 0
    assert np.all(result.sequences[1] == 0)
    assert result.absorbed_at[1] == 4  # never absorbed (steps+1)


def test_batch_walk_with_rewards():
    """Verify reward traces are populated when reward funcs are provided."""
    n = 3
    n_iters = 4
    matrices = np.zeros((n_iters, n, n))
    for i in range(n_iters):
        matrices[i] = np.eye(n)
    matrices[:, 2, :] = 0
    matrices[:, 2, 2] = 1.0

    mask = np.zeros((n_iters, n), dtype=bool)
    mask[:, 2] = True

    batch = BatchMarkovChain(
        matrices=matrices, absorbing_mask=mask, death_idx=2, lt_bleeding_idx=-1
    )

    def reward_fn(step, state_idx, store_arrays, shared_kwargs, rng):
        # Constant reward 1.0
        return np.ones(state_idx.shape[0], dtype=np.float64)

    def store_fn(step, state_idx, store_arrays, shared_kwargs, rng):
        # Step index as the store value
        return np.full(state_idx.shape, float(step), dtype=np.float64)

    rng = np.random.default_rng(0)
    result = batch.walk_batch(
        steps=3,
        entrance_idx=0,
        rng=rng,
        store_funcs={"my_store": store_fn},
        reward_funcs={"my_reward": reward_fn},
    )
    assert result.rewards["my_store"].shape == (n_iters, 4)  # steps+1
    assert result.rewards["my_reward"].shape == (n_iters, 4)
    # All rewards should be 1.0
    assert np.allclose(result.rewards["my_reward"], 1.0)
    # Store values should be [0, 1, 2, 3] for each iter
    expected = np.tile([0, 1, 2, 3], (n_iters, 1))
    assert np.allclose(result.rewards["my_store"], expected)


def test_batch_walk_zero_steps():
    """Zero-step run should return single-step trace at entrance."""
    n = 3
    n_iters = 5
    matrices = np.tile(np.eye(n), (n_iters, 1, 1))
    mask = np.zeros((n_iters, n), dtype=bool)
    batch = BatchMarkovChain(
        matrices=matrices, absorbing_mask=mask, death_idx=-1, lt_bleeding_idx=-1
    )
    rng = np.random.default_rng(0)
    result = batch.walk_batch(steps=0, entrance_idx=1, rng=rng)
    assert result.sequences.shape == (n_iters, 1)
    assert np.all(result.sequences[:, 0] == 1)


def test_batch_walk_mortality_applied_every_week():
    """Mortality adjustment applies every week with the weekly death
    probability of the patient's current age band (not only at year
    boundaries). With an identity matrix and a 0.5 weekly death
    probability, all iters must die within a few weeks."""
    n = 3
    n_iters = 2
    matrices = np.tile(np.eye(n), (n_iters, 1, 1))
    mask = np.zeros((n_iters, n), dtype=bool)
    mask[:, 2] = True  # 'dead' absorbing

    # Weekly death probabilities per age (already converted upstream).
    mortality = np.array([0.5, 0.5] + [0.0] * 118)
    batch = BatchMarkovChain(
        matrices=matrices,
        absorbing_mask=mask,
        death_idx=2,
        lt_bleeding_idx=-1,
        mortality_weekly_prob_per_age=mortality,
    )
    rng = np.random.default_rng(0)
    result = batch.walk_batch(steps=10, entrance_idx=0, rng=rng)
    # Every iter must have transitioned to the absorbing death state.
    assert np.all(result.sequences[:, -1] == 2)
    assert np.all(result.absorbed_at <= 10)


def test_batch_walk_mortality_uses_baseline_age_offset():
    """The mortality age band must include the baseline age at model
    entry: with baseline_age=2, step 0 already looks up age band 2."""
    n = 3
    n_iters = 2
    matrices = np.tile(np.eye(n), (n_iters, 1, 1))
    mask = np.zeros((n_iters, n), dtype=bool)
    mask[:, 2] = True

    # Only age band 2 carries a death probability; bands 0-1 are zero.
    mortality = np.array([0.0, 0.0, 1.0] + [0.0] * 117)
    batch = BatchMarkovChain(
        matrices=matrices,
        absorbing_mask=mask,
        death_idx=2,
        lt_bleeding_idx=-1,
        mortality_weekly_prob_per_age=mortality,
        baseline_age=2,
    )
    rng = np.random.default_rng(0)
    result = batch.walk_batch(steps=10, entrance_idx=0, rng=rng)
    # Age band 2 applies from step 0 -> everyone dies immediately.
    assert np.all(result.sequences[:, -1] == 2)
    assert np.all(result.absorbed_at <= 10)


def test_batch_walk_invalid_input_raises():
    with pytest.raises(ValueError, match="matrices must be"):
        BatchMarkovChain(
            matrices=np.eye(3),  # 2D instead of 3D
            absorbing_mask=np.zeros((1, 3), dtype=bool),
            death_idx=0,
            lt_bleeding_idx=-1,
        )


def test_batch_walk_progress_callback_fires():
    """Progress callback should fire every progress_every steps."""
    n = 3
    n_iters = 2
    matrices = np.tile(np.eye(n), (n_iters, 1, 1))
    matrices[:, 2, :] = 0
    matrices[:, 2, 2] = 1.0
    mask = np.zeros((n_iters, n), dtype=bool)

    batch = BatchMarkovChain(
        matrices=matrices, absorbing_mask=mask, death_idx=2, lt_bleeding_idx=-1
    )

    calls = []

    def cb(step, total_steps):
        calls.append(step)

    rng = np.random.default_rng(0)
    batch.walk_batch(
        steps=20,
        entrance_idx=0,
        rng=rng,
        progress_callback=cb,
        progress_every=5,
    )
    # Callback fires at step % 5 == 0 (before the step >= steps check),
    # so for steps=20 the fires are 0, 5, 10, 15 (step 20 breaks out first).
    assert calls == [0, 5, 10, 15]


def test_batch_walk_progress_callback_none_is_safe():
    """No callback should work fine."""
    n = 3
    n_iters = 2
    matrices = np.tile(np.eye(n), (n_iters, 1, 1))
    mask = np.zeros((n_iters, n), dtype=bool)
    batch = BatchMarkovChain(
        matrices=matrices, absorbing_mask=mask, death_idx=2, lt_bleeding_idx=-1
    )
    rng = np.random.default_rng(0)
    result = batch.walk_batch(steps=5, entrance_idx=0, rng=rng)
    assert result.sequences.shape == (n_iters, 6)


def test_aggregate_qaly_is_continuous_not_integer():
    """Regression: total_qaly must remain a float (continuous), not be
    truncated to int. Earlier `_aggregate_vectorized_output` wrapped the
    sum in `int(...)`, which discretized the QALY distribution to whole
    numbers (0, 1, 2, ...) and made the PSA density plots look like
    spikes at integer ticks.
    """
    # Simulate a (n_iters=3, n_steps+1=4) utility trace with non-integer
    # sum, mimicking what walk_batch produces after the discount + utility
    # pipeline.
    from app.domain.enums import HealthStates, Regime
    from app.domain.inputs import ModelInput
    from app.domain.worker import _aggregate_vectorized_output
    from engine.vectorized import BatchResult

    states = [s.value for s in HealthStates]
    n_iters = 3
    rewards = {
        "consumption": np.full((n_iters, 4), 1.0),
        "utility": np.array(
            [
                [0.0173, 0.0173, 0.0173, 0.0],   # sum = 0.0519
                [0.0125, 0.0125, 0.0125, 0.0125],  # sum = 0.05
                [0.015, 0.014, 0.013, 0.012],     # sum = 0.054
            ]
        ),
        "weight": np.full((n_iters, 4), 32.0),
        "event_count": np.zeros((n_iters, 4), dtype=np.int64),
        "pettersson_score": np.zeros((n_iters, 4)),
    }
    batch = BatchResult(
        sequences=np.zeros((n_iters, 4), dtype=np.int32),
        absorbed_at=np.array([4, 4, 4], dtype=np.int32),
        rewards=rewards,
    )
    inputs = [
        ModelInput(
            cycle=3, bleeding_rate=15.0, spontaneous_bleeding_rate=10.0,
            joint_bleeding_rate=5.0, life_threatening_bleeding_rate=1.0,
            ltb_case_fatality=0.35,
            baseline_age=2, weight_factor=1.0, benefits_discount_rate=0.03,
            healthy_utility=0.9, mild_arthropathy_utility=0.85,
            moderate_arthropathy_utility=0.7, severe_arthropathy_utility=0.5,
            spontaneous_bleeding_utility=0.6, joint_bleeding_utility=0.5,
            life_threatening_bleeding_utility=0.3, death_utility=0.0,
            per_unit_price=1000.0, costs_discount_rate=0.0,
            prophylaxis_background_factor_consumption_per_kg=0.5,
            factor_consumption_per_spontaneous_bleeding_per_kg=10.0,
            factor_consumption_per_joint_bleeding_per_kg=20.0,
            factor_consumption_per_life_threatening_bleeding_per_kg=50.0,
        )
        for _ in range(n_iters)
    ]
    outputs = [
        _aggregate_vectorized_output(inputs, batch, states, Regime.PROPHYLAXIS, i)
        for i in range(n_iters)
    ]
    # All three sums are non-integer; aggregator must preserve them.
    assert outputs[0].total_qaly == pytest.approx(0.0519, rel=1e-9)
    assert outputs[1].total_qaly == pytest.approx(0.05, rel=1e-9)
    assert outputs[2].total_qaly == pytest.approx(0.054, rel=1e-9)
    assert all(isinstance(o.total_qaly, float) for o in outputs)
    assert all(isinstance(o.total_factor, float) for o in outputs)
    assert all(isinstance(o.mean_weight, float) for o in outputs)
