import numpy as np

from app.domain.rewards.vectorized import (
    _sample_zero_truncated_poisson,
    register_state_index,
    store_event_count,
)


def test_vectorized_zero_truncated_poisson_is_strictly_positive():
    lam = np.array([0.001, 0.05, 0.5, 2.0, 10.0])
    uniform = np.array([0.0, 0.25, 0.5, 0.75, 0.999999])
    draws = _sample_zero_truncated_poisson(lam, uniform)
    assert np.all(draws >= 1)
    assert np.all(draws == np.floor(draws))


def test_vectorized_zero_truncated_poisson_matches_expected_mean():
    rng = np.random.default_rng(12345)
    n = 200_000
    for rate in (0.05, 0.5, 2.0, 5.0):
        lam = np.full(n, rate)
        draws = _sample_zero_truncated_poisson(lam, rng.random(n))
        expected = rate / (1.0 - np.exp(-rate))
        assert np.isclose(draws.mean(), expected, rtol=0.01)


def test_store_event_count_handles_all_states():
    states = [
        "healthy",
        "bleeding",
        "hemarthrosis",
        "intracranial_hemorrhage",
        "non_ich_major_bleeding",
        "death",
    ]
    for index, state in enumerate(states):
        register_state_index(state, index)

    state_idx = np.arange(len(states), dtype=np.int32)
    shared_kwargs = {
        "per_iter": {
            "lam_bleed": np.full(len(states), 0.5),
            "lam_joint": np.full(len(states), 1.0),
        }
    }
    result = store_event_count(
        step=0,
        state_idx=state_idx,
        store_arrays={},
        shared_kwargs=shared_kwargs,
        rng=np.random.default_rng(7),
    )

    assert result[states.index("healthy")] == 0
    assert result[states.index("death")] == 0
    assert result[states.index("bleeding")] >= 1
    assert result[states.index("hemarthrosis")] >= 1
    assert result[states.index("intracranial_hemorrhage")] == 1
    assert result[states.index("non_ich_major_bleeding")] == 1
