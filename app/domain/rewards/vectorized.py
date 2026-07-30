"""Vectorized versions of the hemophilia reward functions.

Each function operates on (n_iters,) shaped arrays instead of single scalars.
The signature is:
    f(step, state_idx, store_arrays, shared_kwargs, rng) -> (n_iters,) array

`store_arrays` is a dict of previously-computed store results (filled in
left-to-right call order, matching the scalar `add_store_function` order).

`shared_kwargs` provides per-batch and per-iter constants:
    - regime (Regime)
    - per_iter: dict with per-iter arrays
        - weight_factor
        - lam_bleed
        - lam_joint
        - util_healthy, util_bleeding, ..., util_death
        - mild_arthropathy_utility, moderate_..., severe...
        - prophylaxis_background_factor_consumption_per_kg
        - factor_consumption_per_spontaneous_bleeding_per_kg
        - factor_consumption_per_joint_bleeding_per_kg
        - factor_consumption_per_life_threatening_bleeding_per_kg
    - thresholds: dict
        - mild, moderate, max
    - conversion_factor: float
    - weekly_discount: float
    - baseline_age_weeks: float

Returns (n_iters,) ndarray of the per-iter reward value at the current step.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

# State name -> index (filled by setup_vectorized_rewards)
_STATE_IDX: dict[str, int] = {}


def register_state_index(name: str, idx: int) -> None:
    _STATE_IDX[name] = idx


def _sample_zero_truncated_poisson(
    lam: np.ndarray,
    uniform: np.ndarray,
) -> np.ndarray:
    """Draw ``Poisson(lam)`` values conditional on being at least one."""
    lam = np.asarray(lam, dtype=np.float64)
    uniform = np.asarray(uniform, dtype=np.float64)
    if lam.shape != uniform.shape:
        raise ValueError("lam and uniform must have identical shapes")

    # Preserve the legacy fallback: invalid/non-positive rates yield one
    # when an iteration is already known to occupy an event state.
    draws = np.ones(lam.shape, dtype=np.float64)
    valid = np.isfinite(lam) & (lam > 0.0)
    if not valid.any():
        return draws

    valid_lam = lam[valid]
    p_zero = np.exp(-valid_lam)
    quantiles = p_zero + uniform[valid] * (1.0 - p_zero)
    quantiles = np.minimum(quantiles, np.nextafter(1.0, 0.0))
    sampled = poisson.ppf(quantiles, valid_lam)
    draws[valid] = np.maximum(sampled, 1.0)
    return draws


# ----------------------- Store functions -----------------------


def store_weight(step, state_idx, store_arrays, shared_kwargs, rng):
    """Per-iter body weight (kg) at the current step.

    Weight is state-independent at a given step but varies across iters
    through the per-iter ``weight_factor`` (PSA / scenario variation),
    matching the scalar ``weight`` store function iter-for-iter.
    """
    per_iter = shared_kwargs["per_iter"]
    weight_factor = per_iter["weight_factor"]
    base_weight = shared_kwargs["base_weight_by_step"][step]
    return np.round(base_weight * weight_factor, 2)


def store_event_count(step, state_idx, store_arrays, shared_kwargs, rng):
    """Zero-truncated Poisson sample of bleeding/hemarthrosis event count.

    For iters in absorbing or non-bleeding states, returns 0. Iters in the
    life-threatening-bleeding state count exactly one episode (matching
    the scalar ``event_count``); bleeding/hemarthrosis iters draw from a
    zero-truncated Poisson with the per-iter weekly rate.
    """
    per_iter = shared_kwargs["per_iter"]
    lam_bleed = per_iter["lam_bleed"]
    lam_joint = per_iter["lam_joint"]

    bleeding_idx = _STATE_IDX["bleeding"]
    hemarthrosis_idx = _STATE_IDX["hemarthrosis"]
    ich_idx = _STATE_IDX["intracranial_hemorrhage"]
    non_ich_idx = _STATE_IDX["non_ich_major_bleeding"]

    n_iters = state_idx.shape[0]
    out = np.zeros(n_iters, dtype=np.float64)

    in_bleeding = state_idx == bleeding_idx
    in_hemarthrosis = state_idx == hemarthrosis_idx
    in_acute_major = (state_idx == ich_idx) | (state_idx == non_ich_idx)

    # Exactly one LTB episode per LTB-state week (scalar parity)
    out[in_acute_major] = 1.0

    in_any = in_bleeding | in_hemarthrosis

    if not in_any.any():
        return out

    # Build lam array: per-iter lam value depending on state
    lam = np.zeros(n_iters, dtype=np.float64)
    lam[in_bleeding] = lam_bleed[in_bleeding]
    lam[in_hemarthrosis] = lam_joint[in_hemarthrosis]

    # A single vectorized inverse-CDF operation replaces the former Python
    # loop that rebuilt a truncated PMF for every active iteration.
    u = rng.random(n_iters)
    out[in_any] = _sample_zero_truncated_poisson(lam[in_any], u[in_any])

    return out


def store_pettersson_score(step, state_idx, store_arrays, shared_kwargs, rng):
    """Pettersson score = cumulative hemarthrosis event count / conversion factor.

    Stateful: uses store_arrays['_pettersson_count'] (n_iters,) accumulator.
    Increments when state is hemarthrosis, capped at 79.
    """
    conversion_factor = shared_kwargs["conversion_factor"]
    count = store_arrays.get("_pettersson_count")
    if count is None:
        count = np.zeros(state_idx.shape, dtype=np.float64)

    hemarthrosis_idx = _STATE_IDX["hemarthrosis"]
    event_count = store_arrays.get("event_count")
    if event_count is not None:
        in_hem = state_idx == hemarthrosis_idx
        count = count + np.where(in_hem, event_count, 0.0)

    store_arrays["_pettersson_count"] = count
    score = count / conversion_factor
    return np.minimum(score, 79.0)


# ----------------------- Reward functions -----------------------


def reward_consumption(step, state_idx, store_arrays, shared_kwargs, rng):
    """Per-iter factor consumption at the current step."""
    per_iter = shared_kwargs["per_iter"]
    regime = shared_kwargs["regime"]
    weight = store_arrays.get("weight")
    event_count = store_arrays.get("event_count")
    if weight is None:
        weight = np.zeros(state_idx.shape, dtype=np.float64)
    if event_count is None:
        event_count = np.zeros(state_idx.shape, dtype=np.float64)

    death_idx = _STATE_IDX["death"]
    bleeding_idx = _STATE_IDX["bleeding"]
    hemarthrosis_idx = _STATE_IDX["hemarthrosis"]
    ich_idx = _STATE_IDX["intracranial_hemorrhage"]
    non_ich_idx = _STATE_IDX["non_ich_major_bleeding"]

    dose = np.zeros(state_idx.shape, dtype=np.float64)

    in_death = state_idx == death_idx
    if not (~in_death).any():
        return dose

    if regime.value == "prophylaxis":
        dose = dose + np.where(
            ~in_death,
            weight * per_iter["prophylaxis_background_factor_consumption_per_kg"],
            0.0,
        )

    in_bleeding = state_idx == bleeding_idx
    in_hemarthrosis = state_idx == hemarthrosis_idx
    in_ich = state_idx == ich_idx
    in_non_ich = state_idx == non_ich_idx

    dose = dose + np.where(
        in_bleeding & ~in_death,
        weight
        * per_iter["factor_consumption_per_spontaneous_bleeding_per_kg"]
        * event_count,
        0.0,
    )
    dose = dose + np.where(
        in_hemarthrosis & ~in_death,
        weight * per_iter["factor_consumption_per_joint_bleeding_per_kg"] * event_count,
        0.0,
    )
    dose = dose + np.where(
        in_ich & ~in_death,
        weight * per_iter["factor_consumption_per_intracranial_hemorrhage_per_kg"],
        0.0,
    )
    dose = dose + np.where(
        in_non_ich & ~in_death,
        weight * per_iter["factor_consumption_per_non_ich_major_bleeding_per_kg"],
        0.0,
    )
    return dose


def reward_utility(step, state_idx, store_arrays, shared_kwargs, rng):
    """Per-iter weekly QALY at the current step (with discounting)."""
    per_iter = shared_kwargs["per_iter"]
    thresholds = shared_kwargs["thresholds"]
    weekly_discount = shared_kwargs["weekly_discount"]
    pettersson_score = store_arrays.get("pettersson_score")
    if pettersson_score is None:
        pettersson_score = np.zeros(state_idx.shape, dtype=np.float64)

    # Arth severity utility
    mild_thr = thresholds["mild"]
    moderate_thr = thresholds["moderate"]
    max_thr = thresholds["max"]

    util_mild = per_iter["mild_arthropathy_utility"]
    util_moderate = per_iter["moderate_arthropathy_utility"]
    util_severe = per_iter["severe_arthropathy_utility"]
    util_healthy = per_iter["healthy_utility"]

    arth = np.where(
        pettersson_score < mild_thr,
        util_healthy,
        np.where(
            pettersson_score < moderate_thr,
            util_mild,
            np.where(pettersson_score < max_thr, util_moderate, util_severe),
        ),
    )

    healthy_idx = _STATE_IDX["healthy"]
    bleeding_idx = _STATE_IDX["bleeding"]
    hemarthrosis_idx = _STATE_IDX["hemarthrosis"]
    ich_idx = _STATE_IDX["intracranial_hemorrhage"]
    non_ich_idx = _STATE_IDX["non_ich_major_bleeding"]
    death_idx = _STATE_IDX["death"]

    util_bleed = per_iter["spontaneous_bleeding_utility"]
    util_hem = per_iter["joint_bleeding_utility"]
    util_ich = per_iter["intracranial_hemorrhage_utility"]
    util_non_ich = per_iter["non_ich_major_bleeding_utility"]
    util_death = per_iter["death_utility"]

    acute = np.where(
        state_idx == healthy_idx,
        arth,
        np.where(
            state_idx == bleeding_idx,
            util_bleed,
            np.where(
                state_idx == hemarthrosis_idx,
                util_hem,
                np.where(
                    state_idx == ich_idx,
                    util_ich,
                    np.where(
                        state_idx == non_ich_idx,
                        util_non_ich,
                        np.where(state_idx == death_idx, util_death, arth),
                    ),
                ),
            ),
        ),
    )
    u = np.minimum(arth, acute)

    weekly = u / 52.0
    if weekly_discount == 0:
        return weekly
    return weekly / ((1.0 + weekly_discount) ** step)


# Convenience exports
VECTORIZED_STORE_FUNCS = {
    "weight": store_weight,
    "event_count": store_event_count,
    "pettersson_score": store_pettersson_score,
}

VECTORIZED_REWARD_FUNCS = {
    "consumption": reward_consumption,
    "utility": reward_utility,
}
