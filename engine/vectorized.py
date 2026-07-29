"""Vectorized batch Markov chain engine.

Runs n_iters independent Markov simulations in a single numpy call chain by
stacking per-iteration state into (n_iters, ...) arrays. The per-step loop
runs once per step but vectorized across all iters, eliminating the per-iter
Python overhead that dominates the scalar path.

Designed for PSA-style workloads where thousands of structurally identical
simulations are run with different per-iter parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


def _precompute_age_mortality(mortality_file: Any) -> np.ndarray:
    """Precompute per-age *weekly* death probability (length ~120).

    The life table stores annual probabilities of death (q_x). Conversion
    to the weekly cycle uses the constant-hazard identity
    ``p_week = 1 - (1 - q_x) ** (1/52)`` (equivalently 1 - exp(-h/52) with
    h = -ln(1 - q_x)). The result is indexed by integer age so the per-step
    mortality adjustment is a single array lookup.
    """
    max_age = 120
    rates = np.full(max_age, float(mortality_file.crude_annual_rate), dtype=np.float64)
    age_specific = mortality_file.age_specific
    for key, rate in age_specific.items():
        rate = float(rate)
        if "-" in key:
            try:
                low, high = map(int, key.split("-"))
                rates[low : high + 1] = rate
            except ValueError:
                continue
        elif key == "90+":
            rates[90:max_age] = rate
        else:
            try:
                idx = int(key)
                if 0 <= idx < max_age:
                    rates[idx] = rate
            except ValueError:
                continue
    rates = np.clip(rates, 0.0, 0.999999)
    # annual probability -> weekly probability (actuarially consistent)
    return 1.0 - (1.0 - rates) ** (1.0 / 52.0)


@dataclass
class BatchResult:
    """Output of a vectorized batch walk.

    sequences: (n_iters, n_steps+1) int — state index at each step
    absorbed_at: (n_iters,) int — first absorbing step (steps+1 if never)
    rewards: dict[name, (n_iters, n_steps+1) array] — store + reward traces
    """

    sequences: np.ndarray
    absorbed_at: np.ndarray
    rewards: dict[str, np.ndarray]


class BatchMarkovChain:
    """Vectorized across-iterations Markov chain runner.

    All iters advance in lockstep. The per-step Python loop is O(steps)
    instead of O(n_iters * steps), with the inner work being numpy ops
    over (n_iters, n_states) shaped arrays.

    Parameters
    ----------
    matrices : (n_iters, n_states, n_states) array
        Per-iter transition probability matrices. Rows must sum to 1.
    absorbing_mask : (n_iters, n_states) bool array
        Per-iter absorbing-state mask.
    death_idx, lt_bleeding_idx : int
        Indices used to skip mortality adjustment.
    mortality_weekly_prob_per_age : (max_age,) array, optional
        Precomputed weekly death probabilities by age. If None, no
        mortality adjustment is applied.
    baseline_age : int
        Patient age at simulation start; the age band used for the
        mortality lookup only changes at calendar-year boundaries.
    """

    def __init__(
        self,
        matrices: np.ndarray,
        absorbing_mask: np.ndarray,
        death_idx: int,
        lt_bleeding_idx: int,
        mortality_weekly_prob_per_age: np.ndarray | None = None,
        baseline_age: int = 0,
    ) -> None:
        if matrices.ndim != 3:
            raise ValueError("matrices must be (n_iters, n_states, n_states)")
        if absorbing_mask.shape != matrices.shape[:2]:
            raise ValueError("absorbing_mask must be (n_iters, n_states)")
        n_iters, n_states, _ = matrices.shape
        self.n_iters = n_iters
        self.n_states = n_states
        self.matrices = matrices
        self.absorbing_mask = absorbing_mask
        self.death_idx = death_idx
        self.lt_bleeding_idx = lt_bleeding_idx
        self.mortality_weekly_prob_per_age = mortality_weekly_prob_per_age
        self.baseline_age = int(baseline_age)

    @classmethod
    def from_chain(
        cls,
        matrices: np.ndarray,
        states: list[str],
        absorbing_states: set[str] | None,
        mortality_file: Any | None = None,
        auto_detect_absorbing: bool = False,
        baseline_age: float = 0,
    ) -> BatchMarkovChain:
        """Build a BatchMarkovChain from per-iter matrices and state names.

        matrices : (n_iters, n_states, n_states) array
        """
        n_iters, n_states, _ = matrices.shape
        state_to_idx = {s: i for i, s in enumerate(states)}

        # Build absorbing mask per iter
        mask = np.zeros((n_iters, n_states), dtype=bool)
        if absorbing_states:
            for s in absorbing_states:
                if s in state_to_idx:
                    mask[:, state_to_idx[s]] = True
        if auto_detect_absorbing:
            # Self-loop = 1 absorbing
            for i in range(n_states):
                rows = matrices[:, i, i]
                if np.allclose(rows, 1.0, rtol=1e-8) and np.all(
                    np.abs(matrices[:, i, :].sum(axis=1) - 1.0) < 1e-8
                ):
                    # Check the rest of the row is 0
                    rest = matrices[:, i, :].copy()
                    rest[:, i] = 0
                    if np.allclose(rest, 0.0, atol=1e-8):
                        mask[:, i] = True

        death_idx = state_to_idx.get("death", -1)
        lt_bleeding_idx = state_to_idx.get("lt_bleeding", -1)

        mort = (
            _precompute_age_mortality(mortality_file)
            if mortality_file is not None
            else None
        )

        return cls(
            matrices=matrices,
            absorbing_mask=mask,
            death_idx=death_idx,
            lt_bleeding_idx=lt_bleeding_idx,
            mortality_weekly_prob_per_age=mort,
            baseline_age=baseline_age,
        )

    def _apply_mortality(
        self, probs: np.ndarray, step: int, current_state_idx: np.ndarray
    ) -> np.ndarray:
        """Apply age-based mortality adjustment to (n_iters, n_states) probs.

        Applied every week with the weekly death probability of the
        patient's current age band (the band itself only changes at
        calendar-year boundaries). Iters currently in the absorbing or
        life-threatening-bleeding states are left untouched, mirroring
        the scalar ``AgeBasedMortalityModifier``.
        """
        if self.mortality_weekly_prob_per_age is None:
            return probs

        age = max(0, self.baseline_age + step // 52)
        if age >= len(self.mortality_weekly_prob_per_age):
            weekly_death_prob = self.mortality_weekly_prob_per_age[-1]
        else:
            weekly_death_prob = self.mortality_weekly_prob_per_age[age]

        if weekly_death_prob <= 0.0:
            return probs

        base_death = probs[:, self.death_idx]
        # If base death is already 1, skip
        safe = base_death < 1.0
        combined_death = 1.0 - (1.0 - base_death) * (1.0 - weekly_death_prob)
        survival_scale = np.where(
            safe, (1.0 - combined_death) / np.maximum(1.0 - base_death, 1e-12), 1.0
        )

        new_probs = probs.copy()
        new_probs[:, self.death_idx] = combined_death
        # Scale non-death
        non_death = np.ones(self.n_states, dtype=bool)
        non_death[self.death_idx] = False
        new_probs[:, non_death] *= survival_scale[:, None]
        new_probs = np.clip(new_probs, 0.0, 1.0)
        row_sums = new_probs.sum(axis=1, keepdims=True)
        new_probs = np.divide(
            new_probs, row_sums, out=np.zeros_like(new_probs), where=row_sums > 0
        )

        # Iters in lt_bleeding keep their special (case-fatality) row,
        # matching the scalar modifier which skips that state.
        if self.lt_bleeding_idx >= 0:
            in_lt = current_state_idx == self.lt_bleeding_idx
            if in_lt.any():
                new_probs[in_lt] = probs[in_lt]
        return new_probs

    def walk_batch(
        self,
        steps: int,
        entrance_idx: int,
        rng: np.random.Generator,
        store_funcs: dict[str, Callable[..., np.ndarray]] | None = None,
        reward_funcs: dict[str, Callable[..., np.ndarray]] | None = None,
        shared_kwargs: dict[str, Any] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        progress_every: int = 52,
    ) -> BatchResult:
        """Run a vectorized batch walk.

        store_funcs / reward_funcs are callables with signature:
            f(step, state_idx, store_arrays, shared_kwargs, rng) -> (n_iters,) array

        store_arrays is a dict of previously-computed store results
        (e.g. {'weight': (n_iters,)}, {'event_count': (n_iters,)}).

        shared_kwargs is forwarded to each function for per-batch constants
        (utilities, thresholds, regime, per-iter constants, etc.).

        progress_callback(step, total_steps) is called every `progress_every`
        steps when provided. The bar in the caller is expected to translate
        step-progress into iter-progress (each step = n_iters iters
        processed in lockstep).
        """
        if steps <= 0:
            return BatchResult(
                sequences=np.full((self.n_iters, 1), entrance_idx, dtype=np.int32),
                absorbed_at=np.zeros(self.n_iters, dtype=np.int32),
                rewards={},
            )

        store_funcs = store_funcs or {}
        reward_funcs = reward_funcs or {}
        shared_kwargs = shared_kwargs or {}

        # Allocate outputs
        sequences = np.empty((self.n_iters, steps + 1), dtype=np.int32)
        absorbed_at = np.full(self.n_iters, steps + 1, dtype=np.int32)
        rewards: dict[str, np.ndarray] = {}
        for name in store_funcs:
            rewards[name] = np.empty((self.n_iters, steps + 1), dtype=np.float64)
        for name in reward_funcs:
            rewards[name] = np.empty((self.n_iters, steps + 1), dtype=np.float64)

        # Mutable state
        current_state_idx = np.full(self.n_iters, entrance_idx, dtype=np.int32)
        active = np.ones(self.n_iters, dtype=bool)
        store_arrays: dict[str, np.ndarray] = {}

        for step in range(steps + 1):
            sequences[:, step] = current_state_idx

            # Absorbing check
            is_absorbed_now = self.absorbing_mask[
                np.arange(self.n_iters), current_state_idx
            ]
            newly_absorbed = is_absorbed_now & active
            if newly_absorbed.any():
                # Record the step at which each iter first absorbed
                not_yet_recorded = absorbed_at == steps + 1
                record = newly_absorbed & not_yet_recorded
                if record.any():
                    absorbed_at[record] = step
            active = active & ~is_absorbed_now

            # Run store functions
            for name, func in store_funcs.items():
                out = func(
                    step=step,
                    state_idx=current_state_idx,
                    store_arrays=store_arrays,
                    shared_kwargs=shared_kwargs,
                    rng=rng,
                )
                store_arrays[name] = out
                rewards[name][:, step] = out

            # Run reward functions
            for name, func in reward_funcs.items():
                out = func(
                    step=step,
                    state_idx=current_state_idx,
                    store_arrays=store_arrays,
                    shared_kwargs=shared_kwargs,
                    rng=rng,
                )
                rewards[name][:, step] = out

            if step >= steps:
                break

            if not active.any():
                # All absorbed: pad with current state for remaining steps
                if step + 1 <= steps:
                    sequences[:, step + 1 :] = current_state_idx[:, None]
                for name in rewards:
                    if step + 1 <= steps:
                        rewards[name][:, step + 1 :] = rewards[name][:, step : step + 1]
                break

            # Sample next state for active iters
            # Get adjusted probs row for each active iter
            probs = self.matrices[np.arange(self.n_iters), current_state_idx].copy()

            # Apply mortality modifier (vectorized)
            probs = self._apply_mortality(probs, step, current_state_idx)

            # Vectorized categorical sampling via cumsum + uniform
            cumsum = np.cumsum(probs, axis=1)
            u = rng.random(self.n_iters)
            # argmax of (u[:, None] < cumsum) along axis 1
            next_state = (u[:, None] < cumsum).argmax(axis=1).astype(np.int32)

            # For inactive iters, keep current state
            current_state_idx = np.where(active, next_state, current_state_idx)

            # Progress callback (after each step completes).
            if progress_callback is not None and step % progress_every == 0:
                progress_callback(step, steps)

        return BatchResult(
            sequences=sequences, absorbed_at=absorbed_at, rewards=rewards
        )
