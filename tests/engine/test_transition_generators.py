"""Comprehensive math validation tests for all transition generators.

These tests verify the mathematical correctness of the four transition
generators shipped in :mod:`engine.transitions`:

* :class:`DTMCTransitionGenerator` — pure discrete-time, direct probabilities
* :class:`HybridTransitionGenerator` — mixed direct probs + hazards
* :class:`CTMCTransitionGenerator` — continuous-time via matrix exponential
* :class:`IndependentHazardTransitionGenerator` — independent hazard conversion

Each test class targets a specific mathematical property (survival,
proportional allocation, generator row sum, expm equivalence, etc.) so a
future refactor that silently breaks a formula will be caught here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from engine.transitions import (
    CTMCTransitionGenerator,
    DTMCTransitionGenerator,
    HybridTransitionGenerator,
    IndependentHazardTransitionGenerator,
)

# ── helpers ──────────────────────────────────────────────────────────


def assert_row_stochastic(P: np.ndarray, atol: float = 1e-9) -> None:
    """Every row of P must sum to 1 (within tolerance)."""
    sums = P.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=atol)


def assert_nonnegative(P: np.ndarray, atol: float = 1e-12) -> None:
    """All entries of P must be >= -atol (allow tiny FP noise from expm)."""
    assert np.all(P >= -atol), f"Negative entries found: min={P.min()}"


def assert_unit_interval(P: np.ndarray, atol: float = 1e-9) -> None:
    """All entries must lie in [-atol, 1 + atol]."""
    assert np.all(P <= 1.0 + atol), f"Entries > 1: max={P.max()}"
    assert np.all(P >= -atol), f"Negative entries: min={P.min()}"


# ── DTMC math tests ──────────────────────────────────────────────────


class TestDTMCMath:
    """Mathematical correctness of the pure DTMC generator."""

    def test_identity_matrix_when_all_self_loops(self):
        """If every state transitions to itself with prob 1, P = I."""
        tg = DTMCTransitionGenerator(
            states=["A", "B", "C"],
            transition_pairs={
                ("A", "A"): 1.0,
                ("B", "B"): 1.0,
                ("C", "C"): 1.0,
            },
        )
        P = tg.build_matrix()
        np.testing.assert_allclose(P, np.eye(3), atol=1e-12)

    def test_exact_probability_passthrough(self):
        """Direct probabilities are stored verbatim — no transformation."""
        tg = DTMCTransitionGenerator(
            states=["A", "B"],
            transition_pairs={
                ("A", "A"): 0.3,
                ("A", "B"): 0.7,
            },
        )
        P = tg.build_matrix()
        assert P[0, 0] == pytest.approx(0.3, abs=1e-12)
        assert P[0, 1] == pytest.approx(0.7, abs=1e-12)

    def test_row_stochastic_invariant(self):
        """A correctly specified DTMC is always row-stochastic."""
        tg = DTMCTransitionGenerator(
            states=["A", "B", "C", "D"],
            transition_pairs={
                ("A", "A"): 0.1,
                ("A", "B"): 0.2,
                ("A", "C"): 0.3,
                ("A", "D"): 0.4,
                ("B", "A"): 0.25,
                ("B", "B"): 0.25,
                ("B", "C"): 0.25,
                ("B", "D"): 0.25,
                ("C", "D"): 1.0,
                ("D", "D"): 1.0,
            },
        )
        P = tg.build_matrix()
        assert_row_stochastic(P)
        assert_unit_interval(P)

    def test_absorbing_state_self_loop_is_one(self):
        """An absorbing state (no outgoing entries or self-loop=1) stays put."""
        tg = DTMCTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "A"): 0.5, ("A", "B"): 0.5},
        )
        P = tg.build_matrix()
        # B was not specified -> default absorbing
        assert P[1, 1] == pytest.approx(1.0, abs=1e-12)
        assert P[1, 0] == 0.0

    def test_special_transition_overrides_regular(self):
        """``special_transitions`` wins over ``transition_pairs`` for that state."""
        tg = DTMCTransitionGenerator(
            states=["A", "B"],
            transition_pairs={
                ("A", "A"): 0.0,
                ("A", "B"): 1.0,
            },
            special_transitions={
                "A": {"A": 0.7, "B": 0.3},
            },
        )
        P = tg.build_matrix()
        np.testing.assert_allclose(P[0], [0.7, 0.3], atol=1e-12)

    def test_special_absorbing_state(self):
        """An absorbing ``special_transitions`` entry yields P[i,i] = 1."""
        tg = DTMCTransitionGenerator(
            states=["A", "death"],
            transition_pairs={("A", "A"): 0.5, ("A", "death"): 0.5},
            special_transitions={"death": {"A": 0.0, "death": 1.0}},
        )
        P = tg.build_matrix()
        assert P[1, 1] == pytest.approx(1.0, abs=1e-12)
        assert P[1, 0] == 0.0

    def test_markov_property_preserves_distribution(self):
        """A row-stochastic DTMC is a valid Markov kernel: π' = π P.

        This is a sanity check: pick a distribution, multiply by P, sum
        to 1, and all entries must stay non-negative.
        """
        tg = DTMCTransitionGenerator(
            states=["A", "B", "C"],
            transition_pairs={
                ("A", "A"): 0.2,
                ("A", "B"): 0.5,
                ("A", "C"): 0.3,
                ("B", "A"): 0.4,
                ("B", "B"): 0.4,
                ("B", "C"): 0.2,
                ("C", "C"): 1.0,
            },
        )
        P = tg.build_matrix()
        pi = np.array([0.5, 0.3, 0.2])
        pi_next = pi @ P
        assert pi_next.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(pi_next >= -1e-12)
        np.testing.assert_allclose(pi_next.sum(), 1.0, atol=1e-12)

    def test_validation_row_sum_not_one(self):
        """Validation must reject a state whose probabilities don't sum to 1."""
        with pytest.raises(ValueError, match="sum to"):
            DTMCTransitionGenerator(
                states=["A", "B"],
                transition_pairs={("A", "A"): 0.3, ("A", "B"): 0.3},
            )

    def test_validation_probability_out_of_range(self):
        with pytest.raises(ValueError, match="must be in"):
            DTMCTransitionGenerator(
                states=["A", "B"],
                transition_pairs={("A", "A"): 1.5, ("A", "B"): -0.5},
            )

    def test_validation_invalid_state(self):
        with pytest.raises(ValueError, match="Invalid transition pair"):
            DTMCTransitionGenerator(
                states=["A", "B"],
                transition_pairs={("A", "C"): 1.0},
            )

    def test_validation_empty_states(self):
        with pytest.raises(ValueError, match="States list cannot be empty"):
            DTMCTransitionGenerator(states=[], transition_pairs={})

    def test_validation_special_row_sum(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            DTMCTransitionGenerator(
                states=["A", "B"],
                transition_pairs={},
                special_transitions={"A": {"A": 0.5, "B": 0.4}},
            )

    def test_validation_special_prob_out_of_range(self):
        with pytest.raises(ValueError, match="must be in"):
            DTMCTransitionGenerator(
                states=["A", "B"],
                transition_pairs={},
                special_transitions={"A": {"A": 1.5, "B": -0.5}},
            )

    def test_build_matrix_returns_ndarray(self):
        tg = DTMCTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "A"): 1.0},
        )
        M = tg.build_matrix()
        assert isinstance(M, np.ndarray)
        assert M.shape == (2, 2)

    def test_time_step_ignored_for_dtmc(self):
        """For a DTMC, ``time_step`` is accepted but does NOT alter probabilities.

        This is the defining property of a discrete-time chain: there is
        no rate-to-probability conversion.
        """
        states = ["A", "B"]
        transition_pairs = {("A", "A"): 0.4, ("A", "B"): 0.6}
        P_weekly = DTMCTransitionGenerator(
            states=states,
            transition_pairs=transition_pairs,
            time_step="weekly",
        ).build_matrix()
        P_annual = DTMCTransitionGenerator(
            states=states,
            transition_pairs=transition_pairs,
            time_step="annual",
        ).build_matrix()
        np.testing.assert_allclose(P_weekly, P_annual, atol=1e-12)


# ── Hybrid math tests ────────────────────────────────────────────────


class TestHybridMath:
    """Mathematical correctness of the hybrid (direct + hazard) generator."""

    def test_single_hazard_weekly_matches_1_minus_exp(self):
        """For a single weekly hazard: P[0,1] = 1 - exp(-λ)."""
        lam = 0.5
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (lam, "weekly")},
            time_step="weekly",
        )
        P = tg.build_matrix()
        expected = 1 - math.exp(-lam)
        assert P[0, 1] == pytest.approx(expected, abs=1e-9)
        assert P[0, 0] == pytest.approx(math.exp(-lam), abs=1e-9)
        assert_row_stochastic(P)

    def test_annual_hazard_in_weekly_step(self):
        """Annual hazard is divided by 52 for a weekly time step."""
        annual_lam = 52.0  # 52/yr => λ_week = 1.0
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (annual_lam, "annual")},
            time_step="weekly",
        )
        P = tg.build_matrix()
        assert P[0, 1] == pytest.approx(1 - math.exp(-1.0), abs=1e-9)

    def test_weekly_hazard_in_annual_step(self):
        """Weekly hazard is multiplied by 52 for an annual time step."""
        weekly_lam = 1.0  # 1/wk => λ_year = 52.0
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (weekly_lam, "weekly")},
            time_step="annual",
        )
        P = tg.build_matrix()
        assert P[0, 1] == pytest.approx(1 - math.exp(-52.0), abs=1e-9)

    def test_competing_risks_proportional_allocation(self):
        """For competing weekly hazards, the transition mass is split in
        proportion to the per-target rate: P[0,j] = (λ_j/Σλ) * (1 - exp(-Σλ)).
        """
        lam1, lam2 = 0.3, 0.2
        total = lam1 + lam2
        tg = HybridTransitionGenerator(
            states=["A", "B", "C"],
            transition_pairs={
                ("A", "B"): (lam1, "weekly"),
                ("A", "C"): (lam2, "weekly"),
            },
            time_step="weekly",
        )
        P = tg.build_matrix()
        expected_b = (lam1 / total) * (1 - math.exp(-total))
        expected_c = (lam2 / total) * (1 - math.exp(-total))
        assert P[0, 1] == pytest.approx(expected_b, abs=1e-9)
        assert P[0, 2] == pytest.approx(expected_c, abs=1e-9)
        assert P[0, 0] == pytest.approx(math.exp(-total), abs=1e-9)

    def test_direct_probability_unchanged(self):
        """A direct probability (period=None) is stored verbatim."""
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (0.42, None)},
            time_step="weekly",
        )
        P = tg.build_matrix()
        assert P[0, 1] == pytest.approx(0.42, abs=1e-12)
        assert P[0, 0] == pytest.approx(0.58, abs=1e-12)

    def test_mixed_direct_and_hazard(self):
        """When direct probs and hazards coexist, direct are applied first
        and the remaining mass is allocated via competing risks on the
        hazard side. Survival = remaining * exp(-Σλ).
        """
        # Direct A->A = 0.2 (so remaining = 0.8), plus hazard A->B = 0.1.
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={
                ("A", "A"): (0.2, None),  # direct self-loop
                ("A", "B"): (0.1, "weekly"),  # hazard
            },
            time_step="weekly",
        )
        P = tg.build_matrix()
        # All hazard mass goes to B: 0.8 * (1 - exp(-0.1))
        expected_b = 0.8 * (1 - math.exp(-0.1))
        # Self-loop gets the survival portion of the remaining mass
        expected_a = 0.2 + 0.8 * math.exp(-0.1)
        assert P[0, 1] == pytest.approx(expected_b, abs=1e-9)
        assert P[0, 0] == pytest.approx(expected_a, abs=1e-9)
        assert_row_stochastic(P)

    def test_special_transition_overrides_hazard(self):
        """``special_transitions`` replaces the hazard-built row for that state."""
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (10.0, "weekly")},
            special_transitions={"A": [0.25, 0.75]},
            time_step="weekly",
        )
        P = tg.build_matrix()
        np.testing.assert_allclose(P[0], [0.25, 0.75], atol=1e-12)

    def test_zero_hazard_gives_identity(self):
        """A zero weekly hazard yields P = identity (no transitions)."""
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (0.0, "weekly")},
            time_step="weekly",
        )
        P = tg.build_matrix()
        np.testing.assert_allclose(P, np.eye(2), atol=1e-12)

    def test_survival_matches_exponential_formula(self):
        """Self-loop = exp(-Σλ) for any pure-hazard row."""
        tg = HybridTransitionGenerator(
            states=["A", "B", "C"],
            transition_pairs={
                ("A", "B"): (0.4, "weekly"),
                ("A", "C"): (0.1, "weekly"),
            },
            time_step="weekly",
        )
        P = tg.build_matrix()
        expected = math.exp(-0.5)
        assert P[0, 0] == pytest.approx(expected, abs=1e-9)

    def test_get_probability_helper(self):
        """``get_probability`` returns 1 - exp(-λ) when periods match."""
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (1.0, "weekly")},
        )
        p = tg.get_probability(1.0, "weekly")
        assert p == pytest.approx(1 - math.exp(-1.0), abs=1e-9)

    def test_validation_negative_value(self):
        """Negative hazard values are clamped to 0 (no negative probabilities)."""
        tg = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (-0.5, "weekly")},
        )
        P = tg.build_matrix()
        assert P[0, 1] >= 0.0
        assert_row_stochastic(P)


# ── CTMC math tests ─────────────────────────────────────────────────


class TestCTMCMath:
    """Mathematical correctness of the CTMC (matrix exponential) generator."""

    def _q(self, tg: CTMCTransitionGenerator) -> np.ndarray:
        return tg._build_generator_matrix()

    def test_generator_matrix_rows_sum_to_zero(self):
        """Q must have row sums = 0 (CTMC invariant)."""
        tg = CTMCTransitionGenerator(
            states=["A", "B", "C"],
            transition_pairs={
                ("A", "B"): 0.3,
                ("A", "C"): 0.2,
                ("B", "C"): 0.1,
            },
        )
        Q = self._q(tg)
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-12)

    def test_generator_diagonal_is_negative_of_off_diagonal(self):
        """Q[i, i] = -Σ_{j != i} Q[i, j]."""
        tg = CTMCTransitionGenerator(
            states=["A", "B", "C", "D"],
            transition_pairs={
                ("A", "B"): 0.3,
                ("A", "C"): 0.2,
                ("B", "A"): 0.05,
                ("B", "D"): 0.1,
            },
        )
        Q = self._q(tg)
        for i in range(4):
            assert Q[i, i] == pytest.approx(-sum(Q[i, j] for j in range(4) if j != i))

    def test_single_hazard_matches_1_minus_exp(self):
        """For a single hazard, P[0,1] = 1 - exp(-λ*Δt) regardless of expm.

        This is the closed-form 2×2 solution of expm and must hold exactly.
        """
        lam = 0.5
        tg = CTMCTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): lam},
            time_step="weekly",
        )
        P = tg.build_matrix()
        expected = 1 - math.exp(-lam / 52)
        assert P[0, 1] == pytest.approx(expected, abs=1e-9)
        assert P[0, 0] == pytest.approx(math.exp(-lam / 52), abs=1e-9)

    def test_multi_hazard_proportional_for_small_lambda(self):
        """For small λ·Δt, P[i,j] ≈ (λ_j/Σλ)·(1 - exp(-Σλ·Δt))."""
        lam1, lam2 = 0.3, 0.2
        total = lam1 + lam2
        dt = 1 / 52
        tg = CTMCTransitionGenerator(
            states=["A", "B", "C"],
            transition_pairs={
                ("A", "B"): lam1,
                ("A", "C"): lam2,
            },
            time_step="weekly",
        )
        P = tg.build_matrix()
        expected_b = (lam1 / total) * (1 - math.exp(-total * dt))
        expected_c = (lam2 / total) * (1 - math.exp(-total * dt))
        # expm and the small-Δt approximation are both highly accurate
        # at this scale; allow 1e-6 slack.
        assert P[0, 1] == pytest.approx(expected_b, abs=1e-6)
        assert P[0, 2] == pytest.approx(expected_c, abs=1e-6)

    def test_p_equals_expm_of_q_times_dt(self):
        """The matrix returned by build() is expm(Q · Δt), the defining CTMC eq."""
        tg = CTMCTransitionGenerator(
            states=["A", "B", "C"],
            transition_pairs={
                ("A", "B"): 0.3,
                ("A", "C"): 0.2,
                ("B", "C"): 0.05,
            },
            time_step="weekly",
        )
        Q = self._q(tg)
        P_actual = tg.build_matrix()
        P_expected = expm(Q * tg.delta_t)
        # Real part should match (expm can produce tiny imaginary parts)
        P_expected_real = np.real_if_close(P_expected)
        np.testing.assert_allclose(P_actual, P_expected_real, atol=1e-9)

    def test_p_is_stochastic(self):
        """P from a valid Q is a row-stochastic matrix (expm of a generator)."""
        tg = CTMCTransitionGenerator(
            states=["A", "B", "C", "D"],
            transition_pairs={
                ("A", "B"): 0.3,
                ("A", "C"): 0.2,
                ("B", "A"): 0.05,
                ("B", "C"): 0.1,
                ("C", "D"): 0.4,
            },
            time_step="weekly",
        )
        P = tg.build_matrix()
        assert_row_stochastic(P, atol=1e-9)
        assert_nonnegative(P, atol=1e-9)
        assert_unit_interval(P, atol=1e-9)

    def test_special_absorbing_state_yields_zero_q_row(self):
        """Regression: a special transition with p_ii = 1 must have Q[i, :] = 0."""
        tg = CTMCTransitionGenerator(
            states=["A", "death"],
            transition_pairs={("A", "death"): 0.1},
            special_transitions={"death": {"A": 0.0, "death": 1.0}},
            time_step="weekly",
        )
        Q = self._q(tg)
        # Row for 'death' must be all zeros (absorbing state).
        death_idx = tg.state_index["DEATH"]
        np.testing.assert_allclose(Q[death_idx, :], 0.0, atol=1e-12)
        # And P[death, death] = 1.
        P = tg.build_matrix()
        assert P[death_idx, death_idx] == pytest.approx(1.0, abs=1e-9)

    def test_special_non_absorbing_state_q_row_sums_to_zero(self):
        """Regression (the bug we fixed): for a NON-absorbing special
        transition, the generator row must still sum to 0 so expm produces
        a valid stochastic matrix.

        Before the fix, the diagonal Q[i, i] was left at 0, the row sum was
        positive, and the self-loop probability was wrong by ~4×.
        """
        # The "lt_bleeding" row used in the hemophilia domain.
        # 94% chance to go healthy, 6% to death, 0% to self.
        special = {
            "lt_bleeding": {
                "healthy": 0.94,
                "bleeding": 0.0,
                "lt_bleeding": 0.0,
                "death": 0.06,
            }
        }
        states = ["healthy", "bleeding", "lt_bleeding", "death"]
        tg = CTMCTransitionGenerator(
            states=states,
            transition_pairs={},
            special_transitions=special,
            time_step="weekly",
        )
        Q = self._q(tg)
        i = tg.state_index["LT_BLEEDING"]
        # The row sum must be exactly 0 (this was the bug).
        assert Q[i, :].sum() == pytest.approx(0.0, abs=1e-9)
        # The diagonal must be the negative of the off-diagonal sum.
        off_diag = sum(Q[i, j] for j in range(4) if j != i)
        assert Q[i, i] == pytest.approx(-off_diag, abs=1e-9)

        # And the resulting P must match the closed-form competing-risks answer
        # to high precision (no renormalization hackery needed).
        P = tg.build_matrix()
        lam_h = -math.log(0.06) * 52  # 1/Δt = 52
        lam_d = -math.log(0.94) * 52
        total = lam_h + lam_d
        dt = 1 / 52
        expected_h = (lam_h / total) * (1 - math.exp(-total * dt))
        expected_d = (lam_d / total) * (1 - math.exp(-total * dt))
        expected_self = math.exp(-total * dt)
        assert P[i, 0] == pytest.approx(expected_h, abs=1e-6)
        assert P[i, 3] == pytest.approx(expected_d, abs=1e-6)
        assert P[i, 2] == pytest.approx(expected_self, abs=1e-6)

    def test_special_transition_equivalent_constant_hazard(self):
        """Special probs are converted to hazards via λ = -ln(1-p)/Δt."""
        p = 0.5
        tg = CTMCTransitionGenerator(
            states=["A", "B"],
            transition_pairs={},
            # Spec must sum to 1: A->A=0.5 (self-loop), A->B=0.5 (event)
            special_transitions={"A": {"A": 0.5, "B": p}},
            time_step="weekly",
        )
        Q = self._q(tg)
        expected_lam = -math.log(1 - p) / tg.delta_t
        assert Q[0, 1] == pytest.approx(expected_lam, abs=1e-9)
        assert Q[0, 0] == pytest.approx(-expected_lam, abs=1e-9)

    def test_expm_step_counting(self):
        """P over k steps equals expm(Q · k·Δt). The Chapman-Kolmogorov
        equation must hold: P^(k) = P^k.
        """
        tg = CTMCTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): 0.3},
            time_step="weekly",
        )
        P1 = tg.build_matrix()
        P3 = np.linalg.matrix_power(P1, 3)
        Q = self._q(tg)
        P3_expected = expm(Q * 3 * tg.delta_t)
        np.testing.assert_allclose(
            np.real_if_close(P3_expected), P3, atol=1e-7
        )

    def test_no_transitions_gives_identity(self):
        """No hazards + no specials -> Q = 0 -> P = I."""
        tg = CTMCTransitionGenerator(states=["A", "B"], transition_pairs={})
        P = tg.build_matrix()
        np.testing.assert_allclose(P, np.eye(2), atol=1e-12)

    def test_validation_negative_rate(self):
        with pytest.raises(ValueError, match="Negative hazard rate"):
            CTMCTransitionGenerator(
                states=["A", "B"],
                transition_pairs={("A", "B"): -0.1},
            )


# ── Independent-hazard math tests ───────────────────────────────────


class TestIndependentHazardMath:
    """Mathematical correctness of the independent-hazard generator."""

    def test_single_hazard_weekly(self):
        """P[0,1] = 1 - exp(-λ/52) for a single weekly-rate hazard."""
        lam = 0.5
        tg = IndependentHazardTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): lam},
            time_step="weekly",
        )
        P = tg.build_matrix()
        expected = 1 - math.exp(-lam / 52)
        assert P[0, 1] == pytest.approx(expected, abs=1e-9)
        assert P[0, 0] == pytest.approx(math.exp(-lam / 52), abs=1e-9)

    def test_single_hazard_annual(self):
        """Annual time step uses Δt = 1."""
        lam = 1.0
        tg = IndependentHazardTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): lam},
            time_step="annual",
        )
        P = tg.build_matrix()
        assert P[0, 1] == pytest.approx(1 - math.exp(-1.0), abs=1e-9)

    def test_survival_matches_exponential(self):
        """Self-loop = exp(-Σλ·Δt) / Z, where Z is the renormalization factor.

        The independent-hazard approach slightly overestimates the total
        transition mass when many risks compete (sum > 1), so the result is
        renormalized. This test pins the closed-form formula for P[i,i] so
        any future refactor of the renormalization step is caught here.
        """
        lam1, lam2 = 0.4, 0.1
        dt = 1 / 52
        tg = IndependentHazardTransitionGenerator(
            states=["A", "B", "C"],
            transition_pairs={
                ("A", "B"): lam1,
                ("A", "C"): lam2,
            },
            time_step="weekly",
        )
        P = tg.build_matrix()
        raw_p1 = 1 - math.exp(-lam1 * dt)
        raw_p2 = 1 - math.exp(-lam2 * dt)
        raw_survival = math.exp(-(lam1 + lam2) * dt)
        z = raw_p1 + raw_p2 + raw_survival
        expected_self = raw_survival / z
        assert P[0, 0] == pytest.approx(expected_self, abs=1e-9)
        # And the renormalization factor itself is close to 1 for small rates.
        assert z == pytest.approx(1.0, abs=1e-4)

    def test_row_stochastic_after_renormalization(self):
        """Even with strong competing risks, renormalization yields rows=1."""
        tg = IndependentHazardTransitionGenerator(
            states=["A", "B", "C"],
            transition_pairs={
                ("A", "B"): 100.0,
                ("A", "C"): 100.0,
            },
            time_step="weekly",
        )
        P = tg.build_matrix()
        assert_row_stochastic(P, atol=1e-9)
        assert_nonnegative(P, atol=1e-9)

    def test_independent_hazard_overestimates_transition_mass(self):
        """The independent-hazard method produces a *larger* total transition
        mass than the equivalent-constant-hazard CTMC method, because it
        doesn't reconcile competing risks through the matrix exponential.

        We verify the inequality here: for two competing hazards the
        independent-hazard normalization factor (after pre-normalized
        sums) exceeds 1/(1 - exp(-Σλ·Δt)) for the self-loop; i.e. the
        independent-hazard P[i,i] is smaller than exp(-Σλ·Δt).
        """
        lam1, lam2 = 5.0, 5.0
        tg_indep = IndependentHazardTransitionGenerator(
            states=["A", "B", "C"],
            transition_pairs={
                ("A", "B"): lam1,
                ("A", "C"): lam2,
            },
            time_step="weekly",
        )
        P_indep = tg_indep.build_matrix()
        # Pre-renormalization: p_B + p_C + survival_raw
        #  = (1 - exp(-lam1/52)) + (1 - exp(-lam2/52)) + exp(-(lam1+lam2)/52)
        # For lam1=lam2=5, this exceeds 1, so P_indep[0,0] < exp(-Σλ/52)
        raw_self = math.exp(-(lam1 + lam2) / 52)
        assert P_indep[0, 0] < raw_self + 1e-12

    def test_special_transitions_override(self):
        """``special_transitions`` wins over hazards for that state."""
        tg = IndependentHazardTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): 0.5},
            special_transitions={"A": {"A": 0.3, "B": 0.7}},
        )
        P = tg.build_matrix()
        np.testing.assert_allclose(P[0], [0.3, 0.7], atol=1e-12)

    def test_absorbing_special(self):
        """An absorbing special row yields P[i, i] = 1."""
        tg = IndependentHazardTransitionGenerator(
            states=["A", "death"],
            transition_pairs={("A", "death"): 0.1},
            special_transitions={"death": {"A": 0.0, "death": 1.0}},
        )
        P = tg.build_matrix()
        d = tg.state_index["DEATH"]
        assert P[d, d] == pytest.approx(1.0, abs=1e-9)

    def test_validation_negative_rate(self):
        with pytest.raises(ValueError, match="Negative hazard rate"):
            IndependentHazardTransitionGenerator(
                states=["A", "B"],
                transition_pairs={("A", "B"): -0.1},
            )


# ── Cross-generator consistency ─────────────────────────────────────


class TestCrossGeneratorConsistency:
    """All four generators should agree (asymptotically) on small hazards,
    and all should be row-stochastic on valid input.
    """

    @pytest.mark.parametrize(
        "lam_weekly",
        [0.001, 0.01, 0.05, 0.1, 0.5],
    )
    def test_single_hazard_all_generators_agree(self, lam_weekly: float):
        """For a single weekly hazard (no specials, no other transitions),
        all four generators should produce essentially identical matrices
        to expm-precision (because there is no competing risk to differentiate
        them).

        Convention used:
            * Hybrid takes the rate AS the per-time-step rate
              (so we pass (lam_weekly, "weekly") with time_step="weekly").
            * CTMC and IndependentHazard treat the rate as per-year by
              convention and divide internally by 52 for a weekly time step
              (so we pass lam_annual = 52 * lam_weekly).
            * DTMC is direct, so we pre-compute the probability
              p = 1 - exp(-lam_weekly).
        """
        states = ["A", "B"]
        lam_annual = lam_weekly * 52

        P_dtmc = DTMCTransitionGenerator(
            states=states,
            transition_pairs={
                ("A", "A"): math.exp(-lam_weekly),
                ("A", "B"): 1 - math.exp(-lam_weekly),
            },
        ).build_matrix()
        P_hybrid = HybridTransitionGenerator(
            states=states,
            transition_pairs={("A", "B"): (lam_weekly, "weekly")},
            time_step="weekly",
        ).build_matrix()
        P_ctmc = CTMCTransitionGenerator(
            states=states,
            transition_pairs={("A", "B"): lam_annual},
            time_step="weekly",
        ).build_matrix()
        P_indep = IndependentHazardTransitionGenerator(
            states=states,
            transition_pairs={("A", "B"): lam_annual},
            time_step="weekly",
        ).build_matrix()

        # All four should match to expm precision.
        np.testing.assert_allclose(P_hybrid, P_ctmc, atol=1e-9)
        np.testing.assert_allclose(P_hybrid, P_indep, atol=1e-9)
        np.testing.assert_allclose(P_hybrid, P_dtmc, atol=1e-9)

    def test_absorbing_state_identical_across_generators(self):
        """All generators yield P = I (or self-loop = 1) for absorbing states.

        The 'death' state is the second entry in each test (index 1).
        """
        # Hybrid + special (uses list-form special_transitions)
        tg_h = HybridTransitionGenerator(
            states=["A", "death"],
            transition_pairs={("A", "death"): (0.5, "weekly")},
            special_transitions={"death": [0.0, 1.0]},
        )
        # CTMC + special
        tg_c = CTMCTransitionGenerator(
            states=["A", "death"],
            transition_pairs={("A", "death"): 0.5},
            special_transitions={"death": {"A": 0.0, "death": 1.0}},
        )
        # IndHazard + special
        tg_i = IndependentHazardTransitionGenerator(
            states=["A", "death"],
            transition_pairs={("A", "death"): 0.5},
            special_transitions={"death": {"A": 0.0, "death": 1.0}},
        )
        # DTMC (no special needed — death has no entries -> absorbing default)
        tg_d = DTMCTransitionGenerator(
            states=["A", "death"],
            transition_pairs={("A", "A"): 1.0},
        )
        for tg in (tg_h, tg_c, tg_i, tg_d):
            P = tg.build_matrix()
            # 'death' is always the second state -> index 1
            assert P[1, 1] == pytest.approx(1.0, abs=1e-9), (
                f"{type(tg).__name__} did not produce absorbing death state"
            )
            assert P[1, 0] == pytest.approx(0.0, abs=1e-9)

    def test_all_generators_row_stochastic_on_valid_input(self):
        """Smoke test: every generator returns a row-stochastic matrix
        for a small, well-formed example."""
        pairs = {("A", "A"): 0.7, ("A", "B"): 0.3}

        # DTMC
        P_d = DTMCTransitionGenerator(
            states=["A", "B"], transition_pairs=pairs
        ).build_matrix()
        assert_row_stochastic(P_d)

        # Hybrid
        P_h = HybridTransitionGenerator(
            states=["A", "B"],
            transition_pairs={("A", "B"): (0.3, None)},
        ).build_matrix()
        assert_row_stochastic(P_h)

        # CTMC
        P_c = CTMCTransitionGenerator(
            states=["A", "B"], transition_pairs={("A", "B"): 0.5}
        ).build_matrix()
        assert_row_stochastic(P_c)

        # IndHazard
        P_i = IndependentHazardTransitionGenerator(
            states=["A", "B"], transition_pairs={("A", "B"): 0.5}
        ).build_matrix()
        assert_row_stochastic(P_i)


# ── Engine integration: domain/transitions.py still works ───────────


class TestDomainTransitionMatrixBuilder:
    """The :func:`domain.transitions.build_transition_matrix` factory is the
    real-world entry point. Verify it still produces a valid stochastic
    matrix after the CTMC fix.
    """

    def test_build_transition_matrix_stochastic(self):
        """Smoke test: build the real hemophilia transition matrix and
        confirm it is row-stochastic on the actual data files.
        """
        from app.domain.enums import HealthStates
        from app.domain.inputs import ModelInput
        from app.domain.transition_builder import build_transition_matrix
        from app.persistence.context import ModelContext

        # Just load the context to ensure data files are reachable.
        ModelContext.load()
        # Realistic input mirroring ``test_seeding_reproducibility._make_input``.
        inputs = ModelInput(
            cycle=520,
            bleeding_rate=15.0,
            spontaneous_bleeding_rate=10.0,
            joint_bleeding_rate=4.0,
            life_threatening_bleeding_rate=1.0,
            ltb_case_fatality=0.35,
            baseline_age=2.0,
            weight_factor=1.0,
            benefits_discount_rate=0.0,
            healthy_utility=0.9,
            mild_arthropathy_utility=0.85,
            moderate_arthropathy_utility=0.7,
            severe_arthropathy_utility=0.5,
            spontaneous_bleeding_utility=0.6,
            joint_bleeding_utility=0.5,
            life_threatening_bleeding_utility=0.3,
            death_utility=0.0,
            per_unit_price=1000.0,
            costs_discount_rate=0.0,
            prophylaxis_background_factor_consumption_per_kg=0.0,
            factor_consumption_per_spontaneous_bleeding_per_kg=10.0,
            factor_consumption_per_joint_bleeding_per_kg=20.0,
            factor_consumption_per_life_threatening_bleeding_per_kg=50.0,
        )
        states = [s.value for s in HealthStates]
        P = build_transition_matrix(inputs, states)
        assert P.shape == (len(states), len(states))
        # The matrix must be row-stochastic — the bug we fixed would have
        # silently introduced drift in the lt_bleeding row.
        assert_row_stochastic(P, atol=1e-7)
