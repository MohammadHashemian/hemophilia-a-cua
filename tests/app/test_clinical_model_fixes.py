"""Regression tests for the clinical-model fixes (2026-07 audit).

Pins the corrected behavior of:
- ``AgeBasedMortalityModifier`` (baseline-age offset, every-week
  application, actuarially correct annual->weekly conversion)
- ``build_transition_matrix`` (background mortality fully delegated to
  the modifier; LTB case-fatality parameterization)
- ``_aggregate_vectorized_output`` (never-absorbed sentinel -> None)
"""
import numpy as np
import pytest

from app.domain.enums import HealthStates
from app.domain.inputs import ModelInput
from app.domain.transition_builder import (
    AgeBasedMortalityModifier,
    build_transition_matrix,
)
from app.persistence.schemas.mortality import MortalityFile


def _mortality_file() -> MortalityFile:
    return MortalityFile(
        use_age_specific=True,
        crude_annual_rate=0.005,
        age_specific={
            "0": 0.010,
            "1-4": 0.0004,
            "5-9": 0.0005,
            "10-14": 0.0006,
            "15-19": 0.001,
            "20-24": 0.002,
            "90+": 0.25,
        },
    )


def _make_input(**overrides) -> ModelInput:
    base = dict(
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
    base.update(overrides)
    return ModelInput(**base)


class TestAgeBasedMortalityModifier:
    STATES = ["healthy", "bleeding", "hemarthrosis", "lt_bleeding", "death"]

    def _base_probs(self):
        return np.array([0.98, 0.01, 0.005, 0.005, 0.0])

    def test_uses_baseline_age_at_step_zero(self):
        """A 2-year-old entrant must be looked up in the '1-4' band at
        step 0, not in the (23x higher) infant band '0'."""
        mod = AgeBasedMortalityModifier(self._file(), baseline_age=2)
        probs = mod.adjust_transition(
            base_probs=self._base_probs(),
            current_state="healthy",
            current_chain_name="main",
            step=0,
            states=self.STATES,
        )
        q = 0.0004  # band '1-4'
        expected_weekly = 1.0 - (1.0 - q) ** (1.0 / 52.0)
        assert probs[4] == pytest.approx(expected_weekly, rel=1e-9)

    def test_applies_every_week_not_only_year_boundary(self):
        """The corrected modifier applies the weekly death probability at
        every step (the age band only changes at year boundaries)."""
        mod = AgeBasedMortalityModifier(self._file(), baseline_age=2)
        for step in (1, 17, 51):
            probs = mod.adjust_transition(
                base_probs=self._base_probs(),
                current_state="healthy",
                current_chain_name="main",
                step=step,
                states=self.STATES,
            )
            assert probs[4] > 0.0, f"no mortality applied at step {step}"

    def test_age_band_changes_at_year_boundary(self):
        """Age = baseline_age + step//52 (2y entrant: band '1-4' in sim
        years 0-2, band '5-9' from sim year 3 onwards)."""
        mod = AgeBasedMortalityModifier(self._file(), baseline_age=2)
        p_year2 = mod.adjust_transition(
            base_probs=self._base_probs(), current_state="healthy",
            current_chain_name="main", step=104, states=self.STATES,
        )
        p_year3 = mod.adjust_transition(
            base_probs=self._base_probs(), current_state="healthy",
            current_chain_name="main", step=156, states=self.STATES,
        )
        q_young = 1.0 - (1.0 - 0.0004) ** (1.0 / 52.0)
        q_older = 1.0 - (1.0 - 0.0005) ** (1.0 / 52.0)
        assert p_year2[4] == pytest.approx(q_young, rel=1e-9)
        assert p_year3[4] == pytest.approx(q_older, rel=1e-9)

    def test_skips_death_and_lt_bleeding_states(self):
        mod = AgeBasedMortalityModifier(self._file(), baseline_age=2)
        base = np.array([0.5, 0.1, 0.05, 0.05, 0.3])
        for state in ("death", "lt_bleeding"):
            out = mod.adjust_transition(
                base_probs=base, current_state=state,
                current_chain_name="main", step=0, states=self.STATES,
            )
            assert np.array_equal(out, base)

    def test_row_sums_to_one(self):
        mod = AgeBasedMortalityModifier(self._file(), baseline_age=2)
        probs = mod.adjust_transition(
            base_probs=self._base_probs(), current_state="healthy",
            current_chain_name="main", step=0, states=self.STATES,
        )
        assert probs.sum() == pytest.approx(1.0, abs=1e-12)

    def _file(self) -> MortalityFile:
        return _mortality_file()


class TestTransitionBuilderMortalityAndLTB:
    STATES = [s.value for s in HealthStates]

    def test_base_matrix_has_no_background_death_hazard(self):
        """Background mortality is owned by the modifier: the base matrix
        must carry zero death probability out of healthy/bleeding/
        hemarthrosis (otherwise it double-counts)."""
        P = build_transition_matrix(_make_input(), self.STATES)
        death_idx = self.STATES.index("death")
        for state in ("healthy", "bleeding", "hemarthrosis"):
            i = self.STATES.index(state)
            assert P[i, death_idx] == pytest.approx(0.0, abs=1e-15)

    def test_ltb_row_death_equals_case_fatality(self):
        """The LTB special row must use the parameterized case fatality,
        not the historical hardcoded 0.06."""
        for fatality in (0.2, 0.35, 0.5):
            P = build_transition_matrix(
                _make_input(ltb_case_fatality=fatality), self.STATES
            )
            lt_idx = self.STATES.index("lt_bleeding")
            death_idx = self.STATES.index("death")
            assert P[lt_idx, death_idx] == pytest.approx(fatality, abs=1e-12)

    def test_death_row_absorbing(self):
        P = build_transition_matrix(_make_input(), self.STATES)
        death_idx = self.STATES.index("death")
        assert P[death_idx, death_idx] == pytest.approx(1.0)
        assert P[death_idx].sum() == pytest.approx(1.0)

    def test_rows_stochastic(self):
        P = build_transition_matrix(_make_input(), self.STATES)
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-9)


class TestAggregateAbsorbedSentinel:
    def test_never_absorbed_maps_to_none(self):
        from app.domain.enums import Regime
        from app.domain.worker import _aggregate_vectorized_output
        from engine.vectorized import BatchResult

        n_iters, n_steps = 2, 5
        rewards = {
            "consumption": np.zeros((n_iters, n_steps + 1)),
            "utility": np.zeros((n_iters, n_steps + 1)),
            "weight": np.ones((n_iters, n_steps + 1)),
            "event_count": np.zeros((n_iters, n_steps + 1)),
            "pettersson_score": np.zeros((n_iters, n_steps + 1)),
        }
        batch = BatchResult(
            sequences=np.zeros((n_iters, n_steps + 1), dtype=np.int32),
            # iter 0 never absorbs (steps+1 sentinel); iter 1 absorbs at 3
            absorbed_at=np.array([n_steps + 1, 3], dtype=np.int32),
            rewards=rewards,
        )
        inputs = [_make_input(cycle=n_steps) for _ in range(n_iters)]
        states = [s.value for s in HealthStates]
        out0 = _aggregate_vectorized_output(inputs, batch, states, Regime.ON_DEMAND, 0)
        out1 = _aggregate_vectorized_output(inputs, batch, states, Regime.ON_DEMAND, 1)
        assert out0.absorbed_at is None
        assert out1.absorbed_at == 3
