from typing import Any

import numpy as np

from app.domain.enums import HealthStates
from app.domain.inputs import ModelInput
from app.persistence.schemas.mortality import MortalityFile
from engine.modifier import TransitionModifier
from utils.logging import setup_root_logger


class AgeBasedMortalityModifier(TransitionModifier):
    """Applies age-specific background (all-cause) mortality every week.

    The life table stores *annual probabilities of death* (q_x) per age band.
    The patient's age is derived from the simulation step and the baseline
    age at model entry, and is only updated at calendar-year boundaries
    (i.e. the rate is piecewise-constant within each year of age), which
    keeps the computation cheap while remaining actuarially consistent.

    Conversion: annual probability q -> constant annual hazard h = -ln(1-q)
    -> weekly probability p_w = 1 - exp(-h/52). The weekly probability is
    then combined with any pre-existing death probability in the row under
    the standard independent competing-risks formula

        P_death = 1 - (1 - P_base) * (1 - p_w).
    """

    def __init__(
        self, mortality_file: MortalityFile, baseline_age: float = 0
    ) -> None:
        super().__init__()
        self.context = mortality_file
        self.age_specific_rates = self.context.age_specific
        self.crude_annual_rate = self.context.crude_annual_rate
        self.baseline_age = int(baseline_age)
        # Cache: age -> weekly death probability (ages change only once a year)
        self._weekly_prob_cache: dict[int, float] = {}

    def _get_annual_mortality(self, age: int):
        # Exact match for single years (e.g. "0")
        if str(age) in self.age_specific_rates:
            return self.age_specific_rates[str(age)]

        # Check range bands (e.g. "1-4", "5-9", ...)
        for key, rate in self.age_specific_rates.items():
            if "-" in key:
                try:
                    low, high = map(int, key.split("-"))
                    if low <= age <= high:
                        return rate
                except ValueError:
                    continue
            elif key == "90+" and age >= 90:
                return rate
        # Fallback to crude rate
        return float(self.crude_annual_rate)

    def _weekly_death_prob(self, age: int) -> float:
        """Annual death probability q_x -> weekly death probability.

        Uses the actuarially correct constant-hazard conversion
        p_w = 1 - (1 - q_x)^(1/52), which equals 1 - exp(-h/52) with
        h = -ln(1 - q_x). Cached per age (computed at most once per year).
        """
        cached = self._weekly_prob_cache.get(age)
        if cached is not None:
            return cached
        annual_prob = float(self._get_annual_mortality(age))
        annual_prob = min(max(annual_prob, 0.0), 0.999999)
        prob = 1.0 - (1.0 - annual_prob) ** (1.0 / 52.0)
        self._weekly_prob_cache[age] = prob
        return prob

    def adjust_transition(
        self,
        base_probs: np.ndarray,
        current_state: str,
        current_chain_name: str,
        step: int,
        states: list[str],
        **kwargs: Any,
    ) -> np.ndarray:

        if current_state == "death":
            return base_probs

        # Patient age at this step; the age band only changes at
        # calendar-year boundaries (weekly rate constant within the year).
        age = max(0, self.baseline_age + int(step / 52))
        probs = base_probs.copy().astype(float)
        weekly_death_prob = self._weekly_death_prob(age)
        if weekly_death_prob <= 0.0:
            return probs

        death_idx = states.index("death")
        # Existing death probability
        base_death = probs[death_idx]
        if base_death >= 1.0:  # Protection
            return probs
        # Competing risk combination
        combined_death = 1 - (1 - base_death) * (1 - weekly_death_prob)
        # Scale all non-death transitions
        survival_scale = (1 - combined_death) / (1 - base_death)

        for i in range(len(probs)):
            if i != death_idx:
                probs[i] *= survival_scale

        probs[death_idx] = combined_death

        # Re-normalize the row so probabilities sum to 1.0
        probs = np.clip(probs, 0.0, 1.0)
        probs /= probs.sum()
        return probs


def _weekly_event_distribution(
    inputs: ModelInput,
    states: list[str],
) -> np.ndarray:
    """Return one fresh weekly competing-risk distribution.

    ``healthy`` is retained as the serialized key for the conceptual
    No Bleeding state. The remaining destinations are mutually exclusive
    bleeding-event outcomes within a weekly cycle.
    """
    state_idx = {state: i for i, state in enumerate(states)}
    event_hazards = {
        HealthStates.BLEEDING.value: inputs.spontaneous_bleeding_rate / 52.0,
        HealthStates.HEMARTHROSIS.value: inputs.joint_bleeding_rate / 52.0,
        HealthStates.INTRACRANIAL_HEMORRHAGE.value: (
            inputs.intracranial_hemorrhage_rate / 52.0
        ),
        HealthStates.NON_ICH_MAJOR_BLEEDING.value: (
            inputs.non_ich_major_bleeding_rate / 52.0
        ),
    }
    if any(hazard < 0.0 for hazard in event_hazards.values()):
        raise ValueError("Annual bleeding-event hazards must be non-negative.")

    row = np.zeros(len(states), dtype=float)
    total_hazard = sum(event_hazards.values())
    no_bleeding = HealthStates.NO_BLEEDING.value
    if np.isclose(total_hazard, 0.0):
        row[state_idx[no_bleeding]] = 1.0
        return row

    survival = float(np.exp(-total_hazard))
    event_mass = 1.0 - survival
    row[state_idx[no_bleeding]] = survival
    for destination, hazard in event_hazards.items():
        row[state_idx[destination]] = hazard / total_hazard * event_mass
    return row


def _with_case_fatality(
    ordinary_row: np.ndarray,
    case_fatality: float,
    states: list[str],
) -> np.ndarray:
    """Apply acute fatality once, then allocate survivors to the next week."""
    state_idx = {state: i for i, state in enumerate(states)}
    p_death = float(np.clip(case_fatality, 0.0, 1.0))
    row = ordinary_row * (1.0 - p_death)
    row[state_idx[HealthStates.DEATH.value]] += p_death
    return row


def build_transition_matrices(
    inputs: list[ModelInput],
    states: list[str],
) -> np.ndarray:
    """Build all PSA transition matrices with NumPy broadcasting."""
    if not inputs:
        return np.empty((0, len(states), len(states)), dtype=np.float64)

    required_states = {state.value for state in HealthStates}
    missing = required_states.difference(states)
    if missing:
        raise ValueError(f"Transition matrix is missing required states: {missing}")

    state_idx = {state: i for i, state in enumerate(states)}
    n_iters = len(inputs)
    n_states = len(states)
    annual_hazards = np.column_stack(
        (
            np.fromiter(
                (item.spontaneous_bleeding_rate for item in inputs),
                dtype=np.float64,
                count=n_iters,
            ),
            np.fromiter(
                (item.joint_bleeding_rate for item in inputs),
                dtype=np.float64,
                count=n_iters,
            ),
            np.fromiter(
                (item.intracranial_hemorrhage_rate for item in inputs),
                dtype=np.float64,
                count=n_iters,
            ),
            np.fromiter(
                (item.non_ich_major_bleeding_rate for item in inputs),
                dtype=np.float64,
                count=n_iters,
            ),
        )
    )
    if np.any(annual_hazards < 0.0):
        raise ValueError("Annual bleeding-event hazards must be non-negative.")

    weekly_hazards = annual_hazards / 52.0
    total_hazard = weekly_hazards.sum(axis=1)
    survival = np.exp(-total_hazard)
    event_mass = 1.0 - survival
    event_probs = np.divide(
        weekly_hazards,
        total_hazard[:, None],
        out=np.zeros_like(weekly_hazards),
        where=total_hazard[:, None] > 0.0,
    )
    event_probs *= event_mass[:, None]

    ordinary_rows = np.zeros((n_iters, n_states), dtype=np.float64)
    ordinary_rows[:, state_idx[HealthStates.NO_BLEEDING.value]] = survival
    destinations = (
        HealthStates.BLEEDING.value,
        HealthStates.HEMARTHROSIS.value,
        HealthStates.INTRACRANIAL_HEMORRHAGE.value,
        HealthStates.NON_ICH_MAJOR_BLEEDING.value,
    )
    for column, destination in enumerate(destinations):
        ordinary_rows[:, state_idx[destination]] = event_probs[:, column]

    matrices = np.zeros((n_iters, n_states, n_states), dtype=np.float64)
    for state in (
        HealthStates.NO_BLEEDING.value,
        HealthStates.BLEEDING.value,
        HealthStates.HEMARTHROSIS.value,
    ):
        matrices[:, state_idx[state], :] = ordinary_rows

    death_idx = state_idx[HealthStates.DEATH.value]
    fatality_specs = (
        (
            HealthStates.INTRACRANIAL_HEMORRHAGE.value,
            np.fromiter(
                (item.ich_case_fatality for item in inputs),
                dtype=np.float64,
                count=n_iters,
            ),
        ),
        (
            HealthStates.NON_ICH_MAJOR_BLEEDING.value,
            np.fromiter(
                (item.non_ich_case_fatality for item in inputs),
                dtype=np.float64,
                count=n_iters,
            ),
        ),
    )
    for state, fatality in fatality_specs:
        fatality = np.clip(fatality, 0.0, 1.0)
        row = ordinary_rows * (1.0 - fatality[:, None])
        row[:, death_idx] += fatality
        matrices[:, state_idx[state], :] = row

    matrices[:, death_idx, death_idx] = 1.0
    return matrices


def build_transition_matrix(
    inputs: "ModelInput",
    states: list[str],
) -> np.ndarray:
    """Build a weekly matrix with resolved acute-event semantics.

    Ordinary bleeding and hemarthrosis resolve within their observed week and
    do not protect the next week. Those rows therefore use the same fresh
    competing-risk draw as No Bleeding. ICH and non-ICH first apply their case
    fatality once and then allocate survivors using that same distribution.

    Background mortality is deliberately added later by
    :class:`AgeBasedMortalityModifier` to avoid double counting.
    """
    required_states = {state.value for state in HealthStates}
    missing = required_states.difference(states)
    if missing:
        raise ValueError(f"Transition matrix is missing required states: {missing}")

    state_idx = {state: i for i, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)), dtype=float)
    ordinary_row = _weekly_event_distribution(inputs, states)

    for state in (
        HealthStates.NO_BLEEDING.value,
        HealthStates.BLEEDING.value,
        HealthStates.HEMARTHROSIS.value,
    ):
        matrix[state_idx[state]] = ordinary_row

    matrix[state_idx[HealthStates.INTRACRANIAL_HEMORRHAGE.value]] = (
        _with_case_fatality(ordinary_row, inputs.ich_case_fatality, states)
    )
    matrix[state_idx[HealthStates.NON_ICH_MAJOR_BLEEDING.value]] = (
        _with_case_fatality(ordinary_row, inputs.non_ich_case_fatality, states)
    )
    matrix[
        state_idx[HealthStates.DEATH.value],
        state_idx[HealthStates.DEATH.value],
    ] = 1.0

    # Quick sanity check
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-6):
        logger = setup_root_logger()
        logger.warning("Warning: Transition matrix rows do not sum to 1:", row_sums)

    return matrix
