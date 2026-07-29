from typing import Any

import numpy as np

from app.domain.inputs import ModelInput
from app.persistence.schemas.mortality import MortalityFile
from engine.modifier import TransitionModifier
from engine.transitions import HybridTransitionGenerator
from utils.logging import setup_root_logger
from utils.math import prob_at_least_one, to_weekly


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

        if current_state in ["death", "lt_bleeding"]:
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


def _add_death_transitions(
    states: list[str], special_transitions: dict = {}, dt=1.0, *args
):
    state_idx = {state: i for i, state in enumerate(states)}
    n = len(states)
    # Absorbing death state
    death_row = [0.0] * n
    death_row[state_idx["death"]] = 1.0
    special_transitions["death"] = death_row


def _add_ltb_transitions(
    states: list[str], special_transitions: dict = {}, dt=1.0, *args
):
    p_no_event, spont_rate, joint_rate, life_rate, p_death = args
    state_idx = {state: i for i, state in enumerate(states)}
    n = len(states)

    # Event probabilities
    # p_death = conditional case-fatality of a life-threatening bleeding
    # episode (evidence: Zwagemaker et al. 2021, ICH mortality/incidence
    # ~= 0.8/2.3 ~= 0.35). Applied once per LTB-state week.
    p_healthy = p_no_event  # probability to stay healthy

    p_spont = prob_at_least_one(spont_rate * dt)
    p_joint = prob_at_least_one(joint_rate * dt)
    p_life = prob_at_least_one(life_rate * dt)

    lt_row = np.zeros(n)

    remaining_mass = 1.0 - p_healthy - p_death
    if remaining_mass < 0:
        p_survive_death = 1 - p_death
        p_healthy = p_no_event * p_survive_death
        remaining_mass = p_survive_death - p_healthy

    # fixed assignments
    lt_row[state_idx["healthy"]] = p_healthy
    lt_row[state_idx["death"]] = p_death

    raw = np.array([p_spont, p_joint, p_life], dtype=float)
    raw_sum = raw.sum()

    if raw_sum > 0 and remaining_mass > 0:
        scaled = raw * (remaining_mass / raw_sum)

        lt_row[state_idx["bleeding"]] = scaled[0]
        lt_row[state_idx["hemarthrosis"]] = scaled[1]
        lt_row[state_idx["lt_bleeding"]] = scaled[2]

    special_transitions["lt_bleeding"] = lt_row.tolist()


def build_transition_matrix(
    inputs: "ModelInput",
    states: list[str],
) -> np.ndarray:

    # Compute weekly survival probability for recovery
    wbr = to_weekly(inputs.bleeding_rate)
    p_no_event = np.exp(-wbr)  # P(no bleeding in one week)

    spont_rate = to_weekly(inputs.spontaneous_bleeding_rate)
    joint_rate = to_weekly(inputs.joint_bleeding_rate)
    life_rate = to_weekly(inputs.life_threatening_bleeding_rate)
    # Background (all-cause) mortality is owned entirely by
    # ``AgeBasedMortalityModifier``, which injects the age-specific
    # life-table hazard every week. Keeping a non-zero constant death
    # hazard here would double-count background mortality.
    death_rate = 0.0

    # Transition pairs
    transition_pairs = {
        # HEALTHY - competing risks (all rates)
        ("healthy", "bleeding"): (spont_rate, "weekly"),
        ("healthy", "hemarthrosis"): (joint_rate, "weekly"),
        ("healthy", "lt_bleeding"): (life_rate, "weekly"),
        ("healthy", "death"): (death_rate, "weekly"),
        # BLEEDING
        ("bleeding", "healthy"): (p_no_event, None),  # Direct probability
        ("bleeding", "hemarthrosis"): (joint_rate, "weekly"),
        ("bleeding", "lt_bleeding"): (life_rate, "weekly"),
        ("bleeding", "death"): (death_rate, "weekly"),
        # HEMARTHROSIS
        ("hemarthrosis", "healthy"): (p_no_event, None),  # Direct probability
        ("hemarthrosis", "bleeding"): (
            to_weekly(inputs.spontaneous_bleeding_rate),
            "weekly",
        ),
        ("hemarthrosis", "lt_bleeding"): (
            to_weekly(inputs.life_threatening_bleeding_rate),
            "weekly",
        ),
        ("hemarthrosis", "death"): (death_rate, "weekly"),
        # LT_BLEEDING & DEATH (SPECIAL TRANSITIONS)
    }

    special_transitions = {}

    _add_ltb_transitions(
        states,
        special_transitions,
        1.0,
        p_no_event,
        spont_rate,
        joint_rate,
        life_rate,
        inputs.ltb_case_fatality,
    )
    _add_death_transitions(
        states,
        special_transitions,
        1.0,
        p_no_event,
        spont_rate,
        joint_rate,
        life_rate,
        inputs.ltb_case_fatality,
    )

    # Build using your original TransitionGenerator
    builder = HybridTransitionGenerator(
        states=states,
        transition_pairs=transition_pairs,
        special_transitions=special_transitions,
        time_step="weekly",
    )
    matrix = builder.build_matrix()

    # Quick sanity check
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-6):
        logger = setup_root_logger()
        logger.warning("Warning: Transition matrix rows do not sum to 1:", row_sums)

    return matrix
