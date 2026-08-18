from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from modular_models.state_transition.context import StudyContext
from modular_models.state_transition.kernels import apply_utility_interval
from modular_models.state_transition.results import SimulationResult
from modular_models.state_transition.rewards import (
    ComputeBackend,
    create_reward_integrator,
)
from modular_models.state_transition.types import (
    EVENT_ORDER,
    AcuteEvent,
    ChronicState,
    DeathCause,
    Strategy,
)

if TYPE_CHECKING:
    from modular_models.state_transition.trace import TraceSession


_STREAM_EVENT_COUNT = 100
_STREAM_EVENT_ONSET = 500
_STREAM_ICH_FATALITY = 900
_STREAM_BACKGROUND_DEATH = 1000
_STREAM_BACKGROUND_DEATH_TIME = 1001
_STREAM_SEQUELA = 1100

_MORTALITY_BANDS = (
    ("age_1_to_lt5", 1.0, 5.0, "background_mortality_age_1_4"),
    ("age_5_to_lt10", 5.0, 10.0, "background_mortality_age_5_9"),
    ("age_10_to_lt12", 10.0, 12.0, "background_mortality_age_10_lt12"),
)


@dataclass(frozen=True, slots=True)
class EventRates:
    annual: dict[AcuteEvent, float]

    @property
    def weekly(self) -> dict[AcuteEvent, float]:
        return {event: rate / 52.0 for event, rate in self.annual.items()}

    def validate(self, abr: float) -> None:
        if any(not np.isfinite(value) or value < 0 for value in self.annual.values()):
            raise ValueError(
                f"Event rates must be finite and non-negative: {self.annual}"
            )
        total = sum(self.annual.values())
        if not np.isclose(total, abr, rtol=0, atol=1e-9):
            raise ValueError(f"Event rates must sum to ABR ({total} != {abr})")


def derive_event_rates(
    values: dict[str, float],
    options: dict[str, Any],
    strategy: Strategy,
) -> EventRates:
    suffix = "prophylaxis" if strategy is Strategy.PROPHYLAXIS else "on_demand"
    abr = float(values[f"abr_{suffix}"])
    if options.get("joint_rate_method", "direct") == "fraction":
        joint = abr * float(values["joint_bleed_fraction"])
    else:
        joint = float(values[f"ajbr_{suffix}"])

    if options.get("ich_rate_method", "direct") == "fraction":
        ich = abr * float(values["ich_fraction"])
    else:
        ich = float(values[f"ich_rate_{suffix}"])

    non_ich_major = abr * float(values["non_ich_major_fraction"])
    residual = abr - joint - ich - non_ich_major
    if residual < -1e-12:
        raise ValueError(
            "Inconsistent bleeding inputs: ABR - AJBR - ICH - non-ICH major is negative "
            f"for {strategy.value} ({residual:.8g})."
        )
    residual = max(0.0, residual)
    rates = EventRates(
        annual={
            AcuteEvent.JOINT_BLEED: joint,
            AcuteEvent.NON_MAJOR_NON_JOINT: residual,
            AcuteEvent.NON_ICH_MAJOR: non_ich_major,
            AcuteEvent.ICH: ich,
        }
    )
    rates.validate(abr)
    return rates


class StateTransitionEngine:
    """Vectorized individual-level microsimulation with patient memory.

    The engine contains no hidden analysis constants. Every numerical input arrives
    through a validated :class:`StudyContext` and a resolved value mapping.
    Chronic state progression is deterministic conditional on patient history;
    acute bleeds are recurrent within-cycle Poisson events.
    """

    def __init__(
        self,
        context: StudyContext,
        values: dict[str, float],
        options: dict[str, Any],
        *,
        scenario_id: str,
        seed: int,
        compute_backend: ComputeBackend = "cpu",
    ) -> None:
        self.context = context
        self.values = dict(values)
        self.options = dict(options)
        self.scenario_id = scenario_id
        self.seed = int(seed)
        self.compute_backend = compute_backend
        self._validate_runtime_inputs()

    def _validate_runtime_inputs(self) -> None:
        entry = self.values["entry_age_years"]
        exit_age = self.values["exit_age_years"]
        if not 0 <= entry < exit_age:
            raise ValueError(
                "entry_age_years must be non-negative and below exit_age_years"
            )
        if not np.isclose(entry, 1.0) or not np.isclose(exit_age, 12.0):
            raise ValueError(
                "The pediatric analysis enters at age 1 and exits immediately before "
                "the 12th birthday"
            )
        if (
            int(self.values["cycles_per_year"]) != 52
            or int(self.values["days_per_cycle"]) != 7
        ):
            raise ValueError("This analysis requires 52 seven-day cycles per year")
        step = self.values["utility_integration_step_days"]
        if step <= 0 or 7.0 / step != round(7.0 / step):
            raise ValueError(
                "utility integration step must divide the seven-day cycle exactly"
            )
        for key in (
            "utility_anchor",
            "utility_mild",
            "utility_moderate",
            "utility_severe",
            "non_ich_major_utility_cap",
            "ich_acute_utility_cap",
            "post_ich_utility_cap",
        ):
            if not 0 <= self.values[key] <= 1:
                raise ValueError(f"{key} must lie in [0, 1]")

    def _rng(self, cycle: int, stream: int) -> np.random.Generator:
        # Counter-style stream identifiers keep paired strategies aligned even
        # when their event rates and internal Poisson branches differ.
        return np.random.default_rng(np.random.SeedSequence([self.seed, cycle, stream]))

    def _weight(self, age_years: float) -> float:
        ages = np.arange(1, 13, dtype=np.float64)
        weights = np.array([self.values[f"weight_age_{age}"] for age in range(1, 13)])
        return float(np.interp(age_years, ages, weights))

    def _background_mortality_hazard(self, age_years: float) -> float:
        if age_years < 5:
            return self.values["background_mortality_age_1_4"]
        if age_years < 10:
            return self.values["background_mortality_age_5_9"]
        return self.values["background_mortality_age_10_lt12"]

    @staticmethod
    def _mortality_band(age_years: float) -> str:
        if age_years < 5:
            return "age_1_to_lt5"
        if age_years < 10:
            return "age_5_to_lt10"
        return "age_10_to_lt12"

    def _chronic_state(self, pettersson: np.ndarray, alive: np.ndarray) -> np.ndarray:
        state = np.full(
            pettersson.shape, ChronicState.NO_MINIMAL_ARTHROPATHY, dtype=np.int8
        )
        state[pettersson >= 13] = ChronicState.MILD_ARTHROPATHY
        state[pettersson >= 22] = ChronicState.MODERATE_ARTHROPATHY
        state[pettersson >= 40] = ChronicState.SEVERE_ARTHROPATHY
        state[~alive] = ChronicState.DEATH
        return state

    def _chronic_utility(
        self,
        state: np.ndarray,
        sequela: np.ndarray,
    ) -> NDArray[np.float64]:
        mild = (
            self.values["utility_anchor"]
            if self.options.get("mild_utility_rule") == "anchor"
            else self.values["utility_mild"]
        )
        utilities = np.array(
            [
                self.values["utility_anchor"],
                mild,
                self.values["utility_moderate"],
                self.values["utility_severe"],
                0.0,
            ],
            dtype=np.float64,
        )
        chronic = utilities[state]
        post_ich_cap = (
            self.values["post_ich_mild_utility_cap"]
            if self.options.get("post_ich_utility_rule") == "mild"
            else self.values["post_ich_utility_cap"]
        )
        resolved: NDArray[np.float64] = np.where(
            sequela, np.minimum(chronic, post_ich_cap), chronic
        )
        return resolved

    @staticmethod
    def _apply_interval(
        schedule: NDArray[np.float64],
        base: NDArray[np.float64],
        candidate: NDArray[np.float64],
        starts: NDArray[np.float64],
        durations: NDArray[np.float64],
        active: NDArray[np.bool_],
        step_days: float,
    ) -> None:
        """Retain the reference interval operation for audit tests and extensions."""
        if np.any(active):
            apply_utility_interval(
                schedule,
                base,
                candidate,
                starts,
                durations,
                active,
                step_days,
            )

    def _allocate_major_course(
        self,
        current_factor: np.ndarray,
        pending_factor: np.ndarray,
        total_iu: np.ndarray,
        initial_fraction: float,
        starts: np.ndarray,
        duration_days: float,
        active: np.ndarray,
        death_time: np.ndarray,
    ) -> None:
        """Allocate initial and maintenance FVIII over the modeled course.

        The 45 IU/kg initial dose is placed at event onset.  The remaining
        model-derived course is spread over its hemostatic-coverage interval,
        which preserves the total and its timing across weekly discount cycles.
        Treatment is truncated at death.
        """
        if not np.any(active):
            return
        initial_fraction = float(np.clip(initial_fraction, 0.0, 1.0))
        initial = total_iu * initial_fraction * active
        current_factor += initial
        maintenance = (total_iu - total_iu * initial_fraction) * active
        end = np.where(
            death_time < 7.0,
            np.minimum(starts + duration_days, death_time),
            starts + duration_days,
        )
        effective = np.maximum(0.0, end - starts)
        for offset in range(3):
            left = offset * 7.0
            right = left + 7.0
            overlap = np.clip(
                np.minimum(end, right) - np.maximum(starts, left), 0.0, None
            )
            share = np.divide(
                overlap,
                effective,
                out=np.zeros_like(overlap),
                where=effective > 0,
            )
            allocation = maintenance * share
            if offset == 0:
                current_factor += allocation
            else:
                pending_factor[:, offset - 1] += allocation

    def run(
        self,
        strategy: Strategy | str,
        *,
        n_patients: int | None = None,
        retain_patient_level: bool = False,
        trace: TraceSession | None = None,
    ) -> SimulationResult:
        strategy = Strategy(strategy)
        n = int(n_patients or self.values["default_patients"])
        if n <= 0:
            raise ValueError("n_patients must be positive")

        cycles_per_year = int(self.values["cycles_per_year"])
        days_per_cycle = float(self.values["days_per_cycle"])
        entry_age = float(self.values["entry_age_years"])
        exit_age = float(self.values["exit_age_years"])
        cycle_count = (exit_age - entry_age) * cycles_per_year
        if not np.isclose(cycle_count, round(cycle_count)):
            raise ValueError(
                "The exclusive age horizon must contain a whole number of cycles"
            )
        n_cycles = int(round(cycle_count))
        step_days = float(self.values["utility_integration_step_days"])
        n_bins = int(round(days_per_cycle / step_days))
        reward_integrator = create_reward_integrator(
            self.compute_backend,  # type: ignore
            n_bins,
            step_days,
        )
        rates = derive_event_rates(self.values, self.options, strategy)
        weekly_rates = rates.weekly

        alive = np.ones(n, dtype=bool)
        death_cycle = np.full(n, -1, dtype=np.int32)
        death_age_years = np.full(n, np.nan, dtype=np.float64)
        death_cause = np.full(n, DeathCause.ALIVE, dtype=np.int8)
        cumulative_joint_bleeds = np.zeros(n, dtype=np.int32)
        pettersson = np.zeros(n, dtype=np.int16)
        sequela = np.zeros(n, dtype=bool)
        ever_post_ich = np.zeros(n, dtype=bool)
        ever_survived_ich = np.zeros(n, dtype=bool)
        sequela_activation = np.full(n, np.inf, dtype=np.float64)
        carryover = np.zeros((len(EVENT_ORDER), n), dtype=np.float64)
        pending_factor = np.zeros((n, 2), dtype=np.float64)

        total_factor = np.zeros(n, dtype=np.float64)
        total_cost = np.zeros(n, dtype=np.float64)
        total_qaly = np.zeros(n, dtype=np.float64)
        total_life_years = np.zeros(n, dtype=np.float64)
        event_totals = np.zeros((len(EVENT_ORDER), n), dtype=np.int32)
        mortality_band_counts = {
            name: {
                "exposed_patient_cycles": 0,
                "background_deaths": 0,
                "ich_deaths": 0,
            }
            for name, _, _, _ in _MORTALITY_BANDS
        }

        if trace is not None:
            trace.begin(
                strategy,
                n,
                rates,
                self.values,
                self.options,
                provenance={
                    key: {
                        "unit": parameter.unit,
                        "references": list(parameter.references),
                        "assumption": parameter.assumption,
                    }
                    for key, parameter in self.context.parameters.items()
                },
            )

        event_duration = {
            AcuteEvent.JOINT_BLEED: self.values["minor_bleed_duration_days"],
            AcuteEvent.NON_MAJOR_NON_JOINT: self.values["minor_bleed_duration_days"],
            AcuteEvent.NON_ICH_MAJOR: self.values["non_ich_major_duration_days"],
            AcuteEvent.ICH: self.values["ich_duration_days"],
        }

        for cycle in range(n_cycles):
            alive_at_start = alive.copy()
            if not np.any(alive_at_start):
                break
            age = entry_age + cycle / cycles_per_year
            weight = self._weight(age)

            activate_now = alive & (sequela_activation <= 0.0)
            sequela[activate_now] = True
            ever_post_ich[activate_now] = True
            sequela_activation[activate_now] = np.inf

            factor_cycle = pending_factor[:, 0].copy()
            pending_factor[:, 0] = pending_factor[:, 1]
            pending_factor[:, 1] = 0.0

            counts = np.zeros((len(EVENT_ORDER), n), dtype=np.int32)
            occurrences: dict[AcuteEvent, list[np.ndarray]] = {}
            for event_index, event in enumerate(EVENT_ORDER):
                count_rng = self._rng(cycle, _STREAM_EVENT_COUNT + event_index)
                count = count_rng.poisson(weekly_rates[event], n).astype(np.int32)
                count[~alive_at_start] = 0
                counts[event_index] = count
                event_occurrences: list[np.ndarray] = []
                for occurrence_index in range(int(count.max(initial=0))):
                    onset_rng = self._rng(
                        cycle,
                        _STREAM_EVENT_ONSET + event_index * 50 + occurrence_index,
                    )
                    event_occurrences.append(onset_rng.uniform(0.0, days_per_cycle, n))
                occurrences[event] = event_occurrences

            fatal_ich_time = np.full(n, days_per_cycle, dtype=np.float64)
            ich_index = EVENT_ORDER.index(AcuteEvent.ICH)
            for occurrence_index, onset in enumerate(occurrences[AcuteEvent.ICH]):
                present = counts[ich_index] > occurrence_index
                fatal_rng = self._rng(cycle, _STREAM_ICH_FATALITY + occurrence_index)
                fatal = (
                    alive_at_start
                    & present
                    & (fatal_rng.random(n) < self.values["ich_case_fatality"])
                )
                fatal_ich_time = np.where(
                    fatal,
                    np.minimum(fatal_ich_time, onset),
                    fatal_ich_time,
                )

            hazard = self._background_mortality_hazard(age)
            p_background = 1.0 - np.exp(-hazard / cycles_per_year)
            background_death = alive_at_start & (
                self._rng(cycle, _STREAM_BACKGROUND_DEATH).random(n) < p_background
            )
            sampled_background_time = self._rng(
                cycle, _STREAM_BACKGROUND_DEATH_TIME
            ).uniform(0.0, days_per_cycle, n)
            background_time = np.where(
                background_death, sampled_background_time, days_per_cycle
            )
            death_time = np.minimum(fatal_ich_time, background_time)
            ich_death_this_cycle = alive_at_start & (fatal_ich_time < background_time)
            background_death_this_cycle = alive_at_start & (
                (background_time <= fatal_ich_time) & (background_time < days_per_cycle)
            )

            band_name = self._mortality_band(age)
            band_record = mortality_band_counts[band_name]
            band_record["exposed_patient_cycles"] += int(alive_at_start.sum())
            band_record["background_deaths"] += int(background_death_this_cycle.sum())
            band_record["ich_deaths"] += int(ich_death_this_cycle.sum())

            valid_occurrences: dict[AcuteEvent, list[np.ndarray]] = {}
            for event_index, event in enumerate(EVENT_ORDER):
                masks: list[np.ndarray] = []
                for occurrence_index, onset in enumerate(occurrences[event]):
                    valid = (counts[event_index] > occurrence_index) & (
                        onset <= death_time
                    )
                    masks.append(valid)
                    event_totals[event_index] += valid.astype(np.int32)
                    if event is AcuteEvent.JOINT_BLEED:
                        cumulative_joint_bleeds += valid.astype(np.int32)
                valid_occurrences[event] = masks

            pettersson = np.floor(
                cumulative_joint_bleeds
                / max(1e-12, self.values["joint_bleeds_per_pettersson_point"])
            ).astype(np.int16)
            np.minimum(pettersson, int(self.values["pettersson_max"]), out=pettersson)
            chronic_state = self._chronic_state(pettersson, alive_at_start)
            chronic_utility = self._chronic_utility(chronic_state, sequela)
            reward_integrator.begin(chronic_utility)

            # Continuing acute effects are applied before new occurrences.
            next_carryover = np.maximum(carryover - days_per_cycle, 0.0)
            for event_index, event in enumerate(EVENT_ORDER):
                duration = carryover[event_index]
                active = alive_at_start & (duration > 0)
                candidate = self._acute_candidate(event, chronic_utility)
                reward_integrator.apply(
                    candidate,
                    np.zeros(n),
                    duration,
                    active,
                )

            # Persistent sequela can begin midway through a cycle as the acute ICH phase ends.
            activation_in_cycle = (
                alive_at_start
                & (sequela_activation > 0)
                & (sequela_activation < days_per_cycle)
            )
            post_cap = (
                self.values["post_ich_mild_utility_cap"]
                if self.options.get("post_ich_utility_rule") == "mild"
                else self.values["post_ich_utility_cap"]
            )
            reward_integrator.apply(
                np.minimum(chronic_utility, post_cap),
                np.where(np.isfinite(sequela_activation), sequela_activation, 0.0),
                np.where(
                    np.isfinite(sequela_activation),
                    np.maximum(days_per_cycle - sequela_activation, 0.0),
                    0.0,
                ),
                activation_in_cycle,
            )

            # Background prophylaxis is accrued only while alive in the cycle.
            alive_fraction = alive_at_start.astype(np.float64) * np.clip(
                death_time / days_per_cycle, 0.0, 1.0
            )
            total_life_years += alive_fraction / cycles_per_year
            if strategy is Strategy.PROPHYLAXIS:
                factor_cycle += (
                    self.values["prophylaxis_iu_per_kg_week"] * weight * alive_fraction
                )

            for event_index, event in enumerate(EVENT_ORDER):
                candidate = self._acute_candidate(event, chronic_utility)
                duration_value = float(event_duration[event])
                for occurrence_index, onset in enumerate(occurrences[event]):
                    active = valid_occurrences[event][occurrence_index]
                    duration = np.full(n, duration_value, dtype=np.float64)
                    reward_integrator.apply(
                        candidate,
                        onset,
                        duration,
                        active,
                    )
                    spill = np.maximum(onset + duration_value - days_per_cycle, 0.0)
                    next_carryover[event_index] = np.maximum(
                        next_carryover[event_index],
                        np.where(active, spill, 0.0),
                    )

                    if event is AcuteEvent.JOINT_BLEED:
                        factor_cycle += (
                            active * weight * self.values["joint_bleed_iu_per_kg"]
                        )
                    elif event is AcuteEvent.NON_MAJOR_NON_JOINT:
                        factor_cycle += (
                            active
                            * weight
                            * self.values["non_major_non_joint_iu_per_kg"]
                        )
                    else:
                        dose_key = (
                            "non_ich_major_iu_per_kg"
                            if event is AcuteEvent.NON_ICH_MAJOR
                            else "ich_iu_per_kg"
                        )
                        total_iu = np.full(n, weight * self.values[dose_key])
                        self._allocate_major_course(
                            factor_cycle,
                            pending_factor,
                            total_iu,
                            45.0 / self.values[dose_key],
                            onset,
                            duration_value,
                            active,
                            death_time,
                        )

                    if event is AcuteEvent.ICH:
                        survived_ich = active & (onset < death_time)
                        ever_survived_ich |= survived_ich
                        survived = survived_ich & ~sequela
                        seq_rng = self._rng(cycle, _STREAM_SEQUELA + occurrence_index)
                        will_sequela = survived & (
                            seq_rng.random(n)
                            < self.values["post_ich_sequela_probability"]
                        )
                        activation = onset + duration_value
                        sequela_activation = np.where(
                            will_sequela,
                            np.minimum(sequela_activation, activation),
                            sequela_activation,
                        )

            # Truncate QALY at the continuous time of death.
            qaly_cycle = reward_integrator.finish(death_time)
            cost_cycle = (
                np.maximum(factor_cycle, 0.0) * self.values["factor_price_irr_per_iu"]
            )
            discount_time = cycle / cycles_per_year
            qaly_discount = (1.0 + self.values["qaly_discount_rate"]) ** discount_time
            cost_discount = (1.0 + self.values["cost_discount_rate"]) ** discount_time
            total_qaly += qaly_cycle / qaly_discount
            total_cost += cost_cycle / cost_discount
            total_factor += np.maximum(factor_cycle, 0.0)

            died = alive_at_start & (death_time < days_per_cycle)
            death_cycle[died] = cycle
            death_age_years[died] = (
                entry_age
                + (cycle + death_time[died] / days_per_cycle) / cycles_per_year
            )
            death_cause[ich_death_this_cycle] = DeathCause.ICH
            death_cause[background_death_this_cycle] = DeathCause.BACKGROUND
            alive[died] = False
            pending_factor[died] = 0.0
            next_carryover[:, died] = 0.0
            sequela_activation[died] = np.inf

            activate_by_end = alive & (sequela_activation <= days_per_cycle)
            sequela[activate_by_end] = True
            ever_post_ich[activate_by_end] = True
            sequela_activation[activate_by_end] = np.inf
            waiting = np.isfinite(sequela_activation)
            sequela_activation[waiting] -= days_per_cycle
            carryover = next_carryover

            if trace is not None:
                trace.record_cycle(
                    cycle=cycle,
                    age=age,
                    alive_at_start=int(alive_at_start.sum()),
                    alive_at_end=int(alive.sum()),
                    mean_weight=weight,
                    background_annual_hazard=hazard,
                    background_weekly_probability=p_background,
                    background_deaths=int(background_death_this_cycle.sum()),
                    ich_deaths=int(ich_death_this_cycle.sum()),
                    post_ich_flag_count=int(ever_post_ich.sum()),
                    counts={
                        event.value: int(event_totals[index].sum())
                        for index, event in enumerate(EVENT_ORDER)
                    },
                    mean_cycle_factor=float(factor_cycle.mean()),
                    mean_cycle_qaly=float(qaly_cycle.mean()),
                )

        final_state = self._chronic_state(pettersson, alive)
        state_counts = {
            ChronicState(code).name.lower(): int(np.sum(final_state == code))
            for code in range(len(ChronicState))
        }
        deaths_background = int(np.sum(death_cause == DeathCause.BACKGROUND))
        deaths_ich = int(np.sum(death_cause == DeathCause.ICH))
        deaths_total = deaths_background + deaths_ich
        ich_events = int(event_totals[3].sum())
        ich_survivors = int(ever_survived_ich.sum())
        post_ich_count = int(ever_post_ich.sum())

        age_specific_background: dict[str, dict[str, float | int]] = {}
        for name, start_age, end_age, parameter_id in _MORTALITY_BANDS:
            annual_hazard = float(self.values[parameter_id])
            scheduled_cycles = sum(
                start_age <= entry_age + cycle / cycles_per_year < end_age
                for cycle in range(n_cycles)
            )
            record = mortality_band_counts[name]
            age_specific_background[name] = {
                "start_age_years": start_age,
                "end_age_years_exclusive": end_age,
                "annual_hazard": annual_hazard,
                "weekly_probability": 1.0 - np.exp(-annual_hazard / cycles_per_year),
                "scheduled_cycles": scheduled_cycles,
                "cumulative_probability_if_alive_for_entire_band": 1.0
                - np.exp(-annual_hazard * scheduled_cycles / cycles_per_year),
                "exposed_patient_cycles": record["exposed_patient_cycles"],
                "background_deaths": record["background_deaths"],
                "ich_deaths": record["ich_deaths"],
                "all_cause_deaths": record["background_deaths"] + record["ich_deaths"],
            }

        mortality = {
            "overall": {
                "initial_patients": n,
                "alive_at_end": int(alive.sum()),
                "deaths_total": deaths_total,
                "deaths_background": deaths_background,
                "deaths_ich": deaths_ich,
                "all_cause_mortality_probability": deaths_total / n,
                "background_mortality_probability": deaths_background / n,
                "ich_mortality_probability": deaths_ich / n,
            },
            "ich_event_mortality": {
                "ich_events": ich_events,
                "case_fatality_probability_input": float(
                    self.values["ich_case_fatality"]
                ),
                "ich_deaths": deaths_ich,
                "observed_deaths_per_ich_event": (
                    deaths_ich / ich_events if ich_events else 0.0
                ),
            },
            "age_specific_background": age_specific_background,
        }
        summary = {
            "entry_age_years": entry_age,
            "exit_age_years_exclusive": exit_age,
            "follow_up_years": exit_age - entry_age,
            "last_cycle_start_age_years": entry_age + (n_cycles - 1) / cycles_per_year,
            "n_cycles": n_cycles,
            "mean_cost_irr": float(total_cost.mean()),
            "sd_cost_irr": float(total_cost.std(ddof=1)) if n > 1 else 0.0,
            "mean_qaly": float(total_qaly.mean()),
            "sd_qaly": float(total_qaly.std(ddof=1)) if n > 1 else 0.0,
            "mean_factor_iu": float(total_factor.mean()),
            "mean_life_years": float(total_life_years.mean()),
            "mean_joint_bleeds": float(event_totals[0].mean()),
            "mean_non_major_non_joint_bleeds": float(event_totals[1].mean()),
            "mean_non_ich_major_bleeds": float(event_totals[2].mean()),
            "mean_ich": float(event_totals[3].mean()),
            "mean_total_bleeds": float(event_totals.sum(axis=0).mean()),
            "joint_bleed_rate_per_person_year": float(
                event_totals[0].sum() / max(total_life_years.sum(), 1e-12)
            ),
            "non_major_non_joint_rate_per_person_year": float(
                event_totals[1].sum() / max(total_life_years.sum(), 1e-12)
            ),
            "non_ich_major_rate_per_person_year": float(
                event_totals[2].sum() / max(total_life_years.sum(), 1e-12)
            ),
            "ich_rate_per_person_year": float(
                event_totals[3].sum() / max(total_life_years.sum(), 1e-12)
            ),
            "total_bleed_rate_per_person_year": float(
                event_totals.sum() / max(total_life_years.sum(), 1e-12)
            ),
            "mean_pettersson_score": float(pettersson.mean()),
            "survival_probability": float(alive.mean()),
            "initial_patients": n,
            "alive_at_end": int(alive.sum()),
            "deaths_total": deaths_total,
            "deaths_background": deaths_background,
            "deaths_ich": deaths_ich,
            "all_cause_mortality_probability": deaths_total / n,
            "background_mortality_probability": deaths_background / n,
            "ich_mortality_probability": deaths_ich / n,
            "post_ich_sequela_prevalence": float(sequela.mean()),
            "post_ich_ever_count": post_ich_count,
            "post_ich_ever_probability": post_ich_count / n,
            "patients_with_surviving_ich_count": ich_survivors,
            "post_ich_probability_among_patients_with_surviving_ich": (
                post_ich_count / ich_survivors if ich_survivors else 0.0
            ),
            "mcse_cost_irr": (
                float(total_cost.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            ),
            "mcse_qaly": float(total_qaly.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        }
        patient_data = None
        if retain_patient_level:
            patient_data = {
                "patient_id": np.arange(n, dtype=np.int64),
                "total_cost_irr": total_cost,
                "total_qaly": total_qaly,
                "total_factor_iu": total_factor,
                "life_years": total_life_years,
                "joint_bleeds": event_totals[0],
                "non_major_non_joint_bleeds": event_totals[1],
                "non_ich_major_bleeds": event_totals[2],
                "ich_events": event_totals[3],
                "pettersson_score": pettersson,
                "post_ich_sequela": sequela,
                "ever_post_ich": ever_post_ich,
                "death_cycle": death_cycle,
                "death_age_years": death_age_years,
                "death_cause": np.array(
                    [DeathCause(int(code)).name.lower() for code in death_cause]
                ),
                "final_state": final_state,
            }

        result = SimulationResult(
            strategy=strategy,
            scenario_id=self.scenario_id,
            seed=self.seed,
            n_patients=n,
            n_cycles=n_cycles,
            summary=summary,
            state_counts=state_counts,
            mortality=mortality,
            patient_data=patient_data,
        )
        if trace is not None:
            trace.finish(result)
        return result

    def _acute_candidate(
        self, event: AcuteEvent, chronic: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if event in {AcuteEvent.JOINT_BLEED, AcuteEvent.NON_MAJOR_NON_JOINT}:
            candidate: NDArray[np.float64] = np.maximum(
                0.0, chronic - self.values["minor_bleed_decrement"]
            )
        elif event is AcuteEvent.NON_ICH_MAJOR:
            candidate = np.minimum(chronic, self.values["non_ich_major_utility_cap"])
        else:
            candidate = np.minimum(chronic, self.values["ich_acute_utility_cap"])
        return candidate
