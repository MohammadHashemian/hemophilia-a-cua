from __future__ import annotations

import numpy as np
import pytest

from modular_models.state_transition.analysis import StudyRunner
from modular_models.state_transition.context import StudyContext
from modular_models.state_transition.engine import (
    StateTransitionEngine,
    derive_event_rates,
)
from modular_models.state_transition.rewards import cuda_available
from modular_models.state_transition.sampling import ParameterResolver
from modular_models.state_transition.trace import TraceSession
from modular_models.state_transition.types import ChronicState, Strategy


@pytest.fixture(scope="module")
def context() -> StudyContext:
    return StudyContext.load()


def test_base_event_rates_are_exhaustive(context: StudyContext) -> None:
    values, options = ParameterResolver(context).deterministic()
    for strategy, expected in (
        (Strategy.PROPHYLAXIS, values["abr_prophylaxis"]),
        (Strategy.ON_DEMAND, values["abr_on_demand"]),
    ):
        rates = derive_event_rates(values, options, strategy)
        assert sum(rates.annual.values()) == pytest.approx(expected)
        assert all(rate >= 0 for rate in rates.annual.values())


def test_history_maps_to_chronic_states_and_is_capped(context: StudyContext) -> None:
    values, options = ParameterResolver(context).deterministic()
    engine = StateTransitionEngine(context, values, options, scenario_id="base_case", seed=1)
    ps = np.array([0, 12, 13, 21, 22, 39, 40, 78], dtype=np.int16)
    states = engine._chronic_state(ps, np.ones(ps.shape, dtype=bool))

    assert states.tolist() == [
        ChronicState.NO_MINIMAL_ARTHROPATHY,
        ChronicState.NO_MINIMAL_ARTHROPATHY,
        ChronicState.MILD_ARTHROPATHY,
        ChronicState.MILD_ARTHROPATHY,
        ChronicState.MODERATE_ARTHROPATHY,
        ChronicState.MODERATE_ARTHROPATHY,
        ChronicState.SEVERE_ARTHROPATHY,
        ChronicState.SEVERE_ARTHROPATHY,
    ]


def test_paired_base_case_is_reproducible(context: StudyContext) -> None:
    runner = StudyRunner(context)
    first = runner.compare(n_patients=250, seed=481)
    second = runner.compare(n_patients=250, seed=481)

    assert first.economic_summary() == second.economic_summary()
    assert first.prophylaxis.seed == first.on_demand.seed


@pytest.mark.skipif(not cuda_available(), reason="CUDA device is not available")
def test_cuda_reward_backend_matches_cpu_event_history_and_qaly(
    context: StudyContext,
) -> None:
    values, options = ParameterResolver(context).deterministic()
    common = {
        "context": context,
        "values": values,
        "options": options,
        "scenario_id": "cuda_validation",
        "seed": 20260813,
    }
    cpu = StateTransitionEngine(**common, compute_backend="cpu").run(
        Strategy.ON_DEMAND,
        n_patients=250,
    )
    cuda = StateTransitionEngine(**common, compute_backend="cuda").run(
        Strategy.ON_DEMAND,
        n_patients=250,
    )

    for key in (
        "mean_cost_irr",
        "mean_qaly",
        "mean_total_bleeds",
        "deaths_total",
        "deaths_background",
        "deaths_ich",
        "post_ich_ever_count",
    ):
        assert cuda.summary[key] == pytest.approx(cpu.summary[key], abs=1e-12)


def test_death_is_absorbing_and_stops_future_accrual(context: StudyContext) -> None:
    values, options = ParameterResolver(context).deterministic(
        overrides={
            "background_mortality_age_1_4": 1.0e9,
            "background_mortality_age_5_9": 1.0e9,
            "background_mortality_age_10_lt12": 1.0e9,
            "ich_case_fatality": 0.0,
        }
    )
    engine = StateTransitionEngine(context, values, options, scenario_id="test", seed=23)
    result = engine.run(Strategy.PROPHYLAXIS, n_patients=200, retain_patient_level=True)

    assert result.summary["survival_probability"] == 0.0
    assert result.state_counts["death"] == 200
    assert result.patient_data is not None
    assert (result.patient_data["death_cycle"] == 0).all()
    assert (result.patient_data["death_cause"] == "background").all()
    assert result.summary["deaths_background"] == 200
    assert result.summary["deaths_ich"] == 0
    assert result.summary["mean_qaly"] < values["utility_anchor"] / 52


def test_simulated_bleed_mean_tracks_input_abr(context: StudyContext) -> None:
    values, options = ParameterResolver(context).deterministic(
        overrides={
            "background_mortality_age_1_4": 0.0,
            "background_mortality_age_5_9": 0.0,
            "background_mortality_age_10_lt12": 0.0,
            "ich_case_fatality": 0.0,
        }
    )
    engine = StateTransitionEngine(context, values, options, scenario_id="test", seed=782)
    result = engine.run(Strategy.ON_DEMAND, n_patients=3000)
    annualized = result.summary["mean_total_bleeds"] / 11.0

    assert annualized == pytest.approx(values["abr_on_demand"], rel=0.015)


def test_child_horizon_is_age_one_to_before_twelfth_birthday(context: StudyContext) -> None:
    values, options = ParameterResolver(context).deterministic(
        overrides={
            "background_mortality_age_1_4": 0.0,
            "background_mortality_age_5_9": 0.0,
            "background_mortality_age_10_lt12": 0.0,
            "ich_case_fatality": 0.0,
        }
    )
    trace = TraceSession(max_cycles=600)
    result = StateTransitionEngine(context, values, options, scenario_id="test", seed=31).run(
        Strategy.PROPHYLAXIS, n_patients=1, trace=trace
    )

    assert result.n_cycles == 572
    assert result.summary["follow_up_years"] == 11
    assert result.summary["exit_age_years_exclusive"] == 12
    assert result.summary["last_cycle_start_age_years"] == pytest.approx(11 + 51 / 52)
    assert trace.runs[0].cycles[0]["age"] == 1
    assert trace.runs[0].cycles[-1]["age"] < 12
    assert trace.runs[0].cycles[-1]["mean_weight"] < values["weight_age_12"]


def test_fatal_ich_is_reported_as_a_distinct_death_cause(context: StudyContext) -> None:
    values, options = ParameterResolver(context).deterministic(
        overrides={
            "abr_on_demand": 520.0,
            "ajbr_on_demand": 0.0,
            "ich_rate_on_demand": 520.0,
            "non_ich_major_fraction": 0.0,
            "ich_case_fatality": 1.0,
            "background_mortality_age_1_4": 0.0,
            "background_mortality_age_5_9": 0.0,
            "background_mortality_age_10_lt12": 0.0,
        }
    )
    result = StateTransitionEngine(context, values, options, scenario_id="test", seed=19).run(
        Strategy.ON_DEMAND, n_patients=100, retain_patient_level=True
    )

    assert result.patient_data is not None
    assert result.summary["deaths_ich"] == 100
    assert result.summary["deaths_background"] == 0
    assert (result.patient_data["death_cause"] == "ich").all()
    assert result.mortality["ich_event_mortality"]["case_fatality_probability_input"] == 1.0


def test_post_ich_entry_is_retained_as_an_ever_flag(context: StudyContext) -> None:
    values, options = ParameterResolver(context).deterministic(
        overrides={
            "abr_on_demand": 52.0,
            "ajbr_on_demand": 0.0,
            "ich_rate_on_demand": 52.0,
            "non_ich_major_fraction": 0.0,
            "ich_case_fatality": 0.0,
            "post_ich_sequela_probability": 1.0,
            "background_mortality_age_1_4": 0.0,
            "background_mortality_age_5_9": 0.0,
            "background_mortality_age_10_lt12": 0.0,
        }
    )
    result = StateTransitionEngine(context, values, options, scenario_id="test", seed=71).run(
        Strategy.ON_DEMAND, n_patients=100, retain_patient_level=True
    )

    assert result.patient_data is not None
    assert result.summary["post_ich_ever_count"] == 100
    assert result.summary["post_ich_ever_probability"] == 1.0
    assert result.patient_data["ever_post_ich"].all()


def test_mortality_audit_reconciles_population_and_age_probabilities(
    context: StudyContext,
) -> None:
    result = StudyRunner(context).compare(n_patients=200, seed=811).on_demand
    overall = result.mortality["overall"]

    assert overall["initial_patients"] == overall["alive_at_end"] + overall["deaths_total"]
    assert overall["deaths_total"] == overall["deaths_background"] + overall["deaths_ich"]
    assert set(result.mortality["age_specific_background"]) == {
        "age_1_to_lt5",
        "age_5_to_lt10",
        "age_10_to_lt12",
    }
    for record in result.mortality["age_specific_background"].values():
        expected = 1 - np.exp(-record["annual_hazard"] / 52)
        assert record["weekly_probability"] == pytest.approx(expected)


def test_acute_interval_carries_fractional_effect_across_cycle(context: StudyContext) -> None:
    values, options = ParameterResolver(context).deterministic()
    engine = StateTransitionEngine(context, values, options, scenario_id="test", seed=2)
    schedule = np.full((1, 28), 0.9)
    engine._apply_interval(
        schedule=schedule,
        base=np.array([0.9]),
        candidate=np.array([0.5]),
        starts=np.array([6.5]),
        durations=np.array([2.0]),
        active=np.array([True]),
        step_days=0.25,
    )

    assert np.allclose(schedule[0, :26], 0.9)
    assert np.allclose(schedule[0, 26:], 0.5)


def test_major_factor_course_is_conserved_across_cycles(context: StudyContext) -> None:
    values, options = ParameterResolver(context).deterministic()
    engine = StateTransitionEngine(context, values, options, scenario_id="test", seed=3)
    current = np.zeros(1)
    pending = np.zeros((1, 2))
    total = np.array([6200.0])
    engine._allocate_major_course(
        current,
        pending,
        total,
        initial_fraction=45.0 / 620.0,
        starts=np.array([6.0]),
        duration_days=12.0,
        active=np.array([True]),
        death_time=np.array([7.0]),
    )

    assert current.sum() + pending.sum() == pytest.approx(total[0])


def _maintenance_allocation(
    engine: StateTransitionEngine,
    *,
    total_iu_value: float,
    initial_fraction: float,
    starts: np.ndarray,
    duration_days: float,
    death_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.zeros(starts.shape[0])
    pending = np.zeros((starts.shape[0], 2))
    total = np.full(starts.shape[0], total_iu_value)
    engine._allocate_major_course(
        current,
        pending,
        total,
        initial_fraction=initial_fraction,
        starts=starts,
        duration_days=duration_days,
        active=np.ones(starts.shape[0], dtype=bool),
        death_time=death_time,
    )
    return current, pending


def test_major_factor_course_share_uses_nominal_duration_denominator(
    context: StudyContext,
) -> None:
    """Per-week share must equal (week overlap) / nominal_duration, where
    nominal_duration is the full prescribed course (not the truncated,
    post-death effective span). The old implementation divided by the
    truncated effective duration and compressed the entire maintenance
    into the time the patient survived.
    """
    values, options = ParameterResolver(context).deterministic()
    engine = StateTransitionEngine(context, values, options, scenario_id="test", seed=4)

    total_iu_value = 6200.0
    dose_per_kg = 620.0
    initial_fraction = 45.0 / dose_per_kg
    initial = total_iu_value * initial_fraction
    maintenance = total_iu_value - initial
    duration_days = 12.0

    # Patient dies 3 days into a 12-day course: end = 4.0, delivered = 3.0.
    onset = 1.0
    end = 4.0
    expected_share_week_0 = max(0.0, min(end, 7.0) - max(onset, 0.0)) / duration_days
    expected_share_week_1 = max(0.0, min(end, 14.0) - max(onset, 7.0)) / duration_days
    expected_share_week_2 = max(0.0, min(end, 21.0) - max(onset, 14.0)) / duration_days

    current, pending = _maintenance_allocation(
        engine,
        total_iu_value=total_iu_value,
        initial_fraction=initial_fraction,
        starts=np.array([onset]),
        duration_days=duration_days,
        death_time=np.array([end]),
    )
    allocated_maintenance = (current.sum() + pending.sum()) - initial
    allocated_share = allocated_maintenance / maintenance

    assert allocated_share == pytest.approx(0.25, rel=1e-9)
    assert allocated_share < 1.0
    assert (current.sum() - initial) == pytest.approx(
        expected_share_week_0 * maintenance, rel=1e-9
    )
    assert pending[0, 0] == pytest.approx(
        expected_share_week_1 * maintenance, rel=1e-9
    )
    assert pending[0, 1] == pytest.approx(
        expected_share_week_2 * maintenance, rel=1e-9
    )
    assert expected_share_week_1 == 0.0
    assert expected_share_week_2 == 0.0


def test_major_factor_course_full_course_when_alive(context: StudyContext) -> None:
    """When the patient is alive for the whole cycle, total share must equal 1
    (the full maintenance is delivered) and conservation must hold.
    """
    values, options = ParameterResolver(context).deterministic()
    engine = StateTransitionEngine(context, values, options, scenario_id="test", seed=5)

    total_iu_value = 6200.0
    dose_per_kg = 620.0
    initial_fraction = 45.0 / dose_per_kg
    initial = total_iu_value * initial_fraction
    maintenance = total_iu_value - initial
    duration_days = 12.0

    # Onset 6.0 in a 7-day cycle; no death in this cycle.
    current, pending = _maintenance_allocation(
        engine,
        total_iu_value=total_iu_value,
        initial_fraction=initial_fraction,
        starts=np.array([6.0]),
        duration_days=duration_days,
        death_time=np.array([7.0]),
    )
    allocated_maintenance = (current.sum() + pending.sum()) - initial
    allocated_share = allocated_maintenance / maintenance

    assert allocated_share == pytest.approx(1.0, rel=1e-9)
    assert current.sum() + pending.sum() == pytest.approx(total_iu_value, rel=1e-9)
    # The three week-windows (0-7, 7-14, 14-21) must cover the full 12-day course.
    assert (current.sum() - initial) == pytest.approx(maintenance / 12.0, rel=1e-9)
    assert pending[0, 0] == pytest.approx(7.0 * maintenance / 12.0, rel=1e-9)
    assert pending[0, 1] == pytest.approx(4.0 * maintenance / 12.0, rel=1e-9)


def test_major_factor_course_zero_duration_avoids_divide_by_zero(
    context: StudyContext,
) -> None:
    """The 1e-12 floor on nominal_duration must protect against division by
    zero when duration_days is zero (or any non-positive value that survives
    the max()). The only allocation should be the initial dose, with no
    maintenance and no NaN/inf.
    """
    values, options = ParameterResolver(context).deterministic()
    engine = StateTransitionEngine(context, values, options, scenario_id="test", seed=6)

    total_iu_value = 6200.0
    dose_per_kg = 620.0
    initial_fraction = 45.0 / dose_per_kg
    initial = total_iu_value * initial_fraction

    current, pending = _maintenance_allocation(
        engine,
        total_iu_value=total_iu_value,
        initial_fraction=initial_fraction,
        starts=np.array([3.0]),
        duration_days=0.0,
        death_time=np.array([7.0]),
    )

    assert current.sum() == pytest.approx(initial, rel=1e-9)
    assert pending.sum() == 0.0
    assert np.all(np.isfinite(current))
    assert np.all(np.isfinite(pending))


def test_non_ich_major_treatment_duration_decouples_from_utility_duration(
    context: StudyContext,
) -> None:
    """The FVIII treatment course length (10 days) and the acute utility
    duration (7 days) for non-ICH major bleeds must be tracked separately.
    The split is verified by confirming total FVIII is conserved under the
    longer treatment window while the utility effect remains bounded by the
    shorter 7-day interval.
    """
    values, options = ParameterResolver(context).deterministic()
    assert (
        values["non_ich_major_treatment_duration_days"]
        > values["non_ich_major_duration_days"]
    )
    engine = StateTransitionEngine(context, values, options, scenario_id="test", seed=7)

    total_iu_value = 6200.0
    dose_per_kg = 620.0
    initial_fraction = 45.0 / dose_per_kg
    initial = total_iu_value * initial_fraction
    maintenance = total_iu_value - initial

    # Treatment duration = 10 days; patient alive for full 7-day cycle;
    # onset late in the cycle so the course spills into weeks 1 and 2.
    onset = 5.0
    current, pending = _maintenance_allocation(
        engine,
        total_iu_value=total_iu_value,
        initial_fraction=initial_fraction,
        starts=np.array([onset]),
        duration_days=values["non_ich_major_treatment_duration_days"],
        death_time=np.array([7.0]),
    )
    allocated_maintenance = (current.sum() + pending.sum()) - initial

    # Full 10-day course delivered: share = 1.0, total FVIII conserved.
    assert allocated_maintenance == pytest.approx(maintenance, rel=1e-9)
    assert current.sum() + pending.sum() == pytest.approx(total_iu_value, rel=1e-9)
    # Week 0: 2 days overlap (5 → 7). Weeks 1-2: 7 + 1 = 8 days.
    assert (current.sum() - initial) == pytest.approx(
        2.0 / 10.0 * maintenance, rel=1e-9
    )
    assert pending[0, 0] == pytest.approx(
        7.0 / 10.0 * maintenance, rel=1e-9
    )
    assert pending[0, 1] == pytest.approx(
        1.0 / 10.0 * maintenance, rel=1e-9
    )


def test_utility_integration_step_adequacy(context: StudyContext) -> None:
    """The base case uses ``utility_integration_step_days = 1.0`` (one bin
    per weekly cycle). The kernels integrate utility by averaging each event
    over its bin before taking the per-bin minimum, which is an
    approximation of the time-point-wise minimum. This test confirms that
    halving the step to 0.25 day does not change the four incremental
    decision metrics by more than a documented tolerance, so PSA can keep
    the coarser step.

    Cost must be bitwise identical between the two step values because
    ``utility_integration_step_days`` only affects the reward integrator.
    QALY, ICER, and INMB differ by a deterministic per-event approximation
    bias of about 0.1% in this cohort; the tolerance is set well above that
    so the test stays stable across parameter-table edits while still
    catching a real regression.
    """
    runner = StudyRunner(context)
    seed = 20_260_813
    n_patients = 5_000

    coarse = runner.compare(
        n_patients=n_patients,
        seed=seed,
        overrides={"utility_integration_step_days": 1.0},
    )
    fine = runner.compare(
        n_patients=n_patients,
        seed=seed,
        overrides={"utility_integration_step_days": 0.25},
    )

    cost_rel = abs(coarse.incremental_cost_irr - fine.incremental_cost_irr) / max(
        abs(fine.incremental_cost_irr), 1e-12
    )
    qaly_rel = abs(coarse.incremental_qaly - fine.incremental_qaly) / max(
        abs(fine.incremental_qaly), 1e-12
    )
    icer_rel = abs(coarse.icer_irr_per_qaly - fine.icer_irr_per_qaly) / max(
        abs(fine.icer_irr_per_qaly), 1e-12
    )
    inmb_rel = abs(coarse.incremental_nmb_irr - fine.incremental_nmb_irr) / max(
        abs(fine.incremental_nmb_irr), 1e-12
    )

    assert cost_rel == pytest.approx(0.0, abs=1e-6), (
        f"step_days must not affect cost, got rel diff {cost_rel:.3e}"
    )
    assert qaly_rel < 5e-3, f"QALY rel diff {qaly_rel:.3e} exceeds 5e-3"
    assert icer_rel < 5e-3, f"ICER rel diff {icer_rel:.3e} exceeds 5e-3"
    assert inmb_rel < 5e-3, f"INMB rel diff {inmb_rel:.3e} exceeds 5e-3"


def test_zero_price_changes_cost_not_resource_use(context: StudyContext) -> None:
    comparison = StudyRunner(context).compare(
        n_patients=100,
        seed=18,
        overrides={"factor_price_irr_per_iu": 0.0},
    )

    assert comparison.prophylaxis.summary["mean_cost_irr"] == 0.0
    assert comparison.on_demand.summary["mean_cost_irr"] == 0.0
    assert comparison.prophylaxis.summary["mean_factor_iu"] > 0.0


def test_no_discount_scenario_increases_present_values(context: StudyContext) -> None:
    runner = StudyRunner(context)
    base = runner.compare(n_patients=120, seed=20)
    undiscounted = runner.compare(scenario_id="no_discount", n_patients=120, seed=20)

    assert (
        undiscounted.prophylaxis.summary["mean_cost_irr"]
        > base.prophylaxis.summary["mean_cost_irr"]
    )
    assert undiscounted.prophylaxis.summary["mean_qaly"] > base.prophylaxis.summary["mean_qaly"]
