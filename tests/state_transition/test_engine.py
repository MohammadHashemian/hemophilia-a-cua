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
