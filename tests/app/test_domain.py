import numpy as np
import pytest

from app.domain.enums import ArthropathySeverity, HealthStates, Regime
from app.domain.modifiers import HemophiliaMortalityModifier
from app.domain.rewards.pettersson import pettersson_to_severity
from app.domain.rewards.scalar import (
    consumption,
    event_count,
    make_pettersson_score,
    utility,
    weight,
)
from app.domain.scenario import Scenario, ScenarioBundle
from engine.modifier import NoOpModifier


class TestEnums:
    def test_regime_values(self):
        assert Regime.ON_DEMAND.value == "on_demand"
        assert Regime.PROPHYLAXIS.value == "prophylaxis"

    def test_health_states(self):
        assert HealthStates.HEALTHY.value == "healthy"
        assert HealthStates.DEATH.value == "death"


class TestScenario:
    def test_basic_scenario(self):
        s = Scenario(name="test", regime=Regime.ON_DEMAND)
        assert s.name == "test"
        assert s.regime == Regime.ON_DEMAND

    def test_scenario_with_overrides(self):
        overrides = {"bleeding_rate": {"value": 15.0}}
        s = Scenario(name="test", regime=Regime.ON_DEMAND, overrides=overrides)
        assert s.overrides["bleeding_rate"]["value"] == 15.0

    def test_scenario_bundle(self):
        s = Scenario(name="test", regime=Regime.ON_DEMAND)
        bundle = ScenarioBundle(scenario=s, inputs=[1, 2, 3])
        assert bundle.scenario == s
        assert bundle.inputs == [1, 2, 3]


class TestMakePetterssonScore:
    def test_no_hemarthrosis(self):
        pet = make_pettersson_score(factor=1.0)
        assert pet(step=0, state="healthy", event_count=0) == 0

    def test_single_hemarthrosis(self):
        pet = make_pettersson_score(factor=1.0)
        assert pet(step=0, state="hemarthrosis", event_count=1) == 1

    def test_accumulation(self):
        pet = make_pettersson_score(factor=2.0)
        pet(step=0, state="hemarthrosis", event_count=2)
        result = pet(step=1, state="healthy", event_count=0)
        assert result == 1

    def test_capped_at_79(self):
        pet = make_pettersson_score(factor=1.0)
        score = pet(step=0, state="hemarthrosis", event_count=200)
        assert score == 79

    def test_only_hemarthrosis_counts(self):
        pet = make_pettersson_score(factor=1.0)
        pet(step=0, state="bleeding", event_count=5)
        assert pet(step=1, state="healthy", event_count=0) == 0

    def test_separate_instances_independent(self):
        pet1 = make_pettersson_score(factor=1.0)
        pet2 = make_pettersson_score(factor=1.0)
        pet1(step=0, state="hemarthrosis", event_count=5)
        pet2(step=0, state="hemarthrosis", event_count=10)
        assert pet1(step=1, state="healthy", event_count=0) == 5
        assert pet2(step=1, state="healthy", event_count=0) == 10


class TestPetterssonToSeverity:
    def test_healthy(self):
        assert pettersson_to_severity(0) == ArthropathySeverity.HEALTHY
        assert pettersson_to_severity(3) == ArthropathySeverity.HEALTHY

    def test_mild(self):
        assert pettersson_to_severity(4) == ArthropathySeverity.MILD
        assert pettersson_to_severity(26) == ArthropathySeverity.MILD

    def test_moderate(self):
        assert pettersson_to_severity(27) == ArthropathySeverity.MODERATE
        assert pettersson_to_severity(78) == ArthropathySeverity.MODERATE

    def test_severe(self):
        assert pettersson_to_severity(79) == ArthropathySeverity.SEVERE


class TestEventCount:
    def test_major_bleeding_always_one(self):
        count = event_count(step=0, state="intracranial_hemorrhage", const={"lam_bleed": 1.0, "lam_joint": 1.0}, rng=np.random.default_rng(42))
        assert count == 1

    def test_healthy_returns_zero(self):
        count = event_count(step=0, state="healthy", const={"lam_bleed": 1.0, "lam_joint": 1.0}, rng=np.random.default_rng(42))
        assert count == 0

    def test_death_returns_zero(self):
        count = event_count(step=0, state="death", const={"lam_bleed": 1.0, "lam_joint": 1.0}, rng=np.random.default_rng(42))
        assert count == 0

    def test_bleeding_returns_positive(self):
        count = event_count(step=0, state="bleeding", const={"lam_bleed": 5.0, "lam_joint": 1.0}, rng=np.random.default_rng(42))
        assert count >= 1

    def test_hemarthrosis_returns_positive(self):
        count = event_count(step=0, state="hemarthrosis", const={"lam_bleed": 1.0, "lam_joint": 5.0}, rng=np.random.default_rng(42))
        assert count >= 1

    def test_higher_lambda_more_events_on_average(self):
        rng = np.random.default_rng(42)
        counts_low = [event_count(0, "bleeding", const={"lam_bleed": 0.5, "lam_joint": 0.5}, rng=rng) for _ in range(1000)]
        rng = np.random.default_rng(42)
        counts_high = [event_count(0, "bleeding", const={"lam_bleed": 5.0, "lam_joint": 0.5}, rng=rng) for _ in range(1000)]
        assert np.mean(counts_high) > np.mean(counts_low)


class TestWeight:
    def test_weight_positive(self):
        w = weight(step=0, state="healthy", const={"baseline_age_weeks": 0}, inputs=type("obj", (object,), {"weight_factor": 1.0})())
        assert w > 0

    def test_weight_scales_with_factor(self):
        inputs_low = type("obj", (object,), {"weight_factor": 1.0})()
        inputs_high = type("obj", (object,), {"weight_factor": 2.0})()
        w1 = weight(step=1000, state="healthy", const={"baseline_age_weeks": 0}, inputs=inputs_low)
        w2 = weight(step=1000, state="healthy", const={"baseline_age_weeks": 0}, inputs=inputs_high)
        # cal_body_weight rounds to 2 decimals, so allow one rounding unit
        assert abs(w2 - 2 * w1) <= 0.02


class TestConsumption:
    def test_death_no_consumption(self):
        inputs = type("obj", (object,), {"prophylaxis_background_factor_consumption_per_kg": 0, "factor_consumption_per_spontaneous_bleeding_per_kg": 10, "factor_consumption_per_joint_bleeding_per_kg": 20, "factor_consumption_per_life_threatening_bleeding_per_kg": 50})()
        c = consumption(step=0, state="death", regime=Regime.ON_DEMAND, weight=70.0, event_count=0, inputs=inputs)
        assert c == 0.0

    def test_prophylaxis_background_consumption(self):
        inputs = type("obj", (object,), {"prophylaxis_background_factor_consumption_per_kg": 5, "factor_consumption_per_spontaneous_bleeding_per_kg": 10, "factor_consumption_per_joint_bleeding_per_kg": 20, "factor_consumption_per_life_threatening_bleeding_per_kg": 50})()
        c = consumption(step=0, state="healthy", regime=Regime.PROPHYLAXIS, weight=70.0, event_count=0, inputs=inputs)
        assert abs(c - 70.0 * 5) < 1e-6

    def test_bleeding_consumption(self):
        inputs = type("obj", (object,), {"prophylaxis_background_factor_consumption_per_kg": 0, "factor_consumption_per_spontaneous_bleeding_per_kg": 10, "factor_consumption_per_joint_bleeding_per_kg": 20, "factor_consumption_per_life_threatening_bleeding_per_kg": 50})()
        c = consumption(step=0, state="bleeding", regime=Regime.ON_DEMAND, weight=70.0, event_count=3, inputs=inputs)
        assert abs(c - 70.0 * 10 * 3) < 1e-6

    def test_ich_consumption(self):
        inputs = type("obj", (object,), {"prophylaxis_background_factor_consumption_per_kg": 0, "factor_consumption_per_spontaneous_bleeding_per_kg": 10, "factor_consumption_per_joint_bleeding_per_kg": 20, "factor_consumption_per_intracranial_hemorrhage_per_kg": 50})()
        c = consumption(step=0, state="intracranial_hemorrhage", regime=Regime.ON_DEMAND, weight=70.0, event_count=0, inputs=inputs)
        assert abs(c - 70.0 * 50) < 1e-6


def _make_utility_consts(healthy=1.0, mild=0.9, moderate=0.7, severe=0.5, bleeding=0.6, hemarthrosis=0.4, lt_bleeding=0.3, death=0.0):
    utils = type("obj", (object,), {
        "healthy": healthy, "mild_arthropathy": mild, "moderate_arthropathy": moderate,
        "severe_arthropathy": severe, "bleeding": bleeding, "hemarthrosis": hemarthrosis,
        "lt_bleeding": lt_bleeding, "death": death,
    })
    return {
        "utilities": utils,
        "threshold_mild": 10,
        "threshold_moderate": 30,
        "threshold_max": 50,
        "weekly_discount": 0.0,
    }


class TestUtility:
    def test_healthy_no_discount(self):
        consts = _make_utility_consts()
        u = utility(step=0, state="healthy", const=consts, pettersson_score=0)
        assert abs(u - 1.0 / 52) < 1e-6

    def test_severe_arthropathy_lowers_utility(self):
        consts = _make_utility_consts()
        u = utility(step=0, state="healthy", const=consts, pettersson_score=60)
        assert u < 1.0 / 52

    def test_bleeding_applies_acute_decrement(self):
        consts = _make_utility_consts()
        u_bleeding = utility(step=0, state="bleeding", const=consts, pettersson_score=0)
        u_healthy = utility(step=0, state="healthy", const=consts, pettersson_score=0)
        assert u_bleeding < u_healthy


class TestNoOpModifier:
    def test_returns_copy(self):
        modifier = NoOpModifier()
        base = np.array([0.9, 0.1, 0.0])
        adjusted = modifier.adjust_transition(
            base_probs=base, current_state="A", current_chain_name="X", step=0, states=["A", "B", "C"]
        )
        assert np.array_equal(adjusted, base)
        assert id(adjusted) != id(base)


class TestHemophiliaMortalityModifier:
    @pytest.fixture
    def mortality_func(self):
        def func(age):
            if age < 30:
                return 1.0
            elif age < 60:
                return 5.0
            else:
                return 20.0
        return func

    def test_applies_on_yearly_boundary(self, mortality_func):
        modifier = HemophiliaMortalityModifier(
            mortality_func=mortality_func, start_age=25, dead_state="Dead", enable_logger=False,
        )
        base = np.array([0.9, 0.1, 0.0])
        adjusted = modifier.adjust_transition(
            base_probs=base, current_state="Healthy", current_chain_name="X", step=52, states=["Healthy", "Sick", "Dead"],
        )
        assert np.isclose(adjusted.sum(), 1.0, rtol=1e-8)
        assert adjusted[2] > base[2]

    def test_skips_non_target_states(self, mortality_func):
        modifier = HemophiliaMortalityModifier(
            mortality_func=mortality_func, start_age=1, adjust_only_states=["Healthy"], enable_logger=False,
        )
        base = np.array([0.8, 0.2, 0.0])
        adjusted = modifier.adjust_transition(
            base_probs=base, current_state="Sick", current_chain_name="X", step=52, states=["Healthy", "Sick", "Dead"],
        )
        assert np.array_equal(adjusted, base)

    def test_only_applies_yearly(self, mortality_func):
        modifier = HemophiliaMortalityModifier(
            mortality_func=mortality_func, start_age=1, enable_logger=False,
        )
        base = np.array([0.95, 0.05, 0.0])
        adjusted = modifier.adjust_transition(
            base_probs=base, current_state="Healthy", current_chain_name="X", step=10, states=["Healthy", "Sick", "Dead"],
        )
        assert np.allclose(adjusted, base)
