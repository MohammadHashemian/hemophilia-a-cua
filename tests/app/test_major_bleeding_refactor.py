import numpy as np

from app.analysis.psa.parameter_resolver import ParameterResolver
from app.domain.enums import HealthStates
from app.domain.inputs import ModelInput
from app.domain.transition_builder import AgeBasedMortalityModifier, build_transition_matrix
from app.persistence.schemas.mortality import MortalityFile


def _input(**overrides) -> ModelInput:
    values = dict(
        cycle=52,
        bleeding_rate=20.0,
        spontaneous_bleeding_rate=4.962725,
        joint_bleeding_rate=14.888175,
        intracranial_hemorrhage_rate=0.010,
        non_ich_major_bleeding_rate=0.1491,
        ich_case_fatality=0.10,
        non_ich_case_fatality=0.0,
        baseline_age=1.0,
        weight_factor=1.0,
        benefits_discount_rate=0.0,
        healthy_utility=0.915,
        mild_arthropathy_utility=0.85,
        moderate_arthropathy_utility=0.78,
        severe_arthropathy_utility=0.68,
        spontaneous_bleeding_utility=0.6,
        joint_bleeding_utility=0.5,
        intracranial_hemorrhage_utility=0.25,
        non_ich_major_bleeding_utility=0.25,
        death_utility=0.0,
        per_unit_price=58_000.0,
        costs_discount_rate=0.0,
        prophylaxis_background_factor_consumption_per_kg=0.0,
        factor_consumption_per_spontaneous_bleeding_per_kg=120.0,
        factor_consumption_per_joint_bleeding_per_kg=60.0,
        factor_consumption_per_intracranial_hemorrhage_per_kg=550.0,
        factor_consumption_per_non_ich_major_bleeding_per_kg=550.0,
    )
    values.update(overrides)
    return ModelInput(**values)


def test_health_states_are_split_and_old_ltb_state_is_removed():
    states = [state.value for state in HealthStates]
    assert "intracranial_hemorrhage" in states
    assert "non_ich_major_bleeding" in states
    assert "lt_bleeding" not in states


def test_non_ich_is_removed_from_abr_but_ich_is_absolute():
    resolved = ParameterResolver.resolve_samples(
        {
            "bleeding_rate": np.array([20.0]),
            "joint_bleeding_fraction": np.array([0.75]),
            "gi_neck_bleeding_fraction": np.array([95 / 20_295]),
            "iliopsoas_bleeding_fraction": np.array([9 / 3_244]),
            "intracranial_hemorrhage_rate": np.array([0.010]),
        }
    )
    routine = (
        resolved["spontaneous_bleeding_rate"]
        + resolved["joint_bleeding_rate"]
    )
    assert np.allclose(
        routine + resolved["non_ich_major_bleeding_rate"], [20.0]
    )
    assert np.allclose(resolved["intracranial_hemorrhage_rate"], [0.010])


def test_transition_rows_are_stochastic_and_death_is_absorbing():
    states = [state.value for state in HealthStates]
    matrix = build_transition_matrix(_input(), states)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    death = states.index("death")
    assert np.array_equal(matrix[death], np.eye(len(states))[death])


def test_ordinary_event_rows_reuse_fresh_weekly_distribution():
    states = [state.value for state in HealthStates]
    matrix = build_transition_matrix(_input(), states)
    no_bleeding = states.index(HealthStates.NO_BLEEDING.value)
    bleeding = states.index(HealthStates.BLEEDING.value)
    hemarthrosis = states.index(HealthStates.HEMARTHROSIS.value)
    assert np.allclose(matrix[bleeding], matrix[no_bleeding])
    assert np.allclose(matrix[hemarthrosis], matrix[no_bleeding])
    assert matrix[bleeding, bleeding] > 0.0
    assert matrix[hemarthrosis, hemarthrosis] > 0.0


def test_acute_rows_apply_case_fatality_then_next_week_risks():
    states = [state.value for state in HealthStates]
    matrix = build_transition_matrix(
        _input(ich_case_fatality=0.10, non_ich_case_fatality=0.02), states
    )
    death = states.index("death")
    no_bleeding = states.index(HealthStates.NO_BLEEDING.value)
    ich = states.index("intracranial_hemorrhage")
    non_ich = states.index("non_ich_major_bleeding")
    assert np.isclose(matrix[ich, death], 0.10)
    assert np.isclose(matrix[non_ich, death], 0.02)
    assert np.isclose(
        matrix[ich, no_bleeding],
        0.90 * matrix[no_bleeding, no_bleeding],
    )
    assert np.isclose(
        matrix[non_ich, no_bleeding],
        0.98 * matrix[no_bleeding, no_bleeding],
    )
    for destination in range(len(states)):
        if destination != death:
            assert np.isclose(
                matrix[ich, destination],
                0.90 * matrix[no_bleeding, destination],
            )
            assert np.isclose(
                matrix[non_ich, destination],
                0.98 * matrix[no_bleeding, destination],
            )


def test_living_rows_use_one_competing_risk_event_mass():
    states = [state.value for state in HealthStates]
    inp = _input()
    matrix = build_transition_matrix(inp, states)
    no_bleeding = states.index(HealthStates.NO_BLEEDING.value)
    total_annual_hazard = (
        inp.spontaneous_bleeding_rate
        + inp.joint_bleeding_rate
        + inp.intracranial_hemorrhage_rate
        + inp.non_ich_major_bleeding_rate
    )
    assert np.isclose(
        matrix[no_bleeding, no_bleeding],
        np.exp(-total_annual_hazard / 52),
    )


def test_background_mortality_combines_with_ich_fatality():
    states = [state.value for state in HealthStates]
    matrix = build_transition_matrix(_input(ich_case_fatality=0.10), states)
    modifier = AgeBasedMortalityModifier(
        MortalityFile(
            use_age_specific=True,
            source={"name": "test"},
            age_specific={"1-4": 0.01},
            crude_annual_rate=0.01,
        ),
        baseline_age=1,
    )
    ich = states.index("intracranial_hemorrhage")
    death = states.index("death")
    adjusted = modifier.adjust_transition(
        matrix[ich], "intracranial_hemorrhage", "main", 0, states
    )
    weekly_background = 1 - (1 - 0.01) ** (1 / 52)
    expected = 1 - (1 - 0.10) * (1 - weekly_background)
    assert np.isclose(adjusted[death], expected)
