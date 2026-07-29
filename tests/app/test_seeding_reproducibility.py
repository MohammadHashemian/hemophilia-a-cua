"""End-to-end seeded reproducibility tests for the worker entry points.

These tests exercise the same code path the PSA/OWSA notebooks hit when they
run ``worker_function`` (scalar) and ``worker_function_batch`` (vectorized),
and assert that two runs with the same ``context.simulation.environment.seed``
produce bit-for-bit identical outputs. This locks in the contract that the
provided seed is honoured across the full pipeline.
"""
import numpy as np
import pytest

from app.domain.enums import HealthStates, Regime
from app.domain.inputs import ModelInput
from app.domain.scenario import Scenario
from app.domain.worker import worker_function, worker_function_batch
from app.persistence.context import ModelContext
from engine.chains import Chain

# Eight states, in the order expected by ``build_transition_matrix`` and the
# vectorized reward registries.
STATES = [s.value for s in HealthStates]


def _make_chain():
    return Chain(name="main", states=STATES, matrix=np.eye(len(STATES)))


def _make_input(cycle: int = 100) -> ModelInput:
    """A small but non-trivial ModelInput so the worker actually exercises
    rewards and the mortality modifier."""
    return ModelInput(
        cycle=cycle,
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


def _make_scenario() -> Scenario:
    return Scenario(name="test on_demand bayesian", regime=Regime.ON_DEMAND)


@pytest.fixture
def context() -> ModelContext:
    return ModelContext.load()


def test_scalar_worker_function_is_reproducible_with_seed(context):
    """Two scalar runs with the same env seed must produce identical output."""
    chain = _make_chain()
    inputs = _make_input()
    scenario = _make_scenario()

    out_a = worker_function(chain, inputs, scenario, context, worker_id=0)
    out_b = worker_function(chain, inputs, scenario, context, worker_id=0)

    assert out_a.sequence == out_b.sequence
    assert out_a.absorbed_at == out_b.absorbed_at
    assert out_a.total_factor == pytest.approx(out_b.total_factor)
    assert out_a.total_qaly == pytest.approx(out_b.total_qaly)
    assert out_a.event_count == out_b.event_count


def test_scalar_worker_function_different_worker_ids_advance_rng(context):
    """Two scalar runs with the same env seed but different worker_ids must
    not accidentally produce the same draws (per-worker seeds must vary)."""
    chain = _make_chain()
    inputs = _make_input(cycle=200)
    scenario = _make_scenario()

    out_a = worker_function(chain, inputs, scenario, context, worker_id=0)
    out_b = worker_function(chain, inputs, scenario, context, worker_id=1)

    # At least one of sequence, absorbed_at, or event_count should differ
    # between the two workers because their per-worker seeds differ.
    assert (out_a.sequence != out_b.sequence) or (out_a.event_count != out_b.event_count)


def test_vectorized_worker_batch_is_reproducible_with_seed(context):
    """Vectorized batch: two runs with the same seed must be bit-for-bit identical."""
    chain = _make_chain()
    inputs = [_make_input(cycle=80) for _ in range(16)]
    scenario = _make_scenario()

    out_a = worker_function_batch(chain, inputs, scenario, context, worker_id=0)
    out_b = worker_function_batch(chain, inputs, scenario, context, worker_id=0)

    assert len(out_a) == len(out_b) == len(inputs)
    for a, b in zip(out_a, out_b, strict=True):
        assert a.sequence == b.sequence
        assert a.absorbed_at == b.absorbed_at
        assert a.event_count == b.event_count
        assert a.total_factor == pytest.approx(b.total_factor)
        assert a.total_qaly == pytest.approx(b.total_qaly)


def test_scalar_and_vectorized_paths_are_each_self_consistent(context):
    """Both paths must be individually seedable from the same env seed. The
    scalar and vectorized sampling algorithms are not bit-for-bit comparable
    (they use different np.random.Generator methods internally), so this test
    only checks that each path is self-consistent — not that the two paths
    produce identical sequences for the same seed."""
    chain = _make_chain()
    inp = _make_input(cycle=80)
    scenario = _make_scenario()

    scalar_a = worker_function(chain, inp, scenario, context, worker_id=0)
    scalar_b = worker_function(chain, inp, scenario, context, worker_id=0)
    vector_a = worker_function_batch(chain, [inp], scenario, context, worker_id=0)[0]
    vector_b = worker_function_batch(chain, [inp], scenario, context, worker_id=0)[0]

    # Self-consistency of the scalar path.
    assert scalar_a.sequence == scalar_b.sequence
    assert scalar_a.absorbed_at == scalar_b.absorbed_at
    # Self-consistency of the vectorized path.
    assert vector_a.sequence == vector_b.sequence
    assert vector_a.absorbed_at == vector_b.absorbed_at
