from dataclasses import replace
from threading import Lock
from time import sleep

import numpy as np
import polars as pl

from app.domain.enums import Regime
from app.domain.inputs import ModelInput
from app.domain.scenario import Scenario, ScenarioBundle
from app.notebook.psa.workflow import scenario_cache_fingerprint
from app.notebook.scenario_runner import _run_batch, run_scenarios_in_batches
from app.persistence.context import ModelContext
from engine.chains import Chain


def _input() -> ModelInput:
    return ModelInput(
        cycle=52,
        bleeding_rate=10.0,
        spontaneous_bleeding_rate=2.0,
        joint_bleeding_rate=7.9,
        intracranial_hemorrhage_rate=0.01,
        non_ich_major_bleeding_rate=0.09,
        ich_case_fatality=0.1,
        non_ich_case_fatality=0.0,
        baseline_age=1.0,
        weight_factor=1.0,
        benefits_discount_rate=0.0,
        healthy_utility=0.9,
        mild_arthropathy_utility=0.8,
        moderate_arthropathy_utility=0.7,
        severe_arthropathy_utility=0.6,
        spontaneous_bleeding_utility=0.5,
        joint_bleeding_utility=0.4,
        intracranial_hemorrhage_utility=0.2,
        non_ich_major_bleeding_utility=0.2,
        death_utility=0.0,
        per_unit_price=1.0,
        costs_discount_rate=0.0,
        prophylaxis_background_factor_consumption_per_kg=0.0,
        factor_consumption_per_spontaneous_bleeding_per_kg=1.0,
        factor_consumption_per_joint_bleeding_per_kg=1.0,
        factor_consumption_per_intracranial_hemorrhage_per_kg=1.0,
        factor_consumption_per_non_ich_major_bleeding_per_kg=1.0,
    )


def test_scenario_fingerprint_changes_for_small_input_change():
    context = ModelContext.load()
    scenario = Scenario(
        name="childhood on-demand bayesian",
        regime=Regime.ON_DEMAND,
    )
    original = ScenarioBundle(scenario=scenario, inputs=[_input()])
    changed = ScenarioBundle(
        scenario=scenario,
        inputs=[replace(_input(), per_unit_price=1.000000001)],
    )
    original_hash = scenario_cache_fingerprint(
        original,
        context,
        code_digest="same-code",
    )
    assert original_hash == scenario_cache_fingerprint(
        original,
        context,
        code_digest="same-code",
    )
    assert original_hash != scenario_cache_fingerprint(
        changed,
        context,
        code_digest="same-code",
    )
    assert original_hash != scenario_cache_fingerprint(
        original,
        context,
        code_digest="changed-code",
    )


def test_scenario_cache_reuses_only_matching_fingerprints(tmp_path, monkeypatch):
    context = ModelContext.load()
    names = [
        "childhood on-demand bayesian",
        "childhood prophylaxis bayesian",
    ]
    bundles = [
        ScenarioBundle(
            scenario=Scenario(
                name=name,
                regime=(
                    Regime.ON_DEMAND if "on-demand" in name else Regime.PROPHYLAXIS
                ),
            ),
            inputs=[_input()],
        )
        for name in names
    ]
    calls: list[str] = []

    def fake_run_batch(*, batch, **kwargs):
        calls.extend(bundle.scenario.name for bundle in batch)
        return [bundle.scenario.name for bundle in batch]

    def fake_build_df(results, context):
        return pl.DataFrame(
            {
                "scenario": results,
                "value": [1.0] * len(results),
            },
            schema={"scenario": pl.String, "value": pl.Float64},
        )

    monkeypatch.setattr(
        "app.notebook.scenario_runner._run_batch",
        fake_run_batch,
    )
    monkeypatch.setattr(
        "app.notebook.scenario_runner.build_df",
        fake_build_df,
    )
    chain = Chain(
        name="main",
        states=["healthy", "death"],
        matrix=np.eye(2),
    )
    common = dict(
        bundles=bundles,
        context=context,
        identity_chain=chain,
        worker_function=lambda: None,
        batch_size=2,
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
        engine="batch",
        batch_worker_function=lambda: None,
    )

    fingerprints = {names[0]: "a", names[1]: "b"}
    run_scenarios_in_batches(
        **common,
        scenario_fingerprints=fingerprints,
    )
    assert calls == names

    calls.clear()
    run_scenarios_in_batches(
        **common,
        scenario_fingerprints=fingerprints,
    )
    assert calls == []

    calls.clear()
    changed = {names[0]: "changed", names[1]: "b"}
    run_scenarios_in_batches(
        **common,
        scenario_fingerprints=changed,
    )
    assert calls == [names[0]]


def test_vectorized_scenarios_run_in_parallel_and_preserve_order():
    context = ModelContext.load()
    names = [
        "childhood on-demand bayesian",
        "childhood prophylaxis bayesian",
    ]
    bundles = [
        ScenarioBundle(
            scenario=Scenario(
                name=name,
                regime=(
                    Regime.ON_DEMAND if "on-demand" in name else Regime.PROPHYLAXIS
                ),
            ),
            inputs=[_input()],
        )
        for name in names
    ]
    lock = Lock()
    active = 0
    peak_active = 0

    def fake_worker(*, inputs, scenario, **kwargs):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        sleep(0.05)
        with lock:
            active -= 1
        return [scenario.name for _ in inputs]

    chain = Chain(
        name="main",
        states=["healthy", "death"],
        matrix=np.eye(2),
    )
    results = _run_batch(
        batch=bundles,
        context=context,
        identity_chain=chain,
        batch_worker_function=fake_worker,
        max_parallel_scenarios=2,
    )

    assert peak_active == 2
    assert [result.scenario for result in results] == names
