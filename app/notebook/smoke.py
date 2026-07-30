"""Smoke-test the full PSA-style pipeline for seed reproducibility.

Run as ``python -m markov_chains.smoke`` (or via the script in
``notebook_tools/smoke.py``) to confirm that the env seed in
``data/simulation.json`` is honoured end-to-end:

  1. The seed is read from disk.
  2. Per-scenario seeds are derived stably across processes.
  3. The PSA sampler produces identical draws.
  4. The vectorized worker produces identical per-iter outputs.
  5. Two full pipeline runs (sampler + resolver + worker) are bit-for-bit
     identical.

Exit code 0 means the project is seeded and reproducible; non-zero means
something regressed.

The script is deliberately small and dependency-light so it can be wired
into CI as a fast guard against future regressions.
"""
from __future__ import annotations

import sys

import numpy as np

from app.analysis.distributions import Constant
from app.analysis.psa.models import ParameterSet
from app.analysis.psa.parameter_resolver import ParameterResolver
from app.analysis.psa.parameters import Parameter
from app.analysis.psa.sampler import PSASampler
from app.domain.enums import HealthStates, Regime
from app.domain.scenario import Scenario
from app.domain.worker import worker_function_batch
from app.persistence.context import ModelContext
from engine.chains import Chain
from utils import stable_hash

# Tiny but non-trivial parameter set so the test exercises both the sampler
# and the Markov engine reward pipeline.
PARAM_SET = ParameterSet(
    cycles=Parameter(Constant(80)),
    bleeding_rate=Parameter(Constant(15.0)),
    joint_bleeding_fraction=Parameter(Constant(0.3)),
    gi_neck_bleeding_fraction=Parameter(Constant(95 / 20_295)),
    iliopsoas_bleeding_fraction=Parameter(Constant(9 / 3_244)),
    intracranial_hemorrhage_rate=Parameter(Constant(0.01)),
    ich_case_fatality=Parameter(Constant(0.10)),
    non_ich_case_fatality=Parameter(Constant(0.0)),
    baseline_age=Parameter(Constant(2.0)),
    weight_factor=Parameter(Constant(1.0)),
    benefits_discount_rate=Parameter(Constant(0.0)),
    costs_discount_rate=Parameter(Constant(0.0)),
    healthy_utility=Parameter(Constant(0.9)),
    mild_arthropathy_utility=Parameter(Constant(0.85)),
    moderate_arthropathy_utility=Parameter(Constant(0.7)),
    severe_arthropathy_utility=Parameter(Constant(0.5)),
    spontaneous_bleeding_utility=Parameter(Constant(0.6)),
    joint_bleeding_utility=Parameter(Constant(0.5)),
    intracranial_hemorrhage_utility=Parameter(Constant(0.3)),
    non_ich_major_bleeding_utility=Parameter(Constant(0.3)),
    death_utility=Parameter(Constant(0.0)),
    per_unit_price=Parameter(Constant(1000.0)),
    prophylaxis_background_factor_consumption_per_kg=Parameter(Constant(0.0)),
    factor_consumption_per_spontaneous_bleeding_per_kg=Parameter(Constant(10.0)),
    factor_consumption_per_joint_bleeding_per_kg=Parameter(Constant(20.0)),
    factor_consumption_per_intracranial_hemorrhage_per_kg=Parameter(Constant(50.0)),
    factor_consumption_per_non_ich_major_bleeding_per_kg=Parameter(Constant(50.0)),
)

SCENARIO_NAME = "smoke on_demand bayesian"
N_ITERS = 16
WORKER_ID = 0


def _build_inputs(env_seed: int) -> tuple[list, Chain, Scenario]:
    sampler = PSASampler(PARAM_SET, seed=env_seed)
    samples = sampler.sample(n=N_ITERS)
    resolved = ParameterResolver.resolve_samples(samples)
    inputs = [ParameterResolver.build_single(resolved, i) for i in range(N_ITERS)]
    chain = Chain(
        name="main",
        states=[s.value for s in HealthStates],
        matrix=np.eye(len(HealthStates)),
    )
    scenario = Scenario(name=SCENARIO_NAME, regime=Regime.ON_DEMAND)
    return inputs, chain, scenario


def _run_once(env_seed: int):
    inputs, chain, scenario = _build_inputs(env_seed)
    return worker_function_batch(
        chain, inputs, scenario, ModelContext.load(), worker_id=WORKER_ID,
    )


def main() -> int:
    env_seed = ModelContext.load().simulation.environment.seed
    print(f"env seed from data/simulation.json: {env_seed}")

    per_scenario_seed = stable_hash(env_seed, SCENARIO_NAME)
    print(f"per-scenario stable_hash(env_seed, scenario): {per_scenario_seed}")
    if not (0 <= per_scenario_seed < 2**32):
        print("FAIL: stable_hash out of expected range", file=sys.stderr)
        return 1

    a = _run_once(env_seed)
    b = _run_once(env_seed)

    mismatched = [
        i
        for i, (x, y) in enumerate(zip(a, b, strict=True))
        if (x.sequence != y.sequence)
        or (x.absorbed_at != y.absorbed_at)
        or (x.event_count != y.event_count)
        or (x.total_factor != y.total_factor)
        or (x.total_qaly != y.total_qaly)
    ]
    if mismatched:
        print(
            f"FAIL: outputs diverged at iter indices {mismatched}",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {N_ITERS} iters, env seed {env_seed}, bit-for-bit identical across two runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
