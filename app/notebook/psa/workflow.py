"""Reusable execution workflow for one PSA horizon."""

from __future__ import annotations

import hashlib
import json
import pickle
import struct
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import polars as pl

from app.analysis.psa.parameter_resolver import ParameterResolver
from app.domain.enums import HealthStates
from app.domain.inputs import ModelInput
from app.domain.scenario import Scenario, ScenarioBundle
from app.domain.worker import worker_function, worker_function_batch
from app.notebook.parameter_sets import HemophiliaParamRepo
from app.notebook.psa.scenarios import HorizonSpec, build_psa_scenarios, get_horizon
from app.notebook.scenario_helpers import parse_scenario
from app.notebook.scenario_runner import run_scenarios_in_batches
from app.persistence.context import ModelContext
from engine.chains import Chain
from utils import stable_hash
from utils.logging import setup_root_logger
from utils.path_utils import get_project_root

_SCENARIO_CACHE_VERSION = "scenario-cache-v1"
_SIMULATION_CODE_PATHS = (
    "app/domain/transition_builder.py",
    "app/domain/worker.py",
    "app/domain/rewards/vectorized.py",
    "app/notebook/dataframe_builders.py",
    "app/notebook/scenario_runner.py",
    "app/analysis/psa/parameter_resolver.py",
    "engine/vectorized.py",
    "utils/math.py",
)


def horizon_cache_dir(root: Path, horizon: str | HorizonSpec) -> Path:
    return root / "app" / "cache" / "psa" / get_horizon(horizon).directory


def horizon_results_path(root: Path, horizon: str | HorizonSpec) -> Path:
    return horizon_cache_dir(root, horizon) / "parquet" / "all_results_combined.parquet"


def load_scenario_inputs(
    horizon: str | HorizonSpec,
    *,
    root: Path | None = None,
    context: ModelContext | None = None,
) -> tuple[list[Scenario], dict, ModelContext]:
    root = root or get_project_root()
    context = context or ModelContext.load()
    repo = HemophiliaParamRepo(
        root=root,
        cache_path=Path("app/cache/samples.pkl"),
        context=context,
    )
    with open(repo.root / repo.cache_path, "rb") as file:
        meta_samples = pickle.load(file)
    base_params = repo.load_psa_parameters()
    scenarios = build_psa_scenarios(
        horizon,
        meta_samples=meta_samples,
        context=context,
    )
    return scenarios, base_params, context


def build_bundles(
    scenarios: list[Scenario],
    base_params: dict,
    context: ModelContext,
    *,
    sample_size: int | None = None,
) -> list[ScenarioBundle[ModelInput]]:
    if sample_size is None:
        sample_size = (
            context.simulation.psa.production
            if context.simulation.environment.mode == "production"
            else context.simulation.psa.development
        )

    seed = context.simulation.environment.seed
    bundles: list[ScenarioBundle[ModelInput]] = []
    for scenario in scenarios:
        horizon, _regime, sampling_method, extension = parse_scenario(scenario.name)
        # Both regimes in a cost-effectiveness pair receive the same random
        # stream for shared PSA parameters. Regime-specific parameters remain
        # different through their scenario overrides.
        pairing_key = f"{horizon}|{sampling_method}|{extension or 'base'}"
        scenario_seed = stable_hash(seed, pairing_key)
        scenario_params = scenario.apply_overrides(base_params)
        raw = {}
        for field in scenario_params.__dataclass_fields__:
            field_seed = stable_hash(scenario_seed, field)
            field_rng = np.random.default_rng(field_seed)
            raw[field] = getattr(scenario_params, field).sample(
                sample_size, field_rng
            )
        resolved = ParameterResolver.resolve_samples(raw)
        inputs = [
            ParameterResolver.build_single(resolved, index)
            for index in range(sample_size)
        ]
        bundles.append(ScenarioBundle(scenario=scenario, inputs=inputs))
    return bundles


def identity_chain() -> Chain:
    states = [state.value for state in HealthStates]
    return Chain(
        name="main",
        states=states,
        matrix=np.eye(len(states), dtype=np.float64),
    )


def _simulation_code_digest(root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_SCENARIO_CACHE_VERSION.encode())
    for relative_path in _SIMULATION_CODE_PATHS:
        path = root / relative_path
        digest.update(relative_path.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def scenario_cache_fingerprint(
    bundle: ScenarioBundle[ModelInput],
    context: ModelContext,
    *,
    code_digest: str,
) -> str:
    """Hash every value that can affect a scenario's cached output."""
    digest = hashlib.sha256()
    digest.update(_SCENARIO_CACHE_VERSION.encode())
    digest.update(code_digest.encode())
    digest.update(
        json.dumps(
            bundle.scenario.model_dump(mode="json", exclude={"overrides"}),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    )
    context_payload = {
        "simulation": context.simulation.model_dump(mode="json"),
        "clinical": context.clinical.model_dump(mode="json"),
        "costs": context.costs.model_dump(mode="json"),
        "economic_policy": context.economic_policy.model_dump(mode="json"),
        "utilities": context.utilities.model_dump(mode="json"),
        "mortality": context.mortality.model_dump(mode="json"),
    }
    digest.update(
        json.dumps(
            context_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    )
    model_fields = fields(ModelInput)
    digest.update(str(len(bundle.inputs)).encode())
    for field in model_fields:
        digest.update(field.name.encode())
    for model_input in bundle.inputs:
        for field in model_fields:
            digest.update(struct.pack("!d", float(getattr(model_input, field.name))))
    return digest.hexdigest()


def run_horizon(
    horizon: str | HorizonSpec,
    *,
    sample_size: int | None = None,
    batch_size: int = 4,
    resume: bool = True,
    max_parallel_scenarios: int = 2,
) -> Path:
    total_started = time.perf_counter()
    spec = get_horizon(horizon)
    root = get_project_root()
    logger = setup_root_logger()
    phase_started = time.perf_counter()
    scenarios, base_params, context = load_scenario_inputs(
        spec,
        root=root,
    )
    load_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    bundles = build_bundles(
        scenarios,
        base_params,
        context,
        sample_size=sample_size,
    )
    bundle_seconds = time.perf_counter() - phase_started
    cache_dir = horizon_cache_dir(root, spec)
    scenario_fingerprints = None
    fingerprint_seconds = 0.0
    if resume:
        phase_started = time.perf_counter()
        code_digest = _simulation_code_digest(root)
        scenario_fingerprints = {
            bundle.scenario.name: scenario_cache_fingerprint(
                bundle,
                context,
                code_digest=code_digest,
            )
            for bundle in bundles
        }
        fingerprint_seconds = time.perf_counter() - phase_started
    logger.info(
        "Preparation phases: load=%.2fs, sampling=%.2fs, fingerprint=%.2fs",
        load_seconds,
        bundle_seconds,
        fingerprint_seconds,
    )
    phase_started = time.perf_counter()
    run_scenarios_in_batches(
        bundles=bundles,
        context=context,
        batch_size=batch_size,
        engine="batch",
        worker_function=worker_function,
        identity_chain=identity_chain(),
        output_dir=cache_dir / "parquet",
        temp_dir=cache_dir / "temp" / "parquet_temp",
        batch_worker_function=worker_function_batch,
        scenario_fingerprints=scenario_fingerprints,
        max_parallel_scenarios=max_parallel_scenarios,
    )
    runner_seconds = time.perf_counter() - phase_started
    path = horizon_results_path(root, spec)
    logger.info("Completed %s PSA: %s", spec.label, path)
    logger.info(
        "Horizon timing: runner=%.2fs, total=%.2fs",
        runner_seconds,
        time.perf_counter() - total_started,
    )
    return path


def load_horizon_results(horizon: str | HorizonSpec) -> pl.DataFrame:
    root = get_project_root()
    spec = get_horizon(horizon)
    path = horizon_results_path(root, spec)
    if path.exists():
        return pl.read_parquet(path)

    # Migration path: permit the new analysis notebooks to read the legacy
    # mixed cache until each separated simulation has been run once.
    legacy = root / "app" / "cache" / "psa" / "parquet" / "all_results_combined.parquet"
    # The legacy childhood results covered ages 2–12 and are not compatible
    # with the current ages 1–15 definition. Lifetime remains unchanged and
    # can safely use its legacy rows during migration.
    if legacy.exists() and spec.key == "lifetime":
        return pl.read_parquet(legacy).filter(pl.col("time_horizon") == spec.key)

    raise FileNotFoundError(
        f"No results at {path}. Run this horizon's 02_simulation.ipynb first."
    )
