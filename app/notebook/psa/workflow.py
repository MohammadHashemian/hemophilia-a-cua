"""Reusable execution workflow for one PSA horizon."""

from __future__ import annotations

import pickle
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


def run_horizon(
    horizon: str | HorizonSpec,
    *,
    sample_size: int | None = None,
    batch_size: int = 4,
) -> Path:
    spec = get_horizon(horizon)
    root = get_project_root()
    logger = setup_root_logger()
    scenarios, base_params, context = load_scenario_inputs(
        spec,
        root=root,
    )
    bundles = build_bundles(
        scenarios,
        base_params,
        context,
        sample_size=sample_size,
    )
    cache_dir = horizon_cache_dir(root, spec)
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
    )
    path = horizon_results_path(root, spec)
    logger.info("Completed %s PSA: %s", spec.label, path)
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
