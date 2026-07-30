"""Reusable execution workflow for horizon-specific OWSA."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from app.analysis.psa.parameter_resolver import ParameterResolver
from app.domain.inputs import ModelInput
from app.domain.scenario import Scenario, ScenarioBundle
from app.domain.worker import worker_function, worker_function_batch
from app.notebook.owsa.scenarios import OWSARange, build_owsa_scenarios
from app.notebook.parameter_sets import HemophiliaParamRepo
from app.notebook.psa.scenarios import HorizonSpec, get_horizon
from app.notebook.psa.workflow import (
    identity_chain,
    scenario_cache_fingerprint,
    simulation_code_digest,
)
from app.notebook.scenario_runner import run_scenarios_in_batches
from app.persistence.context import ModelContext
from utils.logging import setup_root_logger
from utils.path_utils import get_project_root

DEFAULT_OWSA_REPLICATIONS = 1_000


def horizon_cache_dir(root: Path, horizon: str | HorizonSpec) -> Path:
    return root / "app" / "cache" / "owsa" / get_horizon(horizon).directory


def horizon_results_path(root: Path, horizon: str | HorizonSpec) -> Path:
    return horizon_cache_dir(root, horizon) / "parquet" / "all_results_combined.parquet"


def load_owsa_inputs(
    horizon: str | HorizonSpec,
    *,
    root: Path | None = None,
    context: ModelContext | None = None,
) -> tuple[list, object, list[OWSARange], ModelContext]:
    root = root or get_project_root()
    context = context or ModelContext.load()
    repo = HemophiliaParamRepo(
        root=root,
        cache_path=Path("app/cache/samples.pkl"),
        context=context,
    )
    with open(repo.root / repo.cache_path, "rb") as file:
        meta_samples = pickle.load(file)
    owsa_parameters = repo.load_owsa_parameters()
    psa_parameters = repo.load_psa_parameters()
    scenarios, ranges = build_owsa_scenarios(
        horizon,
        meta_samples=meta_samples,
        owsa_parameters=owsa_parameters,
        psa_parameters=psa_parameters,
        parameter_keys=repo.ows_params_keys,
        seed=context.simulation.environment.seed,
    )
    return scenarios, owsa_parameters, ranges, context


def build_owsa_bundles(
    scenarios: list[Scenario],
    parameters,
    *,
    replications: int,
) -> list[ScenarioBundle[ModelInput]]:
    """Replicate deterministic point inputs for stochastic microsimulation.

    OWSA changes exactly one input at a time. ``replications`` controls only
    first-order Monte Carlo noise from individual event histories; it is not
    a PSA parameter-draw count.
    """
    if replications < 1:
        raise ValueError("OWSA replications must be a positive integer")
    bundles = []
    for scenario in scenarios:
        scenario_parameters = scenario.apply_overrides(parameters)
        raw = {
            field: np.full(
                replications,
                getattr(scenario_parameters, field).point(),
                dtype=float,
            )
            for field in scenario_parameters.__dataclass_fields__
        }
        resolved = ParameterResolver.resolve_samples(raw)
        inputs = [
            ParameterResolver.build_single(resolved, index)
            for index in range(replications)
        ]
        bundles.append(ScenarioBundle(scenario=scenario, inputs=inputs))
    return bundles


def run_horizon(
    horizon: str | HorizonSpec,
    *,
    replications: int = DEFAULT_OWSA_REPLICATIONS,
    batch_size: int = 4,
    max_parallel_scenarios: int = 2,
) -> Path:
    spec = get_horizon(horizon)
    root = get_project_root()
    scenarios, parameters, _ranges, context = load_owsa_inputs(spec, root=root)
    bundles = build_owsa_bundles(
        scenarios,
        parameters,
        replications=replications,
    )
    cache_dir = horizon_cache_dir(root, spec)
    code_digest = simulation_code_digest(root)
    stream_key = f"owsa|{spec.key}|common-random-numbers-v1"
    fingerprints = {
        bundle.scenario.name: hashlib.sha256(
            (
                scenario_cache_fingerprint(
                    bundle,
                    context,
                    code_digest=code_digest,
                )
                + stream_key
            ).encode()
        ).hexdigest()
        for bundle in bundles
    }
    # Common random numbers reduce Monte Carlo noise in low/base/high
    # differences. All scenarios in this horizon reuse the same stochastic
    # stream; only their deterministic parameter values differ.
    random_stream_keys = {
        bundle.scenario.name: stream_key for bundle in bundles
    }
    setup_root_logger().info(
        "OWSA design: %d deterministic scenarios, %d stochastic replications "
        "per scenario, common random numbers enabled",
        len(bundles),
        replications,
    )
    run_scenarios_in_batches(
        bundles=bundles,
        context=context,
        identity_chain=identity_chain(),
        worker_function=worker_function,
        batch_size=batch_size,
        output_dir=cache_dir / "parquet",
        temp_dir=cache_dir / "temp" / "parquet_temp",
        engine="batch",
        batch_worker_function=worker_function_batch,
        max_parallel_scenarios=max_parallel_scenarios,
        scenario_fingerprints=fingerprints,
        random_stream_keys=random_stream_keys,
    )
    return horizon_results_path(root, spec)


def load_horizon_results(horizon: str | HorizonSpec) -> pl.DataFrame:
    path = horizon_results_path(get_project_root(), horizon)
    if not path.exists():
        raise FileNotFoundError(
            f"No OWSA results at {path}. Run this horizon's 02_simulation.ipynb first."
        )
    df = pl.read_parquet(path)
    if "iteration_id" not in df.columns:
        df = df.with_columns(
            pl.int_range(pl.len()).over("scenario").alias("iteration_id")
        )
    return df
