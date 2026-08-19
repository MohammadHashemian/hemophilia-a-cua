from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl

from modular_models.state_transition.context import StudyContext
from modular_models.state_transition.engine import (
    StateTransitionEngine,
    derive_event_rates,
)
from modular_models.state_transition.kernels import warm_jit_kernels
from modular_models.state_transition.results import ComparisonResult
from modular_models.state_transition.rewards import (
    ComputeBackend,
    cuda_available,
    cuda_runtime_info,
)
from modular_models.state_transition.sampling import ParameterResolver
from modular_models.state_transition.types import Strategy

_TECHNICAL_OWSA_PARAMETERS = {
    "default_patients",
    "utility_integration_step_days",
}

Compression = Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"]
_PIPELINE_SCHEMA_VERSION = 2

_WORKER_CONTEXT: StudyContext | None = None
_WORKER_SCENARIO_ID = "base_case"
_WORKER_N_PATIENTS = 0
_WORKER_SEED = 0
_WORKER_OPTIONS: dict[str, Any] = {}
_WORKER_COMPUTE_BACKEND: ComputeBackend = "cpu"


def _effective_jobs(requested: int) -> int:
    return max(1, int(os.cpu_count() or 1)) if requested == 0 else requested


def _prepare_process_environment() -> None:
    # Each PSA iteration is already one independent CPU task. Limiting native
    # numerical libraries to one thread prevents N workers from each creating
    # another full set of threads.
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    warm_jit_kernels()


def _initialize_worker(
    data_dir: str,
    scenario_id: str,
    n_patients: int,
    seed: int,
    options: dict[str, Any],
    compute_backend: ComputeBackend,
) -> None:
    global _WORKER_CONTEXT
    global _WORKER_SCENARIO_ID
    global _WORKER_N_PATIENTS
    global _WORKER_SEED
    global _WORKER_OPTIONS
    global _WORKER_COMPUTE_BACKEND
    _WORKER_CONTEXT = StudyContext.load(Path(data_dir))
    _WORKER_SCENARIO_ID = scenario_id
    _WORKER_N_PATIENTS = n_patients
    _WORKER_SEED = seed
    _WORKER_OPTIONS = dict(options)
    _WORKER_COMPUTE_BACKEND = compute_backend


def _worker_context() -> StudyContext:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Production worker was not initialized")
    return _WORKER_CONTEXT


@dataclass(frozen=True, slots=True)
class PSAConfig:
    iterations: int
    n_patients: int
    output_dir: Path
    scenario_id: str = "base_case"
    seed: int = 20260813
    n_jobs: int = 1
    batch_size: int = 25
    compression: Compression = "zstd"
    compute_backend: ComputeBackend = "cpu"

    def validate(self) -> None:
        if self.iterations <= 0 or self.n_patients <= 0:
            raise ValueError("iterations and n_patients must be positive")
        if self.n_jobs < 0 or self.batch_size <= 0:
            raise ValueError(
                "n_jobs must be non-negative and batch_size must be positive"
            )
        if self.compute_backend == "cuda" and not cuda_available():
            raise ValueError(
                "CUDA backend requested but no usable CUDA device was found"
            )


@dataclass(frozen=True, slots=True)
class OWSAConfig:
    n_patients: int
    output_dir: Path
    scenario_id: str = "base_case"
    seed: int = 20260813
    n_jobs: int = 1
    include_technical_parameters: bool = False
    parameter_ids: tuple[str, ...] | None = None
    compute_backend: ComputeBackend = "cpu"

    def validate(self) -> None:
        if self.n_patients <= 0 or self.n_jobs < 0:
            raise ValueError(
                "n_patients must be positive and n_jobs must be non-negative"
            )
        if self.compute_backend == "cuda" and not cuda_available():
            raise ValueError(
                "CUDA backend requested but no usable CUDA device was found"
            )


@dataclass(frozen=True, slots=True)
class PSAInnerLoopConfig:
    """Common-draw diagnostic for selecting patients within each PSA iteration."""

    population_sizes: tuple[int, ...]
    iterations: int
    output_dir: Path
    scenario_id: str = "base_case"
    seed: int = 20260813
    n_jobs: int = 0
    batch_size: int = 24
    relative_mean_threshold: float = 0.01

    def validate(self) -> None:
        if len(self.population_sizes) < 2 or any(
            size <= 0 for size in self.population_sizes
        ):
            raise ValueError("At least two positive population sizes are required")
        if tuple(sorted(set(self.population_sizes))) != self.population_sizes:
            raise ValueError("population_sizes must be unique and sorted")
        if self.iterations < 2:
            raise ValueError("At least two common PSA iterations are required")
        if self.n_jobs < 0 or self.batch_size <= 0:
            raise ValueError(
                "n_jobs must be non-negative and batch_size must be positive"
            )
        if not 0 < self.relative_mean_threshold < 1:
            raise ValueError("relative_mean_threshold must lie between zero and one")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_parquet(
    frame: pl.DataFrame, path: Path, compression: Compression = "zstd"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.write_parquet(temporary, compression=compression, statistics=True)
    os.replace(temporary, path)


def _atomic_csv(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    frame.write_csv(temporary)
    os.replace(temporary, path)


def _data_fingerprint(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("model.json", "references.json", "scenarios.json"):
        digest.update(name.encode())
        digest.update((data_dir / name).read_bytes())
    return digest.hexdigest()


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    backbone_files = (
        "context.py",
        "engine.py",
        "kernels.py",
        "production.py",
        "results.py",
        "sampling.py",
        "schema.py",
        "types.py",
        "rewards.py",
    )
    for name in backbone_files:
        path = package_dir / name
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _manifest_base(
    context: StudyContext, analysis: str, config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": _PIPELINE_SCHEMA_VERSION,
        "analysis": analysis,
        "status": "created",
        "created_at_utc": _now(),
        "updated_at_utc": _now(),
        "data_fingerprint_sha256": _data_fingerprint(context.data_dir),
        "code_fingerprint_sha256": _code_fingerprint(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
        "completed": 0,
    }


def _arm_record(prefix: str, summary: dict[str, float | int]) -> dict[str, float | int]:
    keys = (
        "mean_cost_irr",
        "mean_qaly",
        "mean_life_years",
        "mean_factor_iu",
        "mean_joint_bleeds",
        "mean_non_major_non_joint_bleeds",
        "mean_non_ich_major_bleeds",
        "mean_ich",
        "mean_total_bleeds",
        "joint_bleed_rate_per_person_year",
        "non_major_non_joint_rate_per_person_year",
        "non_ich_major_rate_per_person_year",
        "ich_rate_per_person_year",
        "total_bleed_rate_per_person_year",
        "mean_pettersson_score",
        "survival_probability",
        "alive_at_end",
        "deaths_total",
        "deaths_background",
        "deaths_ich",
        "all_cause_mortality_probability",
        "post_ich_ever_count",
        "post_ich_ever_probability",
    )
    return {f"{prefix}_{key}": summary[key] for key in keys}


def comparison_record(result: ComparisonResult) -> dict[str, float | int | bool | str]:
    p = result.prophylaxis.summary
    o = result.on_demand.summary
    bleeds_avoided = float(o["mean_total_bleeds"]) - float(p["mean_total_bleeds"])
    deaths_avoided = float(o["all_cause_mortality_probability"]) - float(
        p["all_cause_mortality_probability"]
    )
    output: dict[str, float | int | bool | str] = {
        **result.economic_summary(),
        **_arm_record("prophylaxis", p),
        **_arm_record("on_demand", o),
        "mean_bleeds_avoided": bleeds_avoided,
        "relative_bleed_reduction": bleeds_avoided
        / max(float(o["mean_total_bleeds"]), 1e-12),
        "absolute_mortality_reduction": deaths_avoided,
        "relative_mortality_reduction": deaths_avoided
        / max(float(o["all_cause_mortality_probability"]), 1e-12),
        "deaths_avoided_per_1000": deaths_avoided * 1000.0,
        "incremental_cost_per_bleed_avoided_irr": result.incremental_cost_irr
        / max(bleeds_avoided, 1e-12),
    }
    return output


def _run_comparison(
    context: StudyContext,
    values: dict[str, float],
    options: dict[str, Any],
    scenario_id: str,
    seed: int,
    n_patients: int,
    compute_backend: ComputeBackend = "cpu",
) -> ComparisonResult:
    engine = StateTransitionEngine(
        context,
        values,
        options,
        scenario_id=scenario_id,
        seed=seed,
        compute_backend=compute_backend,
    )
    prophylaxis = engine.run(Strategy.PROPHYLAXIS, n_patients=n_patients)
    on_demand = engine.run(Strategy.ON_DEMAND, n_patients=n_patients)
    return ComparisonResult(prophylaxis, on_demand, values["wtp_irr_per_qaly"])


def _execute_psa_worker(row: dict[str, Any]) -> dict[str, Any]:
    values = {
        key: float(value)
        for key, value in row.items()
        if key not in {"iteration", "iteration_seed"}
    }
    comparison = _run_comparison(
        _worker_context(),
        values,
        _WORKER_OPTIONS,
        _WORKER_SCENARIO_ID,
        int(row["iteration_seed"]),
        _WORKER_N_PATIENTS,
        _WORKER_COMPUTE_BACKEND,
    )
    return {
        "iteration": int(row["iteration"]),
        "iteration_seed": int(row["iteration_seed"]),
        **comparison_record(comparison),
    }


def _execute_owsa_worker(task: dict[str, Any]) -> dict[str, Any]:
    context = _worker_context()
    resolver = ParameterResolver(context)
    values, options = resolver.deterministic(
        _WORKER_SCENARIO_ID,
        cast(dict[str, float], task["overrides"]),
    )
    metadata = cast(dict[str, Any], task["metadata"])
    try:
        comparison = _run_comparison(
            context,
            values,
            options,
            _WORKER_SCENARIO_ID,
            _WORKER_SEED,
            _WORKER_N_PATIENTS,
            _WORKER_COMPUTE_BACKEND,
        )
        return {
            **metadata,
            "status": "complete",
            "error": None,
            **comparison_record(comparison),
        }
    except ValueError as exc:
        return {
            **metadata,
            "status": "invalid_input_combination",
            "error": str(exc),
        }


class _RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self) -> _RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.descriptor, f"pid={os.getpid()} created={_now()}".encode())
        except FileExistsError as exc:
            raise RuntimeError(f"Another run owns the lock: {self.path}") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
        self.path.unlink(missing_ok=True)


class PSAProductionPipeline:
    """Checkpointed and resumable second-order PSA using paired simulations."""

    def __init__(self, context: StudyContext, config: PSAConfig) -> None:
        config.validate()
        self.context = context
        self.config = config
        self.resolver = ParameterResolver(context)
        self.run_dir = config.output_dir.resolve()
        self.parts_dir = self.run_dir / "parts"
        self.manifest_path = self.run_dir / "manifest.json"
        self.draws_path = self.run_dir / "parameter_draws.parquet"
        self.results_path = self.run_dir / "psa_results.parquet"

    def _config_payload(self) -> dict[str, Any]:
        payload = asdict(self.config)
        payload["output_dir"] = str(self.run_dir)
        return cast(dict[str, Any], json.loads(json.dumps(payload)))

    def _prepare(self) -> tuple[pl.DataFrame, dict[str, Any]]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = _data_fingerprint(self.context.data_dir)
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != _PIPELINE_SCHEMA_VERSION:
                raise ValueError(
                    "Existing PSA run uses an incompatible pipeline schema"
                )
            if manifest["config"] != self._config_payload():
                raise ValueError(
                    "Existing PSA run configuration does not match the requested run"
                )
            if manifest["data_fingerprint_sha256"] != fingerprint:
                raise ValueError("Model inputs changed after the PSA run was created")
            if manifest.get("code_fingerprint_sha256") != _code_fingerprint():
                raise ValueError("Model code changed after the PSA run was created")
        else:
            manifest = _manifest_base(self.context, "psa", self._config_payload())
            _atomic_json(self.manifest_path, manifest)

        if not self.draws_path.exists():
            sampled, _ = self.resolver.probabilistic(
                self.config.iterations,
                self.config.seed,
                self.config.scenario_id,
            )
            draw_data: dict[str, Any] = {
                "iteration": np.arange(self.config.iterations, dtype=np.int64),
                "iteration_seed": np.array(
                    [
                        int(
                            np.random.SeedSequence(
                                [self.config.seed, index]
                            ).generate_state(1)[0]
                        )
                        for index in range(self.config.iterations)
                    ],
                    dtype=np.uint32,
                ),
            }
            draw_data.update(sampled)
            _atomic_parquet(
                pl.DataFrame(draw_data), self.draws_path, self.config.compression
            )
        return pl.read_parquet(self.draws_path), manifest

    def _completed(self) -> set[int]:
        completed: set[int] = set()
        for path in self.parts_dir.glob("psa_*.parquet"):
            completed.update(
                pl.read_parquet(path, columns=["iteration"])["iteration"].to_list()
            )
        return completed

    def _execute(self, row: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        values = {
            key: float(value)
            for key, value in row.items()
            if key not in {"iteration", "iteration_seed"}
        }
        comparison = _run_comparison(
            self.context,
            values,
            options,
            self.config.scenario_id,
            int(row["iteration_seed"]),
            self.config.n_patients,
            self.config.compute_backend,
        )
        return {
            "iteration": int(row["iteration"]),
            "iteration_seed": int(row["iteration_seed"]),
            **comparison_record(comparison),
        }

    def run(self) -> pl.DataFrame:
        draws, manifest = self._prepare()
        _, options = self.resolver.deterministic(self.config.scenario_id)
        with _RunLock(self.run_dir / ".run.lock"):
            completed = self._completed()
            pending = [
                index
                for index in range(self.config.iterations)
                if index not in completed
            ]
            manifest.update(
                status="running", updated_at_utc=_now(), completed=len(completed)
            )
            jobs = _effective_jobs(self.config.n_jobs)
            manifest["effective_jobs"] = jobs
            manifest["parallel_backend"] = "process" if jobs > 1 else "sequential"
            if self.config.compute_backend == "cuda":
                manifest["cuda"] = cuda_runtime_info()
                manifest["parallel_backend"] = (
                    "cuda_process_pool" if jobs > 1 else "cuda_sequential"
                )
            _atomic_json(self.manifest_path, manifest)
            started = time.perf_counter()
            executor: ProcessPoolExecutor | None = None
            try:
                if jobs > 1:
                    _prepare_process_environment()
                    executor = ProcessPoolExecutor(
                        max_workers=jobs,
                        mp_context=mp.get_context("spawn"),
                        initializer=_initialize_worker,
                        initargs=(
                            str(self.context.data_dir),
                            self.config.scenario_id,
                            self.config.n_patients,
                            self.config.seed,
                            options,
                            self.config.compute_backend,
                        ),
                    )
                for offset in range(0, len(pending), self.config.batch_size):
                    indices = pending[offset : offset + self.config.batch_size]
                    rows = draws.filter(pl.col("iteration").is_in(indices)).to_dicts()
                    if executor is None:
                        records = [self._execute(row, options) for row in rows]
                    else:
                        records = list(
                            executor.map(_execute_psa_worker, rows, chunksize=1)
                        )
                    part = (
                        self.parts_dir
                        / f"psa_{min(indices):06d}_{max(indices):06d}.parquet"
                    )
                    _atomic_parquet(pl.DataFrame(records).sort("iteration"), part)
                    completed.update(indices)
                    elapsed = time.perf_counter() - started
                    manifest.update(
                        status="running",
                        updated_at_utc=_now(),
                        completed=len(completed),
                        elapsed_seconds_this_session=elapsed,
                    )
                    _atomic_json(self.manifest_path, manifest)
            except BaseException:
                manifest.update(status="interrupted", updated_at_utc=_now())
                _atomic_json(self.manifest_path, manifest)
                raise
            finally:
                if executor is not None:
                    executor.shutdown()

            result = self.consolidate()
            manifest.update(
                status="complete",
                updated_at_utc=_now(),
                completed=result.height,
                results_file=str(self.results_path),
            )
            _atomic_json(self.manifest_path, manifest)
            return result

    def consolidate(self) -> pl.DataFrame:
        paths = sorted(self.parts_dir.glob("psa_*.parquet"))
        if not paths:
            return pl.DataFrame()
        frame = pl.concat(
            [pl.read_parquet(path) for path in paths], how="vertical_relaxed"
        )
        frame = frame.unique(subset=["iteration"], keep="last").sort("iteration")
        _atomic_parquet(frame, self.results_path, self.config.compression)
        return frame


class OWSAProductionPipeline:
    """Resumable endpoint analysis with one atomic result per parameter endpoint."""

    def __init__(self, context: StudyContext, config: OWSAConfig) -> None:
        config.validate()
        self.context = context
        self.config = config
        self.resolver = ParameterResolver(context)
        self.run_dir = config.output_dir.resolve()
        self.parts_dir = self.run_dir / "parts"
        self.manifest_path = self.run_dir / "manifest.json"
        self.results_path = self.run_dir / "owsa_results.parquet"

    def _config_payload(self) -> dict[str, Any]:
        payload = asdict(self.config)
        payload["output_dir"] = str(self.run_dir)
        return cast(dict[str, Any], json.loads(json.dumps(payload)))

    def _parameters(self) -> list[str]:
        selected = (
            list(self.config.parameter_ids)
            if self.config.parameter_ids is not None
            else [
                key
                for key, parameter in self.context.parameters.items()
                if parameter.owsa is not None
            ]
        )
        if not self.config.include_technical_parameters:
            selected = [
                key for key in selected if key not in _TECHNICAL_OWSA_PARAMETERS
            ]
        options = self.context.scenario(self.config.scenario_id).options
        if options.get("joint_rate_method") == "fraction":
            selected = [
                key
                for key in selected
                if key not in {"ajbr_prophylaxis", "ajbr_on_demand"}
            ]
        else:
            selected = [key for key in selected if key != "joint_bleed_fraction"]
        if options.get("ich_rate_method") == "fraction":
            selected = [
                key
                for key in selected
                if key not in {"ich_rate_prophylaxis", "ich_rate_on_demand"}
            ]
        else:
            selected = [key for key in selected if key != "ich_fraction"]
        if options.get("post_ich_utility_rule") != "mild":
            selected = [key for key in selected if key != "post_ich_mild_utility_cap"]
        return selected

    def _task_spec(
        self, parameter_id: str, endpoint: str, value: float
    ) -> dict[str, Any]:
        overrides = {parameter_id: value}
        analysis_type = "one_way"
        linked_parameter_id: str | None = None
        linked_endpoint_value: float | None = None

        # Direct AJBR endpoints can exceed the simultaneously fixed ABR. If that
        # happens at the high endpoint, pair it with the documented high ABR.
        # This is retained as a linked endpoint, not mislabeled as pure OWSA.
        if endpoint == "high" and parameter_id in {
            "ajbr_prophylaxis",
            "ajbr_on_demand",
        }:
            suffix = parameter_id.removeprefix("ajbr_")
            strategy = (
                Strategy.PROPHYLAXIS if suffix == "prophylaxis" else Strategy.ON_DEMAND
            )
            values, options = self.resolver.deterministic(
                self.config.scenario_id, overrides
            )
            try:
                derive_event_rates(values, options, strategy)
            except ValueError:
                abr_id = f"abr_{suffix}"
                abr_range = self.context.parameter(abr_id).owsa
                if abr_range is None:
                    raise ValueError(
                        f"{parameter_id} high endpoint requires a linked {abr_id} range"
                    ) from None
                overrides[abr_id] = abr_range.high
                analysis_type = "linked_endpoint"
                linked_parameter_id = abr_id
                linked_endpoint_value = abr_range.high
                linked_values, linked_options = self.resolver.deterministic(
                    self.config.scenario_id,
                    overrides,
                )
                derive_event_rates(linked_values, linked_options, strategy)

        parameter = self.context.parameter(parameter_id)
        metadata: dict[str, Any] = {
            "task_id": f"{parameter_id}__{endpoint}",
            "parameter_id": parameter_id,
            "parameter_description": parameter.description,
            "unit": parameter.unit,
            "endpoint": endpoint,
            "base_value": parameter.value,
            "endpoint_value": value,
            "analysis_type": analysis_type,
            "linked_parameter_id": linked_parameter_id,
            "linked_endpoint_value": linked_endpoint_value,
            "overrides_json": json.dumps(overrides, sort_keys=True),
        }
        return {"metadata": metadata, "overrides": overrides}

    def _task(self, task: dict[str, Any]) -> dict[str, Any]:
        values, options = self.resolver.deterministic(
            self.config.scenario_id,
            cast(dict[str, float], task["overrides"]),
        )
        metadata = cast(dict[str, Any], task["metadata"])
        try:
            comparison = _run_comparison(
                self.context,
                values,
                options,
                self.config.scenario_id,
                self.config.seed,
                self.config.n_patients,
                self.config.compute_backend,
            )
            return {
                **metadata,
                "status": "complete",
                "error": None,
                **comparison_record(comparison),
            }
        except ValueError as exc:
            return {
                **metadata,
                "status": "invalid_input_combination",
                "error": str(exc),
            }

    def run(self) -> pl.DataFrame:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        parameters = self._parameters()
        tasks: list[dict[str, Any]] = []
        for parameter_id in parameters:
            owsa = self.context.parameter(parameter_id).owsa
            if owsa is None:
                continue
            tasks.extend(
                (
                    self._task_spec(parameter_id, "low", owsa.low),
                    self._task_spec(parameter_id, "high", owsa.high),
                )
            )

        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != _PIPELINE_SCHEMA_VERSION:
                raise ValueError(
                    "Existing OWSA run uses an incompatible pipeline schema"
                )
            if manifest["config"] != self._config_payload():
                raise ValueError(
                    "Existing OWSA configuration does not match the requested run"
                )
            if manifest["data_fingerprint_sha256"] != _data_fingerprint(
                self.context.data_dir
            ):
                raise ValueError("Model inputs changed after the OWSA run was created")
            if manifest.get("code_fingerprint_sha256") != _code_fingerprint():
                raise ValueError("Model code changed after the OWSA run was created")
        else:
            manifest = _manifest_base(self.context, "owsa", self._config_payload())
            manifest["total_tasks"] = len(tasks)
            _atomic_json(self.manifest_path, manifest)

        with _RunLock(self.run_dir / ".run.lock"):
            completed = {
                path.stem.removeprefix("owsa_")
                for path in self.parts_dir.glob("*.parquet")
            }
            pending = [
                task
                for task in tasks
                if str(cast(dict[str, Any], task["metadata"])["task_id"])
                not in completed
            ]
            manifest.update(
                status="running", completed=len(completed), updated_at_utc=_now()
            )
            jobs = _effective_jobs(self.config.n_jobs)
            manifest["effective_jobs"] = jobs
            manifest["parallel_backend"] = "process" if jobs > 1 else "sequential"
            if self.config.compute_backend == "cuda":
                manifest["cuda"] = cuda_runtime_info()
                manifest["parallel_backend"] = (
                    "cuda_process_pool" if jobs > 1 else "cuda_sequential"
                )
            _atomic_json(self.manifest_path, manifest)

            executor: ProcessPoolExecutor | None = None
            try:
                if jobs == 1:
                    iterator: Iterable[dict[str, Any]] = map(self._task, pending)
                else:
                    _prepare_process_environment()
                    executor = ProcessPoolExecutor(
                        max_workers=jobs,
                        mp_context=mp.get_context("spawn"),
                        initializer=_initialize_worker,
                        initargs=(
                            str(self.context.data_dir),
                            self.config.scenario_id,
                            self.config.n_patients,
                            self.config.seed,
                            {},
                            self.config.compute_backend,
                        ),
                    )
                    iterator = executor.map(_execute_owsa_worker, pending, chunksize=1)
                for record in iterator:
                    part = self.parts_dir / f"owsa_{record['task_id']}.parquet"
                    _atomic_parquet(pl.DataFrame([record]), part)
                    completed.add(str(record["task_id"]))
                    manifest.update(completed=len(completed), updated_at_utc=_now())
                    _atomic_json(self.manifest_path, manifest)
            except BaseException:
                manifest.update(status="interrupted", updated_at_utc=_now())
                _atomic_json(self.manifest_path, manifest)
                raise
            finally:
                if executor is not None:
                    executor.shutdown()

            paths = sorted(self.parts_dir.glob("owsa_*.parquet"))
            frame = pl.concat(
                [pl.read_parquet(path) for path in paths], how="diagonal_relaxed"
            )
            frame = frame.unique(subset=["task_id"], keep="last").sort(
                ["parameter_id", "endpoint"]
            )
            if "status" in frame.columns:
                frame = frame.with_columns(
                    pl.when(
                        pl.col("status").is_null()
                        & pl.col("incremental_nmb_irr").is_not_null()
                    )
                    .then(pl.lit("complete"))
                    .otherwise(pl.col("status"))
                    .alias("status")
                )
            _atomic_parquet(frame, self.results_path)
            manifest.update(
                status="complete",
                completed=frame.height,
                updated_at_utc=_now(),
                results_file=str(self.results_path),
            )
            _atomic_json(self.manifest_path, manifest)
            return frame


class PSAInnerLoopDiagnostic:
    """Run common second-order draws across candidate inner patient counts."""

    def __init__(self, context: StudyContext, config: PSAInnerLoopConfig) -> None:
        config.validate()
        self.context = context
        self.config = config
        self.run_dir = config.output_dir.resolve()
        self.results_path = self.run_dir / "inner_loop_precision.parquet"
        self.csv_path = self.run_dir / "inner_loop_precision.csv"

    def run(self) -> pl.DataFrame:
        from modular_models.state_transition.reporting import psa_inner_loop_precision

        frames: dict[int, pl.DataFrame] = {}
        for population_size in self.config.population_sizes:
            pipeline = PSAProductionPipeline(
                self.context,
                PSAConfig(
                    iterations=self.config.iterations,
                    n_patients=population_size,
                    output_dir=self.run_dir / f"n_{population_size:06d}",
                    scenario_id=self.config.scenario_id,
                    seed=self.config.seed,
                    n_jobs=self.config.n_jobs,
                    batch_size=self.config.batch_size,
                ),
            )
            frames[population_size] = pipeline.run()
        result = psa_inner_loop_precision(
            frames,
            reference_size=max(self.config.population_sizes),
            relative_mean_threshold=self.config.relative_mean_threshold,
        )
        _atomic_parquet(result, self.results_path)
        _atomic_csv(result, self.csv_path)
        return result
