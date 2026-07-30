import gc
import os
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Literal

import enlighten
import polars as pl

from app.domain.inputs import ModelInput
from app.domain.scenario import ScenarioBundle
from app.notebook.dataframe_builders import build_df
from app.notebook.scenario_helpers import pair_scenarios
from app.persistence.context import ModelContext
from engine.chains import Chain
from engine.runners import ScenarioRunner, SimulationResult
from utils import stable_hash
from utils.logging import setup_root_logger


class _Errors:
    MISSING_OUT_DIR = "Output directory %s does not exist. Attempting to create it."
    MISSING_TEMP_DIR = "Temp directory %s does not exist. Cannot use cached temp files."
    USING_CACHED_TEMP = "Using %d cached temp batch parquet files from %s"
    NO_CACHED_TEMP = "No cached temp parquet found in %s, regenerating batches."
    FAILED_BATCH_SAVE = "Failed to write temp batch parquet %s, reason: \n%s"
    NO_RESULTS_COMBINED = "No results available after combining temp batches. Final parquet will not be created."
    FAILED_TO_SAVE_COMBINED_RESULTS = "Failed to save combined results checkpoint %s"
    FAILED_TO_PAIR = "Failed to pair scenarios for final output"
    NO_PAIRS_FOUND = "No scenario pairs found in combined results. Final parquet file list will be empty."
    SKIPPING_PAIR = "Skipping pair %s vs %s: missing arm data"
    FAILED_TO_SAVE_PAIR_RESULTS = "Failed to write Parquet for pair %s vs %s -> %s"
    FAILED_TO_CLEAN_TEMP_DIRECTORY = "Failed to clean up temp directory %s. Manual clean up may be required to free disk space."
    CLEAN_UP_DISABLED = "Clean up disabled for temp directory %s. Manual clean up may be required to free disk space."
    FAILED_TO_BUILD_DF = "Failed to build DataFrame from batch results"


class _Info:
    STARTING_BATCH_RUNNER = (
        "Starting batch runner with %d scenario, batch size: %d, engine: %s"
    )
    SAVED_BATCH = "Saved temp batch %d to %s"
    BATCH_COMPLETE = "Batch %d/%d complete (%d scenarios processed, %d remaining). "
    ELAPSED = "Elapsed=%0.1fs, avg_batch=%0.1fs, est_remaining=%0.1fs"
    COMBINING_TEMP_BATCHES = "Combining %d temp batch files into a DF for pairing"
    SAVED_COMBINED_RESULTS = "Saved combined results checkpoint to %s"
    SAVED_PAIR_RESULTS = "Saved Parquet for pair %s vs %s -> %s"
    SAVED_FINAL_RESULTS = "Saved %d final parquet files to %s"
    USING_SCENARIO_CACHE = "Reusing scenario cache for %s"
    INVALID_SCENARIO_CACHE = "Ignoring invalid scenario cache for %s"
    SCENARIO_CACHE_SUMMARY = (
        "Scenario cache: %d reusable, %d require simulation"
    )


def batch_generator(bundles: list[ScenarioBundle[ModelInput]], batch_size: int):
    for i in range(0, len(bundles), batch_size):
        yield bundles[i : i + batch_size]


def _safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_")


def _available_memory_bytes() -> int | None:
    """Best-effort available-memory query without an external dependency."""
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available_physical)
        return None
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        return None


def _parallel_worker_count(
    batch: list[ScenarioBundle[ModelInput]],
    requested: int,
) -> int:
    """Cap concurrency when full trace arrays would put RAM at risk."""
    requested = max(1, min(requested, len(batch)))
    if requested == 1 or not batch:
        return requested
    largest = max(
        len(bundle.inputs) * (int(bundle.inputs[0].cycle) + 1) * 96
        for bundle in batch
    )
    available = _available_memory_bytes()
    if available is None or largest <= 0:
        return requested
    memory_limit = max(1, int((available * 0.55) // largest))
    return max(1, min(requested, memory_limit))


def _run_batch(
    batch: list[ScenarioBundle[ModelInput]],
    context: ModelContext,
    identity_chain: Chain,
    batch_worker_function: Callable,
    max_parallel_scenarios: int,
) -> list[SimulationResult]:
    """Run each scenario in the batch via the vectorized batch worker.

    The batch worker processes all inputs for one scenario in a single
    numpy sweep, so we call it once per scenario in the batch and wrap
    each per-iter output in a SimulationResult for the dataframe builder.

    A per-scenario enlighten progress bar is shown (one bar per scenario
    in the batch) reporting completed iters / total iters with an
    ETA. The bar advances once per simulated year (every 52 steps)
    because all n_iters iters run in lockstep — at each year boundary
    we report n_iters / n_years additional iters as "completed work",
    so the user sees their natural unit (iters, not iter-steps).
    """
    run_id = uuid.uuid4().hex
    manager = enlighten.get_manager()
    total_iterations = sum(len(bundle.inputs) for bundle in batch)
    progress = manager.counter(
        total=total_iterations,
        desc="Vectorized scenarios",
        unit="iter",
    )
    progress_lock = Lock()
    reported: dict[str, int] = {}
    workers = _parallel_worker_count(batch, max_parallel_scenarios)
    logger = setup_root_logger()
    if workers < min(max_parallel_scenarios, len(batch)):
        logger.info(
            "Memory guard reduced parallel scenarios from %d to %d",
            min(max_parallel_scenarios, len(batch)),
            workers,
        )

    def run_one(
        position: int,
        bundle: ScenarioBundle[ModelInput],
    ) -> tuple[int, list[SimulationResult], float]:
        scenario = bundle.scenario
        inputs = bundle.inputs
        scenario_name = getattr(scenario, "name", str(scenario))
        worker_id = stable_hash(
            context.simulation.environment.seed,
            scenario_name,
        )
        n_iters = len(inputs)
        steps = int(inputs[0].cycle)
        n_years = max(steps // 52, 1)
        iters_per_year_tick = max(1, n_iters // n_years)

        reported[scenario_name] = 0

        def _on_step(step: int, total_steps: int) -> None:
            with progress_lock:
                remaining = n_iters - reported[scenario_name]
                increment = min(iters_per_year_tick, remaining)
                if increment > 0:
                    progress.update(increment)
                    reported[scenario_name] += increment

        started = time.perf_counter()
        outputs = batch_worker_function(
            chain=identity_chain,
            inputs=inputs,
            scenario=scenario,
            context=context,
            run_id=run_id,
            worker_id=worker_id,
            progress_callback=_on_step,
            progress_every=52,
        )
        elapsed = time.perf_counter() - started
        with progress_lock:
            remaining = n_iters - reported[scenario_name]
            if remaining > 0:
                progress.update(remaining, force=True)
                reported[scenario_name] += remaining

        scenario_results: list[SimulationResult] = []
        for input_data, output in zip(inputs, outputs, strict=True):
            scenario_results.append(
                SimulationResult(
                    run_id=run_id,
                    scenario=scenario_name,
                    worker_id=worker_id,
                    input_data=input_data,
                    output=output,
                )
            )
        return position, scenario_results, elapsed

    ordered_results: dict[int, list[SimulationResult]] = {}
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="psa-scenario",
    ) as executor:
        futures = {
            executor.submit(run_one, position, bundle): bundle.scenario.name
            for position, bundle in enumerate(batch)
        }
        for future in as_completed(futures):
            position, scenario_results, elapsed = future.result()
            ordered_results[position] = scenario_results
            logger.info(
                "Scenario complete: %s in %.2fs",
                futures[future],
                elapsed,
            )

    progress.close()
    return [
        result
        for position in range(len(batch))
        for result in ordered_results[position]
    ]


class _ValidationErrors:
    NO_BUNDLES = "No scenario bundles supplied; nothing to simulate."
    INVALID_BATCH_SIZE = (
        "batch_size must be a positive integer, got %r."
    )
    INVALID_OUTPUT_DIR = "Output directory %s is not writable."
    INVALID_TEMP_DIR = "Temp directory %s is not writable."
    INVALID_ENGINE = (
        "engine must be one of 'pathos', 'multiprocessing', 'batch' (got %r)."
    )
    NO_BATCH_WORKER = (
        "engine='batch' requires batch_worker_function; none was supplied."
    )
    SCHEMA_MISMATCH = (
        "build_df produced a different schema for the first batch of each "
        "scenarios group. The downstream pl.concat would fail. "
        "First batch: %s. Mismatched batch: %s."
    )


def validate_simulation_inputs(
    bundles: list[ScenarioBundle[ModelInput]],
    *,
    context: ModelContext,
    identity_chain: Chain,
    batch_size: int,
    output_dir: Path,
    temp_dir: Path,
    engine: Literal["pathos", "multiprocessing", "batch"],
    batch_worker_function: Callable | None,
    max_parallel_scenarios: int,
) -> None:
    """Pre-flight validation for ``run_scenarios_in_batches``.

    Runs **before** any simulation kicks off so that schema mismatches,
    missing directories, or invalid arguments surface immediately as
    a clear ``ValueError`` (or ``FileNotFoundError`` for the
    directory checks) instead of an opaque ``SchemaError`` minutes
    into the run.

    The check is intentionally cheap: it does **not** run any
    simulation, but it does dry-run ``build_df`` on the first bundle
    of each distinct scenario name and confirms the resulting
    DataFrames share a single schema. That is the failure mode that
    bit the PSA run (a batch with no ``extension`` values produced
    ``Null``-typed columns while a batch with extension values
    produced ``String``-typed columns).
    """
    if not bundles:
        raise ValueError(_ValidationErrors.NO_BUNDLES)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(_ValidationErrors.INVALID_BATCH_SIZE % (batch_size,))
    if engine not in ("pathos", "multiprocessing", "batch"):
        raise ValueError(_ValidationErrors.INVALID_ENGINE % (engine,))
    if engine == "batch" and batch_worker_function is None:
        raise ValueError(_ValidationErrors.NO_BATCH_WORKER)
    if max_parallel_scenarios <= 0:
        raise ValueError("max_parallel_scenarios must be a positive integer")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".write_probe"
    try:
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise OSError(_ValidationErrors.INVALID_OUTPUT_DIR, output_dir) from exc

    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)
    probe = temp_dir / ".write_probe"
    try:
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise OSError(_ValidationErrors.INVALID_TEMP_DIR, temp_dir) from exc

    # Schema dry-run. We do not actually simulate; we use a synthetic
    # "empty" batch per unique scenario name. build_df handles an
    # empty input list by returning an empty DataFrame with the right
    # column names, so the schema check is meaningful.
    seen_schemas: dict[str, dict[str, pl.DataType]] = {}
    for bundle in bundles:
        scenario_name = getattr(bundle.scenario, "name", str(bundle.scenario))
        if scenario_name in seen_schemas:
            continue
        try:
            probe_df = build_df(results=[], context=context)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"build_df dry-run failed for scenario '{scenario_name}': {exc}"
            ) from exc
        seen_schemas[scenario_name] = dict(probe_df.schema)

    schemas = list(seen_schemas.values())
    if schemas and not all(s == schemas[0] for s in schemas):
        first_name, first_schema = next(iter(seen_schemas.items()))
        bad_name, bad_schema = next(
            (name, schema)
            for name, schema in seen_schemas.items()
            if schema != first_schema
        )
        raise ValueError(
            _ValidationErrors.SCHEMA_MISMATCH.format(first_schema, bad_schema)
            + f" (scenarios: '{first_name}' vs '{bad_name}')"
        )


def run_scenarios_in_batches(
    bundles: list[ScenarioBundle[ModelInput]],
    context: ModelContext,
    identity_chain: Chain,
    worker_function: Callable,
    batch_size: int,
    output_dir: Path,
    temp_dir: Path,
    options: dict | None = None,
    engine: Literal["pathos", "multiprocessing", "batch"] = "pathos",
    batch_worker_function: Callable | None = None,
    scenario_fingerprints: dict[str, str] | None = None,
    max_parallel_scenarios: int = 2,
):
    """Run scenario bundles in batches, write per-pair Parquet files, and free memory after each batch.

    When `engine == "batch"`, a vectorized batch worker is used:
    `batch_worker_function(chain, inputs, scenario, context, ...)` is called
    once per scenario with the full list of inputs, and must return a
    list[ModelOutput] of the same length as inputs. This avoids the
    per-iter Python overhead of running 10k simulations one at a time.
    """

    logger = setup_root_logger()

    # Pre-flight validation. Runs before any simulation work so
    # configuration / schema issues fail fast with a clear error
    # rather than mid-run (e.g. the
    # ``SchemaError: type String is incompatible with Null`` that
    # used to surface in the per-batch pl.concat after ~20 minutes
    # of PSA simulation).
    validate_simulation_inputs(
        bundles=bundles,
        context=context,
        identity_chain=identity_chain,
        batch_size=batch_size,
        output_dir=output_dir,
        temp_dir=temp_dir,
        engine=engine,
        batch_worker_function=batch_worker_function,
        max_parallel_scenarios=max_parallel_scenarios,
    )

    logger.info(
        _Info.STARTING_BATCH_RUNNER,
        len(bundles),
        batch_size,
        engine,
    )
    options = options or {}
    use_cached_temp: bool = options.get("use_cache_temp", False)

    if not output_dir.exists():
        logger.info(_Errors.MISSING_OUT_DIR, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    if not temp_dir.exists():
        logger.info(_Errors.MISSING_TEMP_DIR, temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    temp_files: list[Path] = []
    bundles_to_run = bundles
    scenario_cache_paths: dict[str, Path] = {}

    if scenario_fingerprints is not None:
        scenario_cache_dir = temp_dir / "scenario_cache"
        scenario_cache_dir.mkdir(parents=True, exist_ok=True)
        bundles_to_run = []
        for bundle in bundles:
            scenario_name = bundle.scenario.name
            fingerprint = scenario_fingerprints.get(scenario_name)
            if fingerprint is None:
                raise ValueError(
                    f"Missing cache fingerprint for scenario '{scenario_name}'"
                )
            cache_path = scenario_cache_dir / (
                f"{_safe_name(scenario_name)[:40]}__{fingerprint[:32]}.parquet"
            )
            scenario_cache_paths[scenario_name] = cache_path
            if cache_path.exists():
                try:
                    cached_scenarios = pl.read_parquet(
                        cache_path,
                        columns=["scenario"],
                    )
                    valid = (
                        cached_scenarios.height == len(bundle.inputs)
                        and cached_scenarios["scenario"].n_unique() == 1
                        and cached_scenarios["scenario"][0] == scenario_name
                    )
                except Exception:
                    valid = False
                if valid:
                    temp_files.append(cache_path)
                    logger.info(_Info.USING_SCENARIO_CACHE, scenario_name)
                    continue
                logger.warning(_Info.INVALID_SCENARIO_CACHE, scenario_name)
            bundles_to_run.append(bundle)
        logger.info(
            _Info.SCENARIO_CACHE_SUMMARY,
            len(temp_files),
            len(bundles_to_run),
        )
    elif use_cached_temp:
        # Step 0: Load cache from temp storage
        temp_files = sorted(temp_dir.glob("batch_*.parquet"))
        if temp_files:
            logger.info(
                _Errors.USING_CACHED_TEMP,
                len(temp_files),
                temp_dir,
            )
        else:
            logger.info(
                _Errors.NO_CACHED_TEMP,
                temp_dir,
            )

    if bundles_to_run and (
        scenario_fingerprints is not None
        or not use_cached_temp
        or not temp_files
    ):
        # Step 1: Process batches and save each to temp storage
        if scenario_fingerprints is None:
            temp_files = []
        total_batches = (len(bundles_to_run) + batch_size - 1) // batch_size
        batch_start_time = time.perf_counter()
        simulation_seconds = 0.0
        dataframe_seconds = 0.0
        cache_write_seconds = 0.0
        processed_scenarios = 0

        for index, batch in enumerate(batch_generator(bundles_to_run, batch_size)):
            if engine == "batch":
                phase_started = time.perf_counter()
                batch_results = _run_batch(
                    batch=batch,
                    context=context,
                    identity_chain=identity_chain,
                    batch_worker_function=batch_worker_function or worker_function,
                    max_parallel_scenarios=max_parallel_scenarios,
                )
                simulation_seconds += time.perf_counter() - phase_started
            else:
                phase_started = time.perf_counter()
                runner = ScenarioRunner(
                    context=context,
                    scenario_bundles=batch,
                    chain_instance=identity_chain,
                    worker_func=worker_function,
                    backend=engine,
                )
                batch_results = runner.run_all()
                simulation_seconds += time.perf_counter() - phase_started

            # convert to DataFrame using existing helper
            phase_started = time.perf_counter()
            try:
                batch_df = build_df(results=batch_results, context=context)
            except Exception:
                logger.exception(_Errors.FAILED_TO_BUILD_DF)
                batch_df = pl.DataFrame()
            dataframe_seconds += time.perf_counter() - phase_started

            # Save either fingerprint-addressed scenario files or the legacy
            # coarse batch file.
            phase_started = time.perf_counter()
            if not batch_df.is_empty():
                if scenario_fingerprints is not None:
                    for bundle in batch:
                        scenario_name = bundle.scenario.name
                        scenario_df = batch_df.filter(
                            pl.col("scenario") == scenario_name
                        )
                        temp_path = scenario_cache_paths[scenario_name]
                        try:
                            scenario_df.write_parquet(temp_path)
                            temp_files.append(temp_path)
                            logger.info(
                                _Info.SAVED_BATCH,
                                index,
                                temp_path,
                            )
                        except Exception as e:
                            logger.exception(
                                _Errors.FAILED_BATCH_SAVE,
                                temp_path,
                                e.__str__(),
                            )
                else:
                    temp_path = temp_dir / f"batch_{index}.parquet"
                    try:
                        batch_df.write_parquet(temp_path)
                        temp_files.append(temp_path)
                        logger.info(_Info.SAVED_BATCH, index, temp_path)
                    except Exception as e:
                        logger.exception(
                            _Errors.FAILED_BATCH_SAVE,
                            temp_path,
                            e.__str__(),
                        )
            cache_write_seconds += time.perf_counter() - phase_started

            processed_scenarios += len(batch)
            elapsed = time.perf_counter() - batch_start_time
            avg_batch_time = elapsed / (index + 1)
            batches_remaining = total_batches - (index + 1)
            scenarios_remaining = len(bundles_to_run) - processed_scenarios
            remaining_time = avg_batch_time * batches_remaining
            logger.info(
                _Info.BATCH_COMPLETE + _Info.ELAPSED,
                index + 1,
                total_batches,
                processed_scenarios,
                scenarios_remaining,
                elapsed,
                avg_batch_time,
                remaining_time,
            )

            # free memory after processing this batch
            try:
                del batch_results
                del batch_df
            except Exception:
                pass
            gc.collect()

        logger.info(
            "Execution phases: simulation=%.2fs, dataframe=%.2fs, "
            "cache_write=%.2fs",
            simulation_seconds,
            dataframe_seconds,
            cache_write_seconds,
        )

    # Step 2: Read all temp files and combine by scenario pair
    logger.info(_Info.COMBINING_TEMP_BATCHES, len(temp_files))
    combine_started = time.perf_counter()
    if temp_files:
        all_results_df = pl.concat(
            [pl.read_parquet(f) for f in temp_files],
            how="vertical",
        )
    else:
        all_results_df = pl.DataFrame()
    combine_seconds = time.perf_counter() - combine_started

    if all_results_df.is_empty():
        logger.warning(_Errors.NO_RESULTS_COMBINED)
        return []

    # Optional debug output: keep combined data for inspection if needed
    combined_path = output_dir / "all_results_combined.parquet"
    try:
        all_results_df.write_parquet(combined_path)
        logger.info(_Info.SAVED_COMBINED_RESULTS, combined_path)
    except Exception:
        logger.exception(_Errors.FAILED_TO_SAVE_COMBINED_RESULTS, combined_path)

    # Step 3: Group and write per-pair parquet files
    saved_files = []
    final_write_started = time.perf_counter()
    try:
        all_pairs = pair_scenarios(all_results_df["scenario"].unique().to_list())
    except Exception:
        logger.exception(_Errors.FAILED_TO_PAIR)
        raise

    if not all_pairs:
        logger.warning(_Errors.NO_PAIRS_FOUND)

    for control, intervention in all_pairs:
        control_df = all_results_df.filter(pl.col("scenario") == control)
        intervention_df = all_results_df.filter(pl.col("scenario") == intervention)

        if control_df.is_empty() or intervention_df.is_empty():
            logger.warning(_Errors.SKIPPING_PAIR, control, intervention)
            continue

        # combine into a single file with a column indicating arm
        control_df = control_df.with_columns(pl.lit("control").alias("arm"))
        intervention_df = intervention_df.with_columns(pl.lit("intervention").alias("arm"))

        combined = pl.concat([control_df, intervention_df], how="vertical")

        fname = f"{_safe_name(control)}_vs_{_safe_name(intervention)}.parquet"
        path = output_dir / fname
        try:
            combined.write_parquet(path)
            saved_files.append(path)
            logger.info(_Info.SAVED_PAIR_RESULTS, control, intervention, path)
        except Exception:
            logger.exception(_Errors.FAILED_TO_SAVE_PAIR_RESULTS, control, intervention, path)

    # Step 4: Clean up temp files
    try:
        # shutil.rmtree(temp_dir)
        # logger.info("Cleaned up temp directory")
        logger.warning(_Errors.CLEAN_UP_DISABLED, temp_dir)
    except Exception:
        logger.exception(_Errors.FAILED_TO_CLEAN_TEMP_DIRECTORY, temp_dir)

    logger.info(_Info.SAVED_FINAL_RESULTS, len(saved_files), output_dir)
    logger.info(
        "Output phases: combine=%.2fs, final_write=%.2fs",
        combine_seconds,
        time.perf_counter() - final_write_started,
    )
    return saved_files
