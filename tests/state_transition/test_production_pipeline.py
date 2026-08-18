from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from modular_models.state_transition.analysis import StudyRunner
from modular_models.state_transition.context import StudyContext
from modular_models.state_transition.production import (
    OWSAConfig,
    OWSAProductionPipeline,
    PSAConfig,
    PSAProductionPipeline,
)
from modular_models.state_transition.reporting import (
    calibration_table,
    incremental_outcomes,
    psa_inner_loop_precision,
    psa_iteration_convergence,
    strategy_outcomes,
    validation_table,
)


def test_psa_pipeline_checkpoints_resumes_and_preserves_draws(tmp_path: Path) -> None:
    context = StudyContext.load()
    config = PSAConfig(
        iterations=2,
        n_patients=10,
        output_dir=tmp_path / "psa",
        seed=91,
        batch_size=1,
    )
    pipeline = PSAProductionPipeline(context, config)
    first = pipeline.run()
    second = pipeline.run()

    assert first.height == second.height == 2
    assert pipeline.draws_path.is_file()
    assert pipeline.results_path.is_file()
    assert len(list(pipeline.parts_dir.glob("psa_*.parquet"))) == 2
    assert "relative_mortality_reduction" in first.columns
    manifest = json.loads(pipeline.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["completed"] == 2


def test_owsa_pipeline_resumes_endpoint_results(tmp_path: Path) -> None:
    context = StudyContext.load()
    config = OWSAConfig(
        n_patients=10,
        output_dir=tmp_path / "owsa",
        parameter_ids=("factor_price_irr_per_iu",),
    )
    pipeline = OWSAProductionPipeline(context, config)
    first = pipeline.run()
    second = pipeline.run()

    assert first.height == second.height == 2
    assert set(first["endpoint"]) == {"low", "high"}
    assert pipeline.results_path.is_file()


def test_infeasible_high_ajbr_is_linked_to_high_abr(tmp_path: Path) -> None:
    context = StudyContext.load()
    frame = OWSAProductionPipeline(
        context,
        OWSAConfig(
            n_patients=10,
            output_dir=tmp_path / "linked_owsa",
            parameter_ids=("ajbr_on_demand",),
        ),
    ).run()

    high = frame.filter(frame["endpoint"] == "high").row(0, named=True)
    assert high["status"] == "complete"
    assert high["analysis_type"] == "linked_endpoint"
    assert high["linked_parameter_id"] == "abr_on_demand"
    assert high["linked_endpoint_value"] == 15.6
    assert '"abr_on_demand": 15.6' in high["overrides_json"]


def test_process_parallel_psa_matches_sequential_draws(tmp_path: Path) -> None:
    context = StudyContext.load()
    sequential = PSAProductionPipeline(
        context,
        PSAConfig(
            iterations=2,
            n_patients=10,
            output_dir=tmp_path / "sequential",
            seed=771,
            batch_size=2,
            n_jobs=1,
        ),
    ).run()
    parallel_pipeline = PSAProductionPipeline(
        context,
        PSAConfig(
            iterations=2,
            n_patients=10,
            output_dir=tmp_path / "parallel",
            seed=771,
            batch_size=2,
            n_jobs=2,
        ),
    )
    parallel = parallel_pipeline.run()

    assert sequential.select(parallel.columns).equals(parallel)
    manifest = json.loads(parallel_pipeline.manifest_path.read_text(encoding="utf-8"))
    assert manifest["parallel_backend"] == "process"
    assert manifest["effective_jobs"] == 2


def test_reporting_tables_cover_economic_clinical_and_validation_outputs() -> None:
    context = StudyContext.load()
    comparison = StudyRunner(context).compare(n_patients=50, seed=181)

    assert strategy_outcomes(comparison).height == 2
    assert "relative_mortality_reduction" in incremental_outcomes(comparison).columns
    assert calibration_table(context, comparison).height == 10
    validation = validation_table(context, comparison)
    assert validation["passed"].all()


def test_psa_inner_loop_precision_uses_paired_iterations() -> None:
    reference = pl.DataFrame(
        {
            "iteration": [0, 1, 2],
            "incremental_cost_irr": [100.0, 120.0, 80.0],
            "incremental_qaly": [1.0, 1.2, 0.8],
            "prophylaxis_cost_effective": [True, True, False],
        }
    )
    smaller = reference.with_columns(
        (pl.col("incremental_cost_irr") * 1.005).alias("incremental_cost_irr"),
        (pl.col("incremental_qaly") * 0.995).alias("incremental_qaly"),
    )
    result = psa_inner_loop_precision({1000: smaller, 5000: reference})

    first = result.row(0, named=True)
    assert first["common_iterations"] == 3
    assert first["means_within_threshold"]
    assert first["cost_effectiveness_agreement"] == 1.0


def test_psa_iteration_convergence_tracks_checkpoint_changes() -> None:
    frame = pl.DataFrame(
        {
            "iteration": list(range(20)),
            "incremental_cost_irr": [100.0] * 20,
            "incremental_qaly": [1.0] * 20,
            "incremental_nmb_irr": [-20.0] * 20,
        }
    )
    result = psa_iteration_convergence(frame, [5, 10, 20])

    assert result["iterations"].to_list() == [5, 10, 20]
    assert result.row(1, named=True)["converged_from_previous_checkpoint"]
    assert result.row(2, named=True)["probability_mcse"] == 0.0
