from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import polars as pl

from modular_models.state_transition.context import StudyContext
from modular_models.state_transition.engine import derive_event_rates
from modular_models.state_transition.results import ComparisonResult
from modular_models.state_transition.sampling import ParameterResolver
from modular_models.state_transition.types import Strategy


def parameter_table(context: StudyContext) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    for parameter in context.parameters.values():
        records.append(
            {
                "parameter_id": parameter.id,
                "description": parameter.description,
                "base_value": parameter.value,
                "unit": parameter.unit,
                "owsa_low": parameter.owsa.low if parameter.owsa else None,
                "owsa_high": parameter.owsa.high if parameter.owsa else None,
                "psa_distribution": parameter.psa.distribution if parameter.psa else "fixed",
                "psa_parameters": json_compact(parameter.psa.parameters if parameter.psa else {}),
                "references": ", ".join(parameter.references),
                "assumption": parameter.assumption,
            }
        )
    return pl.DataFrame(records)


def json_compact(value: dict[str, float]) -> str:
    return "; ".join(f"{key}={number:g}" for key, number in value.items())


def strategy_outcomes(comparison: ComparisonResult) -> pl.DataFrame:
    rows = []
    for result in (comparison.prophylaxis, comparison.on_demand):
        summary = result.summary
        rows.append(
            {
                "strategy": result.strategy.value,
                "cost_irr": summary["mean_cost_irr"],
                "qaly": summary["mean_qaly"],
                "life_years": summary["mean_life_years"],
                "factor_viii_iu": summary["mean_factor_iu"],
                "total_bleeds": summary["mean_total_bleeds"],
                "joint_bleeds": summary["mean_joint_bleeds"],
                "non_major_non_joint_bleeds": summary["mean_non_major_non_joint_bleeds"],
                "non_ich_major_bleeds": summary["mean_non_ich_major_bleeds"],
                "ich_events": summary["mean_ich"],
                "pettersson_score": summary["mean_pettersson_score"],
                "alive_at_end": summary["alive_at_end"],
                "deaths_total": summary["deaths_total"],
                "deaths_background": summary["deaths_background"],
                "deaths_ich": summary["deaths_ich"],
                "mortality_probability": summary["all_cause_mortality_probability"],
                "post_ich_count": summary["post_ich_ever_count"],
                "post_ich_probability": summary["post_ich_ever_probability"],
            }
        )
    return pl.DataFrame(rows)


def incremental_outcomes(comparison: ComparisonResult) -> pl.DataFrame:
    p = comparison.prophylaxis.summary
    o = comparison.on_demand.summary
    bleeds_avoided = float(o["mean_total_bleeds"]) - float(p["mean_total_bleeds"])
    joint_avoided = float(o["mean_joint_bleeds"]) - float(p["mean_joint_bleeds"])
    mortality_difference = float(o["all_cause_mortality_probability"]) - float(
        p["all_cause_mortality_probability"]
    )
    ich_mortality_difference = float(o["ich_mortality_probability"]) - float(
        p["ich_mortality_probability"]
    )
    return pl.DataFrame(
        [
            {
                "incremental_cost_irr": comparison.incremental_cost_irr,
                "incremental_qaly": comparison.incremental_qaly,
                "icer_irr_per_qaly": comparison.icer_irr_per_qaly,
                "incremental_nmb_irr": comparison.incremental_nmb_irr,
                "cost_effective": comparison.is_prophylaxis_cost_effective,
                "bleeds_avoided": bleeds_avoided,
                "joint_bleeds_avoided": joint_avoided,
                "relative_bleed_reduction": bleeds_avoided
                / max(float(o["mean_total_bleeds"]), 1e-12),
                "incremental_cost_per_bleed_avoided_irr": comparison.incremental_cost_irr
                / max(bleeds_avoided, 1e-12),
                "absolute_mortality_reduction": mortality_difference,
                "relative_mortality_reduction": mortality_difference
                / max(float(o["all_cause_mortality_probability"]), 1e-12),
                "relative_ich_mortality_reduction": ich_mortality_difference
                / max(float(o["ich_mortality_probability"]), 1e-12),
                "deaths_avoided_per_1000": mortality_difference * 1000.0,
                "number_needed_to_treat_to_avoid_one_death": 1.0 / max(mortality_difference, 1e-12),
            }
        ]
    )


def calibration_table(context: StudyContext, comparison: ComparisonResult) -> pl.DataFrame:
    values, options = ParameterResolver(context).deterministic(comparison.prophylaxis.scenario_id)
    records: list[dict[str, Any]] = []
    for strategy, result in (
        (Strategy.PROPHYLAXIS, comparison.prophylaxis),
        (Strategy.ON_DEMAND, comparison.on_demand),
    ):
        rates = derive_event_rates(values, options, strategy)
        simulated = {
            "joint_bleed": result.summary["joint_bleed_rate_per_person_year"],
            "non_major_non_joint_bleed": result.summary["non_major_non_joint_rate_per_person_year"],
            "non_ich_major_bleed": result.summary["non_ich_major_rate_per_person_year"],
            "intracranial_hemorrhage": result.summary["ich_rate_per_person_year"],
            "total_bleed": result.summary["total_bleed_rate_per_person_year"],
        }
        targets = {event.value: rate for event, rate in rates.annual.items()}
        suffix = "prophylaxis" if strategy is Strategy.PROPHYLAXIS else "on_demand"
        targets["total_bleed"] = values[f"abr_{suffix}"]
        for outcome, target in targets.items():
            estimate = float(simulated[outcome])
            relative_error = (estimate - target) / max(abs(target), 1e-12)
            exposure = float(result.summary["mean_life_years"]) * result.n_patients
            expected_events = target * exposure
            observed_events = estimate * exposure
            poisson_z = (observed_events - expected_events) / np.sqrt(
                max(expected_events, 1e-12)
            )
            records.append(
                {
                    "strategy": strategy.value,
                    "calibration_target": outcome,
                    "target_rate_per_person_year": target,
                    "simulated_rate_per_person_year": estimate,
                    "relative_error": relative_error,
                    "absolute_relative_error": abs(relative_error),
                    "within_5_percent": abs(relative_error) <= 0.05,
                    "person_year_exposure": exposure,
                    "expected_events": expected_events,
                    "observed_events": observed_events,
                    "poisson_z_score": poisson_z,
                    "within_poisson_95_percent": abs(poisson_z) <= 1.96,
                }
            )
    return pl.DataFrame(records)


def validation_table(context: StudyContext, comparison: ComparisonResult) -> pl.DataFrame:
    records: list[dict[str, Any]] = []
    for result in (comparison.prophylaxis, comparison.on_demand):
        summary = result.summary
        prefix = result.strategy.value
        checks = {
            "population_reconciles": int(summary["initial_patients"])
            == int(summary["alive_at_end"])
            + int(summary["deaths_background"])
            + int(summary["deaths_ich"]),
            "qaly_non_negative": float(summary["mean_qaly"]) >= 0,
            "qaly_below_undiscounted_horizon": float(summary["mean_qaly"])
            <= float(summary["follow_up_years"]),
            "cost_non_negative": float(summary["mean_cost_irr"]) >= 0,
            "factor_non_negative": float(summary["mean_factor_iu"]) >= 0,
            "exit_age_exclusive": float(summary["last_cycle_start_age_years"]) < 12,
            "cycle_count_572": int(summary["n_cycles"]) == 572,
        }
        for check, passed in checks.items():
            records.append(
                {"scope": prefix, "check": check, "passed": bool(passed), "detail": None}
            )

    values = context.base_values()
    for parameter_id in (
        "background_mortality_age_1_4",
        "background_mortality_age_5_9",
        "background_mortality_age_10_lt12",
    ):
        hazard = values[parameter_id]
        probability = 1.0 - np.exp(-hazard / values["cycles_per_year"])
        records.append(
            {
                "scope": "mortality",
                "check": f"weekly_probability_{parameter_id}",
                "passed": bool(0 <= probability <= 1),
                "detail": f"{probability:.12g}",
            }
        )
    return pl.DataFrame(records)


def psa_summary(frame: pl.DataFrame) -> pl.DataFrame:
    metrics = (
        "incremental_cost_irr",
        "incremental_qaly",
        "icer_irr_per_qaly",
        "incremental_nmb_irr",
        "mean_bleeds_avoided",
        "absolute_mortality_reduction",
        "relative_mortality_reduction",
    )
    records = []
    for metric in metrics:
        if metric not in frame.columns:
            continue
        series = frame[metric].cast(pl.Float64).drop_nans().drop_nulls()
        records.append(
            {
                "metric": metric,
                "n": len(series),
                "mean": series.mean(),
                "sd": series.std(),
                "median": series.median(),
                "p2_5": series.quantile(0.025, interpolation="linear"),
                "p97_5": series.quantile(0.975, interpolation="linear"),
            }
        )
    return pl.DataFrame(records)


def ceac_table(frame: pl.DataFrame, wtp_values: np.ndarray) -> pl.DataFrame:
    delta_cost = frame["incremental_cost_irr"].to_numpy()
    delta_qaly = frame["incremental_qaly"].to_numpy()
    return pl.DataFrame(
        {
            "wtp_irr_per_qaly": wtp_values,
            "probability_cost_effective": [
                float(np.mean(wtp * delta_qaly - delta_cost > 0)) for wtp in wtp_values
            ],
        }
    )


def psa_inner_loop_precision(
    frames: Mapping[int, pl.DataFrame],
    *,
    reference_size: int | None = None,
    relative_mean_threshold: float = 0.01,
) -> pl.DataFrame:
    """Compare first-order patient noise using common PSA draws and seeds.

    The largest supplied population is the default reference. This diagnostic
    does not replace second-order convergence; it identifies an economical
    patient count for each PSA iteration.
    """
    if not frames:
        raise ValueError("At least one PSA frame is required")
    chosen_reference = reference_size or max(frames)
    if chosen_reference not in frames:
        raise KeyError(f"Missing reference population size: {chosen_reference}")
    reference = frames[chosen_reference].select(
        "iteration",
        pl.col("incremental_cost_irr").alias("reference_cost"),
        pl.col("incremental_qaly").alias("reference_qaly"),
        pl.col("prophylaxis_cost_effective").alias("reference_ce"),
    )
    reference_cost_mean = float(cast(float, reference["reference_cost"].mean()))
    reference_qaly_mean = float(cast(float, reference["reference_qaly"].mean()))
    reference_cost_sd = float(cast(float, reference["reference_cost"].std()))
    reference_qaly_sd = float(cast(float, reference["reference_qaly"].std()))

    records: list[dict[str, Any]] = []
    for population_size in sorted(frames):
        joined = frames[population_size].select(
            "iteration",
            "incremental_cost_irr",
            "incremental_qaly",
            "prophylaxis_cost_effective",
        ).join(reference, on="iteration", how="inner")
        if joined.is_empty():
            raise ValueError(f"No common iterations for population size {population_size}")
        cost_error = (
            joined["incremental_cost_irr"] - joined["reference_cost"]
        ).to_numpy()
        qaly_error = (joined["incremental_qaly"] - joined["reference_qaly"]).to_numpy()
        cost_mean = float(cast(float, joined["incremental_cost_irr"].mean()))
        qaly_mean = float(cast(float, joined["incremental_qaly"].mean()))
        relative_cost = abs(cost_mean - reference_cost_mean) / max(
            abs(reference_cost_mean), 1e-12
        )
        relative_qaly = abs(qaly_mean - reference_qaly_mean) / max(
            abs(reference_qaly_mean), 1e-12
        )
        records.append(
            {
                "n_patients_per_strategy": population_size,
                "reference_patients": chosen_reference,
                "common_iterations": joined.height,
                "mean_incremental_cost_irr": cost_mean,
                "mean_incremental_qaly": qaly_mean,
                "relative_mean_difference_cost": relative_cost,
                "relative_mean_difference_qaly": relative_qaly,
                "paired_rmse_cost_irr": float(np.sqrt(np.mean(np.square(cost_error)))),
                "paired_rmse_qaly": float(np.sqrt(np.mean(np.square(qaly_error)))),
                "noise_to_parameter_sd_cost": float(cost_error.std(ddof=1))
                / max(reference_cost_sd, 1e-12),
                "noise_to_parameter_sd_qaly": float(qaly_error.std(ddof=1))
                / max(reference_qaly_sd, 1e-12),
                "cost_effectiveness_agreement": float(
                    np.mean(
                        joined["prophylaxis_cost_effective"].to_numpy()
                        == joined["reference_ce"].to_numpy()
                    )
                ),
                "means_within_threshold": relative_cost <= relative_mean_threshold
                and relative_qaly <= relative_mean_threshold,
            }
        )
    return pl.DataFrame(records)


def psa_iteration_convergence(
    frame: pl.DataFrame,
    checkpoints: list[int] | tuple[int, ...],
    *,
    relative_mean_threshold: float = 0.01,
    probability_change_threshold: float = 0.01,
) -> pl.DataFrame:
    """Track cumulative PSA estimates as second-order iterations increase."""
    if frame.is_empty():
        raise ValueError("PSA frame cannot be empty")
    ordered = frame.sort("iteration")
    selected = sorted({point for point in checkpoints if 1 < point <= ordered.height})
    if not selected or selected[-1] != ordered.height:
        selected.append(ordered.height)
    records: list[dict[str, Any]] = []
    previous_cost: float | None = None
    previous_qaly: float | None = None
    previous_probability: float | None = None
    for point in selected:
        prefix = ordered.head(point)
        cost = prefix["incremental_cost_irr"].to_numpy()
        qaly = prefix["incremental_qaly"].to_numpy()
        inmb = prefix["incremental_nmb_irr"].to_numpy()
        probability = float(np.mean(inmb > 0.0))
        mean_cost = float(np.mean(cost))
        mean_qaly = float(np.mean(qaly))
        relative_cost = (
            None
            if previous_cost is None
            else abs(mean_cost - previous_cost) / max(abs(previous_cost), 1e-12)
        )
        relative_qaly = (
            None
            if previous_qaly is None
            else abs(mean_qaly - previous_qaly) / max(abs(previous_qaly), 1e-12)
        )
        probability_change = (
            None
            if previous_probability is None
            else abs(probability - previous_probability)
        )
        converged = (
            relative_cost is not None
            and relative_qaly is not None
            and probability_change is not None
            and relative_cost <= relative_mean_threshold
            and relative_qaly <= relative_mean_threshold
            and probability_change <= probability_change_threshold
        )
        records.append(
            {
                "iterations": point,
                "mean_incremental_cost_irr": mean_cost,
                "mean_incremental_qaly": mean_qaly,
                "mean_incremental_nmb_irr": float(np.mean(inmb)),
                "p2_5_incremental_cost_irr": float(np.quantile(cost, 0.025)),
                "p97_5_incremental_cost_irr": float(np.quantile(cost, 0.975)),
                "p2_5_incremental_qaly": float(np.quantile(qaly, 0.025)),
                "p97_5_incremental_qaly": float(np.quantile(qaly, 0.975)),
                "probability_cost_effective": probability,
                "probability_mcse": float(
                    np.sqrt(probability * (1.0 - probability) / point)
                ),
                "relative_change_mean_cost": relative_cost,
                "relative_change_mean_qaly": relative_qaly,
                "absolute_change_probability": probability_change,
                "converged_from_previous_checkpoint": converged,
            }
        )
        previous_cost = mean_cost
        previous_qaly = mean_qaly
        previous_probability = probability
    return pl.DataFrame(records)


def cost_effectiveness_quadrants(frame: pl.DataFrame) -> pl.DataFrame:
    cost = frame["incremental_cost_irr"].to_numpy()
    qaly = frame["incremental_qaly"].to_numpy()
    labels = {
        "north_east_more_cost_more_effect": (cost >= 0) & (qaly >= 0),
        "south_east_dominant": (cost < 0) & (qaly >= 0),
        "north_west_dominated": (cost >= 0) & (qaly < 0),
        "south_west_less_cost_less_effect": (cost < 0) & (qaly < 0),
    }
    return pl.DataFrame(
        [
            {"quadrant": label, "count": int(mask.sum()), "proportion": float(mask.mean())}
            for label, mask in labels.items()
        ]
    )
