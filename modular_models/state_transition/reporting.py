from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Base-case economic summary and WTP sensitivity
# ---------------------------------------------------------------------------


def economic_summary_table(comparison: ComparisonResult) -> pl.DataFrame:
    """Per-arm mean cost, QALY, ICER and INMB at the supplied WTP threshold."""
    p = comparison.prophylaxis.summary
    o = comparison.on_demand.summary
    return pl.DataFrame(
        [
            {
                "metric": "Cost - prophylaxis",
                "value": float(p["mean_cost_irr"]),
                "unit": "IRR/patient",
            },
            {
                "metric": "Cost - on demand",
                "value": float(o["mean_cost_irr"]),
                "unit": "IRR/patient",
            },
            {
                "metric": "Incremental cost",
                "value": comparison.incremental_cost_irr,
                "unit": "IRR/patient",
            },
            {
                "metric": "Incremental QALY",
                "value": comparison.incremental_qaly,
                "unit": "QALY/patient",
            },
            {
                "metric": "ICER",
                "value": comparison.icer_irr_per_qaly,
                "unit": "IRR/QALY",
            },
            {
                "metric": "Incremental NMB",
                "value": comparison.incremental_nmb_irr,
                "unit": "IRR/patient",
            },
        ]
    )


def wtp_sensitivity_table(
    delta_cost: float, delta_qaly: float, wtp_values: np.ndarray
) -> pl.DataFrame:
    """INMB and the preferred strategy at each supplied WTP value."""
    wtp_values = np.asarray(wtp_values, dtype=float)
    delta_cost = float(delta_cost)
    delta_qaly = float(delta_qaly)
    inmb = wtp_values * delta_qaly - delta_cost
    return pl.DataFrame(
        {
            "wtp_irr_per_qaly": wtp_values,
            "incremental_nmb_irr": inmb,
        }
    ).with_columns(
        pl.when(pl.col("incremental_nmb_irr") > 0)
        .then(pl.lit("Prophylaxis"))
        .when(pl.col("incremental_nmb_irr") < 0)
        .then(pl.lit("On-demand"))
        .otherwise(pl.lit("Break-even"))
        .alias("preferred_strategy")
    )


def wtp_threshold_summary(
    delta_cost: float, delta_qaly: float, primary_wtp: float
) -> pl.DataFrame:
    """Deterministic break-even WTP and the ratio to the primary threshold."""
    delta_cost = float(delta_cost)
    delta_qaly = float(delta_qaly)
    primary_wtp = float(primary_wtp)
    break_even = delta_cost / delta_qaly if delta_qaly > 0 else float("nan")
    return pl.DataFrame(
        [
            {
                "incremental_cost_irr": delta_cost,
                "incremental_qaly": delta_qaly,
                "primary_wtp_irr_per_qaly": primary_wtp,
                "break_even_wtp_irr_per_qaly": break_even,
                "break_even_wtp_billion_irr_per_qaly": break_even / 1e9,
                "primary_wtp_as_fraction_of_break_even": primary_wtp / break_even,
                "break_even_to_primary_wtp_ratio": break_even / primary_wtp,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Extended patient-level validation
# ---------------------------------------------------------------------------


def extended_validation_table(comparison: ComparisonResult) -> pl.DataFrame:
    """Patient-level invariant audit that requires retained patient records."""
    rows: list[dict[str, Any]] = []
    for result in (comparison.prophylaxis, comparison.on_demand):
        frame = result.to_polars(patient_level=True)
        strategy = result.strategy.value
        summary = result.summary

        calculated_mean_total_bleeds = float(
            frame.select(
                (
                    pl.col("joint_bleeds")
                    + pl.col("non_major_non_joint_bleeds")
                    + pl.col("non_ich_major_bleeds")
                    + pl.col("ich_events")
                ).mean().alias("mean_total")
            )["mean_total"][0]
        )

        rows.extend(
            [
                {
                    "strategy": strategy,
                    "check": "pettersson_within_0_78",
                    "passed": bool(
                        frame["pettersson_score"].min() >= 0
                        and frame["pettersson_score"].max() <= 78
                    ),
                },
                {
                    "strategy": strategy,
                    "check": "patient_qaly_non_negative",
                    "passed": bool((frame["total_qaly"] >= 0).all()),
                },
                {
                    "strategy": strategy,
                    "check": "patient_qaly_not_above_life_years",
                    "passed": bool(
                        (frame["total_qaly"] <= frame["life_years"] + 1e-10).all()
                    ),
                },
                {
                    "strategy": strategy,
                    "check": "patient_cost_non_negative",
                    "passed": bool((frame["total_cost_irr"] >= 0).all()),
                },
                {
                    "strategy": strategy,
                    "check": "patient_factor_non_negative",
                    "passed": bool((frame["total_factor_iu"] >= 0).all()),
                },
                {
                    "strategy": strategy,
                    "check": "bleed_components_reconcile",
                    "passed": bool(
                        np.isclose(
                            calculated_mean_total_bleeds,
                            float(summary["mean_total_bleeds"]),
                            rtol=0,
                            atol=1e-10,
                        )
                    ),
                },
            ]
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Auditable patient and cycle samples
# ---------------------------------------------------------------------------


def auditable_patient_sample(comparison: ComparisonResult, n: int = 10) -> pl.DataFrame:
    """Top-N on-demand patients by joint-bleed count with end-of-horizon outcomes."""
    return (
        comparison.on_demand.to_polars(patient_level=True)
        .sort("joint_bleeds", descending=True)
        .head(n)
        .select(
            "patient_id",
            "life_years",
            "total_qaly",
            "total_cost_irr",
            "total_factor_iu",
            "joint_bleeds",
            "non_major_non_joint_bleeds",
            "non_ich_major_bleeds",
            "ich_events",
            "pettersson_score",
            "ever_post_ich",
            "death_cause",
            "death_age_years",
            "final_state",
        )
    )


def auditable_cycle_sample(trace_json_path: Path, run_index: int = 1) -> pl.DataFrame:
    """Read the trace JSON and return the per-cycle record of the chosen strategy run."""
    payload = json.loads(Path(trace_json_path).read_text(encoding="utf-8"))
    return pl.DataFrame(payload["runs"][run_index]["cycles"])


# ---------------------------------------------------------------------------
# CEAC helpers
# ---------------------------------------------------------------------------


def selected_ceac_table(
    ceac: pl.DataFrame, wtp_billion_values: list[float]
) -> pl.DataFrame:
    """Return the CEAC rows whose WTP (in billion IRR/QALY) matches any input value."""
    annotated = ceac.with_columns(
        (pl.col("wtp_irr_per_qaly") / 1e9).alias("wtp_billion_irr_per_qaly")
    )
    return annotated.filter(pl.col("wtp_billion_irr_per_qaly").is_in(wtp_billion_values))


def exact_ceac_table(
    psa_frame: pl.DataFrame, wtp_values: np.ndarray, primary_wtp: float
) -> pl.DataFrame:
    """Probability cost-effective computed exactly at each WTP using PSA draws."""
    wtp_values = np.asarray(wtp_values, dtype=float)
    primary_wtp = float(primary_wtp)
    rows: list[dict[str, Any]] = []
    inmb_at_primary = psa_frame["incremental_nmb_irr"].to_numpy()
    qaly = psa_frame["incremental_qaly"].to_numpy()
    for wtp in wtp_values:
        shifted_inmb = inmb_at_primary + (wtp - primary_wtp) * qaly
        rows.append(
            {
                "wtp_billion_irr_per_qaly": wtp / 1e9,
                "probability_prophylaxis_cost_effective": float(
                    (shifted_inmb > 0).mean()
                ),
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# PSA precision diagnostics
# ---------------------------------------------------------------------------


def psa_precision_summary(psa_frame: pl.DataFrame) -> pl.DataFrame:
    """Relative MCSE of the mean incremental QALY plus CEAC worst-case margins."""
    qaly_values = psa_frame["incremental_qaly"].to_numpy()
    completed = int(psa_frame.height)
    sd = float(np.std(qaly_values, ddof=1))
    mean_abs = abs(float(np.mean(qaly_values)))
    relative_mcse = sd / np.sqrt(completed) / max(mean_abs, 1e-12)
    iterations_for_one_percent = int(
        np.ceil((sd / (0.01 * max(mean_abs, 1e-12))) ** 2)
    )
    return pl.DataFrame(
        [
            {
                "completed_iterations": completed,
                "relative_mcse_mean_incremental_qaly": relative_mcse,
                "iterations_estimated_for_1pct_relative_mcse_qaly": (
                    iterations_for_one_percent
                ),
                "worst_case_95pct_ceac_margin_at_2500": 1.96
                * np.sqrt(0.25 / 2_500),
                "worst_case_95pct_ceac_margin_at_10000": 1.96
                * np.sqrt(0.25 / 10_000),
                "rule_of_three_upper_probability_if_zero_events": 3 / completed,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Expected Value of Perfect Information
# ---------------------------------------------------------------------------


def evpi_table(
    psa_frame: pl.DataFrame, wtp_billion_values: list[float]
) -> pl.DataFrame:
    """EVPI at each supplied WTP (in billion IRR/QALY)."""
    qaly = psa_frame["incremental_qaly"].to_numpy()
    cost = psa_frame["incremental_cost_irr"].to_numpy()
    rows: list[dict[str, Any]] = []
    for wtp_billion in wtp_billion_values:
        wtp = float(wtp_billion) * 1e9
        draw_inmb = wtp * qaly - cost
        mean_inmb = float(np.mean(draw_inmb))
        evpi_perfect = float(np.mean(np.maximum(draw_inmb, 0.0)))
        evpi = evpi_perfect - max(mean_inmb, 0.0)
        rows.append(
            {
                "wtp_billion_irr_per_qaly": float(wtp_billion),
                "mean_inmb_billion_irr": mean_inmb / 1e9,
                "probability_prophylaxis_cost_effective": float(
                    (draw_inmb > 0).mean()
                ),
                "evpi_billion_irr_per_patient": evpi / 1e9,
                "evpi_irr_per_patient": evpi,
            }
        )
    return pl.DataFrame(rows)


def evpi_grid(psa_frame: pl.DataFrame, wtp_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate EVPI across a dense WTP grid; returns (wtp_grid, evpi_values)."""
    wtp_grid = np.asarray(wtp_grid, dtype=float)
    qaly = psa_frame["incremental_qaly"].to_numpy()
    cost = psa_frame["incremental_cost_irr"].to_numpy()
    values = np.fromiter(
        (
            float(np.mean(np.maximum(wtp * qaly - cost, 0.0)))
            - max(float(np.mean(wtp * qaly - cost)), 0.0)
            for wtp in wtp_grid
        ),
        dtype=float,
        count=len(wtp_grid),
    )
    return wtp_grid, values


def evpi_max(psa_frame: pl.DataFrame, wtp_grid: np.ndarray) -> tuple[float, float]:
    """Return (max_evpi_irr_per_patient, wtp_billion_at_max) on the supplied grid."""
    wtp_grid, values = evpi_grid(psa_frame, wtp_grid)
    max_idx = int(np.argmax(values))
    return float(values[max_idx]), float(wtp_grid[max_idx]) / 1e9


# ---------------------------------------------------------------------------
# Deterministic break-even and FVIII price policy
# ---------------------------------------------------------------------------


def break_even_factor_price(
    base_comparison: ComparisonResult,
    primary_wtp: float,
    base_factor_price: float,
) -> dict[str, float]:
    """Linear-cost approximation of the break-even FVIII unit price at the primary WTP."""
    base_cost = float(base_comparison.incremental_cost_irr)
    base_qaly = float(base_comparison.incremental_qaly)
    required_incremental_cost = float(primary_wtp) * base_qaly
    break_even_price = base_factor_price * required_incremental_cost / base_cost
    required_reduction = 1.0 - break_even_price / base_factor_price
    return {
        "base_factor_price_irr_per_iu": float(base_factor_price),
        "primary_wtp_irr_per_qaly": float(primary_wtp),
        "break_even_factor_price_irr_per_iu": break_even_price,
        "required_price_reduction_fraction": required_reduction,
        "required_price_reduction_percent": required_reduction * 100.0,
    }


def factor_price_policy_table(
    base_comparison: ComparisonResult,
    base_factor_price: float,
    primary_wtp: float,
    reduction_percent: np.ndarray,
) -> pl.DataFrame:
    """Deterministic incremental outcomes at each FVIII price reduction percent."""
    reduction_percent = np.asarray(reduction_percent, dtype=float)
    base_cost = float(base_comparison.incremental_cost_irr)
    base_qaly = float(base_comparison.incremental_qaly)
    rows: list[dict[str, Any]] = []
    for reduction in reduction_percent:
        ratio = 1.0 - reduction / 100.0
        new_price = float(base_factor_price) * ratio
        new_incremental_cost = base_cost * ratio
        new_icer = new_incremental_cost / base_qaly
        new_inmb = float(primary_wtp) * base_qaly - new_incremental_cost
        rows.append(
            {
                "price_reduction_percent": reduction,
                "factor_price_irr_per_iu": new_price,
                "incremental_cost_billion_irr": new_incremental_cost / 1e9,
                "incremental_qaly": base_qaly,
                "icer_billion_irr_per_qaly": new_icer / 1e9,
                "inmb_billion_irr_per_patient": new_inmb / 1e9,
                "cost_effective_at_primary_wtp": float(new_inmb >= 0),
            }
        )
    return pl.DataFrame(rows)


def factor_price_psa_table(
    psa_frame: pl.DataFrame,
    base_factor_price: float,
    primary_wtp: float,
    price_grid: np.ndarray,
) -> pl.DataFrame:
    """Per-PSA-draw INMB at each FVIII unit price on the supplied grid."""
    price_grid = np.asarray(price_grid, dtype=float)
    qaly = psa_frame["incremental_qaly"].to_numpy()
    cost = psa_frame["incremental_cost_irr"].to_numpy()
    rows: list[dict[str, Any]] = []
    for price in price_grid:
        ratio = price / float(base_factor_price)
        adjusted_cost = cost * ratio
        adjusted_inmb = float(primary_wtp) * qaly - adjusted_cost
        rows.append(
            {
                "factor_price_irr_per_iu": float(price),
                "price_reduction_percent": 100.0 * (1.0 - price / float(base_factor_price)),
                "mean_inmb_billion_irr": float(np.mean(adjusted_inmb)) / 1e9,
                "probability_cost_effective": float(np.mean(adjusted_inmb > 0)),
            }
        )
    return pl.DataFrame(rows)


def factor_price_probability_thresholds(
    factor_price_psa_df: pl.DataFrame, targets: list[float]
) -> pl.DataFrame:
    """Maximum FVIII price that achieves each target probability of cost-effectiveness."""
    rows: list[dict[str, Any]] = []
    for target in targets:
        eligible = factor_price_psa_df.filter(
            pl.col("probability_cost_effective") >= target
        ).sort("factor_price_irr_per_iu", descending=True)
        if eligible.height:
            row = eligible.row(0, named=True)
            rows.append(
                {
                    "target_probability": float(target),
                    "maximum_price_irr_per_iu": float(row["factor_price_irr_per_iu"]),
                    "required_price_reduction_percent": float(
                        row["price_reduction_percent"]
                    ),
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# OWSA ranking helpers
# ---------------------------------------------------------------------------


def owsa_parameter_ranking(
    owsa_frame: pl.DataFrame, base_inmb_irr: float, top_n: int = 15
) -> pl.DataFrame:
    """One ranked row per complete OWSA parameter with low/high INMB and ICER.

    The ranking follows ``plot_owsa_frame``'s ordering (largest INMB span first).
    Linked endpoints retain their ``analysis_type``, ``linked_parameter_id`` and
    ``linked_endpoint_value`` metadata.
    """
    base_inmb_irr = float(base_inmb_irr)
    complete = owsa_frame.filter(pl.col("status") == "complete")
    ranking = (
        complete.group_by(["parameter_id", "parameter_description", "unit"])
        .agg(
            [
                pl.col("base_value").first().alias("base_value"),
                pl.col("endpoint_value")
                .filter(pl.col("endpoint") == "low")
                .first()
                .alias("low_value"),
                pl.col("incremental_nmb_irr")
                .filter(pl.col("endpoint") == "low")
                .first()
                .alias("inmb_low_irr"),
                pl.col("icer_irr_per_qaly")
                .filter(pl.col("endpoint") == "low")
                .first()
                .alias("icer_low_irr_per_qaly"),
                pl.col("endpoint_value")
                .filter(pl.col("endpoint") == "high")
                .first()
                .alias("high_value"),
                pl.col("incremental_nmb_irr")
                .filter(pl.col("endpoint") == "high")
                .first()
                .alias("inmb_high_irr"),
                pl.col("icer_irr_per_qaly")
                .filter(pl.col("endpoint") == "high")
                .first()
                .alias("icer_high_irr_per_qaly"),
                pl.col("analysis_type")
                .filter(pl.col("endpoint") == "high")
                .first()
                .alias("high_endpoint_type"),
                pl.col("linked_parameter_id")
                .filter(pl.col("endpoint") == "high")
                .first()
                .alias("linked_parameter_id"),
                pl.col("linked_endpoint_value")
                .filter(pl.col("endpoint") == "high")
                .first()
                .alias("linked_endpoint_value"),
            ]
        )
        .with_columns(
            [
                (
                    pl.max_horizontal("inmb_low_irr", "inmb_high_irr")
                    - pl.min_horizontal("inmb_low_irr", "inmb_high_irr")
                ).alias("inmb_range_irr"),
                (pl.col("inmb_low_irr") - base_inmb_irr).alias("low_change_from_base_irr"),
                (pl.col("inmb_high_irr") - base_inmb_irr).alias("high_change_from_base_irr"),
                ((pl.col("inmb_low_irr") > 0) | (pl.col("inmb_high_irr") > 0)).alias(
                    "any_endpoint_cost_effective"
                ),
            ]
        )
        .sort("inmb_range_irr", descending=True)
        .head(top_n)
        .with_row_index("sensitivity_rank", offset=1)
    )
    return ranking.with_columns(
        [
            (pl.col("inmb_low_irr") / 1e9).alias("INMB_low_billion_IRR"),
            (pl.col("inmb_high_irr") / 1e9).alias("INMB_high_billion_IRR"),
            (pl.col("inmb_range_irr") / 1e9).alias("INMB_range_billion_IRR"),
            (pl.col("icer_low_irr_per_qaly") / 1e9).alias("ICER_low_billion_IRR_per_QALY"),
            (pl.col("icer_high_irr_per_qaly") / 1e9).alias(
                "ICER_high_billion_IRR_per_QALY"
            ),
        ]
    ).select(
        [
            "sensitivity_rank",
            "parameter_id",
            "parameter_description",
            "unit",
            "base_value",
            "low_value",
            "high_value",
            "INMB_low_billion_IRR",
            "INMB_high_billion_IRR",
            "INMB_range_billion_IRR",
            "ICER_low_billion_IRR_per_QALY",
            "ICER_high_billion_IRR_per_QALY",
            "any_endpoint_cost_effective",
            "high_endpoint_type",
            "linked_parameter_id",
            "linked_endpoint_value",
        ]
    )


# ---------------------------------------------------------------------------
# Runtime and convergence audits
# ---------------------------------------------------------------------------


def runtime_audit_table(
    production_manifest: dict[str, Any],
    cpu_manifest: dict[str, Any],
    cuda_manifest: dict[str, Any],
) -> pl.DataFrame:
    """Build the CPU/CUDA benchmark + final PSA runtime comparison table."""
    return pl.DataFrame(
        [
            {
                "run": "CPU benchmark",
                "iterations": cpu_manifest["config"]["iterations"],
                "patients_per_strategy": cpu_manifest["config"]["n_patients"],
                "worker_processes": cpu_manifest["effective_jobs"],
                "backend": "CPU JIT process pool",
                "elapsed_minutes": cpu_manifest["elapsed_seconds_this_session"] / 60.0,
                "status": cpu_manifest["status"],
            },
            {
                "run": "CUDA benchmark",
                "iterations": cuda_manifest["config"]["iterations"],
                "patients_per_strategy": cuda_manifest["config"]["n_patients"],
                "worker_processes": cuda_manifest["effective_jobs"],
                "backend": "CUDA FP64 reward kernel",
                "elapsed_minutes": cuda_manifest["elapsed_seconds_this_session"] / 60.0,
                "status": cuda_manifest["status"],
            },
            {
                "run": "Final PSA",
                "iterations": production_manifest["config"]["iterations"],
                "patients_per_strategy": production_manifest["config"]["n_patients"],
                "worker_processes": production_manifest["effective_jobs"],
                "backend": "CPU JIT process pool",
                "elapsed_minutes": (
                    production_manifest["elapsed_seconds_this_session"] / 60.0
                ),
                "status": production_manifest["status"],
            },
        ]
    )


def monte_carlo_convergence_table(convergence_records: list[dict[str, Any]]) -> pl.DataFrame:
    """Materialise the deterministic Monte Carlo convergence record list."""
    return pl.DataFrame(convergence_records)

