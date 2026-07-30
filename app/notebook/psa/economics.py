"""Reusable paired economic calculations for PSA horizons."""

from __future__ import annotations

import numpy as np
import polars as pl

from app.notebook.scenario_helpers import pair_scenarios, parse_scenario

_EXTENSION_ORDER = {
    None: 0,
    "ich_pooled": 1,
    "weight_reduction_10": 2,
    "weight_reduction_20": 3,
    "weight_reduction_30": 4,
    "is_discounting": 5,
}
_METHOD_ORDER = {"bayesian": 0, "dirichlet": 1}


def scenario_sort_key(scenario: str) -> tuple[int, int, str, int]:
    """Clinical report order: extension first, then Bayesian/Dirichlet."""
    _horizon, regime, method, extension = parse_scenario(scenario)
    return (
        _EXTENSION_ORDER.get(extension, 99),
        _METHOD_ORDER.get(method, 99),
        extension or "",
        0 if regime == "on-demand" else 1,
    )


def pair_sort_key(pair: tuple[str, str]) -> tuple[int, int, str, int]:
    return scenario_sort_key(pair[0])


def scenario_pairs(df: pl.DataFrame) -> list[tuple[str, str]]:
    pairs = pair_scenarios(df["scenario"].unique().to_list())
    return sorted(pairs, key=pair_sort_key)


def sort_report_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Sort a report table using any scenario identifier it contains."""
    for column in ("scenario", "base_scenario", "comparison_scenario"):
        if column in df.columns:
            return (
                df.with_columns(
                    pl.col(column)
                    .map_elements(
                        lambda value: (
                            scenario_sort_key(value)[0] * 10
                            + scenario_sort_key(value)[1]
                        ),
                        return_dtype=pl.Int64,
                    )
                    .alias("_report_order")
                )
                .sort("_report_order")
                .drop("_report_order")
            )
    if "comparison" in df.columns:
        return (
            df.with_columns(
                pl.col("comparison")
                .str.split(" vs ")
                .list.last()
                .map_elements(
                    lambda value: (
                        scenario_sort_key(value)[0] * 10
                        + scenario_sort_key(value)[1]
                    ),
                    return_dtype=pl.Int64,
                )
                .alias("_report_order")
            )
            .sort("_report_order")
            .drop("_report_order")
        )
    return df


def paired_outcomes(
    df: pl.DataFrame,
    base: str,
    comparison: str,
    *,
    wtp: float,
) -> pl.DataFrame:
    """Join two scenario arms one-to-one and calculate incremental outcomes."""
    required = {"iteration_id", "scenario", "total_cost", "total_qaly"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PSA results are missing required columns: {sorted(missing)}")

    base_df = df.filter(pl.col("scenario") == base).select(
        "iteration_id",
        pl.col("total_cost").alias("base_cost"),
        pl.col("total_qaly").alias("base_qaly"),
        *(
            [pl.col("sampled_abr").alias("base_sampled_abr")]
            if "sampled_abr" in df.columns
            else []
        ),
    )
    comp_df = df.filter(pl.col("scenario") == comparison).select(
        "iteration_id",
        pl.col("total_cost").alias("comparison_cost"),
        pl.col("total_qaly").alias("comparison_qaly"),
        *(
            [pl.col("sampled_abr").alias("comparison_sampled_abr")]
            if "sampled_abr" in df.columns
            else []
        ),
    )
    paired = base_df.join(
        comp_df,
        on="iteration_id",
        how="inner",
        validate="1:1",
    )
    if paired.height != base_df.height or paired.height != comp_df.height:
        raise ValueError(
            f"Incomplete iteration pairing for {comparison} vs {base}: "
            f"base={base_df.height}, comparison={comp_df.height}, "
            f"paired={paired.height}"
        )
    return paired.with_columns(
        (pl.col("comparison_cost") - pl.col("base_cost")).alias("delta_cost"),
        (pl.col("comparison_qaly") - pl.col("base_qaly")).alias("delta_qaly"),
    ).with_columns(
        (pl.col("delta_qaly") * wtp - pl.col("delta_cost")).alias("delta_nmb")
    )


def abr_threshold_curve(
    df: pl.DataFrame,
    base: str,
    comparison: str,
    *,
    wtp: float,
    points: int = 21,
    min_pairs: int = 30,
) -> pl.DataFrame:
    """Estimate paired cost effectiveness above baseline-ABR cutoffs.

    The ABR cutoff is applied to the base (on-demand) arm. Both arms then
    retain exactly the same iteration IDs. The ICER is a ratio of cohort
    means, not the mean of unstable iteration-level ICERs.
    """
    if points < 2:
        raise ValueError("points must be at least 2")
    paired = paired_outcomes(df, base, comparison, wtp=wtp)
    if "base_sampled_abr" not in paired.columns:
        raise ValueError("PSA results need sampled_abr for ABR-threshold analysis")

    abr = paired["base_sampled_abr"].to_numpy()
    finite_abr = abr[np.isfinite(abr)]
    if finite_abr.size == 0:
        return pl.DataFrame()

    # Quantile-spaced cutoffs give useful resolution without leaving almost
    # empty high-ABR subsets. Duplicate cutoffs are deliberately collapsed.
    quantiles = np.linspace(0.0, 0.95, points)
    cutoffs = np.unique(np.quantile(finite_abr, quantiles))
    rows: list[dict[str, float | int | str | bool]] = []
    for cutoff in cutoffs:
        subset = paired.filter(pl.col("base_sampled_abr") >= float(cutoff))
        if subset.height < min_pairs:
            continue
        delta_cost = float(subset["delta_cost"].mean())
        delta_qaly = float(subset["delta_qaly"].mean())
        delta_nmb = float(subset["delta_nmb"].mean())
        rows.append(
            {
                "base_scenario": base,
                "comparison_scenario": comparison,
                "abr_cutoff": float(cutoff),
                "paired_iterations": subset.height,
                "mean_base_abr": float(subset["base_sampled_abr"].mean()),
                "delta_cost": delta_cost,
                "delta_qaly": delta_qaly,
                "icer": (
                    delta_cost / delta_qaly
                    if not np.isclose(delta_qaly, 0)
                    else float("nan")
                ),
                "delta_nmb": delta_nmb,
                "probability_cost_effective": float(
                    (subset["delta_nmb"] > 0).mean()
                ),
            }
        )
    return pl.DataFrame(rows)


def _linear_crossing(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return the first interpolated x where y crosses zero."""
    exact = np.flatnonzero(np.isclose(y, 0))
    if exact.size:
        return float(x[exact[0]])
    changes = np.flatnonzero(np.signbit(y[:-1]) != np.signbit(y[1:]))
    if not changes.size:
        return None
    i = int(changes[0])
    return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))


def abr_threshold_analysis(
    df: pl.DataFrame,
    *,
    wtp: float,
    points: int = 21,
    min_pairs: int = 30,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return detailed curves and a decision-focused ABR threshold table."""
    curves: list[pl.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for base, comparison in scenario_pairs(df):
        curve = abr_threshold_curve(
            df,
            base,
            comparison,
            wtp=wtp,
            points=points,
            min_pairs=min_pairs,
        )
        if curve.is_empty():
            continue
        curves.append(curve)
        ordered = curve.sort("abr_cutoff")
        cutoff = ordered["abr_cutoff"].to_numpy()
        nmb = ordered["delta_nmb"].to_numpy()
        crossing = _linear_crossing(cutoff, nmb)
        baseline_abr = float(
            df.filter(pl.col("scenario") == base)["sampled_abr"].mean()
        )
        summaries.append(
            {
                "comparison": f"{comparison} vs {base}",
                "paired_iterations": int(ordered["paired_iterations"].max()),
                "baseline_on_demand_abr": baseline_abr,
                "cost_effective_abr_threshold": crossing,
                "abr_margin": (
                    baseline_abr - crossing if crossing is not None else None
                ),
                "threshold_found": crossing is not None,
                "observed_cutoff_min": float(cutoff.min()),
                "observed_cutoff_max": float(cutoff.max()),
                "probability_ce_at_baseline_cutoff": float(
                    ordered["probability_cost_effective"][0]
                ),
            }
        )
    curve_frame = pl.concat(curves) if curves else pl.DataFrame()
    return curve_frame, pl.DataFrame(summaries)


def _interpret(delta_cost: float, delta_qaly: float, wtp: float) -> str:
    if delta_cost < 0 and delta_qaly > 0:
        return "Dominant"
    if delta_cost > 0 and delta_qaly < 0:
        return "Dominated"
    if np.isclose(delta_qaly, 0):
        return "Undefined"
    if delta_cost < 0 and delta_qaly < 0:
        return "Trade-off (less effective, cheaper)"
    return "Cost-effective" if delta_cost / delta_qaly < wtp else "Not cost-effective"


def economic_summary(df: pl.DataFrame, *, wtp: float) -> pl.DataFrame:
    """Summarize paired incremental cost, QALY, ICER, NMB, and uncertainty."""
    rows = []
    for base, comparison in scenario_pairs(df):
        paired = paired_outcomes(df, base, comparison, wtp=wtp)
        dc = float(paired["delta_cost"].mean())
        dq = float(paired["delta_qaly"].mean())
        cost_ci = paired["delta_cost"].quantile(0.025), paired["delta_cost"].quantile(0.975)
        qaly_ci = paired["delta_qaly"].quantile(0.025), paired["delta_qaly"].quantile(0.975)
        rows.append(
            {
                "comparison": f"{comparison} vs {base}",
                "paired_iterations": paired.height,
                "delta_cost": dc,
                "delta_cost_ci_low": cost_ci[0],
                "delta_cost_ci_high": cost_ci[1],
                "delta_qaly": dq,
                "delta_qaly_ci_low": qaly_ci[0],
                "delta_qaly_ci_high": qaly_ci[1],
                "icer": dc / dq if not np.isclose(dq, 0) else np.nan,
                "interpretation": _interpret(dc, dq, wtp),
                "delta_nmb": paired["delta_nmb"].mean(),
                "probability_cost_effective": (paired["delta_nmb"] > 0).mean(),
            }
        )
    return pl.DataFrame(rows)


def ceac(
    df: pl.DataFrame,
    *,
    thresholds: np.ndarray,
) -> pl.DataFrame:
    """Calculate paired probability cost-effective over WTP thresholds."""
    rows = []
    for base, comparison in scenario_pairs(df):
        paired = paired_outcomes(df, base, comparison, wtp=0)
        dc = paired["delta_cost"].to_numpy()
        dq = paired["delta_qaly"].to_numpy()
        for threshold in thresholds:
            rows.append(
                {
                    "comparison": f"{comparison} vs {base}",
                    "wtp": float(threshold),
                    "probability_cost_effective": float(
                        (dq * threshold - dc > 0).mean()
                    ),
                }
            )
    return pl.DataFrame(rows)


def ceac_threshold_summary(
    df: pl.DataFrame,
    *,
    selected_wtp: float,
    maximum_wtp: float | None = None,
    points: int = 401,
) -> pl.DataFrame:
    """Report the WTP where paired P(cost-effective) first reaches 50%."""
    upper = maximum_wtp or max(selected_wtp * 3, 1)
    thresholds = np.linspace(0, upper, points)
    curves = ceac(df, thresholds=thresholds)
    rows = []
    for base, comparison in scenario_pairs(df):
        label = f"{comparison} vs {base}"
        sub = curves.filter(pl.col("comparison") == label).sort("wtp")
        x = sub["wtp"].to_numpy()
        probability = sub["probability_cost_effective"].to_numpy()
        crossing = _linear_crossing(x, probability - 0.5)
        paired = paired_outcomes(df, base, comparison, wtp=selected_wtp)
        rows.append(
            {
                "comparison": label,
                "paired_iterations": paired.height,
                "selected_wtp": selected_wtp,
                "probability_ce_at_selected_wtp": float(
                    (paired["delta_nmb"] > 0).mean()
                ),
                "wtp_at_50_percent_ce": crossing,
                "crossing_found_in_range": crossing is not None,
                "searched_wtp_max": upper,
            }
        )
    return pl.DataFrame(rows)
