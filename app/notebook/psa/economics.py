"""Reusable paired economic calculations for PSA horizons."""

from __future__ import annotations

import numpy as np
import polars as pl

from app.notebook.scenario_helpers import pair_scenarios


def scenario_pairs(df: pl.DataFrame) -> list[tuple[str, str]]:
    return pair_scenarios(sorted(df["scenario"].unique().to_list()))


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
    )
    comp_df = df.filter(pl.col("scenario") == comparison).select(
        "iteration_id",
        pl.col("total_cost").alias("comparison_cost"),
        pl.col("total_qaly").alias("comparison_qaly"),
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
