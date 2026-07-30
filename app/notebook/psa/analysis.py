"""Shared analysis functions for horizon-specific PSA notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import MaxNLocator, PercentFormatter

from app.notebook.psa.scenarios import HorizonSpec, get_horizon
from app.notebook.psa.workflow import load_horizon_results
from app.persistence.context import ModelContext
from utils.path_utils import get_project_root


def output_dirs(horizon: str | HorizonSpec) -> tuple[Path, Path]:
    spec = get_horizon(horizon)
    root = get_project_root()
    figures = root / "app" / "outputs" / "figures" / "psa" / spec.directory
    sheets = root / "app" / "outputs" / "sheets" / "psa" / spec.directory
    figures.mkdir(parents=True, exist_ok=True)
    sheets.mkdir(parents=True, exist_ok=True)
    return figures, sheets


def prepare_results(horizon: str | HorizonSpec) -> pl.DataFrame:
    """Load one horizon and ensure an explicit within-scenario iteration key."""
    spec = get_horizon(horizon)
    df = load_horizon_results(spec)
    if "iteration_id" not in df.columns:
        df = df.with_columns(
            pl.int_range(pl.len()).over("scenario").alias("iteration_id")
        )
    unexpected = set(df["time_horizon"].unique().to_list()) - {
        spec.key,
        "early" if spec.key == "childhood" else spec.key,
    }
    if unexpected:
        raise ValueError(
            f"{spec.label} cache contains other horizons: {sorted(unexpected)}"
        )
    return df


def base_results(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("extension").is_null())


def descriptive_summary(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["sampling_method", "regime"])
        .agg(
            simulations=pl.len(),
            mean_cost=pl.col("total_cost").mean(),
            mean_qaly=pl.col("total_qaly").mean(),
            mean_person_years=pl.col("person_years").mean(),
            absorbed_rate=pl.col("is_absorbed").mean(),
            mean_abr=pl.col("annual_bleeding_rate").mean(),
        )
        .sort(["sampling_method", "regime"])
    )


def _paired_base_results(
    df: pl.DataFrame,
    sampling_method: str,
) -> pl.DataFrame:
    sub = base_results(df).filter(pl.col("sampling_method") == sampling_method)
    base = sub.filter(pl.col("regime") == "on-demand").select(
        "iteration_id",
        pl.col("total_cost").alias("base_cost"),
        pl.col("total_qaly").alias("base_qaly"),
    )
    comp = sub.filter(pl.col("regime") == "prophylaxis").select(
        "iteration_id",
        pl.col("total_cost").alias("comp_cost"),
        pl.col("total_qaly").alias("comp_qaly"),
    )
    paired = base.join(
        comp,
        on="iteration_id",
        how="inner",
        validate="1:1",
    )
    if paired.height != base.height or paired.height != comp.height:
        raise ValueError(
            f"Incomplete {sampling_method} pairing: "
            f"on-demand={base.height}, prophylaxis={comp.height}, "
            f"paired={paired.height}"
        )
    return paired.with_columns(
        (pl.col("comp_cost") - pl.col("base_cost")).alias("delta_cost"),
        (pl.col("comp_qaly") - pl.col("base_qaly")).alias("delta_qaly"),
    )


def incremental_results(
    df: pl.DataFrame,
    *,
    wtp: float | None = None,
) -> pl.DataFrame:
    if wtp is None:
        policy = ModelContext.load().economic_policy
        wtp = policy.gdp_per_capita.IRR * policy.wtp_multiplier.rare

    rows = []
    for method in sorted(base_results(df)["sampling_method"].unique().to_list()):
        paired = _paired_base_results(df, method).with_columns(
            (pl.col("delta_qaly") * wtp - pl.col("delta_cost")).alias("delta_nmb")
        )
        rows.append(
            {
                "sampling_method": method,
                "paired_iterations": paired.height,
                "delta_cost": paired["delta_cost"].mean(),
                "delta_cost_ci_low": paired["delta_cost"].quantile(0.025),
                "delta_cost_ci_high": paired["delta_cost"].quantile(0.975),
                "delta_qaly": paired["delta_qaly"].mean(),
                "delta_qaly_ci_low": paired["delta_qaly"].quantile(0.025),
                "delta_qaly_ci_high": paired["delta_qaly"].quantile(0.975),
                "icer": (
                    paired["delta_cost"].mean() / paired["delta_qaly"].mean()
                    if not np.isclose(paired["delta_qaly"].mean(), 0)
                    else np.nan
                ),
                "delta_nmb": paired["delta_nmb"].mean(),
                "probability_cost_effective": (paired["delta_nmb"] > 0).mean(),
            }
        )
    return pl.DataFrame(rows)


def plot_cost_effectiveness_plane(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
) -> tuple[plt.Figure, np.ndarray]:
    spec = get_horizon(horizon)
    policy = ModelContext.load().economic_policy
    wtp = policy.gdp_per_capita.IRR * policy.wtp_multiplier.rare
    methods = sorted(base_results(df)["sampling_method"].unique().to_list())
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    axes = np.atleast_1d(axes)

    for ax, method in zip(axes, methods, strict=True):
        paired = _paired_base_results(df, method)
        delta_cost = paired["delta_cost"].to_numpy()
        delta_qaly = paired["delta_qaly"].to_numpy()
        delta_nmb = delta_qaly * wtp - delta_cost
        ax.scatter(delta_qaly, delta_cost, alpha=0.35, s=12, edgecolor="none")
        ax.axhline(0, color="#2E3440", linewidth=1.2)
        ax.axvline(0, color="#2E3440", linewidth=1.2)
        x = np.linspace(delta_qaly.min(), delta_qaly.max(), 100)
        ax.plot(x, x * wtp, "--", color="#272727", linewidth=2)
        ax.set_title(f"{spec.label}\n{method}")
        ax.set_xlabel("Δ QALY (prophylaxis − on-demand)")
        ax.set_ylabel("Δ cost (prophylaxis − on-demand)")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.98,
            0.02,
            f"P(cost-effective) = {(delta_nmb > 0).mean():.1%}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )
    fig.tight_layout()
    return fig, axes


def plot_survival(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
) -> tuple[plt.Figure, plt.Axes]:  # type: ignore
    spec = get_horizon(horizon)
    fig, ax = plt.subplots(figsize=(10, 6))
    curves: list[np.ndarray] = []
    styles = {
        "on-demand": {"color": "#D55E00", "linestyle": "-", "marker": "o"},
        "prophylaxis": {"color": "#0072B2", "linestyle": "--", "marker": "s"},
    }
    for regime in sorted(df["regime"].unique().to_list()):
        sub = base_results(df).filter(pl.col("regime") == regime)
        horizon_weeks = int(sub["cycles"].max())
        survival = np.array(
            [
                ((~sub["is_absorbed"]) | (sub["observed_cycles"] > week)).sum()
                / sub.height
                for week in range(horizon_weeks + 1)
            ],
            dtype=float,
        )
        curves.append(survival)
        ages = spec.start_age + np.arange(horizon_weeks + 1) / 52
        style = styles.get(regime, {})
        ax.plot(
            ages,
            survival,
            linewidth=2.2,
            drawstyle="steps-post",
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker"),
            markevery=52,
            markersize=4,
            label=(
                f"{regime} — final survival " f"{survival[-1]:.2%} (n={sub.height})"
            ),
        )

    minimum_survival = min(float(curve.min()) for curve in curves)
    if spec.key == "childhood" and minimum_survival >= 0.90:
        observed_loss = 1.0 - minimum_survival
        padding = max(0.002, observed_loss * 0.20)
        ax.set_ylim(max(0.0, minimum_survival - padding), 1.001)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        ax.text(
            0.01,
            0.02,
            "Zoomed vertical axis (does not start at 0)",
            transform=ax.transAxes,
            fontsize=9,
            alpha=0.75,
        )
        ylabel = "Proportion alive (zoomed)"
    else:
        ax.set_ylim(0, 1.01)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ylabel = "Proportion alive"

    ax.set(
        title=f"Survival — {spec.label}",
        xlabel="Age (years)",
        ylabel=ylabel,
        xlim=(spec.start_age, spec.end_age),
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig, ax
