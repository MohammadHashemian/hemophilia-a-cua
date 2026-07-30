"""Horizon-agnostic figures used by PSA report notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from app.notebook.psa.analysis import base_results, plot_survival
from app.notebook.psa.economics import ceac, paired_outcomes, scenario_pairs
from app.notebook.psa.scenarios import HorizonSpec, get_horizon
from app.notebook.psa.tables import state_occupation


def _methods(df: pl.DataFrame) -> list[str]:
    return sorted(base_results(df)["sampling_method"].unique().to_list())


def abr_distribution(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
) -> plt.Figure:
    spec = get_horizon(horizon)
    base = base_results(df)
    methods = _methods(df)
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5))
    for ax, method in zip(np.atleast_1d(axes), methods, strict=True):
        sub = base.filter(pl.col("sampling_method") == method)
        for regime, color in (("on-demand", "#1f77b4"), ("prophylaxis", "#ff7f0e")):
            arm = sub.filter(pl.col("regime") == regime)
            sns.kdeplot(
                x=arm["sampled_abr"].to_numpy(),
                ax=ax,
                color=color,
                linestyle="--",
                label=f"{regime}: sampled",
            )
            sns.kdeplot(
                x=arm["annual_bleeding_rate"].to_numpy(),
                ax=ax,
                color=color,
                label=f"{regime}: simulated",
            )
        ax.set(title=method.title(), xlabel="Annual bleeding rate", ylabel="Density")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Sampled and simulated ABR — {spec.label}")
    fig.tight_layout()
    return fig


def outcome_distribution(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    column: str,
    label: str,
) -> plt.Figure:
    spec = get_horizon(horizon)
    base = base_results(df)
    methods = _methods(df)
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5))
    for ax, method in zip(np.atleast_1d(axes), methods, strict=True):
        sub = base.filter(pl.col("sampling_method") == method)
        for regime, color in (("on-demand", "#1f77b4"), ("prophylaxis", "#ff7f0e")):
            values = sub.filter(pl.col("regime") == regime)[column].to_numpy()
            sns.kdeplot(x=values, ax=ax, fill=True, alpha=0.25, color=color, label=regime)
        ax.set(title=method.title(), xlabel=label, ylabel="Density")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"{label} distribution — {spec.label}")
    fig.tight_layout()
    return fig


def joint_cost_qaly(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
) -> plt.Figure:
    spec = get_horizon(horizon)
    base = base_results(df)
    methods = _methods(df)
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    for ax, method in zip(np.atleast_1d(axes), methods, strict=True):
        sub = base.filter(pl.col("sampling_method") == method)
        for regime, color in (("on-demand", "#1f77b4"), ("prophylaxis", "#ff7f0e")):
            arm = sub.filter(pl.col("regime") == regime)
            ax.scatter(
                arm["total_qaly"],
                arm["total_cost"],
                s=8,
                alpha=0.15,
                color=color,
                label=regime,
            )
        ax.set(title=method.title(), xlabel="Total QALY", ylabel="Total cost")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"Joint cost and QALY — {spec.label}")
    fig.tight_layout()
    return fig


def health_state_distribution(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
) -> plt.Figure:
    spec = get_horizon(horizon)
    occupation = state_occupation(df)
    columns = [
        "healthy",
        "bleeding",
        "hemarthrosis",
        "intracranial_hemorrhage",
        "non_ich_major_bleeding",
        "death",
    ]
    labels = ["Healthy", "Bleeding", "Hemarthrosis", "Life-threatening", "Death"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    fig, ax = plt.subplots(figsize=(12, max(6, occupation.height * 0.35)))
    left = np.zeros(occupation.height)
    y = np.arange(occupation.height)
    for column, label, color in zip(columns, labels, colors, strict=True):
        values = occupation[column].to_numpy()
        ax.barh(y, values, left=left, label=label, color=color, edgecolor="white")
        left += values
    ax.set_yticks(y, occupation["scenario"].to_list(), fontsize=8)
    ax.set_xlabel("Proportion of observed weeks")
    ax.set_title(f"Health-state occupation — {spec.label}")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def cost_effectiveness_planes(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    wtp: float,
) -> plt.Figure:
    spec = get_horizon(horizon)
    pairs = scenario_pairs(df)
    cols = 2
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    flat = np.atleast_1d(axes).ravel()
    for ax, (base, comparison) in zip(flat, pairs, strict=False):
        paired = paired_outcomes(df, base, comparison, wtp=wtp)
        x = paired["delta_qaly"].to_numpy()
        y = paired["delta_cost"].to_numpy()
        ax.scatter(x, y, s=8, alpha=0.25)
        ax.axhline(0, color="#2E3440", linewidth=1)
        ax.axvline(0, color="#2E3440", linewidth=1)
        threshold_x = np.linspace(x.min(), x.max(), 100)
        ax.plot(threshold_x, threshold_x * wtp, "--", color="#272727")
        ax.set_title(f"{comparison}\nvs {base}", fontsize=9)
        ax.set_xlabel("Δ QALY")
        ax.set_ylabel("Δ cost")
        ax.text(
            0.98,
            0.02,
            f"P(CE) = {(paired['delta_nmb'] > 0).mean():.1%}",
            transform=ax.transAxes,
            ha="right",
        )
        ax.grid(alpha=0.25)
    for ax in flat[len(pairs) :]:
        ax.set_visible(False)
    fig.suptitle(f"Cost-effectiveness planes — {spec.label}")
    fig.tight_layout()
    return fig


def icer_distributions(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    wtp: float,
) -> plt.Figure:
    spec = get_horizon(horizon)
    pairs = scenario_pairs(df)
    cols = 2
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    flat = np.atleast_1d(axes).ravel()
    for ax, (base, comparison) in zip(flat, pairs, strict=False):
        paired = paired_outcomes(df, base, comparison, wtp=wtp)
        valid = paired.filter(~pl.col("delta_qaly").is_close(0))
        values = (valid["delta_cost"] / valid["delta_qaly"]).to_numpy()
        values = values[np.isfinite(values)]
        lower, upper = np.quantile(values, [0.01, 0.99])
        sns.histplot(values[(values >= lower) & (values <= upper)], bins=50, ax=ax)
        ax.axvline(wtp, color="red", linestyle="--", label="WTP")
        ax.set_title(f"{comparison}\nvs {base}", fontsize=9)
        ax.set_xlabel("Iteration-level ICER")
        ax.legend()
    for ax in flat[len(pairs) :]:
        ax.set_visible(False)
    fig.suptitle(f"ICER distributions — {spec.label}")
    fig.tight_layout()
    return fig


def ceac_plot(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    wtp: float,
) -> plt.Figure:
    spec = get_horizon(horizon)
    thresholds = np.linspace(0, max(wtp * 2, 1), 101)
    data = ceac(df, thresholds=thresholds)
    fig, ax = plt.subplots(figsize=(11, 7))
    for comparison in data["comparison"].unique().to_list():
        sub = data.filter(pl.col("comparison") == comparison).sort("wtp")
        ax.plot(sub["wtp"], sub["probability_cost_effective"], label=comparison)
    ax.axvline(wtp, color="black", linestyle="--", label="Selected WTP")
    ax.set(
        title=f"Cost-effectiveness acceptability curves — {spec.label}",
        xlabel="Willingness-to-pay per QALY",
        ylabel="Probability cost-effective",
        ylim=(0, 1),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def all_figures(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    wtp: float,
) -> dict[str, plt.Figure]:
    survival, _ = plot_survival(df, horizon)
    return {
        "abr_distribution": abr_distribution(df, horizon),
        "survival_curve": survival,
        "health_state_distribution": health_state_distribution(df, horizon),
        "qaly_distribution": outcome_distribution(
            df, horizon, column="total_qaly", label="Total QALY"
        ),
        "cost_distribution": outcome_distribution(
            df, horizon, column="total_cost", label="Total cost"
        ),
        "joint_cost_qaly": joint_cost_qaly(df, horizon),
        "cost_effectiveness_planes": cost_effectiveness_planes(
            df, horizon, wtp=wtp
        ),
        "icer_distributions": icer_distributions(df, horizon, wtp=wtp),
        "ceac": ceac_plot(df, horizon, wtp=wtp),
    }
