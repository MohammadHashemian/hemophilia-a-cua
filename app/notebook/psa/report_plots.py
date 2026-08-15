"""Horizon-agnostic figures used by PSA report notebooks."""

from __future__ import annotations

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from app.notebook.psa.analysis import base_results, plot_survival
from app.notebook.psa.economics import (
    abr_threshold_analysis,
    ceac,
    ceac_threshold_summary,
    paired_outcomes,
    scenario_pairs,
)
from app.notebook.psa.scenarios import HorizonSpec, get_horizon
from app.notebook.psa.tables import state_occupation
from utils.math import cal_body_weight


def _plot_sample(frame: pl.DataFrame, maximum: int = 3_000) -> pl.DataFrame:
    """Deterministic display sample; calculations continue to use all rows."""
    if frame.height <= maximum:
        return frame
    return frame.sample(n=maximum, seed=20260730, shuffle=True)


def _methods(df: pl.DataFrame) -> list[str]:
    return sorted(base_results(df)["sampling_method"].unique().to_list())


def body_weight_curve(
    horizon: str | HorizonSpec,
) -> plt.Figure:  # type: ignore
    """Plot the deterministic base-weight curve used by simulation workers."""
    spec = get_horizon(horizon)
    milestone_ages = (1, 2, 12, 18)
    display_end_age = max(spec.end_age, max(milestone_ages))
    ages = np.arange(spec.start_age * 52, display_end_age * 52 + 1) / 52
    weights = np.array(
        [cal_body_weight(int(round(age * 52))) for age in ages],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        ages,
        weights,
        color="#2563EB",
        linewidth=2.4,
        label="Base-case male body weight",
        zorder=2,
    )
    if spec.end_age < display_end_age:
        ax.axvspan(
            spec.end_age,
            display_end_age,
            color="#94A3B8",
            alpha=0.14,
            label="Outside simulation horizon",
            zorder=0,
        )

    annotation_offsets = {
        1: (10, 18),
        2: (42, -22),
        12: (-22, 18),
        18: (-112, -18),
    }
    for age in milestone_ages:
        value = cal_body_weight(age * 52)
        ax.scatter(
            age,
            value,
            s=58,
            color="#F59E0B",
            edgecolor="#78350F",
            linewidth=1.2,
            zorder=4,
        )
        ax.annotate(
            f"Age {age}: {value:.2f} kg",
            xy=(age, value),
            xytext=annotation_offsets[age],
            textcoords="offset points",
            fontsize=9,
            fontweight="semibold",
            arrowprops={"arrowstyle": "-", "color": "#64748B", "lw": 0.9},
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": ax.get_facecolor(),
                "edgecolor": "#94A3B8",
                "alpha": 0.92,
            },
            zorder=5,
        )

    ax.set(
        title=f"Patient body-weight trajectory — {spec.label}",
        xlabel="Age (years)",
        ylabel="Assigned body weight (kg)",
        xlim=(spec.start_age, display_end_age),
    )
    ax.margins(y=0.12)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def abr_distribution(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
) -> plt.Figure:  # type: ignore
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
                fill=True,
                alpha=0.25,
                zorder=1,
                linestyle="--",
                label=f"{regime}: sampled",
            )
            sns.kdeplot(
                x=arm["annual_bleeding_rate"].to_numpy(),
                ax=ax,
                color=color,
                fill=True,
                alpha=0.25,
                zorder=2,
                label=f"{regime}: simulated",
            )
        ax.set(title=method.title(), xlabel="Annual bleeding rate", ylabel="Density")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Sampled and simulated ABR — {spec.label}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def outcome_distribution(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    column: str,
    label: str,
) -> plt.Figure:  # type: ignore
    spec = get_horizon(horizon)
    base = base_results(df)
    methods = _methods(df)
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5))
    for ax, method in zip(np.atleast_1d(axes), methods, strict=True):
        sub = base.filter(pl.col("sampling_method") == method)
        for regime, color in (("on-demand", "#1f77b4"), ("prophylaxis", "#ff7f0e")):
            values = sub.filter(pl.col("regime") == regime)[column].to_numpy()
            sns.kdeplot(
                x=values, ax=ax, fill=True, alpha=0.25, color=color, label=regime
            )
        ax.set(title=method.title(), xlabel=label, ylabel="Density")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"{label} distribution — {spec.label}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def joint_cost_qaly(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
) -> plt.Figure:  # type: ignore
    spec = get_horizon(horizon)
    base = base_results(df)
    methods = _methods(df)
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    for ax, method in zip(np.atleast_1d(axes), methods, strict=True):
        sub = base.filter(pl.col("sampling_method") == method)
        for regime, color in (("on-demand", "#1f77b4"), ("prophylaxis", "#ff7f0e")):
            arm = sub.filter(pl.col("regime") == regime)
            shown = _plot_sample(arm)
            ax.scatter(
                shown["total_qaly"],
                shown["total_cost"],
                s=10,
                alpha=0.26,
                color=color,
                edgecolors="none",
                label=regime,
                zorder=1,
            )
            if shown.height >= 10:
                try:
                    before = len(ax.collections)
                    sns.kdeplot(
                        x=shown["total_qaly"].to_numpy(),
                        y=shown["total_cost"].to_numpy(),
                        ax=ax,
                        color=("#33373F" if regime == "on-demand" else "#5672C0"),
                        levels=5,
                        linewidths=1.5,
                        linestyles=("-" if regime == "on-demand" else "--"),
                        alpha=1.0,
                        zorder=3,
                    )
                    for collection in ax.collections[before:]:
                        collection.set_path_effects(
                            [
                                path_effects.Stroke(linewidth=3, foreground="#FFFFFF"),
                                path_effects.Normal(),
                            ]
                        )
                except (np.linalg.LinAlgError, ValueError):
                    pass
        ax.set(title=method.title(), xlabel="Total QALY", ylabel="Total cost")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"Joint cost and QALY — {spec.label}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def health_state_distribution(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
) -> plt.Figure:  # type: ignore
    spec = get_horizon(horizon)
    occupation = state_occupation(df)
    state_styles = [
        ("healthy", "No bleeding", "#4C78A8"),
        ("bleeding", "Spontaneous bleeding", "#F58518"),
        ("hemarthrosis", "Hemarthrosis", "#54A24B"),
        ("intracranial_hemorrhage", "ICH", "#E45756"),
        ("non_ich_major_bleeding", "Non-ICH major bleeding", "#B279A2"),
        ("death", "Death", "#79706E"),
    ]
    fig, ax = plt.subplots(figsize=(12, max(6, occupation.height * 0.35)))
    left = np.zeros(occupation.height)
    y = np.arange(occupation.height)
    for column, label, color in state_styles:
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
) -> plt.Figure:  # type: ignore
    spec = get_horizon(horizon)
    pairs = scenario_pairs(df)
    cols = 2
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    flat = np.atleast_1d(axes).ravel()
    for ax, (base, comparison) in zip(flat, pairs, strict=False):
        paired = paired_outcomes(df, base, comparison, wtp=wtp)
        shown = _plot_sample(paired)
        x = shown["delta_qaly"].to_numpy()
        y = shown["delta_cost"].to_numpy()
        ax.scatter(
            x,
            y,
            s=10,
            alpha=0.30,
            color="#64748B",
            edgecolors="none",
            zorder=1,
        )
        if shown.height >= 10:
            try:
                before = len(ax.collections)
                sns.kdeplot(
                    x=x,
                    y=y,
                    ax=ax,
                    levels=4,
                    color="#665668",
                    linewidths=2,
                    zorder=3,
                )
                for collection in ax.collections[before:]:
                    collection.set_path_effects(
                        [
                            path_effects.Stroke(linewidth=3, foreground="#FFFFFF"),
                            path_effects.Normal(),
                        ]
                    )
            except (np.linalg.LinAlgError, ValueError):
                pass
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
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def icer_distributions(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    wtp: float,
) -> plt.Figure:  # type: ignore
    spec = get_horizon(horizon)
    pairs = scenario_pairs(df)
    cols = 2
    rows = int(np.ceil(len(pairs) / cols))
    fig = plt.figure(figsize=(14, 5 * rows))
    outer = fig.add_gridspec(
        rows,
        cols,
        left=0.07,
        right=0.98,
        bottom=0.06,
        top=0.96,
        hspace=0.58,
        wspace=0.24,
    )
    rng = np.random.default_rng(20260730)
    for index, (base, comparison) in enumerate(pairs):
        row, col = divmod(index, cols)
        inner = outer[row, col].subgridspec(
            2, 1, height_ratios=(1, 3), hspace=0.05
        )
        scatter_ax = fig.add_subplot(inner[0])
        histogram_ax = fig.add_subplot(inner[1], sharex=scatter_ax)
        paired = paired_outcomes(df, base, comparison, wtp=wtp)
        valid = paired.filter(~pl.col("delta_qaly").is_close(0))
        values = (valid["delta_cost"] / valid["delta_qaly"]).to_numpy()
        values = values[np.isfinite(values)]
        lower, upper = np.quantile(values, [0.01, 0.99])
        visible = values[(values >= lower) & (values <= upper)]
        if visible.size:
            if visible.size > 3_000:
                visible = rng.choice(visible, size=3_000, replace=False)
            jitter = rng.uniform(0.15, 0.85, size=visible.size)
            scatter_ax.scatter(
                visible,
                jitter,
                s=6,
                alpha=0.30,
                color="#2563EB",
                edgecolors="none",
            )
            histogram_ax.hist(
                visible,
                bins=50,
                color="#F59E0B",
                alpha=0.70,
                edgecolor="#78350F",
                linewidth=1.6,
            )
        for axis in (scatter_ax, histogram_ax):
            axis.axvline(
                wtp,
                color="#B91C1C",
                linestyle="--",
                linewidth=1.8,
                zorder=5,
            )
            axis.grid(axis="x", alpha=0.20)
        scatter_ax.set(
            title=f"{comparison}\nvs {base}",
            ylim=(0, 1),
            ylabel="Iterations",
        )
        scatter_ax.title.set_fontsize(9)
        scatter_ax.set_yticks([])
        scatter_ax.tick_params(axis="x", labelbottom=False)
        histogram_ax.set(
            xlabel="Iteration-level ICER",
            ylabel="Frequency",
        )
    for index in range(len(pairs), rows * cols):
        row, col = divmod(index, cols)
        unused = fig.add_subplot(outer[row, col])
        unused.set_axis_off()
    fig.suptitle(f"ICER distributions — {spec.label}")
    return fig


def ceac_plot(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    wtp: float,
) -> plt.Figure:  # type: ignore
    spec = get_horizon(horizon)
    thresholds = np.linspace(0, max(wtp * 2, 1), 101)
    data = ceac(df, thresholds=thresholds)
    summary = ceac_threshold_summary(
        df,
        selected_wtp=wtp,
        maximum_wtp=max(wtp * 2, 1),
        points=101,
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    for comparison in data["comparison"].unique().to_list():
        sub = data.filter(pl.col("comparison") == comparison).sort("wtp")
        line = ax.plot(sub["wtp"], sub["probability_cost_effective"], label=comparison)[
            0
        ]
        crossing = summary.filter(pl.col("comparison") == comparison)[
            "wtp_at_50_percent_ce"
        ].item()
        if crossing is not None:
            ax.scatter(
                crossing,
                0.5,
                marker="D",
                s=28,
                color=line.get_color(),
                zorder=4,
            )
    ax.axvline(wtp, color="black", linestyle="--", label="Selected WTP")
    ax.axhline(
        0.5,
        color="#6B7280",
        linestyle=":",
        linewidth=1,
        label="50% probability",
    )
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


def incremental_nmb_distributions(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    wtp: float,
) -> plt.Figure:  # type: ignore
    """Show the paired decision statistic directly at the selected WTP."""
    spec = get_horizon(horizon)
    pairs = scenario_pairs(df)
    cols = 2
    rows = max(1, int(np.ceil(len(pairs) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    flat = np.atleast_1d(axes).ravel()
    for ax, (base, comparison) in zip(flat, pairs, strict=False):
        paired = paired_outcomes(df, base, comparison, wtp=wtp)
        values = paired["delta_nmb"].to_numpy()
        sns.histplot(values, bins=45, stat="density", alpha=0.35, ax=ax)
        if np.unique(values).size > 1:
            sns.kdeplot(x=values, ax=ax, linewidth=1.8)
        ax.axvline(0, color="#B22222", linestyle="--")
        ax.set_title(f"{comparison}\nvs {base}", fontsize=9)
        ax.set_xlabel("Incremental net monetary benefit")
        ax.set_ylabel("Density")
        ax.text(
            0.98,
            0.95,
            f"P(CE) = {(values > 0).mean():.1%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        ax.grid(alpha=0.25)
    for ax in flat[len(pairs) :]:
        ax.set_visible(False)
    fig.suptitle(f"Incremental NMB at selected WTP — {spec.label}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def icer_vs_abr_threshold(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    wtp: float,
) -> plt.Figure:  # type: ignore
    """Plot paired cohort ICER estimates over on-demand ABR cutoffs."""
    spec = get_horizon(horizon)
    curves, _ = abr_threshold_analysis(df, wtp=wtp)
    ordered_pairs = scenario_pairs(df) if not curves.is_empty() else []
    cols = 2
    rows = max(1, int(np.ceil(len(ordered_pairs) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    flat = np.atleast_1d(axes).ravel()
    for ax, (base, comparison) in zip(flat, ordered_pairs, strict=False):
        sub = curves.filter(
            (pl.col("base_scenario") == base)
            & (pl.col("comparison_scenario") == comparison)
        ).sort("abr_cutoff")
        x = sub["abr_cutoff"].to_numpy()
        y = sub["icer"].to_numpy()
        finite = np.isfinite(y)
        ax.plot(x[finite], y[finite], marker="o", markersize=3, linewidth=1.5)
        ax.axhline(wtp, color="#B22222", linestyle="--", label="WTP")

        nmb = sub["delta_nmb"].to_numpy()
        changes = np.flatnonzero(np.signbit(nmb[:-1]) != np.signbit(nmb[1:]))
        if changes.size:
            i = int(changes[0])
            crossing = x[i] - nmb[i] * (x[i + 1] - x[i]) / (nmb[i + 1] - nmb[i])
            ax.axvline(
                crossing,
                color="#D97706",
                linestyle=":",
                label=f"ABR threshold ≈ {crossing:.2f}",
            )
        ax.set_title(f"{comparison}\nvs {base}", fontsize=9)
        ax.set_xlabel("Minimum baseline on-demand ABR")
        ax.set_ylabel("Cohort ICER")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in flat[len(ordered_pairs) :]:
        ax.set_visible(False)
    if not ordered_pairs:
        flat[0].text(
            0.5,
            0.5,
            "Insufficient paired ABR data",
            ha="center",
            va="center",
            transform=flat[0].transAxes,
        )
    fig.suptitle(f"ICER versus baseline ABR threshold — {spec.label}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def all_figures(
    df: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    wtp: float,
) -> dict[str, plt.Figure]:  # type: ignore
    survival, _ = plot_survival(df, horizon)
    return {
        "body_weight_curve": body_weight_curve(horizon),
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
        "cost_effectiveness_planes": cost_effectiveness_planes(df, horizon, wtp=wtp),
        "icer_distributions": icer_distributions(df, horizon, wtp=wtp),
        "ceac": ceac_plot(df, horizon, wtp=wtp),
        "incremental_nmb_distribution": incremental_nmb_distributions(
            df, horizon, wtp=wtp
        ),
        "icer_vs_abr_threshold": icer_vs_abr_threshold(df, horizon, wtp=wtp),
    }
