"""Horizon-agnostic OWSA figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from app.notebook.psa.scenarios import HorizonSpec, get_horizon


def _interval_plot(
    summary: pl.DataFrame,
    *,
    low_column: str,
    base_column: str,
    high_column: str,
    xlabel: str,
    title: str,
    reference: float,
) -> plt.Figure:
    data = summary.sort("nmb_sensitivity").tail(20)
    y = np.arange(data.height)
    low = data[low_column].to_numpy()
    base = data[base_column].to_numpy()
    high = data[high_column].to_numpy()
    # Include the observed base result in the displayed span. If the response
    # is non-linear, the base can legitimately fall outside the two endpoint
    # outcomes and must not be visually detached from its sensitivity bar.
    left = np.minimum(np.minimum(low, high), base)
    right = np.maximum(np.maximum(low, high), base)
    fig, ax = plt.subplots(figsize=(11, max(6, data.height * 0.42)))
    ax.hlines(y, left, right, color="#4C78A8", linewidth=6, alpha=0.75)
    ax.scatter(low, y, marker="|", s=100, color="#1F2937", label="Low input")
    ax.scatter(high, y, marker="|", s=100, color="#B45309", label="High input")
    ax.scatter(base, y, marker="o", s=24, color="#DC2626", label="Base case")
    ax.axvline(reference, color="#111827", linestyle="--", linewidth=1.5)
    ax.set_yticks(y, data["label"].to_list())
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def nmb_tornado(
    summary: pl.DataFrame,
    horizon: str | HorizonSpec,
) -> plt.Figure:
    spec = get_horizon(horizon)
    return _interval_plot(
        summary,
        low_column="low_delta_nmb",
        base_column="base_delta_nmb",
        high_column="high_delta_nmb",
        xlabel="Incremental net monetary benefit (IRR)",
        title=f"One-way sensitivity of incremental NMB — {spec.label}",
        reference=0,
    )


def icer_tornado(
    summary: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    wtp: float,
) -> plt.Figure:
    spec = get_horizon(horizon)
    finite = summary.filter(
        pl.col("low_icer").is_finite()
        & pl.col("base_icer").is_finite()
        & pl.col("high_icer").is_finite()
        & pl.col("icer_tornado_valid")
    )
    excluded = summary.height - finite.height
    return _interval_plot(
        finite,
        low_column="low_icer",
        base_column="base_icer",
        high_column="high_icer",
        xlabel="ICER (IRR/QALY)",
        title=(
            f"One-way sensitivity of ICER — {spec.label}\n"
            f"Conventional ICER quadrant only ({excluded} parameters excluded)"
        ),
        reference=wtp,
    )


def all_figures(
    summary: pl.DataFrame,
    horizon: str | HorizonSpec,
    *,
    wtp: float,
) -> dict[str, plt.Figure]:
    return {
        "nmb_tornado": nmb_tornado(summary, horizon),
        "icer_tornado": icer_tornado(summary, horizon, wtp=wtp),
    }
