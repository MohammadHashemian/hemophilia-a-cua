from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from modular_models.state_transition.results import OWSAResult, PSAResult


def plot_cost_effectiveness_plane(psa: PSAResult, ax=None):
    """Plot incremental QALY against incremental cost in billion IRR."""
    frame = psa.to_polars()
    target = ax or plt.subplots(figsize=(7.5, 6))[1]
    target.scatter(
        frame["incremental_qaly"],
        frame["incremental_cost_irr"] / 1e9,
        s=14,
        alpha=0.35,
        color="#176b87",
        edgecolors="none",
    )
    target.axhline(0, color="#596773", linewidth=0.9)
    target.axvline(0, color="#596773", linewidth=0.9)
    target.set_xlabel("Incremental QALY")
    target.set_ylabel("Incremental cost (billion IRR)")
    target.set_title("Cost-effectiveness plane")
    target.grid(alpha=0.18)
    return target


def plot_ceac(
    psa: PSAResult,
    wtp_values_irr_per_qaly: Iterable[float],
    ax=None,
):
    """Plot the probability that prophylaxis is cost-effective over WTP."""
    wtps = np.asarray(list(wtp_values_irr_per_qaly), dtype=np.float64)
    probabilities = np.array([psa.probability_cost_effective(value) for value in wtps])
    target = ax or plt.subplots(figsize=(7.5, 5))[1]
    target.plot(wtps / 1e9, probabilities, color="#0b8f87", linewidth=2.2)
    target.set_ylim(0, 1)
    target.set_xlabel("Willingness to pay (billion IRR/QALY)")
    target.set_ylabel("Probability cost-effective")
    target.set_title("Cost-effectiveness acceptability curve")
    target.grid(alpha=0.2)
    return target


def plot_owsa_tornado(
    records: Iterable[OWSAResult],
    *,
    base_inmb_irr: float = 0.0,
    ax=None,
    limit: int = 15,
):
    """Plot OWSA endpoints on incremental NMB, ordered by influence."""
    selected = sorted(records, key=lambda item: item.inmb_span, reverse=True)[:limit]
    selected.reverse()
    target = ax or plt.subplots(figsize=(9, max(4.5, len(selected) * 0.42)))[1]
    labels = [item.parameter_id for item in selected]
    low = np.array([min(item.low_inmb, item.high_inmb) for item in selected]) / 1e9
    high = np.array([max(item.low_inmb, item.high_inmb) for item in selected]) / 1e9
    y = np.arange(len(selected))
    target.barh(y, high - low, left=low, color="#5a87b5", alpha=0.88)
    target.axvline(base_inmb_irr / 1e9, color="#c66a25", linewidth=1.6)
    target.set_yticks(y, labels)
    target.set_xlabel("Incremental net monetary benefit (billion IRR)")
    target.set_title("One-way sensitivity analysis")
    target.grid(axis="x", alpha=0.2)
    return target
