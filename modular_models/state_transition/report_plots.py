from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

_PARAMETER_LABELS = {
    "prophylaxis_iu_per_kg_week": "Prophylaxis dose (IU/kg/week)",
    "cost_discount_rate": "Cost discount rate",
    "factor_price_irr_per_iu": "Factor VIII price (IRR/IU)",
    "joint_bleed_iu_per_kg": "Factor VIII per joint bleed (IU/kg)",
    "wtp_irr_per_qaly": "Willingness-to-pay threshold",
    "ich_rate_on_demand": "ICH rate, on-demand",
    "abr_on_demand": "Annual bleed rate, on-demand",
    "ajbr_on_demand": "Annual joint bleed rate, on-demand*",
    "ajbr_prophylaxis": "Annual joint bleed rate, prophylaxis",
    "post_ich_sequela_probability": "Post-ICH sequela probability",
    "qaly_discount_rate": "QALY discount rate",
    "ich_case_fatality": "ICH case fatality",
    "minor_bleed_duration_days": "Minor-bleed duration (days)",
    "abr_prophylaxis": "Annual bleed rate, prophylaxis",
    "non_major_non_joint_iu_per_kg": "Factor VIII per non-joint bleed (IU/kg)",
    "minor_bleed_decrement": "Minor-bleed utility decrement",
    "utility_anchor": "Baseline utility anchor",
}


def plot_clinical_outcomes(outcomes: pl.DataFrame, ax: Any = None) -> Any:
    target = ax or plt.subplots(figsize=(9, 5.5))[1]
    labels = ["Total bleeds", "Joint bleeds", "Post-ICH patients / 100"]
    columns = ["total_bleeds", "joint_bleeds", "post_ich_probability"]
    x = np.arange(len(labels))
    width = 0.36
    for index, row in enumerate(outcomes.iter_rows(named=True)):
        values = [float(row[column]) for column in columns]
        values[2] *= 100.0
        bars = target.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=str(row["strategy"]).replace("_", " ").title(),
        )
        target.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    target.set_xticks(x, labels)
    target.set_ylabel("Mean per patient over the modeled horizon")
    target.set_title("Clinical outcomes by strategy")
    target.legend(frameon=False)
    target.grid(axis="y", alpha=0.2)
    return target


def plot_mortality(outcomes: pl.DataFrame, ax: Any = None) -> Any:
    target = ax or plt.subplots(figsize=(8, 5))[1]
    labels = ["All-cause", "ICH", "Background"]
    columns = ["mortality_probability", "deaths_ich", "deaths_background"]
    initial = outcomes["alive_at_end"] + outcomes["deaths_total"]
    x = np.arange(len(labels))
    width = 0.36
    for index, row in enumerate(outcomes.iter_rows(named=True)):
        denominator = float(initial[index])
        values = [
            float(row[columns[0]]) * 100.0,
            float(row[columns[1]]) / denominator * 100.0,
            float(row[columns[2]]) / denominator * 100.0,
        ]
        bars = target.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=str(row["strategy"]).replace("_", " ").title(),
        )
        target.bar_label(bars, fmt="%.3f%%", padding=3, fontsize=9)
    target.set_xticks(x, labels)
    target.set_ylabel("Probability (%)")
    target.set_title("Cause-specific mortality")
    target.legend(frameon=False)
    target.grid(axis="y", alpha=0.2)
    return target


def plot_psa_plane(frame: pl.DataFrame, wtp_irr_per_qaly: float, ax: Any = None) -> Any:
    target = ax or plt.subplots(figsize=(8, 6))[1]
    qaly = frame["incremental_qaly"].to_numpy()
    cost = frame["incremental_cost_irr"].to_numpy() / 1e9
    target.scatter(qaly, cost, s=16, alpha=0.35, edgecolors="none", label="PSA iterations")
    domain = np.linspace(min(0.0, float(qaly.min())), max(0.0, float(qaly.max())), 100)
    target.plot(
        domain,
        domain * wtp_irr_per_qaly / 1e9,
        linewidth=1.6,
        label=f"WTP = {wtp_irr_per_qaly / 1e9:.1f} bn IRR/QALY",
    )
    mean_qaly = float(qaly.mean())
    mean_cost = float(cost.mean())
    target.scatter([mean_qaly], [mean_cost], marker="X", s=110, label="PSA mean")
    target.annotate(
        f"Mean ({mean_qaly:.3f}, {mean_cost:.2f} bn)",
        (mean_qaly, mean_cost),
        xytext=(8, 8),
        textcoords="offset points",
    )
    target.axhline(0, linewidth=0.8, color="0.45")
    target.axvline(0, linewidth=0.8, color="0.45")
    target.set_xlabel("Incremental QALY")
    target.set_ylabel("Incremental cost (billion IRR)")
    target.set_title("Cost-effectiveness plane")
    target.legend(frameon=False)
    target.grid(alpha=0.18)
    return target


def plot_ceac_frame(ceac: pl.DataFrame, selected_wtp: float, ax: Any = None) -> Any:
    target = ax or plt.subplots(figsize=(8, 5))[1]
    x = ceac["wtp_irr_per_qaly"].to_numpy() / 1e9
    y = ceac["probability_cost_effective"].to_numpy()
    target.plot(x, y, linewidth=2.2)
    selected_probability = float(np.interp(selected_wtp / 1e9, x, y))
    target.scatter([selected_wtp / 1e9], [selected_probability], s=65)
    target.annotate(
        f"{selected_probability:.1%} at {selected_wtp / 1e9:.1f} bn",
        (selected_wtp / 1e9, selected_probability),
        xytext=(8, 8),
        textcoords="offset points",
    )
    target.set_ylim(0, 1)
    target.set_xlabel("Willingness to pay (billion IRR/QALY)")
    target.set_ylabel("Probability cost-effective")
    target.set_title("Cost-effectiveness acceptability curve")
    target.grid(alpha=0.2)
    return target


def plot_owsa_frame(
    frame: pl.DataFrame, base_inmb_irr: float, limit: int = 15, ax: Any = None
) -> Any:
    rows = []
    for parameter_id, group in frame.group_by("parameter_id"):
        group = group.filter(pl.col("status") == "complete")
        values = {row["endpoint"]: row for row in group.iter_rows(named=True)}
        if "low" not in values or "high" not in values:
            continue
        low = float(values["low"]["incremental_nmb_irr"])
        high = float(values["high"]["incremental_nmb_irr"])
        low_row = values["low"]
        high_row = values["high"]
        rows.append(
            {
                "parameter_id": parameter_id[0],
                "minimum": min(low, high),
                "maximum": max(low, high),
                "span": abs(high - low),
                "low_inmb": low,
                "high_inmb": high,
                "low_value": float(low_row["endpoint_value"]),
                "high_value": float(high_row["endpoint_value"]),
            }
        )
    selected = sorted(rows, key=lambda row: row["span"], reverse=True)[:limit]
    selected.reverse()
    target = ax or plt.subplots(figsize=(9, max(5, len(selected) * 0.42)))[1]
    y = np.arange(len(selected))
    minimum = np.array([row["minimum"] for row in selected]) / 1e9
    maximum = np.array([row["maximum"] for row in selected]) / 1e9
    target.barh(y, maximum - minimum, left=minimum, alpha=0.72)
    low_inmb = np.array([row["low_inmb"] for row in selected]) / 1e9
    high_inmb = np.array([row["high_inmb"] for row in selected]) / 1e9
    target.scatter(low_inmb, y, marker="o", s=20, label="Low input")
    target.scatter(high_inmb, y, marker="s", s=20, label="High input")
    target.axvline(base_inmb_irr / 1e9, linewidth=1.6, label="Base-case INMB")
    target.set_yticks(
        y,
        [_PARAMETER_LABELS.get(row["parameter_id"], row["parameter_id"]) for row in selected],
    )
    target.set_xlabel("Incremental NMB (billion IRR)")
    target.set_title("One-way sensitivity analysis")
    if "analysis_type" in frame.columns and frame.filter(
        pl.col("analysis_type") == "linked_endpoint"
    ).height:
        target.text(
            0.0,
            -0.08,
            "* High AJBR is a linked endpoint paired with high ABR; see the audit table.",
            transform=target.transAxes,
            fontsize=8,
            ha="left",
            va="top",
        )
    target.legend(frameon=False)
    target.grid(axis="x", alpha=0.2)
    return target


def plot_calibration(frame: pl.DataFrame, ax: Any = None) -> Any:
    target = ax or plt.subplots(figsize=(7, 6))[1]
    for strategy, group in frame.group_by("strategy"):
        target.scatter(
            group["target_rate_per_person_year"],
            group["simulated_rate_per_person_year"],
            s=55,
            label=strategy[0].replace("_", " ").title(),
        )
        label_rows = group.sort("target_rate_per_person_year").iter_rows(named=True)
        for row in label_rows:
            if strategy[0] != "on_demand":
                continue
            target.annotate(
                str(row["calibration_target"]).replace("_", " "),
                (row["target_rate_per_person_year"], row["simulated_rate_per_person_year"]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=8,
            )
    maximum = max(
        float(cast(float | int, frame["target_rate_per_person_year"].max())),
        float(cast(float | int, frame["simulated_rate_per_person_year"].max())),
    )
    minimum = min(
        float(cast(float | int, frame["target_rate_per_person_year"].min())),
        float(cast(float | int, frame["simulated_rate_per_person_year"].min())),
    )
    lower = minimum / 1.8
    upper = maximum * 1.25
    target.plot([lower, upper], [lower, upper], linestyle="--", color="0.4", label="Perfect fit")
    target.set_xscale("log")
    target.set_yscale("log")
    target.set_xlim(lower, upper)
    target.set_ylim(lower, upper)
    target.set_xlabel("Target annual rate")
    target.set_ylabel("Simulated rate per person-year")
    target.set_title("Internal calibration: event rates")
    target.legend(frameon=False)
    target.grid(alpha=0.2)
    return target
