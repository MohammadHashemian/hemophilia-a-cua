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


# ---------------------------------------------------------------------------
# Notebook-side plot helpers
# ---------------------------------------------------------------------------


def plot_inmb_threshold(
    delta_qaly: float,
    delta_cost: float,
    primary_wtp: float,
    *,
    wtp_max_irr: float = 300_000_000_000.0,
    n_points: int = 301,
    ax: Any = None,
) -> Any:
    """Deterministic INMB across a dense WTP grid with primary and break-even markers."""
    delta_qaly = float(delta_qaly)
    delta_cost = float(delta_cost)
    primary_wtp = float(primary_wtp)
    wtp_grid = np.linspace(0.0, float(wtp_max_irr), int(n_points))
    inmb_grid = wtp_grid * delta_qaly - delta_cost
    break_even_wtp = delta_cost / delta_qaly if delta_qaly > 0 else float("nan")

    target = ax or plt.subplots(figsize=(9, 5.8))[1]
    target.plot(
        wtp_grid / 1e9,
        inmb_grid / 1e9,
        linewidth=2.2,
        label="Incremental NMB",
    )
    target.axhline(0, linestyle="--", linewidth=1.3, label="INMB = 0")
    target.axvline(
        primary_wtp / 1e9,
        linestyle=":",
        linewidth=1.5,
        label=f"Primary WTP = {primary_wtp / 1e9:.0f} billion IRR/QALY",
    )
    target.axvline(
        break_even_wtp / 1e9,
        linestyle="-.",
        linewidth=1.5,
        label=f"Break-even WTP = {break_even_wtp / 1e9:.1f} billion IRR/QALY",
    )
    target.scatter([break_even_wtp / 1e9], [0], s=55, zorder=5)
    target.annotate(
        f"Break-even\n{break_even_wtp / 1e9:.1f} billion",
        xy=(break_even_wtp / 1e9, 0),
        xytext=(12, 18),
        textcoords="offset points",
        ha="left",
    )
    target.set_xlabel("Willingness-to-pay threshold (billion IRR/QALY)")
    target.set_ylabel(
        "Incremental net monetary benefit (billion IRR/patient)"
    )
    target.set_title(
        "Deterministic incremental net monetary benefit "
        "across willingness-to-pay thresholds"
    )
    target.legend(frameon=False, loc="best")
    target.grid(alpha=0.2)
    return target


def plot_scenario_nmb_bars(scenario_table: pl.DataFrame, ax: Any = None) -> Any:
    """Vertical bar chart of incremental NMB for each structural scenario."""
    target = ax or plt.subplots(figsize=(10, 5.5))[1]
    labels = scenario_table["scenario"].to_list()
    values = scenario_table["incremental_nmb_irr"].to_numpy() / 1e9
    bars = target.bar(labels, values)
    target.bar_label(bars, fmt="%.2f", padding=3)
    target.axhline(0, color="0.35", linewidth=0.9)
    target.set_ylabel("Incremental NMB (billion IRR)")
    target.set_title("Structural scenario analysis")
    target.tick_params(axis="x", rotation=30)
    return target


def plot_inner_loop_precision(
    precision_plot: pl.DataFrame,
    *,
    threshold_percent: float = 10.0,
    ax: Any = None,
) -> Any:
    """Paired-QALY-noise vs population size for the inner-loop diagnostic."""
    target = ax or plt.subplots(figsize=(8.5, 5.2))[1]
    target.plot(
        precision_plot["n_patients_per_strategy"],
        precision_plot["qaly_noise_ratio_percent"],
        marker="o",
        linewidth=2,
        label="Paired QALY noise / parameter SD",
    )
    for row in precision_plot.iter_rows(named=True):
        target.annotate(
            f"{row['qaly_noise_ratio_percent']:.1f}%",
            (row["n_patients_per_strategy"], row["qaly_noise_ratio_percent"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    target.axhline(
        threshold_percent,
        color="0.3",
        linestyle="--",
        label=f"{threshold_percent:.0f}% diagnostic threshold",
    )
    target.set_xlabel("Patients per strategy within each PSA draw")
    target.set_ylabel("First-order QALY noise / between-parameter SD (%)")
    target.set_title("Paired PSA inner-loop precision")
    target.legend(frameon=False)
    target.grid(alpha=0.2)
    return target


def plot_psa_convergence(
    convergence_plot: pl.DataFrame,
    *,
    threshold_percent: float = 1.0,
    title: str = "Second-order PSA convergence",
    ax: Any = None,
) -> Any:
    """Change in mean incremental cost/QALY vs cumulative PSA draws."""
    target = ax or plt.subplots(figsize=(9, 5.5))[1]
    target.plot(
        convergence_plot["iterations"],
        convergence_plot["relative_change_mean_cost"] * 100,
        marker="o",
        label="Mean incremental cost",
    )
    target.plot(
        convergence_plot["iterations"],
        convergence_plot["relative_change_mean_qaly"] * 100,
        marker="s",
        label="Mean incremental QALY",
    )
    target.axhline(
        threshold_percent,
        color="0.3",
        linestyle="--",
        label=f"{threshold_percent:.0f}% change threshold",
    )
    target.set_xlabel("Cumulative second-order PSA draws")
    target.set_ylabel("Change from preceding checkpoint (%)")
    target.set_title(title)
    target.legend(frameon=False)
    target.grid(alpha=0.2)
    return target


def plot_evpi(
    wtp_grid: np.ndarray,
    evpi_values: np.ndarray,
    primary_wtp: float,
    ax: Any = None,
) -> Any:
    """EVPI curve with primary-WTP marker and the maximum-EVPI annotation."""
    target = ax or plt.subplots(figsize=(9, 5.8))[1]
    target.plot(
        np.asarray(wtp_grid) / 1e9,
        np.asarray(evpi_values) / 1e9,
        linewidth=2.2,
    )
    target.axvline(
        float(primary_wtp) / 1e9,
        linestyle="--",
        linewidth=1.3,
        label=f"Primary WTP = {float(primary_wtp) / 1e9:.0f} billion IRR/QALY",
    )
    max_idx = int(np.argmax(np.asarray(evpi_values)))
    target.scatter(
        [float(wtp_grid[max_idx]) / 1e9],
        [float(evpi_values[max_idx]) / 1e9],
        s=55,
        zorder=5,
    )
    target.annotate(
        (
            f"Maximum EVPI\n"
            f"{float(evpi_values[max_idx]) / 1e9:.2f} "
            f"billion IRR/patient"
        ),
        xy=(
            float(wtp_grid[max_idx]) / 1e9,
            float(evpi_values[max_idx]) / 1e9,
        ),
        xytext=(15, -15),
        textcoords="offset points",
    )
    target.set_xlabel("Willingness-to-pay threshold (billion IRR/QALY)")
    target.set_ylabel("EVPI (billion IRR/patient)")
    target.set_title("Expected Value of Perfect Information")
    target.legend(frameon=False)
    target.grid(alpha=0.2)
    return target


def plot_factor_price_policy(
    factor_price_psa: pl.DataFrame,
    base_price_irr_per_iu: float,
    break_even_price_irr_per_iu: float,
    ax: Any = None,
) -> Any:
    """Cost-effectiveness probability versus FVIII unit price."""
    target = ax or plt.subplots(figsize=(9, 5.8))[1]
    target.plot(
        factor_price_psa["factor_price_irr_per_iu"].to_numpy(),
        factor_price_psa["probability_cost_effective"].to_numpy() * 100.0,
        linewidth=2.2,
    )
    target.axvline(
        float(base_price_irr_per_iu),
        linestyle="--",
        linewidth=1.3,
        label=f"Base price = {float(base_price_irr_per_iu):,.0f} IRR/IU",
    )
    target.axvline(
        float(break_even_price_irr_per_iu),
        linestyle=":",
        linewidth=1.5,
        label=(
            f"Deterministic break-even = "
            f"{float(break_even_price_irr_per_iu):,.0f} IRR/IU"
        ),
    )
    target.axhline(50, linestyle="-.", linewidth=1.1, label="50% probability")
    target.set_xlabel("FVIII unit price (IRR/IU)")
    target.set_ylabel("Probability prophylaxis is cost-effective (%)")
    target.set_title("Cost-effectiveness probability across FVIII prices")
    target.legend(frameon=False)
    target.grid(alpha=0.2)
    return target


def plot_monte_carlo_convergence(
    convergence_table: pl.DataFrame, ax: Any = None
) -> Any:
    """Relative change in incremental cost and QALY across cohort sizes."""
    target = ax or plt.subplots(figsize=(8, 5))[1]
    plot_data = convergence_table.filter(
        pl.col("relative_change_cost").is_not_null()
    )
    target.plot(
        plot_data["n_patients"],
        plot_data["relative_change_cost"] * 100,
        marker="o",
        label="Incremental cost",
    )
    target.plot(
        plot_data["n_patients"],
        plot_data["relative_change_qaly"] * 100,
        marker="s",
        label="Incremental QALY",
    )
    for row in plot_data.iter_rows(named=True):
        target.annotate(
            f"{row['relative_change_cost'] * 100:.4f}%",
            (row["n_patients"], row["relative_change_cost"] * 100),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
        target.annotate(
            f"{row['relative_change_qaly'] * 100:.3f}%",
            (row["n_patients"], row["relative_change_qaly"] * 100),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    target.axhline(1.0, color="0.35", linestyle="--", label="1% threshold")
    target.set_xlabel("Patients per strategy")
    target.set_ylabel("Relative change from preceding size (%)")
    target.set_title("First-order Monte Carlo convergence")
    target.legend(frameon=False)
    target.grid(alpha=0.2)
    return target
