from src.utils.logger import get_logger
from src.data.loaders import PROJECT_ROOT
from model_dep.schemas import States, Regimes, BaseStates
from model_dep.treatments import initialize_treatments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

logger = get_logger()


def calculate_average_abr_ajbr(
    results: list[dict], num_years: float, regime: Regimes
) -> tuple[float, float]:
    """
    Calculate average ABR and AJBR from simulation results, counting transitions to bleeding states.

    Args:
        results: List of dictionaries with simulation results (state_path).
        num_years: Duration of simulation in years (e.g., 73).
        regime: Treatment regime (ON_DEMAND or PROPHYLAXIS).

    Returns:
        Tuple of (average ABR, average AJBR) across all runs.
    """
    treatments = initialize_treatments()
    treatment = treatments[regime]
    bleeding_states = [
        States.MINOR_BLEEDING.value,
        States.MAJOR_BLEEDING.value,
        States.LT_BLEEDING.value,
    ]
    abr_counts = []
    ajbr_counts = []

    for result in results:
        state_path = result["state_path"]
        # Count transitions to bleeding states
        total_bleed_transitions = sum(
            1 for i in range(1, len(state_path)) if state_path[i] in bleeding_states
        )
        joint_bleed_transitions = sum(
            1
            for i in range(1, len(state_path))
            if state_path[i] == States.MAJOR_BLEEDING.value
        )
        # Convert to annual rates
        abr = total_bleed_transitions / num_years
        ajbr = joint_bleed_transitions / num_years
        abr_counts.append(abr)
        ajbr_counts.append(ajbr)
        logger.debug(
            f"{regime.value} run: {total_bleed_transitions} bleed transitions, {joint_bleed_transitions} joint bleed transitions, ABR={abr:.2f}, AJBR={ajbr:.2f}"
        )

    avg_abr = np.mean(abr_counts)
    avg_ajbr = np.mean(ajbr_counts)
    logger.info(
        f"{regime.value}: Expected ABR={treatment.abr:.2f}, Simulated ABR={avg_abr:.2f}, Expected AJBR={treatment.ajbr:.2f}, Simulated AJBR={avg_ajbr:.2f}"
    )
    return avg_abr, avg_ajbr # type: ignore


def visualize_results(
    od_results: list[dict], pro_results: list[dict], states: list[str], model_name: str
):
    """
    Visualize simulation results over lifetime, saving figures to /outputs/figures/.

    Args:
        od_results: List of dictionaries with on-demand simulation results (state_path).
        pro_results: List of dictionaries with prophylaxis simulation results (state_path).
        states: List of state names for the model.
        model_name: Name of the model (e.g., 'early', 'intermediate', 'end').
    """
    output_dir = PROJECT_ROOT / "outputs" / "figures"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Stacked Bar Plot: Post-ITI state distribution
    def count_post_iti_states(results_list, states):
        counts = {
            BaseStates.NO_BLEEDING.value: 0,
            BaseStates.CHRONIC_ARTHROPATHY.value: 0,
            BaseStates.SEVERE_ARTHROPATHY.value: 0,
        }
        total_inhibitor = 0
        for results in results_list:
            state_path = results["state_path"]
            for i in range(len(state_path) - 1):
                if state_path[i] == States.INHIBITOR.value:
                    total_inhibitor += 1
                    if state_path[i + 1] in counts and state_path[i + 1] in states:
                        counts[state_path[i + 1]] += 1
        logger.info(
            f"{model_name} Total inhibitor state occurrences: {total_inhibitor}"
        )
        total = sum(counts.values())
        if total == 0:
            logger.warning(f"{model_name} No post-ITI transitions found")
            return counts
        return {k: v / total * 100 for k, v in counts.items() if k in states}

    od_iti_counts = count_post_iti_states(od_results, states)
    pro_iti_counts = count_post_iti_states(pro_results, states)

    plt.figure(figsize=(8, 6))
    regimens = ["On-Demand", "Prophylaxis"]
    labels = ["No Arthropathy", "Chronic Arthropathy", "Severe Arthropathy"]
    od_values = [
        (
            od_iti_counts.get(BaseStates.NO_BLEEDING.value, 0)
            if BaseStates.NO_BLEEDING.value in states
            else 0
        ),
        (
            od_iti_counts.get(BaseStates.CHRONIC_ARTHROPATHY.value, 0)
            if BaseStates.CHRONIC_ARTHROPATHY.value in states
            else 0
        ),
        (
            od_iti_counts.get(BaseStates.SEVERE_ARTHROPATHY.value, 0)
            if BaseStates.SEVERE_ARTHROPATHY.value in states
            else 0
        ),
    ]
    pro_values = [
        (
            pro_iti_counts.get(BaseStates.NO_BLEEDING.value, 0)
            if BaseStates.NO_BLEEDING.value in states
            else 0
        ),
        (
            pro_iti_counts.get(BaseStates.CHRONIC_ARTHROPATHY.value, 0)
            if BaseStates.CHRONIC_ARTHROPATHY.value in states
            else 0
        ),
        (
            pro_iti_counts.get(BaseStates.SEVERE_ARTHROPATHY.value, 0)
            if BaseStates.SEVERE_ARTHROPATHY.value in states
            else 0
        ),
    ]
    x = np.arange(len(regimens))
    width = 0.4
    plt.bar(
        x - width / 2 - 0.05,
        [od_values[0], pro_values[0]],
        width,
        label=labels[0],
        color="#1f77b4",
        edgecolor="none",
    )
    if od_values[1] or pro_values[1]:
        plt.bar(
            x - width / 2 - 0.05,
            [od_values[1], pro_values[1]],
            width,
            bottom=[od_values[0], pro_values[0]],
            label=labels[1],
            color="#ff7f0e",
            edgecolor="none",
        )
    if od_values[2] or pro_values[2]:
        plt.bar(
            x - width / 2 - 0.05,
            [od_values[2], pro_values[2]],
            width,
            bottom=[od_values[0] + od_values[1], pro_values[0] + pro_values[1]],
            label=labels[2],
            color="#2ca02c",
            edgecolor="none",
        )
    plt.bar(
        x + width / 2 + 0.05,
        [pro_values[0], 0],
        width,
        color="#1f77b4",
        edgecolor="none",
        alpha=0,
    )
    plt.xticks(x, regimens)
    plt.title(f"Post-ITI State Distribution ({model_name.capitalize()} Model)")
    plt.ylabel("Percentage of Transitions (%)")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / f"post_iti_states_{model_name}.png", dpi=400)
    plt.close()
    logger.info(
        f"Saved post-ITI state distribution plot to post_iti_states_{model_name}.png"
    )

    # 2. Subplot Time Series: State proportions over time
    num_steps = len(od_results[0]["state_path"])
    weeks = np.arange(num_steps)
    focus_states = [
        (
            BaseStates.NO_BLEEDING.value
            if BaseStates.NO_BLEEDING.value in states
            else None
        ),
        (
            BaseStates.CHRONIC_ARTHROPATHY.value
            if BaseStates.CHRONIC_ARTHROPATHY.value in states
            else None
        ),
        (
            BaseStates.SEVERE_ARTHROPATHY.value
            if BaseStates.SEVERE_ARTHROPATHY.value in states
            else None
        ),
        States.INHIBITOR.value if States.INHIBITOR.value in states else None,
    ]
    focus_states = [s for s in focus_states if s is not None]
    labels = [
        "No Arthropathy" if BaseStates.NO_BLEEDING.value in states else None,
        (
            "Chronic Arthropathy"
            if BaseStates.CHRONIC_ARTHROPATHY.value in states
            else None
        ),
        "Severe Arthropathy" if BaseStates.SEVERE_ARTHROPATHY.value in states else None,
        "Inhibitor" if States.INHIBITOR.value in states else None,
    ]
    labels = [lbl for lbl, s in zip(labels, focus_states) if s is not None]

    od_state_counts = np.zeros((num_steps, len(focus_states)))
    pro_state_counts = np.zeros((num_steps, len(focus_states)))
    for results in od_results:
        for t in range(num_steps):
            for i, state in enumerate(focus_states):
                od_state_counts[t, i] += results["state_path"][t] == state
    for results in pro_results:
        for t in range(num_steps):
            for i, state in enumerate(focus_states):
                pro_state_counts[t, i] += results["state_path"][t] == state
    od_state_proportions = od_state_counts / len(od_results)
    pro_state_proportions = pro_state_counts / len(pro_results)

    window = 13
    od_state_proportions_smooth = np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(window) / window, mode="valid"),
        axis=0,
        arr=od_state_proportions,
    )
    pro_state_proportions_smooth = np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(window) / window, mode="valid"),
        axis=0,
        arr=pro_state_proportions,
    )
    valid_weeks = weeks[window - 1 :]

    fig, axes = plt.subplots(
        nrows=len(focus_states),
        ncols=1,
        figsize=(10, 2 * len(focus_states)),
        sharex=True,
    )
    if len(focus_states) == 1:
        axes = [axes]  # Ensure axes is iterable for single subplot
    colors_od = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    colors_pro = ["#66b3ff", "#ffcc99", "#99ff99", "#ff9999"]
    for i, (state, label) in enumerate(zip(focus_states, labels)):
        axes[i].plot(
            valid_weeks / 52,
            od_state_proportions_smooth[:, i],
            label="On-Demand",
            color=colors_od[i],
            linewidth=2.5,
            alpha=0.8,
        )
        axes[i].plot(
            valid_weeks / 52,
            pro_state_proportions_smooth[:, i],
            label="Prophylaxis",
            color=colors_pro[i],
            linewidth=2.5,
            alpha=0.8,
        )
        axes[i].set_title(label, fontsize=10)
        axes[i].set_ylabel("Proportion")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Years")
    plt.suptitle(f"State Proportions ({model_name.capitalize()} Model)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # type: ignore
    plt.savefig(output_dir / f"state_proportions_{model_name}.png", dpi=400)
    plt.close()
    logger.info(
        f"Saved state proportions time series plot to state_proportions_{model_name}.png"
    )
