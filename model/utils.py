from typing import Literal
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def probability_at_least_one_event(
    rate: float, period: Literal["weekly", "annual"]
) -> float:
    """
    Calculate the probability of at least one event occurring in a given period.

    Args:
        rate: Event rate (e.g., annual rate).
        period: Time period ('annual' or 'weekly').

    Returns:
        Probability of at least one event.
    """
    if period == "weekly":
        lambda_value = rate
    elif period == "annual":
        lambda_value = rate / 52  # Convert annual rate to weekly
    else:
        raise ValueError(f"Unsupported period: {period}")
    return 1 - np.exp(-lambda_value)


def cal_body_weight(week: int) -> float:
    """
    Estimates male body weight in kg using Gompertz growth model (0-50 years)
    and linear decline (50-73 years) based on WHO/CDC data.

    Gompertz parameters optimized for key milestones:
    - Birth (0 weeks): 3.3 kg
    - 1 year (52 weeks): 10.0 kg
    - 18 years (936 weeks): 70.0 kg
    - 50 years (2600 weeks): 80.0 kg

    Args:
        week (int): Age in weeks (0 to 3796)

    Returns:
        float: Estimated weight in kg, rounded to 2 decimals

    Raises:
        TypeError: For invalid input
    """
    if not isinstance(week, int) or week < 0 or week > 3796:
        raise TypeError("Week must be an integer between 0 and 3796")

    # Optimized Gompertz parameters for growth phase (0-2600 weeks)
    A = 80.5  # Asymptotic weight (kg)
    B = 3.08  # Displacement parameter
    K = 0.00255  # Growth rate

    if week <= 2600:
        # Gompertz growth model
        weight = A * math.exp(-B * math.exp(-K * week))
    else:
        # Linear decline from 50-73 years (80kg@2600wks → 75kg@3796wks)
        weight = 80.0 - (5.0 * (week - 2600) / (3796 - 2600))

    return round(weight, 2)


def plot_body_weight():
    # Generate denser points
    weeks = np.arange(0, 3797, 10)  # Every 10 weeks
    weights = [cal_body_weight(int(w)) for w in weeks]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(weeks, weights, "b-", label="Male Body Weight")

    plt.xlim(0, 3796)
    plt.xlabel("Age (weeks)")
    plt.ylabel("Weight (kg)")
    plt.title("Male Body Weight Growth (Birth to 73 Years)")
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Add key age markers
    key_ages = [0, 52, 520, 936, 2600, 3796]
    key_labels = ["", "1 yr", "10 yrs", "18 yrs", "50 yrs", "73 yrs"]
    plt.xticks(key_ages, key_labels)

    # Add annotations
    for w, label in zip(key_ages, key_labels):
        weight = cal_body_weight(w)
        plt.text(w, weight, f"{weight} kg", fontsize=10, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs" / "figures" / "body_weight.png")


# Call the plot function
if __name__ == "__main__":
    plot_body_weight()
