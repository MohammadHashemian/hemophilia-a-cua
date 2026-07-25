import math

import numpy as np
from numba import jit, njit
from scipy.stats import poisson


def to_weekly(annual_value: float, weeks_per_year: int = 52) -> float:
    return annual_value / weeks_per_year


@njit(cache=True)
def factorial_numba(n: int):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


@jit(cache=True, fastmath=True)
def cal_body_weight(
    week: int | float, b: int | float = 0, weight_factor: float = 1.0
) -> float:
    """
    Estimates male body weight in kg using Gompertz growth model (0-50 years)
    and gradual decline thereafter, based on real-world data patterns.

    Key milestones (approximate average for modern Western males):
    - Birth (0 weeks): ~3.3 kg
    - 1 year (52 weeks): ~10 kg
    - 18 years (~936 weeks): ~73 kg
    - Peak ~40-50 years (~2600 weeks): ~90-95 kg
    - 80+ years: ~80-85 kg (gradual decline after ~55 years)

    Decline is modeled as exponential decay toward a late-life asymptote (~75 kg),
    providing a smooth, realistic reduction without abrupt drops.

    Args:
        week (int): Age in weeks (0 to ~5200 for 100 years)
        b (int): Offset in weeks (e.g., for adjustment)

    Returns:
        float: Estimated weight in kg, rounded to 2 decimals
    """
    week += b

    # NOTE: Disable if assume valid inputs
    if not isinstance(week, int) or week < 0 or week > 5200:
        raise ValueError(
            "Week must be an integer between 0 and 5200 (approx. 100 years)"
        )

    # Gompertz parameters tuned for realistic growth to adult peak ~93 kg
    # A = 93.0
    # B = 3.15
    # K = 0.00245
    A = 90.0  # Asymptotic adult weight during growth phase
    B = 3.3
    K = 0.0032

    # Transition point: around age 50 years (~2700 weeks)
    transition_week = 2700

    if week <= transition_week:
        # Growth phase (Gompertz)
        weight = A * math.exp(-B * math.exp(-K * week))
    else:
        # Decline phase: exponential decay from peak toward late-life asymptote
        peak_weight = A * math.exp(-B * math.exp(-K * transition_week))
        late_asymptote = 75.0  # Realistic floor for very old age
        decline_rate = 0.00015  # Slow decay for ~15-18 kg drop over 45 years

        weight = late_asymptote + (peak_weight - late_asymptote) * math.exp(
            -decline_rate * (week - transition_week)
        )

    return round(weight * weight_factor, 2)


# @vectorize([float64(float64)], target="cpu")
def prob_at_least_one(lam: float) -> float:
    """
    Calculate the probability of at least one event occurring in a given interval.

    Args:
        lam: Mean number of events occurring within the given interval

    Returns:
        Probability of at least one event.
    """
    # Converse probability
    # P(at least one) = 1 - p(failure)**n
    # n: number of trials
    return 1 - np.exp(-lam)


def expm_prob(rate: float, dt: float = 1.0) -> float:
    """Convert hazard rate to discrete probability."""
    return 1 - np.exp(-rate * dt)


@njit(cache=True)
def zero_truncated_mass_function_numba(lam: float, k: int):
    if k == 0:
        raise ValueError("Zero is truncated")
    return np.power(lam, k) / ((np.exp(lam) - 1) * factorial_numba(k))


@njit(cache=True)
def build_zero_truncated_poisson_probs(lam: float, k_max: int):
    """
    Returns:
        k_values: [1, ..., k_max]
        probs: normalized zero-truncated Poisson probabilities
    """

    # Guards
    if lam <= 0.0 or np.isnan(lam) or np.isinf(lam):
        # degenerate fallback: all mass at 1
        k_values = np.arange(1, k_max + 1)
        probs = np.zeros(k_max)
        probs[0] = 1.0
        return k_values, probs

    # Allocate
    k_values = np.arange(1, k_max + 1)
    probs = np.empty(k_max)

    # Normalization
    # Z = e^λ - 1  (use expm1 for stability)
    Z = np.expm1(lam)

    # First term: P(K=1 | K>=1)
    p = lam / Z
    probs[0] = p

    # Recurrence
    for i in range(1, k_max):
        k = i + 1  # since index 0 → k=1
        p = p * lam / k
        probs[i] = p

    # Protect against drift
    total = probs.sum()

    if total <= 0.0 or np.isnan(total):
        # Fallback (should be extremely rare)
        probs[:] = 0.0
        probs[0] = 1.0
    else:
        probs /= total

    return k_values, probs


def poisson_mass_function(lam: float, k: int, loc: int = 0):
    """
    Poisson mass function(given k): exp(-λ) * ((λ)**k)/ k!
    Args:
        lam: λ
        k: number of expected events
        loc: to shift distribution, 0 to standardized form by default
    """
    return poisson.pmf(k=k, mu=lam, loc=loc)


def zero_truncated_mass_function(
    lam: int | float | np.number | np.int64, k: int | float | np.number
) -> float:
    """
    Zero-Truncated Poisson PMF: (λ**k) / ((e**λ) -1) * k!
    Args:
        k: value(s) to evaluate the PMF at (must be integer >= 1).
        lam: rate parameter of the underlying poisson distribution.
    """
    # The classic ZTP formula
    if not isinstance(k, int | float | np.number):
        raise TypeError(f"Invalid input, expected number value, got {type(k)}")
    if k == 0:
        raise ValueError("zero is truncated")
    numerator = np.power(lam, k)
    denominator = (math.exp(lam) - 1) * math.factorial(int(k))
    res = numerator / denominator
    return res
