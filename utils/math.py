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
    Estimates male body weight in kg using a piecewise Gompertz growth
    model (two consecutive growth segments) with a gradual late-life
    decline, calibrated by least-squares to male median body-weight
    milestones (WHO Child Growth Standards 2006; CDC Clinical Growth
    Charts; Fryar et al. 2021, NHANES anthropometric reference data).

    Structure (weeks):
      - Childhood segment   (0 - 676 wk, 0 - 13 y):  Gompertz A1, B1, K1
      - Adolescent/adult    (676 - 2860 wk, 13 - 55 y): Gompertz A2, B2, K2,
        value-continuous with the childhood segment at the 676 wk joint
      - Late-life decline   (> 2860 wk, > 55 y): exponential decay from
        the 55 y peak toward a late-life asymptote (~75 kg)

    Approximate milestone checks of the calibrated curve (kg):
      - 2 y: ~11   6 y: ~21   10 y: ~33   12 y: ~40   14 y: ~50
      - 18 y: ~69  25 y: ~83  40-55 y: ~88 (peak)     80+ y: ~85

    Patients enter the model at age 2, so the curve is only ever
    evaluated for weeks >= 104, where its absolute error vs. the
    reference medians is typically <= 3%.

    Args:
        week (int): Age in weeks (0 to ~5200 for 100 years)
        b (int): Offset in weeks (e.g., baseline age at model entry)
        weight_factor (float): Multiplicative scaling factor (PSA/scenario)

    Returns:
        float: Estimated weight in kg, rounded to 2 decimals
    """
    week += b

    # NOTE: Disable if assume valid inputs
    if not isinstance(week, int) or week < 0 or week > 5200:
        raise ValueError(
            "Week must be an integer between 0 and 5200 (approx. 100 years)"
        )

    # Childhood segment (0-13 y) — Gompertz parameters fitted to
    # WHO/CDC male median weights at 0-13 y
    A1 = 150.0
    B1 = 3.0116
    K1 = 0.00133

    # Adolescent/adult segment (13-55 y) — value-continuous at 676 wk:
    # v0 = A1*exp(-B1*exp(-K1*676)); B2 = -ln(v0/A2)
    A2 = 88.51
    B2 = 0.6982
    K2 = 0.003955

    # Segment joints
    childhood_end_week = 676  # ~13 years
    transition_week = 2860  # 55 years (start of late-life decline)

    if week <= childhood_end_week:
        weight = A1 * math.exp(-B1 * math.exp(-K1 * week))
    elif week <= transition_week:
        weight = A2 * math.exp(-B2 * math.exp(-K2 * (week - childhood_end_week)))
    else:
        # Decline phase: exponential decay from peak toward late-life asymptote
        peak_weight = A2 * math.exp(
            -B2 * math.exp(-K2 * (transition_week - childhood_end_week))
        )
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
