import numpy as np
from src.utils.logger import get_logger

logger = get_logger()

def probability_at_least_one_event(rate: float, period: str) -> float:
    """
    Calculate the probability of at least one event occurring in a given period.

    Args:
        rate: Event rate (e.g., annual rate).
        period: Time period ('annual' or 'weekly').

    Returns:
        Probability of at least one event.
    """
    if period == "annual":
        return 1 - np.exp(-rate / 52)  # Convert annual rate to weekly
    elif period == "weekly":
        return 1 - np.exp(-rate)
    else:
        logger.error(f"Unsupported period: {period}")
        return 0.0

def probability_no_events(rate: float, period: str) -> float:
    """
    Calculate the probability of no events occurring in a given period.

    Args:
        rate: Event rate (e.g., annual rate).
        period: Time period ('annual' or 'weekly').

    Returns:
        Probability of no events.
    """
    return 1 - probability_at_least_one_event(rate, period)