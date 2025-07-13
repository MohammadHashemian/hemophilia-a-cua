from typing import Literal
from src.utils.logger import get_logger
import math


logger = get_logger()


def probability_at_least_one_event(
    lambda_value: float, interval: Literal["weekly", "annual"]
) -> float:
    """
    Calculate the probability of at least one event occurring in a given interval using a Poisson process.

    Args:
        lambda_value: Expected number of events (e.g., bleeding rate).
        interval: Time interval for the calculation ("weekly" or "annual").

    Returns:
        Probability of at least one event occurring (e.g., 0.4795 for joint bleeds in on-demand).
    """
    if lambda_value < 0:
        logger.warning(f"Negative lambda_value {lambda_value}, setting to 0")
        lambda_value = 0
    if interval == "annual":
        lambda_value /= 52  # Convert annual rate to weekly for weekly cycles
    return 1 - math.exp(-lambda_value)  # P(at least one) = 1 - P(none)


def probability_no_events(
    lambda_value: float, interval: Literal["weekly", "annual"]
) -> float:
    """
    Calculate the probability of no events occurring in a given interval using a Poisson process.

    Args:
        lambda_value: Expected number of events (e.g., total bleeding rate).
        interval: Time interval for the calculation ("weekly" or "annual").

    Returns:
        Probability of no events occurring (e.g., 0.4290 for on-demand no bleeds).
    """
    if lambda_value < 0:
        logger.warning(f"Negative lambda_value {lambda_value}, setting to 0")
        lambda_value = 0
    if interval == "annual":
        lambda_value /= 52  # Convert annual rate to weekly for weekly cycles
    return math.exp(-lambda_value)  # P(none) = e^(-lambda)
