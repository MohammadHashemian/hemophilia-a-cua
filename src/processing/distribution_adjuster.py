import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ..utils.logger import get_logger

logger = get_logger()


def adjust_population_pyramid(
    df: pd.DataFrame,
    population_column: str,
    out_put_column,
    prevalence_factor: float = 1.1 / 100000,
):
    df[out_put_column] = df[population_column].apply(lambda x: x * prevalence_factor)
    return df


def adjust_age_distribution(
    df_known: pd.DataFrame, known_probs: np.ndarray
) -> np.ndarray:
    """
    Adjusts age-group population probabilities to account for unknowns using optimization.

    Parameters:
        df_known: Filtered DataFrame with known age groups (no unknowns).
        known_probs: np.array of observed probabilities in 5 age bins.

    Returns:
        np.ndarray with full population probabilities summing to 1.
    """
    unknown = 1 - np.sum(known_probs)

    logger.info("Calculating mean proportions in known age groups")
    target_means = [
        df_known["0-4"].mean(),
        df_known["5-13"].mean(),
        df_known["14-18"].mean(),
        df_known["19-44"].mean(),
        df_known["45+"].mean(),
    ]

    delta = np.array(target_means) - known_probs

    logger.info("Minimizing difference to estimate unknown distribution portion")

    result = minimize(
        lambda x: np.sum((x - delta) ** 2),
        x0=np.ones(5) * unknown / 5,
        method="SLSQP",
        bounds=[(0, None)] * 5,
        constraints={"type": "eq", "fun": lambda x: np.sum(x) - unknown},
    )

    if not result.success:
        raise ValueError("Optimization failed for adjusting age distribution.")

    full_probabilities = known_probs + result.x
    assert np.isclose(np.sum(full_probabilities), 1.0), "Distribution does not sum to 1"

    return full_probabilities
