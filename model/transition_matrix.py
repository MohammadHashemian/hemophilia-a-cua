from pathlib import Path
from model.schemas import Regimes, Status
from src.utils.logger import get_logger
from src.data.loaders import PROJECT_ROOT
from model.treatments import initialize_treatments
from model.constants import (
    BLEED_RESOLUTION_PROB,
    CRITICAL_BLEED_RESOLUTION_PROB,
    ON_DEMAND_ARTHROPATHY_PROGRESSION_PROB,
    PROPHYLAXIS_ARTHROPATHY_PROGRESSION_PROB,
    CRITICAL_BLEED_DEATH_PROB,
    ON_DEMAND_COMPLICATION_PROB,
    PROPHYLAXIS_COMPLICATION_PROB,
    ITI_COMPLICATION_PROB,
    CHRONIC_ARTHROPATHY_STAY_PROB,
    SEVERE_ARTHROPATHY_STAY_PROB,
    ON_DEMAND_INHIBITOR_PROB,
    PROPHYLAXIS_INHIBITOR_PROB,
    ITI_ANNUAL_EABR,
    ITI_ANNUAL_AJBR,
    ITI_ANNUAL_ALBR,
)
import pandas as pd
import numpy as np

logger = get_logger()


def initialize_transition_matrix(
    path: Path, regime: Regimes, override: bool = True
) -> pd.DataFrame:
    """
    Initialize or load a transition matrix for the given treatment regime and populate it with probabilities.

    Args:
        path: Path to the CSV file storing the transition matrix.
        regime: Treatment regime (ON_DEMAND or PROPHYLAXIS).
        override: Whether to override the existing matrix if it exists.

    Returns:
        DataFrame containing the transition matrix with states as rows and columns.
    """
    states = [state.value for state in Status]
    try:
        if path.exists() and not override:
            logger.info(
                f"Loading transition matrix from: {path.relative_to(PROJECT_ROOT)}"
            )
            df = pd.read_csv(path)
            if "states" in df.columns:
                df.set_index("states", inplace=True)
            elif df.index.name != "states":
                logger.warning(
                    "CSV index does not match 'states', reinitializing matrix."
                )
                df = pd.DataFrame(0.0, index=states, columns=states)
            if not np.array_equal(df.columns, states):
                logger.warning("Column mismatch detected, reinitializing matrix.")
                df = pd.DataFrame(0.0, index=states, columns=states)
        else:
            logger.info(
                f"Creating new transition matrix at: {path.relative_to(PROJECT_ROOT)}"
            )
            df = pd.DataFrame(0.0, index=states, columns=states)
    except pd.errors.EmptyDataError:
        logger.warning(
            f"Empty or malformed CSV at {path.relative_to(PROJECT_ROOT)}, reinitializing matrix."
        )
        df = pd.DataFrame(0.0, index=states, columns=states)

    df.index.name = "states"
    treatments = initialize_treatments()
    treatment = treatments[regime]
    inhibitor_prob = (
        ON_DEMAND_INHIBITOR_PROB
        if regime == Regimes.ON_DEMAND
        else PROPHYLAXIS_INHIBITOR_PROB
    )
    arthropathy_prob = (
        ON_DEMAND_ARTHROPATHY_PROGRESSION_PROB
        if regime == Regimes.ON_DEMAND
        else PROPHYLAXIS_ARTHROPATHY_PROGRESSION_PROB
    )
    complication_prob = (
        ON_DEMAND_COMPLICATION_PROB
        if regime == Regimes.ON_DEMAND
        else PROPHYLAXIS_COMPLICATION_PROB
    )

    # Compute weekly bleeding probabilities
    weekly_abr = treatment.abr / 52
    weekly_eabr = (
        (treatment.eabr / treatment.abr) * weekly_abr if treatment.abr > 0 else 0.0
    )
    weekly_ajbr = (
        (treatment.ajbr / treatment.abr) * weekly_abr if treatment.abr > 0 else 0.0
    )
    weekly_albr = (
        (treatment.albr / treatment.abr) * weekly_abr if treatment.abr > 0 else 0.0
    )

    # Scale bleeding probabilities for on-demand to achieve target ABR
    bleed_scale = 1.8 if regime == Regimes.ON_DEMAND else 1.0
    weekly_eabr *= bleed_scale
    weekly_ajbr *= bleed_scale
    weekly_albr *= bleed_scale

    # Populate transition probabilities
    for state in states:
        if state == Status.NO_BLEEDING.value:
            df.loc[state, Status.MINOR_BLEEDING.value] = weekly_eabr
            df.loc[state, Status.MAJOR_BLEEDING.value] = weekly_ajbr
            df.loc[state, Status.CRITICAL_BLEEDING.value] = weekly_albr
            df.loc[state, Status.CHRONIC_ARTHROPATHY.value] = arthropathy_prob
            df.loc[state, Status.INHIBITOR.value] = inhibitor_prob
            df.loc[state, Status.COMPLICATION.value] = complication_prob
        elif state == Status.MINOR_BLEEDING.value:
            df.loc[state, Status.NO_BLEEDING.value] = BLEED_RESOLUTION_PROB
            df.loc[state, Status.COMPLICATION.value] = complication_prob
            df.loc[state, Status.INHIBITOR.value] = inhibitor_prob
        elif state == Status.MAJOR_BLEEDING.value:
            df.loc[state, Status.NO_BLEEDING.value] = BLEED_RESOLUTION_PROB
            df.loc[state, Status.CHRONIC_ARTHROPATHY.value] = arthropathy_prob
            df.loc[state, Status.COMPLICATION.value] = complication_prob
            df.loc[state, Status.INHIBITOR.value] = inhibitor_prob
        elif state == Status.CRITICAL_BLEEDING.value:
            df.loc[state, Status.NO_BLEEDING.value] = CRITICAL_BLEED_RESOLUTION_PROB
            df.loc[state, Status.DEATH.value] = CRITICAL_BLEED_DEATH_PROB
            df.loc[state, Status.COMPLICATION.value] = complication_prob
            df.loc[state, Status.INHIBITOR.value] = inhibitor_prob
        elif state == Status.CHRONIC_ARTHROPATHY.value:
            df.loc[state, Status.CHRONIC_ARTHROPATHY.value] = (
                CHRONIC_ARTHROPATHY_STAY_PROB
            )
            df.loc[state, Status.SEVERE_ARTHROPATHY.value] = arthropathy_prob
            df.loc[state, Status.MINOR_BLEEDING.value] = weekly_eabr
            df.loc[state, Status.MAJOR_BLEEDING.value] = weekly_ajbr
            df.loc[state, Status.CRITICAL_BLEEDING.value] = weekly_albr
            df.loc[state, Status.COMPLICATION.value] = complication_prob
            df.loc[state, Status.INHIBITOR.value] = inhibitor_prob
        elif state == Status.SEVERE_ARTHROPATHY.value:
            df.loc[state, Status.SEVERE_ARTHROPATHY.value] = (
                SEVERE_ARTHROPATHY_STAY_PROB
            )
            df.loc[state, Status.MINOR_BLEEDING.value] = weekly_eabr
            df.loc[state, Status.MAJOR_BLEEDING.value] = weekly_ajbr
            df.loc[state, Status.CRITICAL_BLEEDING.value] = weekly_albr
            df.loc[state, Status.COMPLICATION.value] = complication_prob
            df.loc[state, Status.INHIBITOR.value] = inhibitor_prob
        elif state == Status.COMPLICATION.value or state == Status.DEATH.value:
            df.loc[state, state] = 1.0
        elif state == Status.INHIBITOR.value:
            df.loc[state, Status.INHIBITOR.value] = 1.0
            df.loc[state, Status.MINOR_BLEEDING.value] = ITI_ANNUAL_EABR / 52
            df.loc[state, Status.MAJOR_BLEEDING.value] = ITI_ANNUAL_AJBR / 52
            df.loc[state, Status.CRITICAL_BLEEDING.value] = ITI_ANNUAL_ALBR / 52
            df.loc[state, Status.COMPLICATION.value] = ITI_COMPLICATION_PROB

    # Normalize probabilities to ensure row sums to 1
    for state in states:
        row_sum = df.loc[state].sum()
        if row_sum > 1:  # type: ignore
            logger.warning(
                f"{regime.value} {state} row sum {row_sum:.4f} > 1, normalizing"
            )
            df.loc[state] /= row_sum
        elif row_sum < 1:  # type: ignore
            if state == Status.CHRONIC_ARTHROPATHY.value:
                logger.debug(
                    f"{regime.value} {state} row sum {row_sum:.4f} < 1, adding to SEVERE_ARTHROPATHY"
                )
                df.loc[state, Status.SEVERE_ARTHROPATHY.value] += 1 - row_sum
            elif state == Status.SEVERE_ARTHROPATHY.value:
                logger.debug(
                    f"{regime.value} {state} row sum {row_sum:.4f} < 1, adding to SEVERE_ARTHROPATHY"
                )
                df.loc[state, Status.SEVERE_ARTHROPATHY.value] += 1 - row_sum
            else:
                logger.debug(
                    f"{regime.value} {state} row sum {row_sum:.4f} < 1, adding to NO_BLEEDING"
                )
                df.loc[state, Status.NO_BLEEDING.value] += 1 - row_sum

    # Log transition probabilities
    for state in states:
        logger.debug(f"{regime.value} transition probabilities from {state}:")
        for target_state in states:
            if df.loc[state, target_state] > 0:  # type: ignore
                logger.debug(f"  To {target_state}: {df.loc[state, target_state]:.4f}")
        logger.debug(f"  Row sum: {df.loc[state].sum():.4f}")

    logger.info(f"Saving {regime.value} transition matrix")
    df.to_csv(path, index=True)
    return df
