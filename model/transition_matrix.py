from pathlib import Path
from model.schemas import Regimes, States, BaseStates, Model
from src.utils.logger import get_logger
from src.data.loaders import PROJECT_ROOT
from model.treatments import initialize_treatments
from model.constants import (
    BLEED_RESOLUTION_PROB,
    CRITICAL_BLEED_RESOLUTION_PROB,
    ON_DEMAND_ARTHROPATHY_PROGRESSION_PROB,
    PROPHYLAXIS_ARTHROPATHY_PROGRESSION_PROB,
    CRITICAL_BLEED_DEATH_PROB,
    ON_DEMAND_INHIBITOR_PROB_EARLY,
    PROPHYLAXIS_INHIBITOR_PROB_EARLY,
    ITI_ANNUAL_EABR,
    ITI_ANNUAL_AJBR,
    ITI_ANNUAL_ALBR,
)
import pandas as pd
import numpy as np

logger = get_logger()


def initialize_transition_matrix(
    path: Path,
    regime: Regimes,
    model: Model,
    override: bool = True,
) -> pd.DataFrame:
    """
    Initialize or load a transition matrix for a given treatment regime and model.

    Args:
        path: Path to the CSV file for storing the transition matrix.
        regime: Treatment regime (ON_DEMAND or PROPHYLAXIS).
        model: Model defining the states (EARLY_MODEL, INTERMEDIATE_MODEL, or END_MODEL).
        override: Whether to override the existing matrix if it exists.

    Returns:
        DataFrame containing the transition matrix with states as rows and columns.
    """
    states = model.states_value

    # Load or initialize the transition matrix
    try:
        if path.exists() and not override:
            logger.info(
                f"Loading transition matrix from: {path.relative_to(PROJECT_ROOT)}"
            )
            df = pd.read_csv(path)
            if "states" in df.columns:
                df.set_index("states", inplace=True)
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

    # Get treatment and define probabilities
    treatments = initialize_treatments()
    treatment = treatments[regime]
    inhibitor_prob = (
        ON_DEMAND_INHIBITOR_PROB_EARLY
        if regime == Regimes.ON_DEMAND
        else PROPHYLAXIS_INHIBITOR_PROB_EARLY
    )
    arthropathy_prob = (
        ON_DEMAND_ARTHROPATHY_PROGRESSION_PROB
        if regime == Regimes.ON_DEMAND
        else PROPHYLAXIS_ARTHROPATHY_PROGRESSION_PROB
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

    # Scale bleeding probabilities for on-demand
    bleed_scale = 1.8 if regime == Regimes.ON_DEMAND else 1.0
    weekly_eabr *= bleed_scale
    weekly_ajbr *= bleed_scale
    weekly_albr *= bleed_scale

    # Define transition rules as a dictionary: {source_state: {target_state: probability}}
    transition_rules = {
        BaseStates.NO_BLEEDING.value: {
            States.MINOR_BLEEDING.value: weekly_eabr,
            States.MAJOR_BLEEDING.value: weekly_ajbr,
            States.LT_BLEEDING.value: weekly_albr,
            BaseStates.CHRONIC_ARTHROPATHY.value: (
                arthropathy_prob
                if BaseStates.CHRONIC_ARTHROPATHY.value in states
                else 0.0
            ),
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
        },
        States.MINOR_BLEEDING.value: {
            BaseStates.NO_BLEEDING.value: BLEED_RESOLUTION_PROB,
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
        },
        States.MAJOR_BLEEDING.value: {
            BaseStates.NO_BLEEDING.value: BLEED_RESOLUTION_PROB,
            BaseStates.CHRONIC_ARTHROPATHY.value: (
                arthropathy_prob
                if BaseStates.CHRONIC_ARTHROPATHY.value in states
                else 0.0
            ),
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
        },
        States.LT_BLEEDING.value: {
            BaseStates.NO_BLEEDING.value: CRITICAL_BLEED_RESOLUTION_PROB,
            States.DEATH.value: (
                CRITICAL_BLEED_DEATH_PROB if States.DEATH.value in states else 0.0
            ),
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
        },
        BaseStates.CHRONIC_ARTHROPATHY.value: {
            BaseStates.SEVERE_ARTHROPATHY.value: (
                arthropathy_prob
                if BaseStates.SEVERE_ARTHROPATHY.value in states
                else 0.0
            ),
            States.MINOR_BLEEDING.value: weekly_eabr,
            States.MAJOR_BLEEDING.value: weekly_ajbr,
            States.LT_BLEEDING.value: weekly_albr,
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
        },
        BaseStates.SEVERE_ARTHROPATHY.value: {
            States.MINOR_BLEEDING.value: weekly_eabr,
            States.MAJOR_BLEEDING.value: weekly_ajbr,
            States.LT_BLEEDING.value: weekly_albr,
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
        },
        States.SURGERY.value: {
            BaseStates.NO_BLEEDING.value: BLEED_RESOLUTION_PROB,
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
        },
        States.REPLACEMENT.value: {
            BaseStates.SEVERE_ARTHROPATHY.value: (
                BLEED_RESOLUTION_PROB
                if BaseStates.SEVERE_ARTHROPATHY.value in states
                else 0.0
            ),
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
        },
        States.INHIBITOR.value: {
            States.MINOR_BLEEDING.value: ITI_ANNUAL_EABR / 52,
            States.MAJOR_BLEEDING.value: ITI_ANNUAL_AJBR / 52,
            States.LT_BLEEDING.value: ITI_ANNUAL_ALBR / 52,
        },
        States.DEATH.value: {
            States.DEATH.value: 1.0,
        },
    }

    # Populate transition matrix
    for state in states:
        if state in transition_rules:
            for target_state, prob in transition_rules[state].items():
                if target_state in states:
                    df.loc[state, target_state] = prob

    # Normalize probabilities to ensure row sums to 1
    for state in states:
        row_sum = df.loc[state].sum()
        if row_sum > 1:
            logger.warning(
                f"{regime.value} {state} row sum {row_sum:.4f} > 1, normalizing"
            )
            df.loc[state] /= row_sum
        elif row_sum < 1:
            default_state = (
                BaseStates.CHRONIC_ARTHROPATHY.value
                if state == BaseStates.CHRONIC_ARTHROPATHY.value
                and BaseStates.CHRONIC_ARTHROPATHY.value in states
                else (
                    BaseStates.SEVERE_ARTHROPATHY.value
                    if state == BaseStates.SEVERE_ARTHROPATHY.value
                    and BaseStates.SEVERE_ARTHROPATHY.value in states
                    else (
                        States.INHIBITOR.value
                        if state == States.INHIBITOR.value
                        and States.INHIBITOR.value in states
                        else BaseStates.NO_BLEEDING.value
                    )
                )
            )
            if default_state in states:
                logger.debug(
                    f"{regime.value} {state} row sum {row_sum:.4f} < 1, adding to {default_state}"
                )
                df.loc[state, default_state] += 1 - row_sum

    # Log transition probabilities
    for state in states:
        logger.debug(f"{regime.value} transition probabilities from {state}:")
        for target_state in states:
            if df.loc[state, target_state] > 0: # type: ignore
                logger.debug(f"  To {target_state}: {df.loc[state, target_state]:.4f}")
        logger.debug(f"  Row sum: {df.loc[state].sum():.4f}")

    logger.info(f"Saving {regime.value} transition matrix")
    df.to_csv(path, index=True)
    return df
