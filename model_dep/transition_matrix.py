from pathlib import Path
from model_dep.schemas import Regimes, States, BaseStates, BaseModel, Treatment
from src.utils.logger import get_logger
from src.data.loaders import PROJECT_ROOT
from model_dep.utils import probability_at_least_one_event, probability_no_events
from model_dep.constants import (
    BLEED_RESOLUTION_PROB,
    CRITICAL_BLEED_RESOLUTION_PROB,
    ON_DEMAND_ARTHROPATHY_PROGRESSION_PROB,
    PROPHYLAXIS_ARTHROPATHY_PROGRESSION_PROB,
    CRITICAL_BLEED_DEATH_PROB,
    ON_DEMAND_INHIBITOR_PROB_EARLY,
    PROPHYLAXIS_INHIBITOR_PROB_EARLY,
    ITI_RESOLUTION_PROB,
)
import pandas as pd
import numpy as np

logger = get_logger()


def initialize_transition_matrix(
    path: Path,
    regime: Regimes,
    model: BaseModel,
    treatments: dict[Regimes, Treatment],
    override: bool = True,
) -> pd.DataFrame:
    """
    Initialize or load a transition matrix for a given treatment regime and model.

    Args:
        path: Path to the CSV file for storing the transition matrix.
        regime: Treatment regime (ON_DEMAND or PROPHYLAXIS).
        model: Model defining the states (EARLY_MODEL, INTERMEDIATE_MODEL, or END_MODEL).
        treatments: Initialized treatments
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
                return df
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
    treatment = treatments[regime]
    inhibitor_prob = (
        ON_DEMAND_INHIBITOR_PROB_EARLY
        if regime == Regimes.ON_DEMAND
        else PROPHYLAXIS_INHIBITOR_PROB_EARLY
    )
    chronic_arthropathy_prob = (
        ON_DEMAND_ARTHROPATHY_PROGRESSION_PROB
        if regime == Regimes.ON_DEMAND
        else PROPHYLAXIS_ARTHROPATHY_PROGRESSION_PROB
    )

    # Compute weekly bleeding probabilities
    weekly_eabr_prob = probability_at_least_one_event(
        period="annual", rate=treatment.eabr
    )
    weekly_ajbr_prob = probability_at_least_one_event(
        period="annual", rate=treatment.ajbr
    )
    weekly_albr_prob = probability_at_least_one_event(
        period="annual", rate=treatment.albr
    )
    weekly_no_bleed_prob = probability_at_least_one_event(
        period="annual", rate=(52 - treatment.abr)
    )  # Expected 0.1426

    # Log probabilities for debugging
    logger.debug(f"weekly_no_bleed_prob: {weekly_no_bleed_prob:.4f}")
    logger.debug(f"weekly_eabr_prob: {weekly_eabr_prob:.4f}")
    logger.debug(f"weekly_ajbr_prob: {weekly_ajbr_prob:.4f}")
    logger.debug(f"weekly/albr_prob: {weekly_albr_prob:.4f}")
    logger.debug(f"chronic_arthropathy_prob: {chronic_arthropathy_prob:.4f}")
    logger.debug(f"inhibitor_prob: {inhibitor_prob:.4f}")

    # Calculate sum of non-NO_BLEEDING probabilities for NO_BLEEDING, MINOR_BLEEDING, MAJOR_BLEEDING
    other_probs_sum = (
        weekly_eabr_prob
        + weekly_ajbr_prob
        + weekly_albr_prob
        + (
            chronic_arthropathy_prob
            if BaseStates.CHRONIC_ARTHROPATHY.value in states
            else 0.0
        )
        + (inhibitor_prob if States.INHIBITOR.value in states else 0.0)
    )
    # Scale other probabilities if sum exceeds 1 - weekly_no_bleed_prob
    if other_probs_sum > 1.0 - weekly_no_bleed_prob:
        scale_factor = (1.0 - weekly_no_bleed_prob) / other_probs_sum
        weekly_eabr_prob *= scale_factor
        weekly_ajbr_prob *= scale_factor
        weekly_albr_prob *= scale_factor
        chronic_arthropathy_prob *= scale_factor
        inhibitor_prob *= scale_factor
        logger.debug(
            f"Scaled other probabilities by {scale_factor:.4f} to sum to {1.0 - weekly_no_bleed_prob:.4f}"
        )

    # Define transition rules
    transition_rules = {
        BaseStates.NO_BLEEDING.value: {
            BaseStates.NO_BLEEDING.value: weekly_no_bleed_prob,  # 0.1426
            States.MINOR_BLEEDING.value: weekly_eabr_prob,
            States.MAJOR_BLEEDING.value: weekly_ajbr_prob,
            States.LT_BLEEDING.value: weekly_albr_prob,
            BaseStates.CHRONIC_ARTHROPATHY.value: (
                chronic_arthropathy_prob
                if BaseStates.CHRONIC_ARTHROPATHY.value in states
                else 0.0
            ),
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
            States.DEATH.value: 0.0 if States.DEATH.value in states else 0.0,
        },
        States.MINOR_BLEEDING.value: {
            BaseStates.NO_BLEEDING.value: weekly_no_bleed_prob,  # 0.1426
            States.MINOR_BLEEDING.value: weekly_eabr_prob,
            States.MAJOR_BLEEDING.value: weekly_ajbr_prob,
            States.LT_BLEEDING.value: weekly_albr_prob,
            BaseStates.CHRONIC_ARTHROPATHY.value: (
                chronic_arthropathy_prob
                if BaseStates.CHRONIC_ARTHROPATHY.value in states
                else 0.0
            ),
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
            States.DEATH.value: 0.0 if States.DEATH.value in states else 0.0,
        },
        States.MAJOR_BLEEDING.value: {
            BaseStates.NO_BLEEDING.value: weekly_no_bleed_prob,  # 0.1426
            States.MINOR_BLEEDING.value: weekly_eabr_prob,
            States.MAJOR_BLEEDING.value: weekly_ajbr_prob,
            States.LT_BLEEDING.value: weekly_albr_prob,
            BaseStates.CHRONIC_ARTHROPATHY.value: (
                chronic_arthropathy_prob
                if BaseStates.CHRONIC_ARTHROPATHY.value in states
                else 0.0
            ),
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
            States.DEATH.value: 0.0 if States.DEATH.value in states else 0.0,
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
                chronic_arthropathy_prob
                if BaseStates.SEVERE_ARTHROPATHY.value in states
                else 0.0
            ),
            States.MINOR_BLEEDING.value: weekly_eabr_prob,
            States.MAJOR_BLEEDING.value: weekly_ajbr_prob,
            States.LT_BLEEDING.value: weekly_albr_prob,
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
            States.DEATH.value: 0.0 if States.DEATH.value in states else 0.0,
        },
        BaseStates.SEVERE_ARTHROPATHY.value: {
            States.MINOR_BLEEDING.value: weekly_eabr_prob,
            States.MAJOR_BLEEDING.value: weekly_ajbr_prob,
            States.LT_BLEEDING.value: weekly_albr_prob,
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
            States.DEATH.value: 0.0 if States.DEATH.value in states else 0.0,
        },
        States.SURGERY.value: {
            BaseStates.NO_BLEEDING.value: BLEED_RESOLUTION_PROB,
            States.INHIBITOR.value: (
                inhibitor_prob if States.INHIBITOR.value in states else 0.0
            ),
            States.DEATH.value: 0.0 if States.DEATH.value in states else 0.0,
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
            States.DEATH.value: 0.0 if States.DEATH.value in states else 0.0,
        },
        States.INHIBITOR.value: {
            States.INHIBITOR.value: 1 - ITI_RESOLUTION_PROB,
            States.MINOR_BLEEDING.value: ITI_RESOLUTION_PROB,
            States.DEATH.value: 0.0 if States.DEATH.value in states else 0.0,
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

    # Log pre-normalization probabilities
    for state in states:
        logger.debug(f"Pre-normalization {regime.value} {state}: {dict(df.loc[state])}")
        logger.debug(
            f"Pre-normalization row sum for {state}: {df.loc[state].sum():.4f}"
        )

    # Normalize probabilities, preserving transitions to NO_BLEEDING
    for state in states:
        row_sum = df.loc[state].sum()
        if abs(row_sum - 1.0) > 1e-6:  # Allow small numerical errors
            # Preserve transitions to NO_BLEEDING where applicable
            if state in [
                BaseStates.NO_BLEEDING.value,
                States.MINOR_BLEEDING.value,
                States.MAJOR_BLEEDING.value,
                States.SURGERY.value,
            ]:
                no_bleed_prob = (
                    df.loc[state, BaseStates.NO_BLEEDING.value]
                    if BaseStates.NO_BLEEDING.value in states
                    else 0.0
                )
                other_states = [s for s in states if s != BaseStates.NO_BLEEDING.value]
                if other_states:
                    other_sum = sum(df.loc[state, s] for s in other_states)
                    if other_sum > 0:  # type: ignore
                        # Scale other probabilities
                        scale_factor = (
                            (1.0 - no_bleed_prob) / other_sum if other_sum > 0 else 0  # type: ignore
                        )
                        for s in other_states:
                            df.loc[state, s] *= scale_factor
                        logger.debug(
                            f"Normalized {regime.value} {state} preserving NO_BLEEDING prob: {no_bleed_prob:.4f}"
                        )
                    else:
                        # Assign remaining probability to another state
                        default_state = (
                            BaseStates.CHRONIC_ARTHROPATHY.value
                            if BaseStates.CHRONIC_ARTHROPATHY.value in states
                            else States.MINOR_BLEEDING.value
                        )
                        if default_state in states:
                            df.loc[state, default_state] = 1.0 - no_bleed_prob  # type: ignore
                            logger.debug(
                                f"Assigned remaining prob to {default_state}: {1.0 - no_bleed_prob:.4f}"  # type: ignore
                            )
                else:
                    df.loc[state, state] = 1.0
                    logger.debug(f"Set {state} to 1.0 as only state")
            else:
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
                                else States.MINOR_BLEEDING.value  # Changed to avoid NO_BLEEDING
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
            if df.loc[state, target_state] > 0:  # type: ignore
                logger.debug(f"  To {target_state}: {df.loc[state, target_state]:.4f}")
        logger.debug(f"  Row sum: {df.loc[state].sum():.4f}")

    logger.info(f"Saving {regime.value} transition matrix")
    df.to_csv(path, index=True)
    return df
