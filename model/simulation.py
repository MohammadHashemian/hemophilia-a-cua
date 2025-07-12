from enum import Enum
from typing import Literal
from pathlib import Path
import pandas as pd
import numpy as np
import math
from model.markov import MarkovChain
from src.utils.logger import get_logger
from src.data.loaders import PROJECT_ROOT

# Initialize logger for tracking operations and debugging
logger = get_logger()

# Constants for simulation parameters
NUMBER_OF_CYCLES = 73 * 52  # Total weeks in 73 years for Markov chain cycles
# Clinical parameters (events per year)
ON_DEMAND_ANNUAL_ABR = 44  # Annual bleeding rate for on-demand treatment
ON_DEMAND_ANNUAL_AJBR = 34  # Annual joint bleeding rate for on-demand treatment
PROPHYLAXIS_ANNUAL_ABR = 3.76  # Annual bleeding rate for prophylaxis
PROPHYLAXIS_ANNUAL_AJBR = 3.76  # Annual joint bleeding rate
# Life-threatening bleeding rates (derived from table: 1.5% intracranial + 1% neck/throat + 2% gastrointestinal = 4.5%)
LIFE_THREATENING_BLEEDING_FRACTION = 0.045
ON_DEMAND_ANNUAL_ALBR = (
    ON_DEMAND_ANNUAL_ABR * LIFE_THREATENING_BLEEDING_FRACTION
)  # 1.98 events/year
PROPHYLAXIS_ANNUAL_ALBR = (
    PROPHYLAXIS_ANNUAL_ABR * LIFE_THREATENING_BLEEDING_FRACTION
)  # 0.1692 events/year
# Extra-articular bleeding rates (muscles + mucous membranes: 5% + 10% = 15%)
EXTRA_ARTICULAR_BLEEDING_FRACTION = 0.15
ON_DEMAND_ANNUAL_EABR = (
    ON_DEMAND_ANNUAL_ABR * EXTRA_ARTICULAR_BLEEDING_FRACTION
)  # 6.6 events/year
PROPHYLAXIS_ANNUAL_EABR = (
    PROPHYLAXIS_ANNUAL_ABR * EXTRA_ARTICULAR_BLEEDING_FRACTION
)  # 0.564 events/year
# Cost parameters
HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL = (
    58_000  # Cost per unit of factor VIII (Rial)
)
PROPHYLAXIS_FACTOR_DOSING_PER_INJECTION_PER_KG = 30  # Dose per injection (IU/kg)
# Placeholder transition probabilities (to be replaced with clinical data)
BLEED_RESOLUTION_PROB = (
    0.9  # Probability of resolving minor/major bleeds to NO_BLEEDING
)
CRITICAL_BLEED_RESOLUTION_PROB = 0.7  # Lower probability for critical bleeds
ARTHROPATHY_PROGRESSION_PROB = 0.01  # Probability of developing/worsening arthropathy
CRITICAL_BLEED_DEATH_PROB = 0.05  # Probability of death from critical bleeds
COMPLICATION_PROB = 0.01  # Probability of complications (e.g., surgery)
CHRONIC_ARTHROPATHY_STAY_PROB = 0.9  # Probability of staying in CHRONIC_ARTHROPATHY
SEVERE_ARTHROPATHY_STAY_PROB = 0.95  # Probability of staying in SEVERE_ARTHROPATHY


class Status(Enum):
    """Enum representing possible health states for a hemophilia A patient."""

    NO_BLEEDING = "alive_wo_arthropathy"
    MINOR_BLEEDING = "extra_articular_bleeding"
    MAJOR_BLEEDING = "articular_bleeding"
    CRITICAL_BLEEDING = "life_threatening_bleeding"
    SEVERE_ARTHROPATHY = "alive_w_severe_arthropathy"
    CHRONIC_ARTHROPATHY = "alive_w_arthropathy"
    COMPLICATION = "surgery_or_injury_or_infection"
    DEATH = "dies_any_reason"
    INHIBITOR = "developing_inhibitor"


class State:
    """Represents a health state with associated utility and cost."""

    def __init__(self, status: Status, utility_score: float, costs: int):
        self.status = status
        self.utility = utility_score
        self.costs = costs


class Treatment:
    """Defines a treatment regime with dosing and bleeding rates."""

    def __init__(
        self,
        name: str,
        dose: int,
        frequency: int,
        abr: float,
        ajbr: float,
        albr: float,
        eabr: float,
    ):
        self.name = name
        self.dose = dose  # Dose per injection (IU/kg)
        self.frequency = frequency  # Injections per week
        self.abr = abr  # Annual bleeding rate
        self.ajbr = ajbr  # Annual joint bleeding rate
        self.albr = albr  # Annual life-threatening bleeding rate
        self.eabr = eabr  # Annual extra-articular bleeding rate


class Regimes(Enum):
    """Enum for treatment regimes."""

    PROPHYLAXIS = "prophylaxis"
    ON_DEMAND = "on-demand"


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


def initialize_treatments() -> dict[Regimes, Treatment]:
    """Initialize treatment regimes with their respective parameters."""
    return {
        Regimes.ON_DEMAND: Treatment(
            name=Regimes.ON_DEMAND.value,
            dose=50,
            frequency=3,
            abr=ON_DEMAND_ANNUAL_ABR,
            ajbr=ON_DEMAND_ANNUAL_AJBR,
            albr=ON_DEMAND_ANNUAL_ALBR,
            eabr=ON_DEMAND_ANNUAL_EABR,
        ),
        Regimes.PROPHYLAXIS: Treatment(
            name=Regimes.PROPHYLAXIS.value,
            dose=35,
            frequency=3,
            abr=PROPHYLAXIS_ANNUAL_ABR,
            ajbr=PROPHYLAXIS_ANNUAL_AJBR,
            albr=PROPHYLAXIS_ANNUAL_ALBR,
            eabr=PROPHYLAXIS_ANNUAL_EABR,
        ),
    }


def initialize_transition_matrix(path: Path, regime: Regimes) -> pd.DataFrame:
    """
    Initialize or load a transition matrix for the given treatment regime and populate it with probabilities.

    Args:
        path: Path to the CSV file storing the transition matrix.
        regime: Treatment regime (ON_DEMAND or PROPHYLAXIS).

    Returns:
        DataFrame containing the transition matrix with states as rows and columns.
    """
    # Define states from Status enum
    states = [state.value for state in Status]

    # Initialize or load the transition matrix
    try:
        if path.exists():
            logger.info(
                f"Loading transition matrix from: {path.relative_to(PROJECT_ROOT)}"
            )
            df = pd.read_csv(path)
            # Check if 'states' is a column or the index
            if "states" in df.columns:
                df.set_index("states", inplace=True)
            elif df.index.name != "states":
                logger.warning(
                    "CSV index does not match 'states', reinitializing matrix."
                )
                df = pd.DataFrame(0.0, index=states, columns=states)
            # Verify columns match expected states
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

    # Get treatment parameters
    treatments = initialize_treatments()
    treatment = treatments[regime]

    # Populate transition probabilities based on regime
    for state in states:
        if state == Status.NO_BLEEDING.value:
            # Transitions from NO_BLEEDING
            df.loc[state, Status.NO_BLEEDING.value] = probability_no_events(
                treatment.abr, "annual"
            )
            df.loc[state, Status.MINOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.eabr, "annual"
            )
            df.loc[state, Status.MAJOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.ajbr, "annual"
            )
            df.loc[state, Status.CRITICAL_BLEEDING.value] = (
                probability_at_least_one_event(treatment.albr, "annual")
            )
            df.loc[state, Status.CHRONIC_ARTHROPATHY.value] = (
                ARTHROPATHY_PROGRESSION_PROB  # Placeholder: joint bleeds may lead to arthropathy
            )
        elif state == Status.MINOR_BLEEDING.value:
            # Transitions from MINOR_BLEEDING
            df.loc[state, Status.NO_BLEEDING.value] = (
                BLEED_RESOLUTION_PROB  # Assume minor bleeds resolve quickly
            )
            df.loc[state, Status.MINOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.eabr, "annual"
            )
            df.loc[state, Status.MAJOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.ajbr, "annual"
            )
            df.loc[state, Status.CRITICAL_BLEEDING.value] = (
                probability_at_least_one_event(treatment.albr, "annual")
            )
            df.loc[state, Status.COMPLICATION.value] = (
                COMPLICATION_PROB  # Placeholder: minor bleeds may lead to complications
            )
        elif state == Status.MAJOR_BLEEDING.value:
            # Transitions from MAJOR_BLEEDING
            df.loc[state, Status.NO_BLEEDING.value] = (
                BLEED_RESOLUTION_PROB  # Assume joint bleeds resolve with treatment
            )
            df.loc[state, Status.MINOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.eabr, "annual"
            )
            df.loc[state, Status.MAJOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.ajbr, "annual"
            )
            df.loc[state, Status.CRITICAL_BLEEDING.value] = (
                probability_at_least_one_event(treatment.albr, "annual")
            )
            df.loc[state, Status.CHRONIC_ARTHROPATHY.value] = (
                ARTHROPATHY_PROGRESSION_PROB  # Joint bleeds cause arthropathy
            )
            df.loc[state, Status.COMPLICATION.value] = (
                COMPLICATION_PROB  # Placeholder: bleeds may lead to surgery
            )
        elif state == Status.CRITICAL_BLEEDING.value:
            # Transitions from CRITICAL_BLEEDING
            df.loc[state, Status.NO_BLEEDING.value] = (
                CRITICAL_BLEED_RESOLUTION_PROB  # Lower resolution probability
            )
            df.loc[state, Status.MINOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.eabr, "annual"
            )
            df.loc[state, Status.MAJOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.ajbr, "annual"
            )
            df.loc[state, Status.CRITICAL_BLEEDING.value] = (
                probability_at_least_one_event(treatment.albr, "annual")
            )
            df.loc[state, Status.DEATH.value] = (
                CRITICAL_BLEED_DEATH_PROB  # Higher risk of death
            )
            df.loc[state, Status.COMPLICATION.value] = (
                COMPLICATION_PROB  # Placeholder: critical bleeds may lead to complications
            )
        elif state == Status.CHRONIC_ARTHROPATHY.value:
            # Transitions from CHRONIC_ARTHROPATHY
            df.loc[state, Status.CHRONIC_ARTHROPATHY.value] = (
                CHRONIC_ARTHROPATHY_STAY_PROB  # Persistent condition
            )
            df.loc[state, Status.SEVERE_ARTHROPATHY.value] = (
                ARTHROPATHY_PROGRESSION_PROB  # Worsening to severe
            )
            df.loc[state, Status.MINOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.eabr, "annual"
            )
            df.loc[state, Status.MAJOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.ajbr, "annual"
            )
            df.loc[state, Status.CRITICAL_BLEEDING.value] = (
                probability_at_least_one_event(treatment.albr, "annual")
            )
            df.loc[state, Status.COMPLICATION.value] = (
                COMPLICATION_PROB  # Placeholder: arthropathy may require surgery
            )
        elif state == Status.SEVERE_ARTHROPATHY.value:
            # Transitions from SEVERE_ARTHROPATHY
            df.loc[state, Status.SEVERE_ARTHROPATHY.value] = (
                SEVERE_ARTHROPATHY_STAY_PROB  # Highly persistent
            )
            df.loc[state, Status.MINOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.eabr, "annual"
            )
            df.loc[state, Status.MAJOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.ajbr, "annual"
            )
            df.loc[state, Status.CRITICAL_BLEEDING.value] = (
                probability_at_least_one_event(treatment.albr, "annual")
            )
            df.loc[state, Status.COMPLICATION.value] = (
                COMPLICATION_PROB  # Placeholder: severe arthropathy may require surgery
            )
        elif state in [
            Status.COMPLICATION.value,
            Status.DEATH.value,
            Status.INHIBITOR.value,
        ]:
            # Placeholder for absorbing or complex states
            df.loc[state, state] = (
                1.0  # Assume absorbing for simplicity (e.g., DEATH stays in DEATH)
            )

    # Normalize probabilities to ensure row sums to 1
    for state in states:
        row_sum = df.loc[state].sum()
        if row_sum > 1:  # type: ignore
            logger.warning(
                f"{regime.value} {state} row sum {row_sum:.4f} > 1, normalizing"
            )
            df.loc[state] /= row_sum
        elif row_sum < 1:  # type: ignore
            logger.debug(
                f"{regime.value} {state} row sum {row_sum:.4f} < 1, adding to {state}"
            )
            df.loc[state, state] += 1 - row_sum

    # Log the transition probabilities for debugging
    for state in states:
        logger.debug(f"{regime.value} transition probabilities from {state}:")
        for target_state in states:
            if df.loc[state, target_state] > 0:  # type: ignore
                logger.debug(f"  To {target_state}: {df.loc[state, target_state]:.4f}")
        logger.debug(f"  Row sum: {df.loc[state].sum():.4f}")

    # Save the updated matrix
    logger.info(f"Saving {regime.value} transition matrix")
    df.to_csv(path, index=True)
    return df


def run():
    """
    Main function to set up and run the Markov chain simulation for hemophilia A.

    Initializes treatment regimes and transition matrices, preparing for state transitions.
    """
    # Define paths for transition matrices
    od_matrix_path = PROJECT_ROOT / "data" / "processed" / "od_transition_matrix.csv"
    pro_matrix_path = PROJECT_ROOT / "data" / "processed" / "pro_transition_matrix.csv"

    # Initialize treatments
    treatments = initialize_treatments()

    # Initialize transition matrices for both regimes
    od_matrix = initialize_transition_matrix(od_matrix_path, Regimes.ON_DEMAND)
    pro_matrix = initialize_transition_matrix(pro_matrix_path, Regimes.PROPHYLAXIS)

    # Placeholder for Markov chain simulation
    # TODO: Implement MarkovChain from model.markov using od_matrix and pro_matrix
    logger.info("Transition matrices initialized. Ready for Markov chain simulation.")

    # Example placeholder transition matrix (to be replaced with actual logic)
    transition_matrix = np.array([[0.1, 0.9], [0.9, 0.1]])
    logger.debug(f"Placeholder transition matrix: {transition_matrix}")

    # Note: The MarkovChain class and further simulation logic need to be implemented
    # to use the transition matrices for state transitions over NUMBER_OF_CYCLES.
