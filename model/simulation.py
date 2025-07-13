from pathlib import Path
from model.markov import MarkovChain
from model.schemas import Regimes, Status, Treatment
from src.utils.logger import get_logger
from src.data.loaders import PROJECT_ROOT
from src.utils.utils import probability_at_least_one_event, probability_no_events
import pandas as pd
import numpy as np

# Initialize logger for tracking operations and debugging
logger = get_logger()

# Constants for simulation parameters
NUMBER_OF_CYCLES = 73 * 52  # Total weeks in 73 years for Markov chain cycles
# Clinical parameters (events per year)
ON_DEMAND_ANNUAL_ABR = 44  # Annual bleeding rate for on-demand treatment
ON_DEMAND_ANNUAL_AJBR = 34  # Annual joint bleeding rate for on-demand treatment
PROPHYLAXIS_ANNUAL_ABR = 3.76  # Annual bleeding rate for prophylaxis
PROPHYLAXIS_ANNUAL_AJBR = 3.66  # Annual joint bleeding rate
# Life-threatening bleeding rates
LIFE_THREATENING_BLEEDING_FRACTION = 0.045
ON_DEMAND_ANNUAL_ALBR = (
    ON_DEMAND_ANNUAL_ABR * LIFE_THREATENING_BLEEDING_FRACTION
)  # 1.98 events/year
PROPHYLAXIS_ANNUAL_ALBR = (
    PROPHYLAXIS_ANNUAL_ABR * LIFE_THREATENING_BLEEDING_FRACTION
)  # 0.1692 events/year
# Extra-articular bleeding rates
EXTRA_ARTICULAR_BLEEDING_FRACTION = 0.15
ON_DEMAND_ANNUAL_EABR = (
    ON_DEMAND_ANNUAL_ABR * EXTRA_ARTICULAR_BLEEDING_FRACTION
)  # 6.6 events/year
PROPHYLAXIS_ANNUAL_EABR = (
    PROPHYLAXIS_ANNUAL_ABR * EXTRA_ARTICULAR_BLEEDING_FRACTION
)  # 0.564 events/year
# ITI-specific bleeding rates (based on prophylaxis, reduced for severe bleeds)
ITI_ANNUAL_ABR = PROPHYLAXIS_ANNUAL_ABR  # 3.76
ITI_ANNUAL_AJBR = PROPHYLAXIS_ANNUAL_AJBR * 0.5  # 1.83
ITI_ANNUAL_ALBR = PROPHYLAXIS_ANNUAL_ALBR * 0.5  # 0.0846
ITI_ANNUAL_EABR = PROPHYLAXIS_ANNUAL_EABR  # 0.564
# Cost parameters
HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL = (
    58_000  # Cost per unit of factor VIII (Rial)
)
AVG_DOSE_PER_BLEED = 25.805  # Weighted average: 0.8*25 + 0.05*31.5 + 0.1*25 + 0.015*32 + 0.01*35 + 0.02*45
# Placeholder transition probabilities
BLEED_RESOLUTION_PROB = 0.9  # Probability of resolving minor/major bleeds
CRITICAL_BLEED_RESOLUTION_PROB = 0.7  # Lower probability for critical bleeds
ARTHROPATHY_PROGRESSION_PROB = 0.01  # Probability of developing/worsening arthropathy
CRITICAL_BLEED_DEATH_PROB = 0.05  # Probability of death from critical bleeds
COMPLICATION_PROB = 0.01  # Probability of complications (e.g., surgery)
ITI_COMPLICATION_PROB = COMPLICATION_PROB * 0.5  # Reduced for ITI
CHRONIC_ARTHROPATHY_STAY_PROB = 0.9  # Probability of staying in CHRONIC_ARTHROPATHY
SEVERE_ARTHROPATHY_STAY_PROB = 0.95  # Probability of staying in SEVERE_ARTHROPATHY
# Inhibitor development probabilities (weekly)
ON_DEMAND_INHIBITOR_PROB = (
    7.94e-5  # Weekly probability for 30% lifetime risk over 73 years
)
PROPHYLAXIS_INHIBITOR_PROB = (
    4.29e-5  # Weekly probability for 15% lifetime risk over 73 years
)
# ITI parameters
ITI_SUCCESS_PROB_6_MONTHS = 0.8  # 80% success rate over 6 months
WEEKS_PER_6_MONTHS = 26
ITI_WEEKLY_SUCCESS_PROB = (
    1 - (1 - ITI_SUCCESS_PROB_6_MONTHS) ** (1 / WEEKS_PER_6_MONTHS)
) * 0.5  # ~0.0303, scaled to reduce early transitions


def initialize_treatments() -> dict[Regimes, Treatment]:
    """Initialize treatment regimes with their respective parameters."""
    return {
        Regimes.ON_DEMAND: Treatment(
            name=Regimes.ON_DEMAND.value,
            dose_joint=25,
            dose_muscle=31.5,
            dose_mucous=25,
            dose_intracranial=32,
            dose_neck_throat=35,
            dose_gastro=45,
            duration_joint=2,
            duration_muscle=5,
            duration_mucous=3,
            duration_intracranial=21,
            duration_neck_throat=14,
            duration_gastro=11,
            avg_dose_per_bleed=AVG_DOSE_PER_BLEED,
            abr=ON_DEMAND_ANNUAL_ABR,
            ajbr=ON_DEMAND_ANNUAL_AJBR,
            albr=ON_DEMAND_ANNUAL_ALBR,
            eabr=ON_DEMAND_ANNUAL_EABR,
        ),
        Regimes.PROPHYLAXIS: Treatment(
            name=Regimes.PROPHYLAXIS.value,
            dose_joint=25,  # Use high-dose for all bleeds
            dose_muscle=31.5,
            dose_mucous=25,
            dose_intracranial=32,
            dose_neck_throat=35,
            dose_gastro=45,
            duration_joint=2,
            duration_muscle=5,
            duration_mucous=3,
            duration_intracranial=21,
            duration_neck_throat=14,
            duration_gastro=11,
            avg_dose_per_bleed=AVG_DOSE_PER_BLEED,
            abr=PROPHYLAXIS_ANNUAL_ABR,
            ajbr=PROPHYLAXIS_ANNUAL_AJBR,
            albr=PROPHYLAXIS_ANNUAL_ALBR,
            eabr=PROPHYLAXIS_ANNUAL_EABR,
        ),
    }


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
    # Define states from Status enum
    states = [state.value for state in Status]

    # Initialize or load the transition matrix
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

    # Get treatment parameters and inhibitor probability
    treatments = initialize_treatments()
    treatment = treatments[regime]
    inhibitor_prob = (
        ON_DEMAND_INHIBITOR_PROB
        if regime == Regimes.ON_DEMAND
        else PROPHYLAXIS_INHIBITOR_PROB
    )

    # Populate transition probabilities based on regime
    for state in states:
        if state == Status.NO_BLEEDING.value:  # alive_wo_arthropathy
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
                ARTHROPATHY_PROGRESSION_PROB
            )
            df.loc[state, Status.INHIBITOR.value] = inhibitor_prob
        elif state == Status.MINOR_BLEEDING.value:  # minor_bleeding
            df.loc[state, Status.NO_BLEEDING.value] = BLEED_RESOLUTION_PROB
            df.loc[state, Status.MINOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.eabr, "annual"
            )
            df.loc[state, Status.MAJOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.ajbr, "annual"
            )
            df.loc[state, Status.CRITICAL_BLEEDING.value] = (
                probability_at_least_one_event(treatment.albr, "annual")
            )
            df.loc[state, Status.COMPLICATION.value] = COMPLICATION_PROB
            df.loc[state, Status.INHIBITOR.value] = inhibitor_prob
        elif state == Status.MAJOR_BLEEDING.value:  # articular_bleeding
            df.loc[state, Status.NO_BLEEDING.value] = BLEED_RESOLUTION_PROB
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
                ARTHROPATHY_PROGRESSION_PROB
            )
            df.loc[state, Status.COMPLICATION.value] = COMPLICATION_PROB
            df.loc[state, Status.INHIBITOR.value] = inhibitor_prob
        elif state == Status.CRITICAL_BLEEDING.value:  # surgery_or_injury_or_infection
            df.loc[state, Status.NO_BLEEDING.value] = CRITICAL_BLEED_RESOLUTION_PROB
            df.loc[state, Status.MINOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.eabr, "annual"
            )
            df.loc[state, Status.MAJOR_BLEEDING.value] = probability_at_least_one_event(
                treatment.ajbr, "annual"
            )
            df.loc[state, Status.CRITICAL_BLEEDING.value] = (
                probability_at_least_one_event(treatment.albr, "annual")
            )
            df.loc[state, Status.DEATH.value] = CRITICAL_BLEED_DEATH_PROB
            df.loc[state, Status.COMPLICATION.value] = COMPLICATION_PROB
            df.loc[state, Status.INHIBITOR.value] = inhibitor_prob
        elif state == Status.CHRONIC_ARTHROPATHY.value:  # chronic_arthropathy
            df.loc[state, Status.CHRONIC_ARTHROPATHY.value] = (
                CHRONIC_ARTHROPATHY_STAY_PROB
            )
            df.loc[state, Status.SEVERE_ARTHROPATHY.value] = (
                ARTHROPATHY_PROGRESSION_PROB
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
            df.loc[state, Status.COMPLICATION.value] = COMPLICATION_PROB
            df.loc[state, Status.INHIBITOR.value] = inhibitor_prob
        elif state == Status.SEVERE_ARTHROPATHY.value:  # severe_arthropathy
            df.loc[state, Status.SEVERE_ARTHROPATHY.value] = (
                SEVERE_ARTHROPATHY_STAY_PROB
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
            df.loc[state, Status.COMPLICATION.value] = COMPLICATION_PROB
            df.loc[state, Status.INHIBITOR.value] = inhibitor_prob
        elif state == Status.COMPLICATION.value or state == Status.DEATH.value:
            # Absorbing states
            df.loc[state, state] = 1.0
        elif state == Status.INHIBITOR.value:  # inhibitor
            df.loc[state, Status.NO_BLEEDING.value] = ITI_WEEKLY_SUCCESS_PROB
            df.loc[state, Status.MINOR_BLEEDING.value] = probability_at_least_one_event(
                ITI_ANNUAL_EABR, "annual"
            )
            df.loc[state, Status.MAJOR_BLEEDING.value] = probability_at_least_one_event(
                ITI_ANNUAL_AJBR, "annual"
            )
            df.loc[state, Status.CRITICAL_BLEEDING.value] = (
                probability_at_least_one_event(ITI_ANNUAL_ALBR, "annual")
            )
            df.loc[state, Status.COMPLICATION.value] = ITI_COMPLICATION_PROB
            df.loc[state, Status.INHIBITOR.value] = 0.0  # Adjusted during normalization

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
    """
    # Define paths for transition matrices
    od_matrix_path = PROJECT_ROOT / "data" / "processed" / "od_transition_matrix.csv"
    pro_matrix_path = PROJECT_ROOT / "data" / "processed" / "pro_transition_matrix.csv"

    # Initialize treatments
    treatments = initialize_treatments()

    # Define states from Status enum
    states = [state.value for state in Status]

    # Initialize transition matrices for both regimes and convert to 2D NumPy arrays
    od_matrix = initialize_transition_matrix(
        od_matrix_path, Regimes.ON_DEMAND
    ).to_numpy()
    pro_matrix = initialize_transition_matrix(
        pro_matrix_path, Regimes.PROPHYLAXIS
    ).to_numpy()

    # Initialize initial state probabilities (all patients start in alive_wo_arthropathy)
    initial_state_probs = np.zeros(len(states))
    no_bleeding_idx = states.index(Status.NO_BLEEDING.value)
    initial_state_probs[no_bleeding_idx] = 1.0

    # Create MarkovChain instances for both regimes
    od_markov_chain = MarkovChain(
        states=states,
        transition_matrix=od_matrix.tolist(),
        initial_state_probs=initial_state_probs.tolist(),
        treatment=treatments[Regimes.ON_DEMAND],
        price_per_unit=HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL,
    )
    pro_markov_chain = MarkovChain(
        states=states,
        transition_matrix=pro_matrix.tolist(),
        initial_state_probs=initial_state_probs.tolist(),
        treatment=treatments[Regimes.PROPHYLAXIS],
        price_per_unit=HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL,
    )

    # Run simulation
    logger.info("Starting Markov chain simulation with dose and cost calculations")
    num_steps = NUMBER_OF_CYCLES
    od_results = od_markov_chain.simulate(num_steps)
    pro_results = pro_markov_chain.simulate(num_steps)

    # Log summary statistics
    total_dose_od = sum(od_results["weekly_doses"])
    total_cost_od = sum(od_results["weekly_costs"])
    total_dose_pro = sum(pro_results["weekly_doses"])
    total_cost_pro = sum(pro_results["weekly_costs"])
    logger.info(
        f"Total factor VIII dose over {num_steps/52:.2f} years: "
        f"On-demand = {total_dose_od:.2f} IU, Prophylaxis = {total_dose_pro:.2f} IU"
    )
    logger.info(
        f"Total cost over {num_steps/52:.2f} years: "
        f"On-demand = {total_cost_od:.2f} Rial, Prophylaxis = {total_cost_pro:.2f} Rial"
    )

    # TODO: Use FileHandler
    # Log sample weekly results for debugging
    for week in range(num_steps):
        logger.debug(
            f"Week {week}, Age {week/52:.2f} years, "
            f"OD State: {od_results['state_path'][week]}, OD Dose: {od_results['weekly_doses'][week]} IU, "
            f"OD Cost: {od_results['weekly_costs'][week]} Rial, "
            f"PRO State: {pro_results['state_path'][week]}, PRO Dose: {pro_results['weekly_doses'][week]} IU, "
            f"PRO Cost: {pro_results['weekly_costs'][week]} Rial"
        )
