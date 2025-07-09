from enum import Enum
from markov import MarkovChain
from src.utils.logger import get_logger
from src.data.loaders import PROJECT_ROOT
import pandas as pd
import numpy as np


# SHORT CYCLES AS STATES ARE TEMPORARY
class State(Enum):
    NO_BLEEDING = "alive_wo_arthropathy"
    SEVERE_ARTHROPATHY = "alive_w_severe_arthropathy"
    CHRONIC_ARTHROPATHY = "alive_w_arthropathy"
    MINOR_BLEEDING = "extra_articular_bleeding"
    MAJOR_BLEEDING = "articular_bleeding"
    CRITICAL_BLEEDING = "life_threating_bleeding"
    COMPLICATION = "surgery_or_injury_or_infection"
    DEATH = "dies_from_bleeding"
    INHIBITER = "developing_inhibiter"


class Treatments(Enum):
    PROPHYLAXIS = "prophylaxis"
    ON_DEMAND = "on-demand"


logger = get_logger()


def run():
    # STATES GOES HERE
    states = np.array([state.value for state in State])
    treatmentOptions = [option.value for option in Treatments]

    df_transitions = pd.DataFrame(columns=states)
    df_transitions.insert(loc=0, column="states", value=states)
    df_transitions.to_csv(
        PROJECT_ROOT / "data" / "processed" / "transition_matrix.csv",
        sep=",",
        index=False,
    )

    # TRANSITION BETWEEN STATES GOES HERE
    # 2D ARRAY
    transition_matrix = np.array([[0.1, 0.9], [0.9, 0.1]])

    # Memory-less Markov Processing avoided
    # Instead Dynamic Switching Dynamic Regression or Hidden Markov Model considered

    # Discrete Time Markov chain
    # A system which is in a certain state at each steps,
    # with the state changing randomly between steps.

    # Defining hemophilia A states
    # Each states shall be a particular condition that
    # a patient can occupy during his life
