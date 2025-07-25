from dataclasses import dataclass
from model_dep.constants import HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL
from enum import Enum


class Regimes(Enum):
    ON_DEMAND = "on_demand"
    PROPHYLAXIS = "prophylaxis"


class BaseStates(Enum):
    NO_BLEEDING = "alive_wo_arthropathy"
    SEVERE_ARTHROPATHY = "severe_arthropathy"
    CHRONIC_ARTHROPATHY = "chronic_arthropathy"


class States(Enum):
    MINOR_BLEEDING = "minor_bleeding"
    MAJOR_BLEEDING = "articular_bleeding"
    LT_BLEEDING = "life_treating_bleeding"
    SURGERY = "surgery"
    REPLACEMENT = "joint_replacement"
    INHIBITOR = "inhibitor"
    DEATH = "death"


EARLY_STATES = [
    BaseStates.NO_BLEEDING,
    States.MINOR_BLEEDING,
    States.MAJOR_BLEEDING,
    States.LT_BLEEDING,
    States.SURGERY,
    States.INHIBITOR,
    States.DEATH,
    BaseStates.CHRONIC_ARTHROPATHY,
]

INTERMEDIATE_STATES = [
    BaseStates.CHRONIC_ARTHROPATHY,
    States.MINOR_BLEEDING,
    States.MAJOR_BLEEDING,
    States.LT_BLEEDING,
    States.SURGERY,
    States.INHIBITOR,
    States.DEATH,
    BaseStates.SEVERE_ARTHROPATHY,
]

END_STATES = [
    BaseStates.SEVERE_ARTHROPATHY,
    States.MINOR_BLEEDING,
    States.MAJOR_BLEEDING,
    States.LT_BLEEDING,
    States.SURGERY,
    States.INHIBITOR,
    States.REPLACEMENT,
    States.DEATH,
]


class BaseModel:
    """
    Args:
        stage: Disease progression, [early, intermediate or end]
        states: List of disease stage available states as Enum
        states_value: List of disease stage as state.value returned
    """

    def __init__(
        self,
        stage: str,
        states: list[Enum],
        transition_matrix: list[float] | None = None,
    ) -> None:
        self.stage = stage
        self.states = states
        self.states_value = [state.value for state in states]
        self.transition_matrix = transition_matrix

    def update(self, transition_matrix: list[float]):
        """
        Updates transition matrix with new values
        """
        # TODO: Validation here
        self.transition_matrix = transition_matrix
        return self


EARLY_MODEL = BaseModel("early", EARLY_STATES)
INTERMEDIATE_MODEL = BaseModel("intermediate", INTERMEDIATE_STATES)
END_MODEL = BaseModel("end", END_STATES)


@dataclass
class Treatment:
    """Defines a treatment regime with dosing and bleeding rates."""

    name: str
    dose_joint: float  # Dose per injection for joints (IU/kg)
    dose_muscle: float  # Dose per injection for muscles (IU/kg)
    dose_mucous: float  # Dose per injection for mucous membranes (IU/kg)
    dose_intracranial: float  # Dose per injection for intracranial (IU/kg)
    dose_neck_throat: float  # Dose per injection for neck/throat (IU/kg)
    dose_gastro: float  # Dose per injection for gastrointestinal (IU/kg)
    duration_joint: float  # Treatment duration for joints (days)
    duration_muscle: float  # Treatment duration for muscles (days)
    duration_mucous: float  # Treatment duration for mucous membranes (days)
    duration_intracranial: float  # Treatment duration for intracranial (days)
    duration_neck_throat: float  # Treatment duration for neck/throat (days)
    duration_gastro: float  # Treatment duration for gastrointestinal (days)
    avg_dose_per_bleed: float  # Weighted average dose per bleed (IU/kg)
    abr: float  # Annual bleeding rate
    ajbr: float  # Annual joint bleeding rate
    albr: float  # Annual life-threatening bleeding rate
    eabr: float  # Annual extra-articular bleeding rate
    price_per_unit: float = HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL
