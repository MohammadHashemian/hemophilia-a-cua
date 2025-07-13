from dataclasses import dataclass
from enum import Enum

class Regimes(Enum):
    ON_DEMAND = "on_demand"
    PROPHYLAXIS = "prophylaxis"

class Status(Enum):
    NO_BLEEDING = "alive_wo_arthropathy"
    MINOR_BLEEDING = "minor_bleeding"
    MAJOR_BLEEDING = "articular_bleeding"
    CRITICAL_BLEEDING = "surgery_or_injury_or_infection"
    CHRONIC_ARTHROPATHY = "chronic_arthropathy"
    SEVERE_ARTHROPATHY = "severe_arthropathy"
    COMPLICATION = "complication"
    INHIBITOR = "inhibitor"
    DEATH = "death"

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