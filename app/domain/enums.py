from enum import StrEnum


class Regime(StrEnum):
    ON_DEMAND = "on_demand"
    PROPHYLAXIS = "prophylaxis"


class HealthStates(StrEnum):
    # ``healthy`` remains the serialized value so existing cached results
    # stay readable. Conceptually this is the event-free No Bleeding state;
    # arthropathy severity is tracked separately.
    NO_BLEEDING = "healthy"
    HEALTHY = "healthy"  # Backward-compatible alias.
    BLEEDING = "bleeding"
    HEMARTHROSIS = "hemarthrosis"
    INTRACRANIAL_HEMORRHAGE = "intracranial_hemorrhage"
    NON_ICH_MAJOR_BLEEDING = "non_ich_major_bleeding"
    DEATH = "death"


class UtilityStates(StrEnum):
    HEALTHY = "healthy"
    MILD_ARTHROPATHY = "mild_arthropathy"
    MODERATE_ARTHROPATHY = "moderate_arthropathy"
    SEVERE_ARTHROPATHY = "severe_arthropathy"
    BLEEDING = "bleeding"
    HEMARTHROSIS = "hemarthrosis"
    INTRACRANIAL_HEMORRHAGE = "intracranial_hemorrhage"
    NON_ICH_MAJOR_BLEEDING = "non_ich_major_bleeding"
    DEATH = "death"


class ArthropathySeverity(StrEnum):
    HEALTHY = "healthy"
    MILD = "mild_arthropathy"
    MODERATE = "moderate_arthropathy"
    SEVERE = "severe_arthropathy"
