from app.persistence.schemas.metadata import InputMetadata


class StateUtilities(InputMetadata):
    healthy: float
    mild_arthropathy: float
    moderate_arthropathy: float
    severe_arthropathy: float
    bleeding: float
    hemarthrosis: float
    intracranial_hemorrhage: float
    non_ich_major_bleeding: float
    death: float


class EventDisutilities(InputMetadata):
    severe_arthropathy_bleeding: float


class UtilityFile(InputMetadata):
    state_utilities: StateUtilities
    event_disutilities: EventDisutilities
