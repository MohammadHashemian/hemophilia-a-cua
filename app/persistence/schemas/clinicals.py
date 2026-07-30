from app.persistence.schemas.metadata import InputMetadata


class EventFractions(InputMetadata):
    ajbr_fraction: float
    non_ich_major_bleeding_fraction: float


class ICHRate(InputMetadata):
    on_demand: float
    prophylaxis: float


class EventRates(InputMetadata):
    intracranial_hemorrhage_rate: ICHRate


class Epidemiology(InputMetadata):
    event_fractions: EventFractions
    event_rates: EventRates
    ich_case_fatality: float = 0.10
    ich_case_fatality_description: str | None = None
    ich_case_fatality_reference: str | list[str] | None = None
    non_ich_case_fatality: float = 0.0
    non_ich_case_fatality_description: str | None = None
    non_ich_case_fatality_reference: str | list[str] | None = None


class UtilityDecrements(InputMetadata):
    on_demand: float
    prophylaxis: float


class Utilities(InputMetadata):
    decrements: UtilityDecrements


class StudyEstimate(InputMetadata):
    mean: float
    sd: float
    size: float
    source: str | None = None
    doi: str | None = None


class ABREvidence(InputMetadata):
    on_demand: list[StudyEstimate]
    prophylaxis: list[StudyEstimate]


class Evidence(InputMetadata):
    abr: ABREvidence


class PetterssonThresholds(InputMetadata):
    mild: int
    moderate: int
    max: int


class PetterssonScore(InputMetadata):
    conversion_factor: float
    thresholds: PetterssonThresholds


class ClinicalScoring(InputMetadata):
    pettersson_score: PetterssonScore


class Dosing(InputMetadata):
    ir_prophylaxis_weekly_dose_ui: float
    standard_prophylaxis_weekly_dose_ui: float
    bleeding_dose_ui: float
    joint_bleeding_dose_ui: float
    intracranial_hemorrhage_dose_ui: float
    non_ich_major_bleeding_dose_ui: float


class Treatment(InputMetadata):
    dosing: Dosing


class ClinicalFile(InputMetadata):
    epidemiology: Epidemiology
    utilities: Utilities
    clinical_scoring: ClinicalScoring
    treatment: Treatment
    evidence: Evidence
