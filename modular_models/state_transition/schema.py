from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReferenceRecord(FrozenModel):
    id: str
    citation: str
    evidence: str
    doi_or_url: str | None = None
    limitation: str | None = None


class OWSARange(FrozenModel):
    low: float
    high: float

    @model_validator(mode="after")
    def ordered(self) -> OWSARange:
        if self.low > self.high:
            raise ValueError("OWSA low must not exceed high")
        return self


class DistributionSpec(FrozenModel):
    distribution: Literal["fixed", "uniform", "beta", "gamma", "triangular", "beta_pert"]
    parameters: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters(self) -> DistributionSpec:
        required = {
            "fixed": set(),
            "uniform": {"low", "high"},
            "beta": {"alpha", "beta"},
            "gamma": {"shape", "scale"},
            "triangular": {"low", "mode", "high"},
            "beta_pert": {"minimum", "mode", "maximum"},
        }[self.distribution]
        missing = required.difference(self.parameters)
        if missing:
            raise ValueError(f"{self.distribution} distribution is missing {sorted(missing)}")

        p = self.parameters
        if self.distribution == "uniform" and p["low"] > p["high"]:
            raise ValueError("uniform low must not exceed high")
        if self.distribution in {"beta", "gamma"} and any(value <= 0 for value in p.values()):
            raise ValueError(f"{self.distribution} parameters must be positive")
        if self.distribution == "triangular" and not (p["low"] <= p["mode"] <= p["high"]):
            raise ValueError("triangular parameters must satisfy low <= mode <= high")
        if self.distribution == "beta_pert" and not (p["minimum"] <= p["mode"] <= p["maximum"]):
            raise ValueError("beta-PERT parameters must satisfy minimum <= mode <= maximum")
        return self


class ParameterSpec(FrozenModel):
    id: str
    description: str
    value: float
    unit: str
    references: tuple[str, ...] = ()
    assumption: str | None = None
    owsa: OWSARange | None = None
    psa: DistributionSpec | None = None

    @model_validator(mode="after")
    def provenance_and_range(self) -> ParameterSpec:
        if not self.references and not self.assumption:
            raise ValueError(f"Parameter {self.id!r} needs a reference or an explicit assumption")
        if self.owsa is not None and not self.owsa.low <= self.value <= self.owsa.high:
            raise ValueError(f"Base value for {self.id!r} must lie inside its OWSA range")
        return self


class StudyMetadata(FrozenModel):
    study_id: str
    title: str
    model_type: Literal["individual_level_state_transition_microsimulation"]
    perspective: Literal["payer"]
    currency: Literal["IRR"]
    specification_version: str
    python_version: str


class ModelSettings(FrozenModel):
    entry_age_years: str
    exit_age_years: str
    cycles_per_year: str
    days_per_cycle: str
    default_patients: str
    convergence_threshold: str
    psa_iterations: str
    seed: str
    utility_integration_step_days: str
    chronic_states: tuple[str, ...]
    acute_events: tuple[str, ...]


class ModelFile(FrozenModel):
    metadata: StudyMetadata
    settings: ModelSettings
    parameters: dict[str, ParameterSpec]
    formulas: dict[str, str]
    validation_rules: tuple[str, ...]

    @model_validator(mode="after")
    def ids_match_keys(self) -> ModelFile:
        mismatches = [key for key, value in self.parameters.items() if key != value.id]
        if mismatches:
            raise ValueError(f"Parameter ids do not match JSON keys: {mismatches}")
        return self


ScenarioValue = Annotated[str | float | int | bool, Field(union_mode="left_to_right")]


class ScenarioDefinition(FrozenModel):
    id: str
    label: str
    description: str
    parameter_overrides: dict[str, float] = Field(default_factory=dict)
    options: dict[str, ScenarioValue] = Field(default_factory=dict)
    references: tuple[str, ...] = ()
    assumption: str | None = None


class ScenarioFile(FrozenModel):
    base_case: str
    scenarios: tuple[ScenarioDefinition, ...]

    @model_validator(mode="after")
    def unique_scenarios(self) -> ScenarioFile:
        ids = [item.id for item in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("Scenario ids must be unique")
        if self.base_case not in ids:
            raise ValueError("base_case must refer to a declared scenario")
        return self


class ReferenceFile(FrozenModel):
    references: tuple[ReferenceRecord, ...]

    @model_validator(mode="after")
    def unique_references(self) -> ReferenceFile:
        ids = [item.id for item in self.references]
        if len(ids) != len(set(ids)):
            raise ValueError("Reference ids must be unique")
        return self


JsonObject = dict[str, Any]
