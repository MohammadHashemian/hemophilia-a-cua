from typing import Literal

from pydantic import Field, model_validator

from app.persistence.schemas.metadata import InputMetadata


class UtilityUncertainty(InputMetadata):
    """PSA distribution declared alongside the utility evidence."""

    distribution: Literal["beta_from_mean_sd", "triangular", "constant"]
    sd: float | None = Field(default=None, gt=0)
    low: float | None = Field(default=None, ge=0, le=1)
    high: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_distribution_parameters(self):
        if self.distribution == "beta_from_mean_sd" and self.sd is None:
            raise ValueError("beta_from_mean_sd requires sd")
        if self.distribution == "triangular":
            if self.low is None or self.high is None:
                raise ValueError("triangular requires low and high")
            if self.low >= self.high:
                raise ValueError("triangular low must be below high")
        return self


class UtilityValue(InputMetadata):
    mean: float = Field(ge=0, le=1)
    uncertainty: UtilityUncertainty

    @model_validator(mode="after")
    def validate_range_contains_mean(self):
        uncertainty = self.uncertainty
        if uncertainty.distribution == "triangular":
            assert uncertainty.low is not None and uncertainty.high is not None
            if not uncertainty.low <= self.mean <= uncertainty.high:
                raise ValueError("triangular range must contain the utility mean/mode")
        return self


class StateUtilities(InputMetadata):
    # Chronic joint-health utilities
    healthy: UtilityValue
    mild_arthropathy: UtilityValue
    moderate_arthropathy: UtilityValue
    severe_arthropathy: UtilityValue
    advanced_arthropathy: UtilityValue
    end_stage_arthropathy: UtilityValue

    # Acute event utilities
    bleeding: UtilityValue
    hemarthrosis: UtilityValue
    intracranial_hemorrhage: UtilityValue
    non_ich_major_bleeding: UtilityValue
    death: UtilityValue


class PetterssonUtilityThresholds(InputMetadata):
    early_arthropathy: int
    moderate_arthropathy: int
    severe_arthropathy: int
    advanced_arthropathy: int
    end_stage_arthropathy: int


class UtilityFile(InputMetadata):
    pettersson_utility_thresholds: PetterssonUtilityThresholds
    state_utilities: StateUtilities
