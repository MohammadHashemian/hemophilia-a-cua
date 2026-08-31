from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from pydantic import ValidationError

from modular_models.state_transition.schema import (
    ModelFile,
    ParameterSpec,
    ReferenceFile,
    ReferenceRecord,
    ScenarioDefinition,
    ScenarioFile,
)


class ContextValidationError(ValueError):
    """Raised when typed files are valid individually but inconsistent together."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required model data file was not found: {path}")
    try:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ContextValidationError(f"Invalid JSON in {path}: {exc}") from exc


class StudyContext:
    """Immutable, fully validated boundary between JSON data and model code."""

    __slots__ = ("data_dir", "model", "parameters", "references", "scenarios")

    def __init__(
        self,
        *,
        data_dir: Path,
        model: ModelFile,
        references: Mapping[str, ReferenceRecord],
        scenarios: Mapping[str, ScenarioDefinition],
    ) -> None:
        self.data_dir = data_dir
        self.model = model
        self.parameters = MappingProxyType(dict(model.parameters))
        self.references = MappingProxyType(dict(references))
        self.scenarios = MappingProxyType(dict(scenarios))

    @classmethod
    def load(cls, data_dir: str | Path | None = None) -> StudyContext:
        """Load predefined model data from JSON files and validate it."""
        root = (
            Path(data_dir)
            if data_dir is not None
            else Path(__file__).parents[2] / "app/data/state_transition"
        )
        root = root.resolve()
        try:
            model = ModelFile.model_validate(_read_json(root / "model.json"))
            ref_file = ReferenceFile.model_validate(
                _read_json(root / "references.json")
            )
            scenario_file = ScenarioFile.model_validate(
                _read_json(root / "scenarios.json")
            )
        except ValidationError as exc:
            raise ContextValidationError(str(exc)) from exc

        references = {item.id: item for item in ref_file.references}
        scenarios = {item.id: item for item in scenario_file.scenarios}
        cls._cross_validate(model, references, scenarios, scenario_file.base_case)
        return cls(
            data_dir=root,
            model=model,
            references=references,
            scenarios=scenarios,
        )

    @staticmethod
    def _cross_validate(
        model: ModelFile,
        references: Mapping[str, ReferenceRecord],
        scenarios: Mapping[str, ScenarioDefinition],
        base_case: str,
    ) -> None:
        referenced: set[str] = set()
        for parameter in model.parameters.values():
            referenced.update(parameter.references)
        for scenario in scenarios.values():
            referenced.update(scenario.references)
            unknown_parameters = set(scenario.parameter_overrides).difference(
                model.parameters
            )
            if unknown_parameters:
                raise ContextValidationError(
                    f"Scenario {scenario.id!r} overrides unknown parameters: "
                    f"{sorted(unknown_parameters)}"
                )
        missing_references = referenced.difference(references)
        if missing_references:
            raise ContextValidationError(
                f"Unknown reference ids used by model data: {sorted(missing_references)}"
            )
        if base_case not in scenarios:
            raise ContextValidationError(f"Unknown base-case scenario: {base_case}")

        required = {
            "entry_age_years",
            "exit_age_years",
            "cycles_per_year",
            "days_per_cycle",
            "default_patients",
            "convergence_threshold",
            "psa_iterations",
            "seed",
            "utility_integration_step_days",
            "abr_prophylaxis",
            "abr_on_demand",
            "ajbr_prophylaxis",
            "ajbr_on_demand",
            "joint_bleed_fraction",
            "ich_rate_prophylaxis",
            "ich_rate_on_demand",
            "ich_fraction",
            "non_ich_major_fraction",
            "ich_case_fatality",
            "post_ich_sequela_probability",
            "joint_bleeds_per_pettersson_point",
            "pettersson_max",
            "weight_age_1",
            "factor_price_irr_per_iu",
            "prophylaxis_iu_per_kg_week",
            "joint_bleed_iu_per_kg",
            "non_major_non_joint_iu_per_kg",
            "non_ich_major_iu_per_kg",
            "ich_iu_per_kg",
            "utility_anchor",
            "utility_mild",
            "utility_moderate",
            "utility_severe",
            "minor_bleed_decrement",
            "minor_bleed_duration_days",
            "non_ich_major_utility_cap",
            "non_ich_major_duration_days",
            "non_ich_major_treatment_duration_days",
            "ich_acute_utility_cap",
            "ich_duration_days",
            "post_ich_utility_cap",
            "background_mortality_age_1_4",
            "background_mortality_age_5_9",
            "background_mortality_age_10_lt12",
            "cost_discount_rate",
            "qaly_discount_rate",
            "wtp_irr_per_qaly",
        }
        missing = required.difference(model.parameters)
        if missing:
            raise ContextValidationError(
                f"Required parameters are missing: {sorted(missing)}"
            )

    def parameter(self, parameter_id: str) -> ParameterSpec:
        try:
            return self.parameters[parameter_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model parameter: {parameter_id}") from exc

    def scenario(self, scenario_id: str) -> ScenarioDefinition:
        try:
            return self.scenarios[scenario_id]
        except KeyError as exc:
            raise KeyError(f"Unknown scenario: {scenario_id}") from exc

    def base_values(self) -> dict[str, float]:
        return {key: parameter.value for key, parameter in self.parameters.items()}
