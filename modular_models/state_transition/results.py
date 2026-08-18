from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modular_models.state_transition.types import Strategy


@dataclass(frozen=True, slots=True)
class SimulationResult:
    strategy: Strategy
    scenario_id: str
    seed: int
    n_patients: int
    n_cycles: int
    summary: dict[str, float | int]
    state_counts: dict[str, int]
    mortality: dict[str, Any]
    patient_data: dict[str, np.ndarray] | None = field(default=None, repr=False)

    def to_polars(self, *, patient_level: bool = False) -> Any:
        """Return a Polars frame without making Polars an engine dependency."""
        import polars as pl

        if patient_level:
            if self.patient_data is None:
                raise ValueError("Patient-level data were not retained for this run")
            return pl.DataFrame(self.patient_data)
        return pl.DataFrame(
            [{"strategy": self.strategy.value, "scenario": self.scenario_id, **self.summary}]
        )


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    prophylaxis: SimulationResult
    on_demand: SimulationResult
    wtp_irr_per_qaly: float

    def __post_init__(self) -> None:
        if self.prophylaxis.n_patients != self.on_demand.n_patients:
            raise ValueError("Paired strategies must use the same patient count")
        if self.prophylaxis.seed != self.on_demand.seed:
            raise ValueError("Paired strategies must use the same first-order seed")

    @property
    def incremental_cost_irr(self) -> float:
        return self.prophylaxis.summary["mean_cost_irr"] - self.on_demand.summary["mean_cost_irr"]

    @property
    def incremental_qaly(self) -> float:
        return self.prophylaxis.summary["mean_qaly"] - self.on_demand.summary["mean_qaly"]

    @property
    def icer_irr_per_qaly(self) -> float:
        delta_qaly = self.incremental_qaly
        return self.incremental_cost_irr / delta_qaly if delta_qaly != 0 else float("nan")

    @property
    def incremental_nmb_irr(self) -> float:
        return self.wtp_irr_per_qaly * self.incremental_qaly - self.incremental_cost_irr

    @property
    def is_prophylaxis_cost_effective(self) -> bool:
        return self.incremental_nmb_irr > 0

    def economic_summary(self) -> dict[str, float | bool | str]:
        return {
            "scenario": self.prophylaxis.scenario_id,
            "n_patients_per_strategy": self.prophylaxis.n_patients,
            "prophylaxis_cost_irr": self.prophylaxis.summary["mean_cost_irr"],
            "on_demand_cost_irr": self.on_demand.summary["mean_cost_irr"],
            "prophylaxis_qaly": self.prophylaxis.summary["mean_qaly"],
            "on_demand_qaly": self.on_demand.summary["mean_qaly"],
            "incremental_cost_irr": self.incremental_cost_irr,
            "incremental_qaly": self.incremental_qaly,
            "icer_irr_per_qaly": self.icer_irr_per_qaly,
            "wtp_irr_per_qaly": self.wtp_irr_per_qaly,
            "incremental_nmb_irr": self.incremental_nmb_irr,
            "prophylaxis_cost_effective": self.is_prophylaxis_cost_effective,
        }

    def analysis_summary(self) -> dict[str, Any]:
        """Return economic, clinical, mortality and state outputs for archiving."""
        return {
            "economic": self.economic_summary(),
            "prophylaxis": {
                "summary": self.prophylaxis.summary,
                "final_state_counts": self.prophylaxis.state_counts,
                "mortality": self.prophylaxis.mortality,
            },
            "on_demand": {
                "summary": self.on_demand.summary,
                "final_state_counts": self.on_demand.state_counts,
                "mortality": self.on_demand.mortality,
            },
        }

    def to_polars(self) -> Any:
        import polars as pl

        return pl.DataFrame([self.economic_summary()])


@dataclass(frozen=True, slots=True)
class OWSAResult:
    parameter_id: str
    low: float
    high: float
    low_icer: float
    high_icer: float
    low_inmb: float
    high_inmb: float

    @property
    def inmb_span(self) -> float:
        return abs(self.high_inmb - self.low_inmb)


@dataclass(frozen=True, slots=True)
class PSAResult:
    records: tuple[dict[str, Any], ...]

    def to_polars(self) -> Any:
        import polars as pl

        return pl.DataFrame(self.records)

    def probability_cost_effective(self, wtp_irr_per_qaly: float) -> float:
        if not self.records:
            return float("nan")
        values = [
            wtp_irr_per_qaly * float(row["incremental_qaly"]) - float(row["incremental_cost_irr"])
            > 0
            for row in self.records
        ]
        return float(np.mean(values))
