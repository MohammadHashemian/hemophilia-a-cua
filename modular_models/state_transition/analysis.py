from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from modular_models.state_transition.context import StudyContext
from modular_models.state_transition.engine import StateTransitionEngine
from modular_models.state_transition.results import (
    ComparisonResult,
    OWSAResult,
    PSAResult,
)
from modular_models.state_transition.sampling import ParameterResolver
from modular_models.state_transition.types import Strategy

if TYPE_CHECKING:
    from modular_models.state_transition.trace import TraceSession


@dataclass(frozen=True, slots=True)
class ConvergenceRecord:
    n_patients: int
    incremental_cost_irr: float
    incremental_qaly: float
    relative_change_cost: float | None
    relative_change_qaly: float | None
    converged: bool


class StudyRunner:
    """High-level base-case, scenario, OWSA, PSA and convergence workflows."""

    def __init__(self, context: StudyContext) -> None:
        self.context = context
        self.resolver = ParameterResolver(context)

    def compare(
        self,
        *,
        scenario_id: str = "base_case",
        n_patients: int | None = None,
        seed: int | None = None,
        overrides: dict[str, float] | None = None,
        retain_patient_level: bool = False,
        trace: TraceSession | None = None,
    ) -> ComparisonResult:
        values, options = self.resolver.deterministic(scenario_id, overrides)
        actual_seed = int(seed if seed is not None else values["seed"])
        engine = StateTransitionEngine(
            self.context,
            values,
            options,
            scenario_id=scenario_id,
            seed=actual_seed,
        )
        prophylaxis = engine.run(
            Strategy.PROPHYLAXIS,
            n_patients=n_patients,
            retain_patient_level=retain_patient_level,
            trace=trace,
        )
        on_demand = engine.run(
            Strategy.ON_DEMAND,
            n_patients=n_patients,
            retain_patient_level=retain_patient_level,
            trace=trace,
        )
        return ComparisonResult(
            prophylaxis=prophylaxis,
            on_demand=on_demand,
            wtp_irr_per_qaly=values["wtp_irr_per_qaly"],
        )

    def run_scenarios(
        self,
        scenario_ids: Iterable[str] | None = None,
        *,
        n_patients: int | None = None,
        seed: int | None = None,
    ) -> dict[str, ComparisonResult]:
        ids = list(scenario_ids or self.context.scenarios)
        return {
            scenario_id: self.compare(
                scenario_id=scenario_id,
                n_patients=n_patients,
                seed=seed,
            )
            for scenario_id in ids
        }

    def owsa(
        self,
        parameter_ids: Iterable[str] | None = None,
        *,
        scenario_id: str = "base_case",
        n_patients: int | None = None,
        seed: int | None = None,
    ) -> tuple[OWSAResult, ...]:
        selected = (
            list(parameter_ids)
            if parameter_ids is not None
            else [
                key
                for key, parameter in self.context.parameters.items()
                if parameter.owsa is not None
            ]
        )
        records: list[OWSAResult] = []
        for parameter_id in selected:
            parameter = self.context.parameter(parameter_id)
            if parameter.owsa is None:
                raise ValueError(f"{parameter_id} has no declared OWSA range")
            low = self.compare(
                scenario_id=scenario_id,
                n_patients=n_patients,
                seed=seed,
                overrides={parameter_id: parameter.owsa.low},
            )
            high = self.compare(
                scenario_id=scenario_id,
                n_patients=n_patients,
                seed=seed,
                overrides={parameter_id: parameter.owsa.high},
            )
            records.append(
                OWSAResult(
                    parameter_id=parameter_id,
                    low=parameter.owsa.low,
                    high=parameter.owsa.high,
                    low_icer=low.icer_irr_per_qaly,
                    high_icer=high.icer_irr_per_qaly,
                    low_inmb=low.incremental_nmb_irr,
                    high_inmb=high.incremental_nmb_irr,
                )
            )
        return tuple(sorted(records, key=lambda item: item.inmb_span, reverse=True))

    def psa(
        self,
        *,
        iterations: int | None = None,
        n_patients: int,
        scenario_id: str = "base_case",
        seed: int | None = None,
        n_jobs: int = 1,
    ) -> PSAResult:
        base = self.context.base_values()
        actual_iterations = int(iterations or base["psa_iterations"])
        actual_seed = int(seed if seed is not None else base["seed"])
        sampled, options = self.resolver.probabilistic(
            actual_iterations,
            actual_seed,
            scenario_id,
        )

        def execute(index: int) -> dict[str, float | int]:
            values = {key: float(array[index]) for key, array in sampled.items()}
            iteration_seed = int(np.random.SeedSequence([actual_seed, index]).generate_state(1)[0])
            engine = StateTransitionEngine(
                self.context,
                values,
                options,
                scenario_id=scenario_id,
                seed=iteration_seed,
            )
            prophylaxis = engine.run(Strategy.PROPHYLAXIS, n_patients=n_patients)
            on_demand = engine.run(Strategy.ON_DEMAND, n_patients=n_patients)
            result = ComparisonResult(
                prophylaxis=prophylaxis,
                on_demand=on_demand,
                wtp_irr_per_qaly=values["wtp_irr_per_qaly"],
            )
            return {
                "iteration": index,
                "incremental_cost_irr": result.incremental_cost_irr,
                "incremental_qaly": result.incremental_qaly,
                "icer_irr_per_qaly": result.icer_irr_per_qaly,
                "incremental_nmb_irr": result.incremental_nmb_irr,
                "prophylaxis_alive_at_end": prophylaxis.summary["alive_at_end"],
                "on_demand_alive_at_end": on_demand.summary["alive_at_end"],
                "prophylaxis_deaths_background": prophylaxis.summary["deaths_background"],
                "on_demand_deaths_background": on_demand.summary["deaths_background"],
                "prophylaxis_deaths_ich": prophylaxis.summary["deaths_ich"],
                "on_demand_deaths_ich": on_demand.summary["deaths_ich"],
                "prophylaxis_mortality_probability": prophylaxis.summary[
                    "all_cause_mortality_probability"
                ],
                "on_demand_mortality_probability": on_demand.summary[
                    "all_cause_mortality_probability"
                ],
                "prophylaxis_post_ich_ever_count": prophylaxis.summary["post_ich_ever_count"],
                "on_demand_post_ich_ever_count": on_demand.summary["post_ich_ever_count"],
            }

        if n_jobs <= 1:
            records = tuple(execute(index) for index in range(actual_iterations))
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                records = tuple(executor.map(execute, range(actual_iterations)))
        return PSAResult(records=records)

    def convergence(
        self,
        population_sizes: Iterable[int],
        *,
        scenario_id: str = "base_case",
        seed: int | None = None,
    ) -> tuple[ConvergenceRecord, ...]:
        threshold = self.context.parameter("convergence_threshold").value
        output: list[ConvergenceRecord] = []
        previous_cost: float | None = None
        previous_qaly: float | None = None
        for population in population_sizes:
            comparison = self.compare(
                scenario_id=scenario_id,
                n_patients=int(population),
                seed=seed,
            )
            cost_change = _relative_change(comparison.incremental_cost_irr, previous_cost)
            qaly_change = _relative_change(comparison.incremental_qaly, previous_qaly)
            converged = (
                cost_change is not None
                and qaly_change is not None
                and cost_change < threshold
                and qaly_change < threshold
            )
            output.append(
                ConvergenceRecord(
                    n_patients=int(population),
                    incremental_cost_irr=comparison.incremental_cost_irr,
                    incremental_qaly=comparison.incremental_qaly,
                    relative_change_cost=cost_change,
                    relative_change_qaly=qaly_change,
                    converged=converged,
                )
            )
            previous_cost = comparison.incremental_cost_irr
            previous_qaly = comparison.incremental_qaly
        return tuple(output)


def _relative_change(current: float, previous: float | None) -> float | None:
    if previous is None:
        return None
    denominator = max(abs(previous), 1e-12)
    return abs(current - previous) / denominator
