"""Horizon-aware deterministic OWSA scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.analysis.distributions import Constant
from app.analysis.psa.models import ParameterSet
from app.analysis.psa.parameters import Parameter
from app.domain.enums import Regime
from app.domain.scenario import Scenario
from app.notebook.psa.scenarios import HorizonSpec, get_horizon


@dataclass(frozen=True)
class OWSARange:
    parameter: str
    label: str
    low: float
    base: float
    high: float


_ORDERED_ARTHROPATHY_UTILITIES = (
    "healthy_utility",
    "mild_arthropathy_utility",
    "moderate_arthropathy_utility",
    "severe_arthropathy_utility",
    "advanced_arthropathy_utility",
    "end_stage_arthropathy_utility",
)


def _ordered_utility_bounds(
    parameter: str,
    low: float,
    high: float,
    owsa_parameters: ParameterSet,
) -> tuple[float, float]:
    """Keep severity utilities ordered while varying one input at a time."""
    if parameter not in _ORDERED_ARTHROPATHY_UTILITIES:
        return low, high
    index = _ORDERED_ARTHROPATHY_UTILITIES.index(parameter)
    upper = (
        1.0
        if index == 0
        else getattr(
            owsa_parameters,
            _ORDERED_ARTHROPATHY_UTILITIES[index - 1],
        ).point()
    )
    lower = (
        0.0
        if index == len(_ORDERED_ARTHROPATHY_UTILITIES) - 1
        else getattr(
            owsa_parameters,
            _ORDERED_ARTHROPATHY_UTILITIES[index + 1],
        ).point()
    )
    return max(float(low), float(lower)), min(float(high), float(upper))


def _clone_pair(
    pair: list[Scenario],
    extension: str,
    *,
    parameter: str | None = None,
    value: float | None = None,
    regime: Regime | None = None,
) -> list[Scenario]:
    result = []
    for source in pair:
        scenario = source.model_copy(deep=True)
        scenario.name += f" {extension}"
        if parameter is not None and (regime is None or scenario.regime == regime):
            assert value is not None
            scenario.overrides[parameter] = Parameter(Constant(value))
        result.append(scenario)
    return result


def _base_pair(
    horizon: HorizonSpec,
    meta_samples: dict,
) -> list[Scenario]:
    return [
        Scenario(
            name=f"{horizon.key} on-demand bayesian",
            regime=Regime.ON_DEMAND,
            overrides={
                "baseline_age": Parameter(Constant(horizon.start_age)),
                "cycles": Parameter(Constant(horizon.cycles)),
                "bleeding_rate": Parameter(
                    Constant(float(np.mean(meta_samples["on_demand"]["bayesian"])))
                ),
                "intracranial_hemorrhage_rate": Parameter(Constant(0.010)),
            },
        ),
        Scenario(
            name=f"{horizon.key} prophylaxis bayesian",
            regime=Regime.PROPHYLAXIS,
            overrides={
                "baseline_age": Parameter(Constant(horizon.start_age)),
                "cycles": Parameter(Constant(horizon.cycles)),
                "bleeding_rate": Parameter(
                    Constant(float(np.mean(meta_samples["prophylaxis"]["bayesian"])))
                ),
                "intracranial_hemorrhage_rate": Parameter(Constant(0.00033)),
            },
        ),
    ]


def build_owsa_scenarios(
    horizon: str | HorizonSpec,
    *,
    meta_samples: dict,
    owsa_parameters: ParameterSet,
    psa_parameters: ParameterSet,
    parameter_keys: list[str],
    seed: int,
    range_sample_size: int = 20_000,
) -> tuple[list[Scenario], list[OWSARange]]:
    """Build base and low/high pairs for one horizon.

    All non-varied inputs remain at deterministic OWSA point estimates. The
    low/high values are the 2.5th/97.5th percentiles of the corresponding PSA
    distribution, sampled with a stable parameter-specific random stream.
    """
    spec = get_horizon(horizon)
    base_pair = _base_pair(spec, meta_samples)
    scenarios = list(base_pair)
    ranges: list[OWSARange] = []

    for parameter in parameter_keys:
        distribution = getattr(psa_parameters, parameter)
        rng = np.random.default_rng(np.random.SeedSequence([seed, len(parameter), *map(ord, parameter)]))
        sampled = distribution.sample(range_sample_size, rng)
        low, high = np.quantile(sampled, [0.025, 0.975])
        base = getattr(owsa_parameters, parameter).point()
        low, high = _ordered_utility_bounds(
            parameter,
            float(low),
            float(high),
            owsa_parameters,
        )
        # Fixed inputs do not constitute a one-way sensitivity range and
        # would only duplicate the base scenario.
        if np.isclose(low, high) and np.isclose(low, base):
            continue
        ranges.append(
            OWSARange(
                parameter=parameter,
                label=parameter.replace("_", " ").title(),
                low=float(low),
                base=float(base),
                high=float(high),
            )
        )
        scenarios.extend(
            _clone_pair(
                base_pair,
                f"{parameter}_low",
                parameter=parameter,
                value=float(low),
            )
        )
        scenarios.extend(
            _clone_pair(
                base_pair,
                f"{parameter}_high",
                parameter=parameter,
                value=float(high),
            )
        )

    for regime, token in (
        (Regime.ON_DEMAND, "on_demand_bleeding_rate"),
        (Regime.PROPHYLAXIS, "prophylaxis_bleeding_rate"),
    ):
        sample_key = "on_demand" if regime == Regime.ON_DEMAND else "prophylaxis"
        values = np.asarray(meta_samples[sample_key]["bayesian"], dtype=float)
        low, high = np.quantile(values, [0.025, 0.975])
        base = float(values.mean())
        ranges.append(
            OWSARange(
                parameter=token,
                label=token.replace("_", " ").title(),
                low=float(low),
                base=base,
                high=float(high),
            )
        )
        scenarios.extend(
            _clone_pair(
                base_pair,
                f"{token}_low",
                parameter="bleeding_rate",
                value=float(low),
                regime=regime,
            )
        )
        scenarios.extend(
            _clone_pair(
                base_pair,
                f"{token}_high",
                parameter="bleeding_rate",
                value=float(high),
                regime=regime,
            )
        )

    ranges.append(
        OWSARange(
            parameter="weight_factor",
            label="Body Weight Factor",
            low=0.9,
            base=1.0,
            high=1.1,
        )
    )
    scenarios.extend(
        _clone_pair(
            base_pair,
            "weight_factor_low",
            parameter="weight_factor",
            value=0.9,
        )
    )
    scenarios.extend(
        _clone_pair(
            base_pair,
            "weight_factor_high",
            parameter="weight_factor",
            value=1.1,
        )
    )
    return scenarios, ranges
