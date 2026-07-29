"""Authoritative horizon-specific PSA scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass

from app.analysis.distributions import Constant, TriangularDist
from app.analysis.psa.parameters import Parameter
from app.domain.enums import Regime
from app.domain.scenario import Scenario
from app.notebook.scenario_helpers import define_scenario_extension
from app.persistence.context import ModelContext


@dataclass(frozen=True)
class HorizonSpec:
    key: str
    directory: str
    label: str
    start_age: int
    end_age: int

    @property
    def cycles(self) -> int:
        return (self.end_age - self.start_age) * 52


CHILDHOOD = HorizonSpec(
    key="childhood",
    directory="childhood_age_2_12",
    label="Childhood horizon (ages 2–12)",
    start_age=2,
    end_age=12,
)
LIFETIME = HorizonSpec(
    key="lifetime",
    directory="lifetime_age_2_100",
    label="Lifetime horizon (ages 2–100)",
    start_age=2,
    end_age=100,
)

_HORIZONS = {
    CHILDHOOD.key: CHILDHOOD,
    CHILDHOOD.directory: CHILDHOOD,
    "early": CHILDHOOD,  # legacy scenario/cache name
    LIFETIME.key: LIFETIME,
    LIFETIME.directory: LIFETIME,
}


def get_horizon(value: str | HorizonSpec) -> HorizonSpec:
    if isinstance(value, HorizonSpec):
        return value
    try:
        return _HORIZONS[value]
    except KeyError as exc:
        raise ValueError(
            f"Unknown PSA horizon {value!r}; expected childhood or lifetime"
        ) from exc


def _base_pair(
    horizon: HorizonSpec,
    sampling_method: str,
    meta_samples: dict,
) -> list[Scenario]:
    return [
        Scenario(
            name=f"{horizon.key} on-demand {sampling_method}",
            regime=Regime.ON_DEMAND,
            overrides={
                "cycles": Parameter(distribution=Constant(value=horizon.cycles)),
                "bleeding_rate": Parameter(
                    distribution=Constant(value=0),
                    cache=meta_samples["on_demand"][sampling_method],
                ),
            },
        ),
        Scenario(
            name=f"{horizon.key} prophylaxis {sampling_method}",
            regime=Regime.PROPHYLAXIS,
            overrides={
                "cycles": Parameter(distribution=Constant(value=horizon.cycles)),
                "bleeding_rate": Parameter(
                    distribution=Constant(value=0),
                    cache=meta_samples["prophylaxis"][sampling_method],
                ),
            },
        ),
    ]


def build_psa_scenarios(
    horizon: str | HorizonSpec,
    *,
    meta_samples: dict,
    context: ModelContext,
) -> list[Scenario]:
    """Build the 16 PSA scenarios belonging to one time horizon."""
    spec = get_horizon(horizon)
    ltb_absolute = TriangularDist(left=0.0049, mode=0.0074, right=0.0111)

    bayesian = _base_pair(spec, "bayesian", meta_samples)
    dirichlet = _base_pair(spec, "dirichlet", meta_samples)
    base = bayesian + dirichlet
    scenarios = list(base)

    scenarios.extend(
        define_scenario_extension(
            scenarios=base,
            extensions={
                "ltb_absolute": {
                    "life_threatening_bleeding_rate": ltb_absolute,
                }
            },
        )
    )
    scenarios.extend(
        define_scenario_extension(
            scenarios=bayesian,
            extensions={
                "is_discounting": {
                    "benefits_discount_rate": Constant(
                        value=context.simulation.discounting.utility_rate_annual
                    ),
                    "costs_discount_rate": Constant(
                        value=context.simulation.discounting.cost_rate_annual
                    ),
                }
            },
        )
    )
    for reduction in (10, 20, 30):
        scenarios.extend(
            define_scenario_extension(
                scenarios=bayesian,
                extensions={
                    f"weight_reduction_{reduction}": {
                        "weight_factor": Constant(value=1 - reduction / 100),
                    }
                },
            )
        )

    return scenarios
