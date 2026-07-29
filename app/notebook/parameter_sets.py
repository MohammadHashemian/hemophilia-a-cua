import pickle
from pathlib import Path

import numpy as np

from app.analysis.distributions import (
    BetaFromMeanSD,
    Constant,
    GammaFromMeanCV,
    TriangularDist,
)
from app.analysis.psa.models import ParameterSet
from app.analysis.psa.parameters import Parameter
from app.persistence.context import ModelContext


class HemophiliaParamRepo:
    def __init__(
        self, root: Path, cache_path: Path, context: ModelContext | None = None
    ):
        self.root = root
        self.cache_path = cache_path
        self.context = context if context is not None else ModelContext.load()
        self.ows_params_keys = [
            "joint_bleeding_fraction",
            "life_threatening_bleeding_fraction",
            "ltb_case_fatality",
            "healthy_utility",
            "mild_arthropathy_utility",
            "moderate_arthropathy_utility",
            "severe_arthropathy_utility",
            "spontaneous_bleeding_utility",
            "joint_bleeding_utility",
            "life_threatening_bleeding_utility",
            "per_unit_price",
            "prophylaxis_background_factor_consumption_per_kg",
            "factor_consumption_per_spontaneous_bleeding_per_kg",
            "factor_consumption_per_joint_bleeding_per_kg",
            "factor_consumption_per_life_threatening_bleeding_per_kg",
        ]

    def load_psa_parameters(self) -> ParameterSet:
        utils = self.context.utilities.state_utilities
        ltb_rate = self.context.clinical.epidemiology.event_rates.ltb_rate
        ltb_fatality = self.context.clinical.epidemiology.ltb_case_fatality
        ltb_fraction = self.context.clinical.epidemiology.event_fractions.ltb_fraction
        with open(self.root / self.cache_path, "rb") as f:
            samples = pickle.load(f)
            params = ParameterSet(
                cycles=Parameter(
                    distribution=Constant(value=10 * 52)
                ),  # EARLY SCENARIO
                baseline_age=Parameter(distribution=Constant(value=2)),
                weight_factor=Parameter(distribution=Constant(value=1.0)),
                benefits_discount_rate=Parameter(distribution=Constant(value=0)),
                costs_discount_rate=Parameter(distribution=Constant(value=0)),
                # Clinical
                bleeding_rate=Parameter(
                    distribution=Constant(value=0),
                    cache=samples["on_demand"]["bayesian"],
                ),  # Cache Data from meta_analysis
                joint_bleeding_fraction=Parameter(
                    distribution=BetaFromMeanSD(mean=0.75, sd=0.0255)
                ),
                life_threatening_bleeding_fraction=Parameter(
                    distribution=TriangularDist(
                        left=0.01, mode=ltb_fraction, right=0.05
                    )
                ),
                # Absolute annual LTB incidence is retained for the
                # evidence-based structural sensitivity scenario. The
                # thesis base case uses the triangular LTB fraction above.
                life_threatening_bleeding_rate=Parameter(
                    distribution=TriangularDist(
                        left=0.0049,
                        mode=ltb_rate.on_demand,
                        right=0.0111,
                    )
                ),
                # Conditional case-fatality per LTB episode
                # (Zwagemaker et al. 2021: 0.8/2.3 per 1000 PY ~= 0.35)
                ltb_case_fatality=Parameter(
                    distribution=BetaFromMeanSD(mean=ltb_fatality, cv=0.20)
                ),
                # Benefits (means from data/utilities.json)
                healthy_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=utils.healthy, cv=0.05)
                ),
                mild_arthropathy_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=utils.mild_arthropathy, cv=0.05)
                ),
                moderate_arthropathy_utility=Parameter(
                    distribution=BetaFromMeanSD(
                        mean=utils.moderate_arthropathy, cv=0.05
                    )
                ),
                severe_arthropathy_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=utils.severe_arthropathy, cv=0.05)
                ),
                spontaneous_bleeding_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=utils.bleeding, cv=0.05)
                ),
                joint_bleeding_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=utils.hemarthrosis, cv=0.05)
                ),
                life_threatening_bleeding_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=utils.lt_bleeding, cv=0.05)
                ),
                death_utility=Parameter(distribution=Constant(value=utils.death)),
                # Costs
                per_unit_price=Parameter(
                    distribution=GammaFromMeanCV(mean=58_000, cv=0.05)
                ),
                prophylaxis_background_factor_consumption_per_kg=Parameter(
                    distribution=GammaFromMeanCV(mean=75, cv=0.15)
                ),
                factor_consumption_per_spontaneous_bleeding_per_kg=Parameter(
                    distribution=GammaFromMeanCV(mean=120, cv=0.15)
                ),
                factor_consumption_per_joint_bleeding_per_kg=Parameter(
                    distribution=GammaFromMeanCV(mean=60, cv=0.15)
                ),
                factor_consumption_per_life_threatening_bleeding_per_kg=Parameter(
                    distribution=GammaFromMeanCV(mean=550, cv=0.15)
                ),
            )
        return params

    def load_owsa_parameters(self) -> ParameterSet:
        utils = self.context.utilities.state_utilities
        ltb_rate = self.context.clinical.epidemiology.event_rates.ltb_rate
        ltb_fatality = self.context.clinical.epidemiology.ltb_case_fatality
        ltb_fraction = self.context.clinical.epidemiology.event_fractions.ltb_fraction

        with open(self.root / self.cache_path, "rb") as f:
            samples = pickle.load(f)
            # on_demand base scenario
            params = ParameterSet(
                cycles=Parameter(Constant(value=10 * 52)),  # EARLY SCENARIO
                baseline_age=Parameter(Constant(value=2)),
                weight_factor=Parameter(Constant(value=1.0)),
                benefits_discount_rate=Parameter(Constant(value=0)),
                costs_discount_rate=Parameter(Constant(value=0)),
                # Clinical
                bleeding_rate=Parameter(
                    Constant(value=np.mean(samples["on_demand"]["bayesian"]))
                ),
                joint_bleeding_fraction=Parameter(Constant(value=0.75)),  # MEAN
                life_threatening_bleeding_fraction=Parameter(
                    Constant(value=ltb_fraction)
                ),  # MODE
                life_threatening_bleeding_rate=Parameter(
                    Constant(value=ltb_rate.on_demand)
                ),  # BASE
                ltb_case_fatality=Parameter(Constant(value=ltb_fatality)),  # BASE
                # Benefits (means from data/utilities.json)
                healthy_utility=Parameter(Constant(value=utils.healthy)),  # MEAN
                mild_arthropathy_utility=Parameter(
                    Constant(value=utils.mild_arthropathy)
                ),  # MEAN
                moderate_arthropathy_utility=Parameter(
                    Constant(value=utils.moderate_arthropathy)
                ),  # MEAN
                severe_arthropathy_utility=Parameter(
                    Constant(value=utils.severe_arthropathy)
                ),  # MEAN
                spontaneous_bleeding_utility=Parameter(
                    Constant(value=utils.bleeding)
                ),  # MEAN
                joint_bleeding_utility=Parameter(
                    Constant(value=utils.hemarthrosis)
                ),  # MEAN
                life_threatening_bleeding_utility=Parameter(
                    Constant(value=utils.lt_bleeding)
                ),  # MEAN
                death_utility=Parameter(Constant(value=utils.death)),
                # Costs
                per_unit_price=Parameter(distribution=Constant(value=58_000)),
                prophylaxis_background_factor_consumption_per_kg=Parameter(
                    Constant(value=75)
                ),
                factor_consumption_per_spontaneous_bleeding_per_kg=Parameter(
                    Constant(120)
                ),
                factor_consumption_per_joint_bleeding_per_kg=Parameter(
                    Constant(value=60)
                ),
                factor_consumption_per_life_threatening_bleeding_per_kg=Parameter(
                    Constant(value=550)
                ),
            )
        return params
