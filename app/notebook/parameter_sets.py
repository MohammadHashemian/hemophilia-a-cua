import pickle
from pathlib import Path

import numpy as np

from app.analysis.distributions import (
    BetaDist,
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
            "gi_neck_bleeding_fraction",
            "iliopsoas_bleeding_fraction",
            "ich_case_fatality",
            "non_ich_case_fatality",
            "healthy_utility",
            "mild_arthropathy_utility",
            "moderate_arthropathy_utility",
            "severe_arthropathy_utility",
            "spontaneous_bleeding_utility",
            "joint_bleeding_utility",
            "intracranial_hemorrhage_utility",
            "non_ich_major_bleeding_utility",
            "per_unit_price",
            "prophylaxis_background_factor_consumption_per_kg",
            "factor_consumption_per_spontaneous_bleeding_per_kg",
            "factor_consumption_per_joint_bleeding_per_kg",
            "factor_consumption_per_intracranial_hemorrhage_per_kg",
            "factor_consumption_per_non_ich_major_bleeding_per_kg",
        ]

    def load_psa_parameters(self) -> ParameterSet:
        utils = self.context.utilities.state_utilities
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
                gi_neck_bleeding_fraction=Parameter(
                    distribution=BetaDist(alpha=95.5, beta=20_200.5)
                ),
                iliopsoas_bleeding_fraction=Parameter(
                    distribution=BetaDist(alpha=9.5, beta=3_235.5)
                ),
                intracranial_hemorrhage_rate=Parameter(
                    distribution=TriangularDist(
                        left=0.005, mode=0.010, right=0.017
                    )
                ),
                ich_case_fatality=Parameter(
                    distribution=BetaDist(alpha=2.5, beta=22.5)
                ),
                non_ich_case_fatality=Parameter(distribution=Constant(value=0.0)),
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
                intracranial_hemorrhage_utility=Parameter(
                    distribution=BetaFromMeanSD(
                        mean=utils.intracranial_hemorrhage, cv=0.05
                    )
                ),
                non_ich_major_bleeding_utility=Parameter(
                    distribution=BetaFromMeanSD(
                        mean=utils.non_ich_major_bleeding, cv=0.05
                    )
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
                factor_consumption_per_intracranial_hemorrhage_per_kg=Parameter(
                    distribution=GammaFromMeanCV(mean=550, cv=0.15)
                ),
                factor_consumption_per_non_ich_major_bleeding_per_kg=Parameter(
                    distribution=GammaFromMeanCV(mean=550, cv=0.15)
                ),
            )
        return params

    def load_owsa_parameters(self) -> ParameterSet:
        utils = self.context.utilities.state_utilities

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
                gi_neck_bleeding_fraction=Parameter(
                    Constant(value=95 / 20_295)
                ),
                iliopsoas_bleeding_fraction=Parameter(
                    Constant(value=9 / 3_244)
                ),
                intracranial_hemorrhage_rate=Parameter(Constant(value=0.010)),
                ich_case_fatality=Parameter(Constant(value=0.10)),
                non_ich_case_fatality=Parameter(Constant(value=0.0)),
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
                intracranial_hemorrhage_utility=Parameter(
                    Constant(value=utils.intracranial_hemorrhage)
                ),  # MEAN
                non_ich_major_bleeding_utility=Parameter(
                    Constant(value=utils.non_ich_major_bleeding)
                ),
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
                factor_consumption_per_intracranial_hemorrhage_per_kg=Parameter(
                    Constant(value=550)
                ),
                factor_consumption_per_non_ich_major_bleeding_per_kg=Parameter(
                    Constant(value=550)
                ),
            )
        return params
