from typing import Literal

from app.persistence.schemas.metadata import InputMetadata

MortalitySource = Literal["iran", "poland", "default"]


class Environment(InputMetadata):
    mode: Literal["development", "production"]
    seed: int


class Discounting(InputMetadata):
    enable: bool
    cost_rate_annual: float
    utility_rate_annual: float


class PSA(InputMetadata):
    development: int
    production: int

    def sample_size(self, mode: str) -> int:
        return getattr(self, mode)


class Mortality(InputMetadata):
    """Selects which mortality table the model loads at startup.

    ``"iran"``     -> ``data/mortality_iran.json``  (UN WPP 2024, Male, Iran)
    ``"poland"``   -> ``data/mortality.json``        (default placeholder)
    ``"default"``  -> ``data/mortality.json``        (alias for ``"poland"``)
    """

    source: MortalitySource = "iran"


class Time(InputMetadata):
    weeks_per_year: int


class SimulationFile(InputMetadata):
    environment: Environment
    discounting: Discounting
    psa: PSA
    mortality: Mortality = Mortality()
    time: Time

    @property
    def is_development(self) -> bool:
        return self.environment.mode == "development"

    @property
    def sample_size(self) -> int:
        return self.psa.sample_size(self.environment.mode)
