from typing import Literal

from app.persistence.schemas.metadata import InputMetadata


class GDPPerCapita(InputMetadata):
    USD: float
    IRR: float
    TOMAN: float


class WTPMultiplier(InputMetadata):
    standard: float
    rare: float


class EconomicPolicyFile(InputMetadata):
    currency: Literal["USD", "IRR", "T"]
    disease_profile: Literal["standard", "rare"]
    gdp_per_capita: GDPPerCapita
    wtp_multiplier: WTPMultiplier
