
from app.persistence.schemas.metadata import InputMetadata


class Currency(InputMetadata):
    name: str
    code: str
    symbol: str


class Assumption(InputMetadata):
    iu_per_microgram: float


class Pricing(InputMetadata):
    per_unit_description: str | None = None
    per_unit_reference: str | list[str] | None = None
    per_unit: dict[str, float]  # IRR, T, USD
    per_microgram_description: str | None = None
    per_microgram_reference: str | list[str] | None = None
    per_microgram: dict[str, float]  # IRR, T, USD


class CostItem(InputMetadata):
    item: str
    assumption: Assumption
    pricing: Pricing


class CostFile(InputMetadata):
    currencies: list[Currency]
    costs: list[CostItem]
