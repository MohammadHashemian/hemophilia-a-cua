from __future__ import annotations

from enum import StrEnum


class DisplayCurrency(StrEnum):
    IRR = "IRR"
    TOMAN = "TOMAN"
    USD = "USD"


def convert_from_irr(
    value_irr: float,
    currency: DisplayCurrency | str,
    *,
    irr_per_usd: float | None = None,
) -> float:
    """Convert for notebook display only; the model always calculates in IRR."""
    target = DisplayCurrency(currency)
    if target is DisplayCurrency.IRR:
        return float(value_irr)
    if target is DisplayCurrency.TOMAN:
        return float(value_irr) / 10.0
    if irr_per_usd is None or irr_per_usd <= 0:
        raise ValueError("A positive, dated irr_per_usd value is required for USD display")
    return float(value_irr) / irr_per_usd
