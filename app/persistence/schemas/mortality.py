"""Pydantic schema for ``data/mortality.json``.

The mortality input file holds age-specific mortality rates (deaths per
person-year) and an overall crude rate. Values are stored as plain
``float`` so the file is consumable by both Pydantic and ad-hoc JSON
parsers.

Note on nulls
-------------
The Iran mortality calculator
(``notebook_tools/iran_mortality.py``) writes ``null`` (Python ``None``)
in ``age_specific`` and ``crude_annual_rate`` for buckets that could
not be estimated by the cohort survival method (e.g. growing young
cohorts where ``P(x+1, t+1) > P(x, t)``). This schema accepts those
nulls so the calculator output is also a valid ``MortalityFile``.
The original Poland values are all finite and still validate.

Citation
--------
* Polish reference values are taken from the Polish life tables used
  in the hemophilia Markov model. See ``data/mortality.json`` for the
  full provenance recorded by the user.
* Iran values are produced by ``notebook_tools/iran_mortality.py``
  using either the cohort survival method (Preston, Heuveline &
  Guillot, 2001) or the UN WPP 2024 abridged life table
  (UN DESA, 2024).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MortalityFile(BaseModel):
    """Typed wrapper for the mortality input file.

    The ``extra="allow"`` config lets producers (notably the Iran
    calculator) attach a ``source`` block for provenance without
    breaking validation of the bare schema.
    """

    model_config = ConfigDict(extra="allow")

    description: str | None = None
    reference: str | list[str] | None = None
    use_age_specific: bool
    crude_annual_rate: float | None = Field(
        default=None,
        description=(
            "Crude annual death rate (deaths per person-year). ``None`` "
            "is permitted when the calculation cannot produce a finite "
            "value (e.g. the cohort method on a growing population)."
        ),
    )
    age_specific: dict[str, float | None] = Field(
        default_factory=dict,
        description=(
            "Age-bucketed mortality rates (deaths per person-year). "
            "Keys are the Poland schema bucket labels "
            "(\"0\", \"1-4\", ..., \"90+\"). ``None`` marks a bucket "
            "that could not be estimated."
        ),
    )
    source: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Provenance block: which input files were used, which "
            "method (cohort / WPP / hybrid), the sex, and any other "
            "context. The schema is intentionally open (any string "
            "keys, any values) to accommodate future methods."
        ),
    )
