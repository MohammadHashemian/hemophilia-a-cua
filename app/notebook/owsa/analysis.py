"""Reusable deterministic OWSA calculations."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import polars as pl

from app.notebook.owsa.scenarios import OWSARange
from app.notebook.psa.economics import paired_outcomes, scenario_pairs
from app.notebook.scenario_helpers import parse_scenario


def parameter_ranges(ranges: list[OWSARange]) -> pl.DataFrame:
    return pl.DataFrame([asdict(item) for item in ranges]).sort("parameter")


def _pair_result(
    df: pl.DataFrame,
    base: str,
    comparison: str,
    *,
    wtp: float,
) -> dict[str, float | int]:
    paired = paired_outcomes(df, base, comparison, wtp=wtp)
    delta_cost = float(paired["delta_cost"].mean())
    delta_qaly = float(paired["delta_qaly"].mean())
    return {
        "paired_iterations": paired.height,
        "delta_cost": delta_cost,
        "delta_qaly": delta_qaly,
        "icer": (
            delta_cost / delta_qaly
            if not np.isclose(delta_qaly, 0)
            else float("nan")
        ),
        "delta_nmb": float(paired["delta_nmb"].mean()),
        "probability_cost_effective": float((paired["delta_nmb"] > 0).mean()),
    }


def base_case(df: pl.DataFrame, *, wtp: float) -> pl.DataFrame:
    pair = next(
        pair
        for pair in scenario_pairs(df)
        if parse_scenario(pair[0])[3] is None
    )
    result = _pair_result(df, *pair, wtp=wtp)
    return pl.DataFrame(
        [
            {
                "comparison": f"{pair[1]} vs {pair[0]}",
                **result,
                "wtp": wtp,
                "cost_effective": result["delta_nmb"] > 0,
            }
        ]
    )


def sensitivity_summary(
    df: pl.DataFrame,
    ranges: list[OWSARange],
    *,
    wtp: float,
) -> pl.DataFrame:
    pairs = scenario_pairs(df)
    base_pair = next(pair for pair in pairs if parse_scenario(pair[0])[3] is None)
    base = _pair_result(df, *base_pair, wtp=wtp)
    results: dict[tuple[str, str], dict[str, float | int]] = {}
    for pair in pairs:
        extension = parse_scenario(pair[0])[3]
        if extension is None:
            continue
        parameter, level = extension.rsplit("_", 1)
        if level not in {"low", "high"}:
            continue
        results[(parameter, level)] = _pair_result(df, *pair, wtp=wtp)

    rows = []
    for item in ranges:
        low = results.get((item.parameter, "low"))
        high = results.get((item.parameter, "high"))
        if low is None or high is None:
            continue
        low_nmb = float(low["delta_nmb"])
        high_nmb = float(high["delta_nmb"])
        low_icer = float(low["icer"])
        high_icer = float(high["icer"])
        base_nmb = float(base["delta_nmb"])
        base_icer = float(base["icer"])
        rows.append(
            {
                **asdict(item),
                "paired_iterations": int(low["paired_iterations"]),
                "base_icer": base_icer,
                "low_icer": low_icer,
                "high_icer": high_icer,
                "base_delta_cost": float(base["delta_cost"]),
                "low_delta_cost": float(low["delta_cost"]),
                "high_delta_cost": float(high["delta_cost"]),
                "base_delta_qaly": float(base["delta_qaly"]),
                "low_delta_qaly": float(low["delta_qaly"]),
                "high_delta_qaly": float(high["delta_qaly"]),
                "low_icer_change": low_icer - base_icer,
                "high_icer_change": high_icer - base_icer,
                "base_delta_nmb": base_nmb,
                "low_delta_nmb": low_nmb,
                "high_delta_nmb": high_nmb,
                "low_nmb_change": low_nmb - base_nmb,
                "high_nmb_change": high_nmb - base_nmb,
                "nmb_sensitivity": max(
                    abs(low_nmb - base_nmb),
                    abs(high_nmb - base_nmb),
                ),
                "base_cost_effective": base_nmb > 0,
                "low_cost_effective": low_nmb > 0,
                "high_cost_effective": high_nmb > 0,
                "decision_changes": (
                    (low_nmb > 0) != (base_nmb > 0)
                    or (high_nmb > 0) != (base_nmb > 0)
                ),
                "icer_tornado_valid": (
                    float(base["delta_cost"]) >= 0
                    and float(low["delta_cost"]) >= 0
                    and float(high["delta_cost"]) >= 0
                    and float(base["delta_qaly"]) > 0
                    and float(low["delta_qaly"]) > 0
                    and float(high["delta_qaly"]) > 0
                ),
            }
        )
    return pl.DataFrame(rows).sort("nmb_sensitivity", descending=True)


def robustness_summary(summary: pl.DataFrame) -> pl.DataFrame:
    return summary.select(
        "parameter",
        "label",
        "base_cost_effective",
        "low_cost_effective",
        "high_cost_effective",
        "decision_changes",
        "nmb_sensitivity",
    )
