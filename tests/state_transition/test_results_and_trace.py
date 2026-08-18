from __future__ import annotations

import json
from pathlib import Path

import pytest

from modular_models.state_transition.analysis import StudyRunner
from modular_models.state_transition.context import StudyContext
from modular_models.state_transition.currency import DisplayCurrency, convert_from_irr
from modular_models.state_transition.trace import TraceSession


def test_currency_conversion_is_display_only_and_explicit() -> None:
    assert convert_from_irr(1000, DisplayCurrency.IRR) == 1000
    assert convert_from_irr(1000, DisplayCurrency.TOMAN) == 100
    assert convert_from_irr(1_000_000, DisplayCurrency.USD, irr_per_usd=100_000) == 10
    with pytest.raises(ValueError, match="dated irr_per_usd"):
        convert_from_irr(1_000_000, DisplayCurrency.USD)


def test_trace_writes_json_and_png(tmp_path: Path) -> None:
    trace = TraceSession(max_cycles=1)
    StudyRunner(StudyContext.load()).compare(n_patients=20, seed=10, trace=trace)

    json_path = trace.write_json(tmp_path / "trace.json")
    png_path = trace.render(tmp_path / "trace.png")

    assert json_path.stat().st_size > 1000
    assert png_path.stat().st_size > 10_000
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["parameter_provenance"]["abr_on_demand"]["references"] == ["ref_44"]
    assert "state_transition_decision_logic" in payload
    for run in payload["runs"]:
        overall = run["output"]["mortality"]["overall"]
        assert overall["initial_patients"] == overall["alive_at_end"] + overall["deaths_total"]
        assert "post_ich_ever_count" in run["output"]["summary"]

    restored = TraceSession.read_json(json_path)
    assert restored.to_dict() == payload


def test_psa_records_economic_mortality_and_post_ich_outputs() -> None:
    result = StudyRunner(StudyContext.load()).psa(
        iterations=2,
        n_patients=20,
        seed=719,
    )

    assert len(result.records) == 2
    for record in result.records:
        assert "incremental_cost_irr" in record
        assert "prophylaxis_deaths_background" in record
        assert "on_demand_deaths_ich" in record
        assert "prophylaxis_post_ich_ever_count" in record
        assert record["prophylaxis_alive_at_end"] <= 20
