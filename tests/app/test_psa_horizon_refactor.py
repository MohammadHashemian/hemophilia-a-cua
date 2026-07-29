import json
from pathlib import Path

import numpy as np

from app.notebook.psa.scenarios import (
    CHILDHOOD,
    LIFETIME,
    build_psa_scenarios,
    get_horizon,
)
from app.notebook.scenario_helpers import parse_scenario
from app.persistence.context import ModelContext


def _meta_samples(n: int = 10) -> dict:
    return {
        "on_demand": {
            "bayesian": np.full(n, 20.0),
            "dirichlet": np.full(n, 20.0),
        },
        "prophylaxis": {
            "bayesian": np.full(n, 4.0),
            "dirichlet": np.full(n, 4.0),
        },
    }


def test_horizon_specs_are_age_explicit():
    assert CHILDHOOD.cycles == 10 * 52
    assert LIFETIME.cycles == 98 * 52
    assert get_horizon("early") is CHILDHOOD
    assert get_horizon("childhood_age_2_12") is CHILDHOOD


def test_each_horizon_builds_only_its_16_scenarios():
    context = ModelContext.load()
    for horizon in (CHILDHOOD, LIFETIME):
        scenarios = build_psa_scenarios(
            horizon,
            meta_samples=_meta_samples(),
            context=context,
        )
        assert len(scenarios) == 16
        assert {parse_scenario(s.name)[0] for s in scenarios} == {horizon.key}
        assert {
            s.overrides["cycles"].distribution.point() for s in scenarios
        } == {horizon.cycles}


def test_new_psa_notebooks_are_valid_and_separated():
    notebooks_root = Path("app/notebooks/psa")
    for horizon in (CHILDHOOD, LIFETIME):
        folder = notebooks_root / horizon.directory
        files = sorted(folder.glob("*.ipynb"))
        assert [file.name for file in files] == [
            "01_scenario_definitions.ipynb",
            "02_simulation.ipynb",
            "03_analysis.ipynb",
        ]
        for file in files:
            notebook = json.loads(file.read_text(encoding="utf-8"))
            assert notebook["nbformat"] == 4
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in notebook["cells"]
            )
            other = LIFETIME if horizon is CHILDHOOD else CHILDHOOD
            assert horizon.directory in str(file)
            assert f"from app.notebook.psa import {horizon.key.upper()}" in source
            assert f"from app.notebook.psa import {other.key.upper()}" not in source
