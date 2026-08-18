from __future__ import annotations

import json

import pytest

from modular_models.state_transition.context import ContextValidationError, StudyContext
from modular_models.state_transition.sampling import ParameterResolver


def test_context_loads_reference_based_typed_inputs() -> None:
    context = StudyContext.load()

    assert context.model.metadata.currency == "IRR"
    assert context.model.metadata.specification_version == "1.1.0"
    assert context.parameter("exit_age_years").value == 12
    assert context.parameter("abr_on_demand").value == pytest.approx(13.8)
    assert context.parameter("abr_on_demand").references == ("ref_44",)
    assert context.parameter("ich_rate_on_demand").psa is not None
    assert context.scenario("ich_fraction").options["ich_rate_method"] == "fraction"


def test_unknown_cross_reference_is_rejected(tmp_path) -> None:
    source = StudyContext.load().data_dir
    for name in ("model.json", "scenarios.json", "references.json"):
        (tmp_path / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    model = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    model["parameters"]["abr_on_demand"]["references"] = ["missing-reference"]
    (tmp_path / "model.json").write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(ContextValidationError, match="Unknown reference"):
        StudyContext.load(tmp_path)


def test_psa_draws_respect_rate_accounting_and_utility_monotonicity() -> None:
    context = StudyContext.load()
    samples, _ = ParameterResolver(context).probabilistic(2000, 991, "base_case")

    for suffix in ("prophylaxis", "on_demand"):
        residual = (
            samples[f"abr_{suffix}"]
            - samples[f"ajbr_{suffix}"]
            - samples[f"ich_rate_{suffix}"]
            - samples[f"abr_{suffix}"] * samples["non_ich_major_fraction"]
        )
        assert (residual >= 0).all()
    assert (samples["utility_anchor"] >= samples["utility_mild"]).all()
    assert (samples["utility_mild"] >= samples["utility_moderate"]).all()
    assert (samples["utility_moderate"] >= samples["utility_severe"]).all()
