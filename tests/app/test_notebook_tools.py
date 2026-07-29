import numpy as np
import pytest

from app.notebook.calibration import _safe_div, classify_calibration
from app.notebook.dataframe_builders import calculate_state_occupation
from app.notebook.scenario_helpers import (
    ltb_mode_for_scenario,
    pair_scenarios,
    parse_scenario,
)
from app.notebook.smoke import main as smoke_main


class TestCalculateStateOccupation:
    def test_all_states_single_step(self):
        seq = ["healthy"]
        occ = calculate_state_occupation(seq, ["healthy", "bleeding", "death"])
        assert occ["healthy_weeks"] == 1
        assert occ["bleeding_weeks"] == 0
        assert occ["death_weeks"] == 0

    def test_mixed_sequence(self):
        seq = ["healthy", "healthy", "bleeding", "death", "death"]
        occ = calculate_state_occupation(seq, ["healthy", "bleeding", "death"])
        assert occ["healthy_weeks"] == 2
        assert occ["bleeding_weeks"] == 1
        assert occ["death_weeks"] == 2

    def test_shares_sum_to_one(self):
        seq = ["healthy"] * 40 + ["bleeding"] * 10 + ["death"] * 50
        occ = calculate_state_occupation(seq, ["healthy", "bleeding", "death"])
        total = occ["healthy_share"] + occ["bleeding_share"] + occ["death_share"]
        assert abs(total - 1.0) < 1e-10

    def test_empty_sequence(self):
        occ = calculate_state_occupation([], ["A", "B"])
        assert occ["A_weeks"] == 0
        assert occ["A_share"] == 0.0
        assert occ["B_weeks"] == 0

    def test_sequence_with_only_death(self):
        seq = ["death"] * 10
        occ = calculate_state_occupation(seq, ["healthy", "bleeding", "death"])
        assert occ["death_weeks"] == 10


class TestClassifyCalibration:
    def test_excellent(self):
        status = classify_calibration(oe_ratio=1.0, relative_error=0.0, distribution_similarity_index=1.0)
        assert status == "Excellent"

    def test_excellent_boundary(self):
        status = classify_calibration(oe_ratio=0.95, relative_error=0.05, distribution_similarity_index=0.90)
        assert status == "Excellent"

    def test_acceptable(self):
        status = classify_calibration(oe_ratio=0.90, relative_error=0.10, distribution_similarity_index=0.85)
        assert status == "Acceptable"

    def test_acceptable_boundary(self):
        status = classify_calibration(oe_ratio=0.85, relative_error=0.15, distribution_similarity_index=0.80)
        assert status == "Acceptable"

    def test_needs_investigation(self):
        status = classify_calibration(oe_ratio=0.70, relative_error=0.30, distribution_similarity_index=0.50)
        assert status == "Needs Investigation"

    def test_needs_investigation_below_acceptable(self):
        status = classify_calibration(oe_ratio=0.84, relative_error=0.16, distribution_similarity_index=0.79)
        assert status == "Needs Investigation"


class TestSafeDiv:
    def test_normal_division(self):
        assert _safe_div(10, 2) == 5.0

    def test_zero_denominator(self):
        assert np.isnan(_safe_div(10, 0))

    def test_zero_numerator(self):
        assert _safe_div(0, 5) == 0.0


class TestParseScenario:
    def test_basic_format(self):
        result = parse_scenario("lifetime on_demand bayesian")
        assert result == ("lifetime", "on-demand", "bayesian", None)

    def test_with_extension(self):
        result = parse_scenario("early prophylaxis dirichlet high_cost")
        assert result == ("early", "prophylaxis", "dirichlet", "high_cost")

    def test_two_part(self):
        result = parse_scenario("lifetime on_demand bayesian low_abr")
        assert result == ("lifetime", "on-demand", "bayesian", "low_abr")


class TestPairScenarios:
    def test_pairs_on_demand_with_prophylaxis(self):
        scenarios = [
            "lifetime on_demand bayesian",
            "lifetime prophylaxis bayesian",
        ]
        pairs = pair_scenarios(scenarios)
        assert len(pairs) == 1
        # parse_scenario normalizes on_demand -> on-demand
        assert pairs[0] == ("lifetime on_demand bayesian", "lifetime prophylaxis bayesian")

    def test_no_matching_pair_raises(self):
        scenarios = ["lifetime on_demand bayesian", "early prophylaxis bayesian"]
        with pytest.raises(ValueError, match="Missing on-demand scenario"):
            pair_scenarios(scenarios)

    def test_multiple_pairs(self):
        scenarios = [
            "lifetime on_demand bayesian",
            "lifetime prophylaxis bayesian",
            "early on_demand dirichlet",
            "early prophylaxis dirichlet",
        ]
        pairs = pair_scenarios(scenarios)
        assert len(pairs) == 2

    def test_with_extensions_must_match(self):
        scenarios = [
            "lifetime on_demand bayesian high_cost",
            "lifetime prophylaxis bayesian high_cost",
        ]
        pairs = pair_scenarios(scenarios)
        assert len(pairs) == 1

    def test_base_and_extended_pairs_can_be_sorted_together(self):
        scenarios = [
            "lifetime prophylaxis bayesian high_cost",
            "lifetime on_demand bayesian",
            "lifetime on_demand bayesian high_cost",
            "lifetime prophylaxis bayesian",
        ]
        assert pair_scenarios(scenarios) == [
            ("lifetime on_demand bayesian", "lifetime prophylaxis bayesian"),
            (
                "lifetime on_demand bayesian high_cost",
                "lifetime prophylaxis bayesian high_cost",
            ),
        ]

    def test_pairs_are_sorted_deterministically(self):
        scenarios = [
            "lifetime prophylaxis dirichlet",
            "early on_demand bayesian",
            "lifetime on_demand bayesian",
            "early prophylaxis bayesian",
            "lifetime on_demand dirichlet",
            "lifetime prophylaxis bayesian",
            "early prophylaxis dirichlet",
            "early on_demand dirichlet",
        ]
        pairs = pair_scenarios(scenarios)
        assert pairs == [
            ("early on_demand bayesian", "early prophylaxis bayesian"),
            ("early on_demand dirichlet", "early prophylaxis dirichlet"),
            ("lifetime on_demand bayesian", "lifetime prophylaxis bayesian"),
            ("lifetime on_demand dirichlet", "lifetime prophylaxis dirichlet"),
        ]


class TestLTBModeForScenario:
    def test_fraction_is_the_default_base_case(self):
        assert ltb_mode_for_scenario("early on_demand bayesian") == "fraction"

    def test_absolute_incidence_requires_explicit_scenario_token(self):
        assert (
            ltb_mode_for_scenario("lifetime prophylaxis bayesian ltb_absolute")
            == "absolute"
        )


class TestSeedSmokeScript:
    """The ``notebook_tools.smoke`` script is the single-line guard that
    CI runs after the test suite to confirm the project is still seeded
    end-to-end. It must return 0 and report OK."""

    def test_smoke_returns_zero_and_reports_ok(self, capsys):
        rc = smoke_main()
        out = capsys.readouterr().out
        assert rc == 0, f"smoke_main() returned {rc}"
        assert "OK" in out
        assert "env seed" in out
