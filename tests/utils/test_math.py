import numpy as np
import pytest

from utils.math import (
    build_zero_truncated_poisson_probs,
    cal_body_weight,
    expm_prob,
    factorial_numba,
    poisson_mass_function,
    prob_at_least_one,
    to_weekly,
    zero_truncated_mass_function,
)


class TestCalBodyWeight:
    def test_milestone_weights_modeled_range(self):
        """Calibrated curve must track male median weight milestones over
        the modeled age range (patients enter at age 2, so the curve is
        calibrated for weeks >= 104; the birth anchor is intentionally
        not a calibration target)."""
        milestones = {
            104: (12.2, 2.5),   # 2 y
            260: (18.4, 2.0),   # 5 y
            520: (32.2, 2.5),   # 10 y
            624: (40.5, 2.5),   # 12 y
            936: (70.0, 3.0),   # 18 y
            2080: (90.0, 3.0),  # 40 y
        }
        for week, (expected, tol) in milestones.items():
            w = cal_body_weight(week)
            assert abs(w - expected) <= tol, (
                f"week {week}: {w} vs milestone {expected} (tol {tol})"
            )

    def test_peak_then_decline_after_55y(self):
        w_peak = cal_body_weight(2860)  # 55 y
        w_old = cal_body_weight(4680)  # 90 y
        assert 85.0 <= w_peak <= 92.0
        assert w_old < w_peak

    def test_positive_weight(self):
        for week in [0, 52, 500, 1000, 2000, 4000, 5000]:
            w = cal_body_weight(week)
            assert w > 0, f"Weight non-positive at week {week}"

    def test_weight_grows_then_declines(self):
        w_young = cal_body_weight(1000)
        w_mid = cal_body_weight(2000)
        w_old = cal_body_weight(4000)
        assert w_mid > w_young
        assert w_old < w_mid

    def test_weight_with_factor(self):
        w_default = cal_body_weight(2000)
        w_scaled = cal_body_weight(2000, weight_factor=2.0)
        assert abs(w_scaled - w_default * 2.0) <= 0.02

    def test_weight_with_offset(self):
        w = cal_body_weight(100, b=50)
        assert w > 0

    def test_invalid_negative_week_raises(self):
        with pytest.raises(ValueError, match="Week must be an integer"):
            cal_body_weight(-1)

    def test_invalid_excessive_week_raises(self):
        with pytest.raises(ValueError, match="Week must be an integer"):
            cal_body_weight(6000)


class TestProbAtLeastOne:
    def test_zero_rate(self):
        assert prob_at_least_one(0.0) == 0.0

    def test_approaches_one(self):
        assert prob_at_least_one(10.0) > 0.999

    def test_small_rate(self):
        p = prob_at_least_one(0.01)
        assert 0.009 < p < 0.011

    def test_monotonic(self):
        rates = [0.1, 0.5, 1.0, 2.0]
        probs = [prob_at_least_one(r) for r in rates]
        assert all(probs[i] < probs[i + 1] for i in range(len(probs) - 1))

    def test_range(self):
        for lam in np.linspace(0, 10, 50):
            p = prob_at_least_one(lam)
            assert 0.0 <= p <= 1.0


class TestExpmProb:
    def test_zero_rate(self):
        assert expm_prob(0.0) == 0.0

    def test_approaches_one(self):
        assert expm_prob(10.0) > 0.999

    def test_dt_scaling(self):
        p1 = expm_prob(0.5, dt=1.0)
        p2 = expm_prob(0.5, dt=2.0)
        assert p2 > p1

    def test_equivalence_with_prob_at_least_one(self):
        rate = 0.5
        assert np.isclose(expm_prob(rate), prob_at_least_one(rate))


class TestToWeekly:
    def test_standard_conversion(self):
        assert to_weekly(52) == 1.0

    def test_custom_weeks(self):
        assert to_weekly(365, weeks_per_year=365) == 1.0

    def test_zero(self):
        assert to_weekly(0) == 0.0


class TestFactorialNumba:
    def test_zero(self):
        assert factorial_numba(0) == 1

    def test_small_values(self):
        assert factorial_numba(1) == 1
        assert factorial_numba(5) == 120
        assert factorial_numba(10) == 3628800


class TestBuildZeroTruncatedPoissonProbs:
    def test_returns_valid_structure(self):
        k_values, probs = build_zero_truncated_poisson_probs(5.0, 20)
        assert len(k_values) == 20
        assert len(probs) == 20
        assert np.isclose(probs.sum(), 1.0, rtol=1e-6)

    def test_no_zero_probability(self):
        k_values, probs = build_zero_truncated_poisson_probs(5.0, 20)
        assert k_values[0] == 1

    def test_small_lambda(self):
        k_values, probs = build_zero_truncated_poisson_probs(0.1, 10)
        assert np.isclose(probs.sum(), 1.0, rtol=1e-6)
        assert probs[0] > 0.9

    def test_large_lambda(self):
        k_values, probs = build_zero_truncated_poisson_probs(50.0, 100)
        assert np.isclose(probs.sum(), 1.0, rtol=1e-6)

    def test_degenerate_lambda_zero(self):
        k_values, probs = build_zero_truncated_poisson_probs(0.0, 10)
        assert np.isclose(probs[0], 1.0)

    def test_degenerate_lambda_nan(self):
        k_values, probs = build_zero_truncated_poisson_probs(np.nan, 10)
        assert np.isclose(probs[0], 1.0)


class TestPoissonMassFunction:
    def test_basic(self):
        p = poisson_mass_function(lam=3.0, k=2)
        assert 0.2 < p < 0.25

    def test_zero_k(self):
        p = poisson_mass_function(lam=3.0, k=0)
        assert np.isclose(p, np.exp(-3.0))


class TestZeroTruncatedMassFunction:
    def test_basic(self):
        p = zero_truncated_mass_function(lam=3.0, k=2)
        assert 0 < p < 1

    def test_k_one_highest_for_small_lambda(self):
        p = zero_truncated_mass_function(lam=0.5, k=1)
        p2 = zero_truncated_mass_function(lam=0.5, k=2)
        assert p > p2

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="zero is truncated"):
            zero_truncated_mass_function(lam=3.0, k=0)
