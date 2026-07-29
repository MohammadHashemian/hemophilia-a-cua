import numpy as np
import pytest

from app.analysis.distributions import (
    Bayesian,
    BetaFromMeanSD,
    Constant,
    DirichletMixture,
    GammaFromMeanCV,
    GammaFromMeanSD,
    MixtureOfStudies,
    TriangularDist,
)
from app.analysis.psa.models import ParameterSet
from app.analysis.psa.parameter_resolver import ParameterResolver
from app.analysis.psa.parameters import Parameter
from app.analysis.psa.sampler import PSASampler
from app.persistence.schemas.clinicals import StudyEstimate

SEED = 42


# ── Constant Distribution ───────────────────────────────────────────


class TestConstant:
    def test_always_returns_value(self):
        dist = Constant(value=5.0)
        samples = dist.sample(100, np.random.default_rng(SEED))
        assert np.allclose(samples, 5.0)

    def test_point(self):
        dist = Constant(value=3.14)
        assert dist.point() == 3.14


# ── GammaFromMeanSD ─────────────────────────────────────────────────


class TestGammaFromMeanSD:
    def test_sample_mean_close_to_expected(self):
        dist = GammaFromMeanSD(mean=10.0, sd=3.0)
        samples = dist.sample(100_000, np.random.default_rng(SEED))
        assert abs(np.mean(samples) - 10.0) < 0.2

    def test_point(self):
        dist = GammaFromMeanSD(mean=10.0, sd=3.0)
        assert dist.point() == 10.0

    def test_all_positive(self):
        dist = GammaFromMeanSD(mean=5.0, sd=2.0)
        samples = dist.sample(1000, np.random.default_rng(SEED))
        assert np.all(samples > 0)


# ── GammaFromMeanCV ─────────────────────────────────────────────────


class TestGammaFromMeanCV:
    def test_sample_mean_close_to_expected(self):
        dist = GammaFromMeanCV(mean=10.0, cv=0.3)
        samples = dist.sample(100_000, np.random.default_rng(SEED))
        assert abs(np.mean(samples) - 10.0) < 0.2

    def test_invalid_cv_raises(self):
        dist = GammaFromMeanCV(mean=10.0, cv=0.0)
        with pytest.raises(ValueError, match="CV must be > 0"):
            dist.sample(10, np.random.default_rng(SEED))


# ── BetaFromMeanSD ──────────────────────────────────────────────────


class TestBetaFromMeanSD:
    def test_sample_mean_close_to_expected(self):
        dist = BetaFromMeanSD(mean=0.3, sd=0.1)
        samples = dist.sample(100_000, np.random.default_rng(SEED))
        assert abs(np.mean(samples) - 0.3) < 0.02

    def test_range(self):
        dist = BetaFromMeanSD(mean=0.5, sd=0.1)
        samples = dist.sample(1000, np.random.default_rng(SEED))
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_cv_instead_of_sd(self):
        dist = BetaFromMeanSD(mean=0.5, cv=0.2)
        samples = dist.sample(1000, np.random.default_rng(SEED))
        assert abs(np.mean(samples) - 0.5) < 0.05

    def test_invalid_variance_raises(self):
        with pytest.raises(ValueError, match="Invalid Beta parameters"):
            BetaFromMeanSD(mean=0.5, sd=0.5).sample(10, np.random.default_rng(SEED))


# ── TriangularDist ──────────────────────────────────────────────────


class TestTriangularDist:
    def test_samples_within_bounds(self):
        dist = TriangularDist(left=0.0, mode=5.0, right=10.0)
        samples = dist.sample(1000, np.random.default_rng(SEED))
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 10.0)

    def test_point(self):
        dist = TriangularDist(left=0.0, mode=5.0, right=10.0)
        assert abs(dist.point() - 5.0) < 1e-6


# ── MixtureOfStudies ────────────────────────────────────────────────


class TestMixtureOfStudies:
    def test_mixture_samples(self):
        comp1 = Constant(0.0)
        comp2 = Constant(10.0)
        mixture = MixtureOfStudies([comp1, comp2])
        samples = mixture.sample(1000, np.random.default_rng(SEED))
        assert 0.0 <= np.mean(samples) <= 10.0

    def test_point(self):
        mixture = MixtureOfStudies([Constant(1.0), Constant(3.0)])
        assert abs(mixture.point() - 2.0) < 1e-6


# ── DirichletMixture ────────────────────────────────────────────────


class TestDirichletMixture:
    def test_mixture_samples(self):
        comp1 = Constant(0.0)
        comp2 = Constant(10.0)
        mixture = DirichletMixture([comp1, comp2], alpha=1.0)
        samples = mixture.sample(1000, np.random.default_rng(SEED))
        assert 0.0 <= np.mean(samples) <= 10.0

    def test_custom_alpha(self):
        comp1 = Constant(0.0)
        comp2 = Constant(10.0)
        mixture = DirichletMixture([comp1, comp2], alpha=np.array([10.0, 1.0]))
        samples = mixture.sample(1000, np.random.default_rng(SEED))
        assert np.mean(samples) < 5.0

    def test_point(self):
        mixture = DirichletMixture([Constant(1.0), Constant(3.0)])
        assert abs(mixture.point() - 2.0) < 1e-6


# ── Parameter ───────────────────────────────────────────────────────


class TestParameter:
    def test_sample(self):
        p = Parameter(distribution=Constant(7.0))
        samples = p.sample(10, np.random.default_rng(SEED))
        assert np.allclose(samples, 7.0)

    def test_cached_sample_returns_same(self):
        cached = np.array([1.0, 2.0, 3.0])
        p = Parameter(distribution=Constant(7.0), cache=cached)
        s1 = p.sample(100, np.random.default_rng(SEED))
        s2 = p.sample(100, np.random.default_rng(SEED))
        assert np.array_equal(s1, cached)
        assert np.array_equal(s1, s2)

    def test_point(self):
        p = Parameter(distribution=Constant(5.0))
        assert p.point() == 5.0


# ── PSASampler ──────────────────────────────────────────────────────


class TestPSASampler:
    def test_sample_returns_dict(self):
        param_set = ParameterSet(
            cycles=Parameter(Constant(520)),
            bleeding_rate=Parameter(Constant(22.0)),
            joint_bleeding_fraction=Parameter(Constant(0.3)),
            life_threatening_bleeding_fraction=Parameter(Constant(0.05)),
            life_threatening_bleeding_rate=Parameter(Constant(0.5)),
            ltb_case_fatality=Parameter(Constant(0.35)),
            baseline_age=Parameter(Constant(2)),
            weight_factor=Parameter(Constant(1.0)),
            benefits_discount_rate=Parameter(Constant(0.0)),
            costs_discount_rate=Parameter(Constant(0.0)),
            healthy_utility=Parameter(Constant(1.0)),
            mild_arthropathy_utility=Parameter(Constant(0.9)),
            moderate_arthropathy_utility=Parameter(Constant(0.7)),
            severe_arthropathy_utility=Parameter(Constant(0.5)),
            spontaneous_bleeding_utility=Parameter(Constant(0.6)),
            joint_bleeding_utility=Parameter(Constant(0.4)),
            life_threatening_bleeding_utility=Parameter(Constant(0.3)),
            death_utility=Parameter(Constant(0.0)),
            per_unit_price=Parameter(Constant(1000.0)),
            prophylaxis_background_factor_consumption_per_kg=Parameter(Constant(0.0)),
            factor_consumption_per_spontaneous_bleeding_per_kg=Parameter(Constant(10.0)),
            factor_consumption_per_joint_bleeding_per_kg=Parameter(Constant(20.0)),
            factor_consumption_per_life_threatening_bleeding_per_kg=Parameter(Constant(50.0)),
        )
        sampler = PSASampler(param_set, seed=SEED)
        samples = sampler.sample(100)
        assert isinstance(samples, dict)
        assert "bleeding_rate" in samples
        assert len(samples["bleeding_rate"]) == 100

    def test_point_returns_deterministic(self):
        param_set = ParameterSet(
            cycles=Parameter(Constant(520)),
            bleeding_rate=Parameter(Constant(22.0)),
            joint_bleeding_fraction=Parameter(Constant(0.3)),
            life_threatening_bleeding_fraction=Parameter(Constant(0.05)),
            life_threatening_bleeding_rate=Parameter(Constant(0.5)),
            ltb_case_fatality=Parameter(Constant(0.35)),
            baseline_age=Parameter(Constant(2)),
            weight_factor=Parameter(Constant(1.0)),
            benefits_discount_rate=Parameter(Constant(0.0)),
            costs_discount_rate=Parameter(Constant(0.0)),
            healthy_utility=Parameter(Constant(1.0)),
            mild_arthropathy_utility=Parameter(Constant(0.9)),
            moderate_arthropathy_utility=Parameter(Constant(0.7)),
            severe_arthropathy_utility=Parameter(Constant(0.5)),
            spontaneous_bleeding_utility=Parameter(Constant(0.6)),
            joint_bleeding_utility=Parameter(Constant(0.4)),
            life_threatening_bleeding_utility=Parameter(Constant(0.3)),
            death_utility=Parameter(Constant(0.0)),
            per_unit_price=Parameter(Constant(1000.0)),
            prophylaxis_background_factor_consumption_per_kg=Parameter(Constant(0.0)),
            factor_consumption_per_spontaneous_bleeding_per_kg=Parameter(Constant(10.0)),
            factor_consumption_per_joint_bleeding_per_kg=Parameter(Constant(20.0)),
            factor_consumption_per_life_threatening_bleeding_per_kg=Parameter(Constant(50.0)),
        )
        sampler = PSASampler(param_set, seed=SEED)
        point = sampler.point()
        assert np.isclose(point["bleeding_rate"], 22.0)


# ── ParameterResolver ──────────────────────────────────────────────


class TestParameterResolver:
    def test_default_mode_is_fraction_of_abr(self):
        samples = {
            "bleeding_rate": np.array([20.0]),
            "joint_bleeding_fraction": np.array([0.75]),
            "life_threatening_bleeding_fraction": np.array([0.025]),
        }
        resolved = ParameterResolver.resolve_samples(samples)
        assert np.allclose(resolved["life_threatening_bleeding_rate"], [0.5])

    def test_resolve_samples_fraction_mode_adds_derived_rates(self):
        samples = {
            "bleeding_rate": np.array([22.0, 15.0]),
            "joint_bleeding_fraction": np.array([0.3, 0.25]),
            "life_threatening_bleeding_fraction": np.array([0.05, 0.03]),
        }
        resolved = ParameterResolver.resolve_samples(samples, ltb_mode="fraction")
        assert "spontaneous_bleeding_rate" in resolved
        assert "joint_bleeding_rate" in resolved
        assert "life_threatening_bleeding_rate" in resolved
        assert np.allclose(resolved["spontaneous_bleeding_rate"], [22.0 * (1 - 0.3 - 0.05), 15.0 * (1 - 0.25 - 0.03)])
        # LTB scales with ABR in fraction mode
        assert np.allclose(resolved["life_threatening_bleeding_rate"], [22.0 * 0.05, 15.0 * 0.03])

    def test_resolve_samples_absolute_mode_uses_sampled_rate(self):
        samples = {
            "bleeding_rate": np.array([22.0, 15.0]),
            "joint_bleeding_fraction": np.array([0.3, 0.25]),
            "life_threatening_bleeding_rate": np.array([0.0074, 0.011]),
        }
        resolved = ParameterResolver.resolve_samples(samples, ltb_mode="absolute")
        # ABR is split only between spontaneous and joint bleeding
        assert np.allclose(resolved["spontaneous_bleeding_rate"], [22.0 * (1 - 0.3), 15.0 * (1 - 0.25)])
        assert np.allclose(resolved["joint_bleeding_rate"], [22.0 * 0.3, 15.0 * 0.25])
        # LTB is independent of ABR in absolute mode
        assert np.allclose(
            resolved["life_threatening_bleeding_rate"], [0.0074, 0.011]
        )

    def test_resolve_samples_invalid_mode_raises(self):
        samples = {"bleeding_rate": np.array([22.0]), "joint_bleeding_fraction": np.array([0.3])}
        with pytest.raises(ValueError, match="Unknown ltb_mode"):
            ParameterResolver.resolve_samples(samples, ltb_mode="bogus")

    def test_build_single_returns_model_input(self):
        from app.domain.inputs import ModelInput
        resolved = {
            "cycles": np.array([520]),
            "bleeding_rate": np.array([22.0]),
            "spontaneous_bleeding_rate": np.array([14.3]),
            "joint_bleeding_rate": np.array([6.6]),
            "life_threatening_bleeding_rate": np.array([1.1]),
            "ltb_case_fatality": np.array([0.35]),
            "baseline_age": np.array([2]),
            "weight_factor": np.array([1.0]),
            "benefits_discount_rate": np.array([0.0]),
            "costs_discount_rate": np.array([0.0]),
            "healthy_utility": np.array([1.0]),
            "mild_arthropathy_utility": np.array([0.9]),
            "moderate_arthropathy_utility": np.array([0.7]),
            "severe_arthropathy_utility": np.array([0.5]),
            "spontaneous_bleeding_utility": np.array([0.6]),
            "joint_bleeding_utility": np.array([0.4]),
            "life_threatening_bleeding_utility": np.array([0.3]),
            "death_utility": np.array([0.0]),
            "per_unit_price": np.array([1000.0]),
            "prophylaxis_background_factor_consumption_per_kg": np.array([0.0]),
            "factor_consumption_per_spontaneous_bleeding_per_kg": np.array([10.0]),
            "factor_consumption_per_joint_bleeding_per_kg": np.array([20.0]),
            "factor_consumption_per_life_threatening_bleeding_per_kg": np.array([50.0]),
        }
        inp = ParameterResolver.build_single(resolved, 0)
        assert isinstance(inp, ModelInput)
        assert inp.bleeding_rate == 22.0


class TestBayesianConvergence:
    """Tests for Bayesian.convergence_diagnostics() with arviz DataTree."""

    def test_diagnostics_returns_correct_keys(self):
        studies = [
            StudyEstimate(mean=20.0, sd=5.0, size=100),
            StudyEstimate(mean=25.0, sd=6.0, size=80),
            StudyEstimate(mean=18.0, sd=4.0, size=120),
        ]
        bayes = Bayesian(studies=studies)
        bayes.configure_mcmc(draws=500, tune=500, chains=2, cores=1, random_seed=42)
        diag = bayes.convergence_diagnostics()
        assert isinstance(diag, dict)
        assert "r_hat" in diag
        assert "ess" in diag
        assert "divergences" in diag
        assert "converged" in diag
        assert isinstance(diag["r_hat"], float)
        assert isinstance(diag["ess"], int)
        assert isinstance(diag["divergences"], int)
        assert isinstance(diag["converged"], bool)
        assert diag["r_hat"] >= 0
        assert diag["ess"] >= 0
