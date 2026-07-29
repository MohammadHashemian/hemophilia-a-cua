"""Validation of the evidence-based absolute LTB incidence scenario.

Zwagemaker et al. (Blood, 2021; DOI 10.1182/blood.2021011849) reported
pooled intracranial-haemorrhage incidence of 7.4 per 1,000 person-years
in children and young adults, with a 95% CI of 4.9 to 11.1.

The structural sensitivity scenario represents those values as a
triangular distribution in events per person-year:

    Triangular(0.0049, 0.0074, 0.0111)

The published point estimate is the *mode*, not the mean, of that
distribution. Its theoretical mean is 0.0078 per person-year.
"""

import numpy as np

from app.analysis.distributions import TriangularDist


LEFT = 0.0049
PUBLISHED_POINT_ESTIMATE = 0.0074
RIGHT = 0.0111
N_DRAWS = 1_000_000
SEED = 468498


def test_absolute_ltb_triangular_psa_matches_its_specification() -> None:
    """Large PSA sample should reconstruct the chosen triangular input."""
    distribution = TriangularDist(
        left=LEFT,
        mode=PUBLISHED_POINT_ESTIMATE,
        right=RIGHT,
    )
    draws = distribution.sample(N_DRAWS, np.random.default_rng(SEED))

    expected_mean = (LEFT + PUBLISHED_POINT_ESTIMATE + RIGHT) / 3

    assert np.isclose(distribution.point(), expected_mean)
    assert np.isclose(draws.mean(), expected_mean, atol=1e-5)
    assert draws.min() >= LEFT
    assert draws.max() <= RIGHT


def test_current_triangle_is_close_but_not_mean_equivalent_to_zwagemaker() -> None:
    """Document the exact interpretation of the current scenario.

    The current triangle is evidence-anchored because 0.0074 is its mode,
    but its expected PSA incidence is 0.0078, not 0.0074.
    """
    expected_psa_incidence = (
        LEFT + PUBLISHED_POINT_ESTIMATE + RIGHT
    ) / 3
    relative_difference = (
        expected_psa_incidence - PUBLISHED_POINT_ESTIMATE
    ) / PUBLISHED_POINT_ESTIMATE

    assert np.isclose(expected_psa_incidence, 0.0078)
    assert np.isclose(relative_difference, 0.05405405405405406)
    assert not np.isclose(
        expected_psa_incidence,
        PUBLISHED_POINT_ESTIMATE,
        rtol=0.01,
    )


def test_weekly_conversion_reconstructs_expected_annual_incidence() -> None:
    """Simulated weekly hazards should reproduce the PSA-expected rate.

    The model converts each sampled annual Poisson rate ``lambda`` to a
    weekly event probability ``1-exp(-lambda/52)``. Recombining 52 weekly
    probabilities gives ``1-exp(-lambda)``, the annual probability of at
    least one event. A seeded Bernoulli simulation should agree with that
    expected annual probability.
    """
    rng = np.random.default_rng(SEED)
    distribution = TriangularDist(
        left=LEFT,
        mode=PUBLISHED_POINT_ESTIMATE,
        right=RIGHT,
    )
    annual_rates = distribution.sample(N_DRAWS, rng)

    weekly_probabilities = 1.0 - np.exp(-annual_rates / 52.0)
    annual_probabilities = 1.0 - (1.0 - weekly_probabilities) ** 52
    simulated_patient_years_with_event = rng.binomial(
        n=1, p=annual_probabilities
    )

    expected_annual_probability = np.mean(1.0 - np.exp(-annual_rates))
    observed_annual_probability = simulated_patient_years_with_event.mean()

    assert np.allclose(
        annual_probabilities,
        1.0 - np.exp(-annual_rates),
        rtol=0,
        atol=1e-14,
    )
    assert np.isclose(
        observed_annual_probability,
        expected_annual_probability,
        atol=3e-4,
    )


def test_mode_required_for_mean_equivalence_is_0_0062() -> None:
    """Show how a mean-centred triangle would be parameterised.

    For a triangular distribution, mean = (left + mode + right) / 3.
    Holding the published confidence limits fixed, a mean of 0.0074
    requires a mode of 0.0062.
    """
    mean_equivalent_mode = (
        3 * PUBLISHED_POINT_ESTIMATE - LEFT - RIGHT
    )

    assert np.isclose(mean_equivalent_mode, 0.0062)
    assert np.isclose(
        (LEFT + mean_equivalent_mode + RIGHT) / 3,
        PUBLISHED_POINT_ESTIMATE,
    )
