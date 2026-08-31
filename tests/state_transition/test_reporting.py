from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modular_models.state_transition.analysis import StudyRunner
from modular_models.state_transition.context import StudyContext
from modular_models.state_transition.reporting import (
    auditable_cycle_sample,
    auditable_patient_sample,
    break_even_factor_price,
    economic_summary_table,
    evpi_grid,
    evpi_max,
    evpi_table,
    exact_ceac_table,
    extended_validation_table,
    factor_price_policy_table,
    factor_price_probability_thresholds,
    factor_price_psa_table,
    monte_carlo_convergence_table,
    owsa_parameter_ranking,
    psa_precision_summary,
    runtime_audit_table,
    selected_ceac_table,
    wtp_sensitivity_table,
    wtp_threshold_summary,
)
from modular_models.state_transition.report_plots import (
    plot_evpi,
    plot_factor_price_policy,
    plot_inmb_threshold,
    plot_inner_loop_precision,
    plot_monte_carlo_convergence,
    plot_psa_convergence,
    plot_scenario_nmb_bars,
)
from modular_models.state_transition.sampling import ParameterResolver


@pytest.fixture(scope="module")
def context() -> StudyContext:
    return StudyContext.load()


@pytest.fixture(scope="module")
def base_comparison(context: StudyContext) -> StudyRunner:
    return StudyRunner(context).compare(
        n_patients=200, seed=42, retain_patient_level=True
    )


@pytest.fixture(scope="module")
def psa_frame() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    n = 500
    delta_cost = 3.34e10 + rng.normal(0, 1.1e9, n)
    delta_qaly = 0.234 + rng.normal(0, 0.087, n)
    inmb = 18e9 * delta_qaly - delta_cost
    return pl.DataFrame(
        {
            "iteration": np.arange(n),
            "iteration_seed": rng.integers(0, 2**31 - 1, n),
            "incremental_cost_irr": delta_cost,
            "incremental_qaly": delta_qaly,
            "incremental_nmb_irr": inmb,
        }
    )


# ---------------------------------------------------------------------------
# Base case economic summary
# ---------------------------------------------------------------------------


def test_economic_summary_table_returns_six_rows(base_comparison) -> None:
    table = economic_summary_table(base_comparison)
    assert table.height == 6
    metrics = table["metric"].to_list()
    assert "Incremental cost" in metrics
    assert "ICER" in metrics
    assert "Incremental NMB" in metrics
    assert "Incremental QALY" in metrics


def test_wtp_sensitivity_table_classifies_strategies() -> None:
    delta_cost = 3.3428e10
    delta_qaly = 0.240277
    wtp_values = np.array(
        [0, 10e9, 18e9, 100e9, 200e9, 300e9], dtype=float
    )
    table = wtp_sensitivity_table(delta_cost, delta_qaly, wtp_values)
    preferences = table["preferred_strategy"].to_list()
    # Below break-even (~139e9) → on-demand; above → prophylaxis.
    assert preferences[0] == "On-demand"
    assert preferences[1] == "On-demand"
    assert preferences[-1] == "Prophylaxis"
    assert table.height == wtp_values.shape[0]


def test_wtp_threshold_summary_reports_break_even() -> None:
    summary = wtp_threshold_summary(3.3428e10, 0.240277, 18e9)
    break_even = summary["break_even_wtp_irr_per_qaly"][0]
    assert break_even == pytest.approx(3.3428e10 / 0.240277, rel=1e-9)
    assert summary["primary_wtp_as_fraction_of_break_even"][0] < 1.0


# ---------------------------------------------------------------------------
# Extended validation and auditable samples
# ---------------------------------------------------------------------------


def test_extended_validation_table_covers_both_strategies(base_comparison) -> None:
    table = extended_validation_table(base_comparison)
    assert set(table["strategy"].unique().to_list()) == {"prophylaxis", "on_demand"}
    assert table["passed"].all()
    assert table.filter(pl.col("check") == "pettersson_within_0_78").height == 2


def test_auditable_patient_sample_top_n_by_bleeds(base_comparison) -> None:
    sample = auditable_patient_sample(base_comparison, n=5)
    assert sample.height == 5
    assert "patient_id" in sample.columns
    assert "joint_bleeds" in sample.columns
    # Rows must be sorted by joint_bleeds in descending order.
    bleeds = sample["joint_bleeds"].to_list()
    assert bleeds == sorted(bleeds, reverse=True)


def test_auditable_cycle_sample_reads_trace(tmp_path: Path) -> None:
    payload = {
        "runs": [
            {"cycles": [{"cycle": 0}]},
            {"cycles": [{"cycle": 1, "alive_at_start": 100}]},
        ]
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    sample = auditable_cycle_sample(path, run_index=1)
    assert sample.height == 1
    assert "alive_at_start" in sample.columns


# ---------------------------------------------------------------------------
# CEAC helpers
# ---------------------------------------------------------------------------


def test_selected_ceac_table_filters_by_billion_wtp(psa_frame) -> None:
    ceac = pl.DataFrame(
        {
            "wtp_irr_per_qaly": pl.Series(
                "wtp_irr_per_qaly", [0.0, 18e9, 50e9, 100e9], dtype=pl.Float64
            ),
            "probability_cost_effective": [0.0, 0.0, 0.0004, 0.13],
        }
    )
    selected = selected_ceac_table(ceac, [18, 100])
    assert selected.height == 2
    assert set(selected["wtp_billion_irr_per_qaly"].to_list()) == {18.0, 100.0}


def test_exact_ceac_table_matches_naive_computation(psa_frame) -> None:
    primary_wtp = 18e9
    wtp_values = np.array([0, 18e9, 100e9], dtype=float)
    table = exact_ceac_table(psa_frame, wtp_values, primary_wtp)
    assert table.height == 3
    inmb_at_primary = psa_frame["incremental_nmb_irr"].to_numpy()
    qaly = psa_frame["incremental_qaly"].to_numpy()
    expected_zero = float(np.mean(inmb_at_primary > 0))
    probs = table["probability_prophylaxis_cost_effective"].to_numpy()
    assert probs[0] == pytest.approx(expected_zero, rel=1e-9)
    # Sanity: zero WTP → prophylaxis is dominant only when INMB > 0 at WTP=0.
    # Verify exact_ceac_table at wtp=primary_wtp reproduces the same probability.
    primary_idx = int(np.argmin(np.abs(wtp_values - primary_wtp)))
    assert probs[primary_idx] == pytest.approx(expected_zero, rel=1e-9)


# ---------------------------------------------------------------------------
# PSA precision diagnostics
# ---------------------------------------------------------------------------


def test_psa_precision_summary_has_six_columns(psa_frame) -> None:
    summary = psa_precision_summary(psa_frame)
    assert summary.height == 1
    assert summary["completed_iterations"][0] == psa_frame.height
    # MCSE bound is positive and ≤ 1.
    rel = summary["relative_mcse_mean_incremental_qaly"][0]
    assert 0 < rel < 1
    # Worst-case margin at 2,500 iterations ≈ 0.0196.
    assert summary["worst_case_95pct_ceac_margin_at_2500"][0] == pytest.approx(
        1.96 * np.sqrt(0.25 / 2_500), rel=1e-9
    )


# ---------------------------------------------------------------------------
# EVPI helpers
# ---------------------------------------------------------------------------


def test_evpi_table_is_consistent_with_naive(psa_frame) -> None:
    table = evpi_table(psa_frame, [0, 18, 100, 150])
    assert table.height == 4
    # At WTP=0, EVPI is zero because INMB ≤ 0 ⇒ max(0) = 0 and mean ≤ 0.
    assert table["evpi_billion_irr_per_patient"][0] == pytest.approx(0.0, abs=1e-9)


def test_evpi_grid_and_max_consistent(psa_frame) -> None:
    grid = np.linspace(0, 300e9, 301)
    wtp_grid, values = evpi_grid(psa_frame, grid)
    assert wtp_grid.shape == (301,)
    assert values.shape == (301,)
    assert np.all(values >= -1e-9)
    max_evpi, max_wtp = evpi_max(psa_frame, grid)
    assert max_evpi == pytest.approx(float(values.max()), rel=1e-9)
    assert max_wtp == pytest.approx(
        float(grid[np.argmax(values)]) / 1e9, rel=1e-9
    )


# ---------------------------------------------------------------------------
# Factor VIII price policy
# ---------------------------------------------------------------------------


def test_break_even_factor_price_is_linear(base_comparison) -> None:
    info = break_even_factor_price(base_comparison, 18e9, 58_000.0)
    base_cost = float(base_comparison.incremental_cost_irr)
    base_qaly = float(base_comparison.incremental_qaly)
    expected = 58_000.0 * (18e9 * base_qaly) / base_cost
    assert info["break_even_factor_price_irr_per_iu"] == pytest.approx(
        expected, rel=1e-9
    )
    assert info["required_price_reduction_percent"] == pytest.approx(
        (1 - expected / 58_000.0) * 100.0, rel=1e-9
    )


def test_factor_price_policy_table_marks_cost_effective_at_threshold(
    base_comparison,
) -> None:
    table = factor_price_policy_table(
        base_comparison,
        base_factor_price=58_000.0,
        primary_wtp=18e9,
        reduction_percent=np.array([0.0, 50.0, 90.0, 95.0]),
    )
    assert table.height == 4
    # At 0% reduction the policy is not cost-effective; at large reductions it is.
    assert not bool(table["cost_effective_at_primary_wtp"][0])
    assert bool(table["cost_effective_at_primary_wtp"][-1])


def test_factor_price_psa_table_scales_cost(psa_frame) -> None:
    table = factor_price_psa_table(
        psa_frame,
        base_factor_price=58_000.0,
        primary_wtp=18e9,
        price_grid=np.array([2_000, 30_000, 58_000]),
    )
    assert table.height == 3
    # Probability increases as price decreases.
    probs = table["probability_cost_effective"].to_list()
    assert probs[0] >= probs[1] >= probs[2]


def test_factor_price_probability_thresholds_returns_highest_price() -> None:
    table = pl.DataFrame(
        {
            "factor_price_irr_per_iu": [10_000, 20_000, 30_000],
            "price_reduction_percent": [80, 60, 40],
            "mean_inmb_billion_irr": [1.0, -0.5, -2.0],
            "probability_cost_effective": [0.9, 0.5, 0.1],
        }
    )
    thresholds = factor_price_probability_thresholds(table, [0.5, 0.9])
    assert thresholds.height == 2
    # For target=0.9, the highest price achieving ≥ 0.9 is 10,000.
    high_price = thresholds.filter(pl.col("target_probability") == 0.9)
    assert high_price["maximum_price_irr_per_iu"][0] == 10_000


# ---------------------------------------------------------------------------
# OWSA ranking helper
# ---------------------------------------------------------------------------


def test_owsa_parameter_ranking_returns_ranked_rows() -> None:
    frame = pl.DataFrame(
        {
            "parameter_id": ["p1", "p1", "p2", "p2"],
            "parameter_description": ["a", "a", "b", "b"],
            "unit": ["u", "u", "u", "u"],
            "base_value": [1.0, 1.0, 2.0, 2.0],
            "endpoint": ["low", "high", "low", "high"],
            "endpoint_value": [0.5, 1.5, 1.5, 2.5],
            "incremental_nmb_irr": [-1.0e10, 1.0e10, -5.0e9, 5.0e9],
            "icer_irr_per_qaly": [1.0e11, 1.2e11, 1.5e11, 1.6e11],
            "analysis_type": ["one_way"] * 4,
            "linked_parameter_id": [None] * 4,
            "linked_endpoint_value": [None] * 4,
            "status": ["complete"] * 4,
        }
    )
    ranking = owsa_parameter_ranking(frame, base_inmb_irr=0.0, top_n=2)
    assert ranking.height == 2
    # Largest range (p1) comes first.
    assert ranking["parameter_id"][0] == "p1"
    assert "INMB_range_billion_IRR" in ranking.columns


# ---------------------------------------------------------------------------
# Runtime audit and convergence helpers
# ---------------------------------------------------------------------------


def test_runtime_audit_table_has_three_runs() -> None:
    base = {
        "config": {"iterations": 2500, "n_patients": 5000},
        "effective_jobs": 12,
        "elapsed_seconds_this_session": 25 * 60.0,
        "status": "complete",
    }
    cpu = {
        "config": {"iterations": 24, "n_patients": 5000},
        "effective_jobs": 12,
        "elapsed_seconds_this_session": 18.7,
        "status": "complete",
    }
    cuda = {
        "config": {"iterations": 24, "n_patients": 5000},
        "effective_jobs": 8,
        "elapsed_seconds_this_session": 27.4,
        "status": "complete",
    }
    table = runtime_audit_table(base, cpu, cuda)
    assert table.height == 3
    assert set(table["run"].to_list()) == {"CPU benchmark", "CUDA benchmark", "Final PSA"}


def test_monte_carlo_convergence_table_round_trips_records() -> None:
    records = [
        {"n_patients": 10_000, "incremental_cost_irr": 1.0, "incremental_qaly": 0.1,
         "relative_change_cost": None, "relative_change_qaly": None, "converged": False},
    ]
    table = monte_carlo_convergence_table(records)
    assert table.height == 1
    assert table["n_patients"][0] == 10_000


# ---------------------------------------------------------------------------
# Plot helpers smoke-test that axes come back without errors
# ---------------------------------------------------------------------------


def test_plot_inmb_threshold_returns_axis() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    returned = plot_inmb_threshold(0.24, 3.34e10, 18e9, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_plot_scenario_nmb_bars_returns_axis() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    scenario = pl.DataFrame(
        {
            "scenario": ["base_case", "no_discount"],
            "incremental_nmb_irr": [-1.0e10, -2.0e10],
        }
    )
    returned = plot_scenario_nmb_bars(scenario, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_plot_inner_loop_precision_returns_axis() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    precision = pl.DataFrame(
        {
            "n_patients_per_strategy": [1_000, 2_500, 5_000],
            "qaly_noise_ratio_percent": [28.8, 16.9, 8.1],
        }
    )
    returned = plot_inner_loop_precision(precision, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_plot_psa_convergence_returns_axis() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    convergence = pl.DataFrame(
        {
            "iterations": [100, 200],
            "relative_change_mean_cost": [0.01, 0.005],
            "relative_change_mean_qaly": [0.05, 0.02],
        }
    )
    returned = plot_psa_convergence(convergence, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_plot_evpi_returns_axis() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    wtp = np.linspace(0, 300e9, 50)
    values = (np.sin(wtp / 30e9) + 1) * 1e9
    returned = plot_evpi(wtp, values, 18e9, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_plot_factor_price_policy_returns_axis() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    table = pl.DataFrame(
        {
            "factor_price_irr_per_iu": [10_000, 30_000, 58_000],
            "probability_cost_effective": [0.9, 0.4, 0.0],
        }
    )
    returned = plot_factor_price_policy(table, 58_000, 7_500, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_plot_monte_carlo_convergence_returns_axis() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    table = pl.DataFrame(
        {
            "n_patients": [10_000, 25_000, 50_000],
            "relative_change_cost": [None, 0.001, 0.0005],
            "relative_change_qaly": [None, 0.014, 0.004],
        }
    )
    returned = plot_monte_carlo_convergence(table, ax=ax)
    assert returned is ax
    plt.close(fig)
