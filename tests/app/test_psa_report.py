import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt

from app.notebook.psa.analysis import plot_survival
from app.notebook.psa.economics import (
    abr_threshold_analysis,
    abr_threshold_curve,
    economic_summary,
    paired_outcomes,
)
from app.notebook.psa.report_plots import health_state_distribution


def _economic_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "scenario": [
                "childhood on-demand bayesian",
                "childhood on-demand bayesian",
                "childhood on-demand bayesian",
                # Deliberately shuffled to prove pairing is by ID, not row.
                "childhood prophylaxis bayesian",
                "childhood prophylaxis bayesian",
                "childhood prophylaxis bayesian",
            ],
            "iteration_id": [0, 1, 2, 2, 0, 1],
            "total_cost": [100.0, 200.0, 300.0, 360.0, 120.0, 240.0],
            "total_qaly": [1.0, 2.0, 3.0, 3.6, 1.2, 2.4],
        }
    )


def test_paired_outcomes_join_by_iteration_id():
    paired = paired_outcomes(
        _economic_frame(),
        "childhood on-demand bayesian",
        "childhood prophylaxis bayesian",
        wtp=1_000,
    ).sort("iteration_id")
    assert paired["delta_cost"].to_list() == [20.0, 40.0, 60.0]
    assert np.allclose(paired["delta_qaly"], [0.2, 0.4, 0.6])


def test_incremental_economic_identities():
    wtp = 1_000
    paired = paired_outcomes(
        _economic_frame(),
        "childhood on-demand bayesian",
        "childhood prophylaxis bayesian",
        wtp=wtp,
    )
    summary = economic_summary(_economic_frame(), wtp=wtp).row(0, named=True)

    assert np.isclose(summary["delta_cost"], paired["delta_cost"].mean())
    assert np.isclose(summary["delta_qaly"], paired["delta_qaly"].mean())
    assert np.isclose(
        summary["delta_nmb"],
        wtp * summary["delta_qaly"] - summary["delta_cost"],
    )
    assert np.isclose(
        summary["icer"],
        summary["delta_cost"] / summary["delta_qaly"],
    )
    assert summary["paired_iterations"] == 3


def test_abr_threshold_curve_preserves_iteration_pairing():
    frame = _economic_frame().with_columns(
        pl.when(pl.col("scenario").str.contains("on-demand"))
        .then(pl.Series([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))
        .otherwise(pl.Series([1.5, 0.5, 1.0, 0.0, 0.0, 0.0]))
        .alias("sampled_abr")
    )
    curve = abr_threshold_curve(
        frame,
        "childhood on-demand bayesian",
        "childhood prophylaxis bayesian",
        wtp=1_000,
        points=3,
        min_pairs=1,
    ).sort("abr_cutoff")

    assert curve["paired_iterations"][0] == 3
    assert curve["paired_iterations"][-1] == 1
    # At the highest cutoff only base iteration 2 and its matching comparison
    # iteration 2 remain: delta cost=60 and delta QALY=0.6.
    assert np.isclose(curve["icer"][-1], 100.0)


def test_abr_threshold_summary_reports_nmb_crossing():
    base = "childhood on-demand bayesian"
    comparison = "childhood prophylaxis bayesian"
    frame = pl.DataFrame(
        {
            "scenario": [base] * 4 + [comparison] * 4,
            "iteration_id": [0, 1, 2, 3, 3, 1, 0, 2],
            "sampled_abr": [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            "total_cost": [0.0] * 4 + [90.0, 90.0, 90.0, 90.0],
            "total_qaly": [0.0] * 4 + [0.12, 0.08, 0.04, 0.10],
        }
    )
    _, summary = abr_threshold_analysis(
        frame,
        wtp=1_000,
        points=4,
        min_pairs=1,
    )
    row = summary.row(0, named=True)

    assert row["threshold_found"]
    assert row["observed_cutoff_min"] <= row["cost_effective_abr_threshold"]
    assert row["cost_effective_abr_threshold"] <= row["observed_cutoff_max"]


def test_health_state_distribution_includes_split_major_bleeding_states():
    frame = pl.DataFrame(
        {
            "scenario": ["childhood on-demand bayesian"],
            "healthy_share": [0.70],
            "bleeding_share": [0.10],
            "hemarthrosis_share": [0.08],
            "intracranial_hemorrhage_share": [0.01],
            "non_ich_major_bleeding_share": [0.01],
            "death_share": [0.10],
        }
    )
    figure = health_state_distribution(frame, "childhood")
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert labels == [
        "No bleeding",
        "Spontaneous bleeding",
        "Hemarthrosis",
        "ICH",
        "Non-ICH major bleeding",
        "Death",
    ]
    plt.close(figure)


def test_childhood_survival_uses_clearly_labelled_adaptive_zoom():
    frame = pl.DataFrame(
        {
            "extension": [None, None, None, None],
            "regime": ["on-demand", "on-demand", "prophylaxis", "prophylaxis"],
            "cycles": [728, 728, 728, 728],
            "is_absorbed": [False, False, False, False],
            "observed_cycles": [728, 728, 728, 728],
        }
    )
    figure, axis = plot_survival(frame, "childhood")
    lower, upper = axis.get_ylim()
    assert lower > 0.0
    assert upper >= 1.0
    assert axis.get_ylabel() == "Proportion alive (zoomed)"
    assert any("does not start at 0" in text.get_text() for text in axis.texts)
    assert all("final survival" in line.get_label() for line in axis.lines)
    plt.close(figure)
