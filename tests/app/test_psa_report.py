import numpy as np
import polars as pl

from app.notebook.psa.economics import economic_summary, paired_outcomes


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
