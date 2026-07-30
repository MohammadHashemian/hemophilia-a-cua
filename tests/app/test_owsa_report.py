import matplotlib
import polars as pl

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt

from app.notebook.owsa.analysis import base_case, sensitivity_summary
from app.notebook.owsa.plots import icer_tornado, nmb_tornado
from app.notebook.owsa.scenarios import OWSARange
from app.notebook.owsa.workflow import DEFAULT_OWSA_REPLICATIONS


def _owsa_frame() -> pl.DataFrame:
    scenarios = {
        "childhood on-demand bayesian": ([100, 110], [1.0, 1.1]),
        "childhood prophylaxis bayesian": ([160, 170], [1.2, 1.3]),
        "childhood on-demand bayesian test_input_low": ([100, 110], [1.0, 1.1]),
        "childhood prophylaxis bayesian test_input_low": (
            [150, 160],
            [1.22, 1.32],
        ),
        "childhood on-demand bayesian test_input_high": ([100, 110], [1.0, 1.1]),
        "childhood prophylaxis bayesian test_input_high": (
            [180, 190],
            [1.18, 1.28],
        ),
    }
    rows = []
    for scenario, (costs, qalys) in scenarios.items():
        for iteration_id, (cost, qaly) in enumerate(zip(costs, qalys, strict=True)):
            rows.append(
                {
                    "scenario": scenario,
                    "iteration_id": iteration_id,
                    "total_cost": cost,
                    "total_qaly": qaly,
                }
            )
    return pl.DataFrame(rows)


def test_owsa_uses_base_low_high_paired_economics():
    frame = _owsa_frame()
    ranges = [OWSARange("test_input", "Test input", 0.8, 1.0, 1.2)]
    base = base_case(frame, wtp=1_000).row(0, named=True)
    summary = sensitivity_summary(frame, ranges, wtp=1_000).row(0, named=True)

    assert base["paired_iterations"] == 2
    assert summary["paired_iterations"] == 2
    assert summary["low_delta_nmb"] > summary["base_delta_nmb"]
    assert summary["high_delta_nmb"] < summary["base_delta_nmb"]
    assert summary["nmb_sensitivity"] > 0


def test_owsa_tornado_figures_render():
    frame = _owsa_frame()
    ranges = [OWSARange("test_input", "Test input", 0.8, 1.0, 1.2)]
    summary = sensitivity_summary(frame, ranges, wtp=1_000)

    nmb = nmb_tornado(summary, "childhood")
    icer = icer_tornado(summary, "childhood", wtp=1_000)
    nmb.canvas.draw()
    icer.canvas.draw()
    assert len(nmb.axes) == 1
    assert len(icer.axes) == 1
    plt.close(nmb)
    plt.close(icer)


def test_owsa_has_dedicated_non_psa_replication_default():
    assert DEFAULT_OWSA_REPLICATIONS == 1_000
