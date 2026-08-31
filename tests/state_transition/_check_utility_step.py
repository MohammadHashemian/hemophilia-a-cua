"""Standalone check: does utility_integration_step_days matter at the base case?

Compare:
  step = 1.0 day  (default)
  step = 0.25 day (finer)

Same seed, same 50,000 patients. If the four incremental metrics
(ΔCost, ΔQALY, ICER, INMB) are within a small relative tolerance, the
1-day integration is adequate and PSA can keep the coarser step.
"""

from __future__ import annotations

import time

from modular_models.state_transition.analysis import StudyRunner
from modular_models.state_transition.context import StudyContext

PRODUCTION_PATIENTS = 50_000
MASTER_SEED = 20_260_813
WTP = 18_000_000_000


def _run(runner: StudyRunner, step: float) -> tuple[float, float, float, float]:
    comparison = runner.compare(
        n_patients=PRODUCTION_PATIENTS,
        seed=MASTER_SEED,
        overrides={"utility_integration_step_days": step},
    )
    return (
        comparison.incremental_cost_irr,
        comparison.incremental_qaly,
        comparison.icer_irr_per_qaly,
        comparison.incremental_nmb_irr,
    )


def main() -> None:
    context = StudyContext.load()
    runner = StudyRunner(context)

    started = time.perf_counter()
    coarse = _run(runner, 1.0)
    coarse_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    fine = _run(runner, 0.25)
    fine_elapsed = time.perf_counter() - started

    labels = ["delta Cost (IRR)", "delta QALY", "ICER (IRR/QALY)", "INMB (IRR)"]
    print()
    print(f"{'metric':<22}{'step=1.0 day':>22}{'step=0.25 day':>22}{'rel diff':>14}")
    print("-" * 80)
    for label, c, f in zip(labels, coarse, fine):
        denom = max(abs(f), 1e-12)
        rel = abs(c - f) / denom
        print(f"{label:<22}{c:>22.6e}{f:>22.6e}{rel:>14.3e}")
    print("-" * 80)
    print(f"coarse runtime: {coarse_elapsed / 60:.2f} min")
    print(f"fine   runtime: {fine_elapsed / 60:.2f} min")


if __name__ == "__main__":
    main()
