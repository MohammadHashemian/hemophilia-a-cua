"""JIT-compiled numerical kernels used inside the weekly simulation loop."""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def apply_utility_interval(
    schedule: NDArray[np.float64],
    base: NDArray[np.float64],
    candidate: NDArray[np.float64],
    starts: NDArray[np.float64],
    durations: NDArray[np.float64],
    active: NDArray[np.bool_],
    step_days: float,
) -> None:
    """Apply the lower interval-average utility without temporary 2-D arrays."""
    for patient in range(schedule.shape[0]):
        if not active[patient]:
            continue
        start = starts[patient]
        end = start + durations[patient]
        for bin_index in range(schedule.shape[1]):
            bin_start = bin_index * step_days
            bin_end = bin_start + step_days
            overlap = min(end, bin_end) - max(start, bin_start)
            if overlap <= 0.0:
                continue
            fraction = min(overlap, step_days) / step_days
            average = base[patient] * (1.0 - fraction)
            average += candidate[patient] * fraction
            if average < schedule[patient, bin_index]:
                schedule[patient, bin_index] = average


@njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def qaly_until_death(
    schedule: NDArray[np.float64],
    death_time: NDArray[np.float64],
    step_days: float,
    model_days_per_year: float,
) -> NDArray[np.float64]:
    """Integrate the within-cycle schedule only over time lived."""
    output = np.zeros(schedule.shape[0], dtype=np.float64)
    for patient in range(schedule.shape[0]):
        lived = death_time[patient]
        total_utility_days = 0.0
        for bin_index in range(schedule.shape[1]):
            bin_start = bin_index * step_days
            overlap = min(lived, bin_start + step_days) - bin_start
            if overlap > 0.0:
                total_utility_days += schedule[patient, bin_index] * min(overlap, step_days)
        output[patient] = total_utility_days / model_days_per_year
    return output


def warm_jit_kernels() -> None:
    """Compile kernels once before worker processes are created."""
    schedule = np.ones((1, 7), dtype=np.float64)
    values = np.ones(1, dtype=np.float64)
    apply_utility_interval(
        schedule,
        values,
        values,
        np.zeros(1, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        np.ones(1, dtype=np.bool_),
        1.0,
    )
    qaly_until_death(schedule, np.full(1, 7.0, dtype=np.float64), 1.0, 364.0)
