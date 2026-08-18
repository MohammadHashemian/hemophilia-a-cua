"""CPU and CUDA implementations of within-cycle utility integration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from modular_models.state_transition.kernels import apply_utility_interval, qaly_until_death

ComputeBackend = Literal["cpu", "cuda"]


class RewardIntegrator(Protocol):
    """Accumulate acute utility intervals and integrate QALYs to death."""

    def begin(self, base: NDArray[np.float64]) -> None: ...

    def apply(
        self,
        candidate: NDArray[np.float64],
        starts: NDArray[np.float64],
        durations: NDArray[np.float64],
        active: NDArray[np.bool_],
    ) -> None: ...

    def finish(self, death_time: NDArray[np.float64]) -> NDArray[np.float64]: ...


class CPURewardIntegrator:
    """JIT-compiled reference implementation used by the validated CPU engine."""

    def __init__(self, n_bins: int, step_days: float) -> None:
        self.n_bins = n_bins
        self.step_days = step_days
        self.base: NDArray[np.float64] | None = None
        self.schedule: NDArray[np.float64] | None = None

    def begin(self, base: NDArray[np.float64]) -> None:
        self.base = base
        self.schedule = np.repeat(base[:, None], self.n_bins, axis=1)

    def apply(
        self,
        candidate: NDArray[np.float64],
        starts: NDArray[np.float64],
        durations: NDArray[np.float64],
        active: NDArray[np.bool_],
    ) -> None:
        if not np.any(active):
            return
        if self.base is None or self.schedule is None:
            raise RuntimeError("Reward cycle has not been initialized")
        apply_utility_interval(
            self.schedule,
            self.base,
            candidate,
            starts,
            durations,
            active,
            self.step_days,
        )

    def finish(self, death_time: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.schedule is None:
            raise RuntimeError("Reward cycle has not been initialized")
        return cast(
            NDArray[np.float64],
            qaly_until_death(self.schedule, death_time, self.step_days, 364.0),
        )


_CUDA_SOURCE = r"""
extern "C" __global__
void integrate_reward(
    const double* base,
    const double* candidates,
    const double* starts,
    const double* durations,
    const unsigned char* active,
    const double* death_time,
    double* output,
    const int patients,
    const int intervals,
    const int bins,
    const double step_days,
    const double model_days_per_year
) {
    const int patient = blockDim.x * blockIdx.x + threadIdx.x;
    if (patient >= patients) return;

    double total_utility_days = 0.0;
    for (int bin_index = 0; bin_index < bins; ++bin_index) {
        const double bin_start = bin_index * step_days;
        const double bin_end = bin_start + step_days;
        double utility = base[patient];

        for (int interval = 0; interval < intervals; ++interval) {
            const int offset = interval * patients + patient;
            if (!active[offset]) continue;
            const double start = starts[offset];
            const double end = start + durations[offset];
            const double overlap = fmin(end, bin_end) - fmax(start, bin_start);
            if (overlap <= 0.0) continue;
            const double fraction = fmin(overlap, step_days) / step_days;
            const double average = base[patient] * (1.0 - fraction)
                + candidates[offset] * fraction;
            utility = fmin(utility, average);
        }

        const double lived_overlap = fmin(death_time[patient], bin_end) - bin_start;
        if (lived_overlap > 0.0) {
            total_utility_days += utility * fmin(lived_overlap, step_days);
        }
    }
    output[patient] = total_utility_days / model_days_per_year;
}
"""


def _configure_cuda_paths() -> None:
    os.environ.setdefault(
        "CUPY_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "hemophilia-state-transition-cupy"),
    )
    if os.environ.get("CUDA_PATH") or os.name != "nt":
        return
    toolkit_root = (
        Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        / "NVIDIA GPU Computing Toolkit"
        / "CUDA"
    )
    candidates = sorted(toolkit_root.glob("v*"), reverse=True)
    if not candidates:
        return
    cuda_path = candidates[0]
    os.environ["CUDA_PATH"] = str(cuda_path)
    binary_path = str(cuda_path / "bin")
    if binary_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = binary_path + os.pathsep + os.environ.get("PATH", "")


def _cupy() -> Any:
    _configure_cuda_paths()
    import cupy as cp

    return cp


def cuda_available() -> bool:
    """Return whether a supported CUDA device is usable from this process."""
    try:
        cp = _cupy()
        return bool(cp.cuda.runtime.getDeviceCount())
    except (ImportError, OSError, RuntimeError):
        return False


def cuda_runtime_info() -> dict[str, Any]:
    """Return auditable CUDA versions and selected device metadata."""
    cp = _cupy()
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode()
    return {
        "cupy_version": cp.__version__,
        "device_id": int(device.id),
        "device_name": str(name),
        "compute_capability": f"{properties['major']}.{properties['minor']}",
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
    }


class CUDARewardIntegrator:
    """Batch all interval effects into one FP64 CUDA kernel per model cycle."""

    threads_per_block = 256

    def __init__(self, n_bins: int, step_days: float) -> None:
        if not cuda_available():
            raise RuntimeError("CUDA backend requested but no usable CUDA device was found")
        self.cp = _cupy()
        self.kernel = self.cp.RawKernel(_CUDA_SOURCE, "integrate_reward")
        self.n_bins = n_bins
        self.step_days = step_days
        self.base: NDArray[np.float64] | None = None
        self.intervals: list[
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.bool_],
            ]
        ] = []

    def begin(self, base: NDArray[np.float64]) -> None:
        self.base = base
        self.intervals.clear()

    def apply(
        self,
        candidate: NDArray[np.float64],
        starts: NDArray[np.float64],
        durations: NDArray[np.float64],
        active: NDArray[np.bool_],
    ) -> None:
        if np.any(active):
            self.intervals.append((candidate, starts, durations, active))

    def finish(self, death_time: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.base is None:
            raise RuntimeError("Reward cycle has not been initialized")
        n = self.base.size
        if self.intervals:
            candidates = np.stack([item[0] for item in self.intervals])
            starts = np.stack([item[1] for item in self.intervals])
            durations = np.stack([item[2] for item in self.intervals])
            active = np.stack([item[3] for item in self.intervals]).astype(np.uint8)
        else:
            candidates = np.empty((0, n), dtype=np.float64)
            starts = np.empty((0, n), dtype=np.float64)
            durations = np.empty((0, n), dtype=np.float64)
            active = np.empty((0, n), dtype=np.uint8)

        cp = self.cp
        base_gpu = cp.asarray(self.base)
        candidates_gpu = cp.asarray(candidates)
        starts_gpu = cp.asarray(starts)
        durations_gpu = cp.asarray(durations)
        active_gpu = cp.asarray(active)
        death_gpu = cp.asarray(death_time)
        output_gpu = cp.empty(n, dtype=cp.float64)
        blocks = (n + self.threads_per_block - 1) // self.threads_per_block
        self.kernel(
            (blocks,),
            (self.threads_per_block,),
            (
                base_gpu,
                candidates_gpu,
                starts_gpu,
                durations_gpu,
                active_gpu,
                death_gpu,
                output_gpu,
                np.int32(n),
                np.int32(len(self.intervals)),
                np.int32(self.n_bins),
                np.float64(self.step_days),
                np.float64(364.0),
            ),
        )
        return cast(NDArray[np.float64], cp.asnumpy(output_gpu))


def create_reward_integrator(
    backend: ComputeBackend,
    n_bins: int,
    step_days: float,
) -> RewardIntegrator:
    if backend == "cuda":
        return CUDARewardIntegrator(n_bins, step_days)
    return CPURewardIntegrator(n_bins, step_days)
