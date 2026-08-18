from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from modular_models.state_transition.context import StudyContext
from modular_models.state_transition.schema import DistributionSpec


def sample_distribution(
    spec: DistributionSpec | None,
    base_value: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if spec is None or spec.distribution == "fixed":
        return np.full(size, base_value, dtype=np.float64)
    p = spec.parameters
    if spec.distribution == "uniform":
        return rng.uniform(p["low"], p["high"], size)
    if spec.distribution == "beta":
        return rng.beta(p["alpha"], p["beta"], size)
    if spec.distribution == "gamma":
        return rng.gamma(p["shape"], p["scale"], size)
    if spec.distribution == "triangular":
        return rng.triangular(p["low"], p["mode"], p["high"], size)
    if spec.distribution == "beta_pert":
        minimum, mode, maximum = p["minimum"], p["mode"], p["maximum"]
        if maximum == minimum:
            return np.full(size, minimum, dtype=np.float64)
        shape = 4.0
        alpha = 1.0 + shape * (mode - minimum) / (maximum - minimum)
        beta = 1.0 + shape * (maximum - mode) / (maximum - minimum)
        return minimum + rng.beta(alpha, beta, size) * (maximum - minimum)
    raise AssertionError(f"Unhandled distribution: {spec.distribution}")


class ParameterResolver:
    """Resolve base, scenario, OWSA and PSA values without mutating context."""

    def __init__(self, context: StudyContext) -> None:
        self.context = context

    def deterministic(
        self,
        scenario_id: str = "base_case",
        overrides: Mapping[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, str | float | int | bool]]:
        scenario = self.context.scenario(scenario_id)
        values = self.context.base_values()
        values.update(scenario.parameter_overrides)
        if overrides:
            unknown = set(overrides).difference(values)
            if unknown:
                raise KeyError(f"Unknown parameter overrides: {sorted(unknown)}")
            values.update(overrides)
        return values, dict(scenario.options)

    def probabilistic(
        self,
        n: int,
        seed: int,
        scenario_id: str = "base_case",
    ) -> tuple[dict[str, np.ndarray], dict[str, str | float | int | bool]]:
        if n <= 0:
            raise ValueError("n must be positive")
        scenario = self.context.scenario(scenario_id)
        rng = np.random.default_rng(seed)
        samples = {
            key: sample_distribution(parameter.psa, parameter.value, n, rng)
            for key, parameter in self.context.parameters.items()
        }
        for key, value in scenario.parameter_overrides.items():
            samples[key] = np.full(n, value, dtype=np.float64)
        self._derive_chronic_utilities(samples)
        self._enforce_rate_consistency(samples, dict(scenario.options), rng)
        return samples, dict(scenario.options)

    @staticmethod
    def _derive_chronic_utilities(samples: dict[str, np.ndarray]) -> None:
        anchor = samples["utility_anchor"]
        reference = (
            samples["fischer_sf6d_0_4"] * 33.0 + samples["fischer_sf6d_5_12"] * 35.0
        ) / 68.0
        reference = np.maximum(reference, 1e-12)
        mild = anchor * samples["fischer_sf6d_13_21"] / reference
        moderate = anchor * samples["fischer_sf6d_22_39"] / reference
        severe = anchor * samples["fischer_sf6d_40_78"] / reference
        # The event-partition constraint is enforced for every parameter draw.
        samples["utility_mild"] = np.minimum(anchor, mild)
        samples["utility_moderate"] = np.minimum(samples["utility_mild"], moderate)
        samples["utility_severe"] = np.minimum(samples["utility_moderate"], severe)

    def _enforce_rate_consistency(
        self,
        samples: dict[str, np.ndarray],
        options: dict[str, str | float | int | bool],
        rng: np.random.Generator,
    ) -> None:
        for suffix in ("prophylaxis", "on_demand"):
            for _ in range(100):
                abr = samples[f"abr_{suffix}"]
                joint = (
                    abr * samples["joint_bleed_fraction"]
                    if options.get("joint_rate_method", "direct") == "fraction"
                    else samples[f"ajbr_{suffix}"]
                )
                ich = (
                    abr * samples["ich_fraction"]
                    if options.get("ich_rate_method", "direct") == "fraction"
                    else samples[f"ich_rate_{suffix}"]
                )
                major = abr * samples["non_ich_major_fraction"]
                invalid = abr - joint - ich - major < 0.0
                if not np.any(invalid):
                    break
                for key in (f"abr_{suffix}", f"ajbr_{suffix}", f"ich_rate_{suffix}"):
                    parameter = self.context.parameter(key)
                    samples[key][invalid] = sample_distribution(
                        parameter.psa,
                        parameter.value,
                        int(invalid.sum()),
                        rng,
                    )
            else:
                raise RuntimeError(
                    "Unable to draw internally consistent event rates for "
                    f"{suffix} after 100 attempts"
                )
