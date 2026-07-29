from dataclasses import Field

import numpy as np

from app.analysis.distributions import Constant
from app.analysis.psa.models import ParameterSet
from app.analysis.psa.parameters import Parameter
from app.domain.scenario import Scenario


def insert_scenario(scenarios: list[Scenario], pair: list[Scenario]):
    """Mutable insertion of a scenario pair into the scenarios list."""
    for scenario in pair:
        scenarios.append(scenario)


def parse_scenario(scenario: str) -> tuple[str, str, str, str | None]:
    """
    Parse scenario format:

    "<time_horizon> <regime> <sampling_method> [extension ...]"

    Examples:
        early on_demand bayesian
        early prophylaxis bayesian abr_low
        lifetime on_demand bayesian factor_consumption_high
    """

    parts = scenario.strip().split()

    if len(parts) < 3:
        raise ValueError(f"Invalid scenario format: {scenario}")

    time_horizon = parts[0]
    regime = parts[1]
    sampling_method = parts[2]

    # Everything after the 3rd token becomes extension
    extension = " ".join(parts[3:]) if len(parts) > 3 else None

    # Normalize regime naming
    regime = regime.replace("_", "-")

    return time_horizon, regime, sampling_method, extension


def _scenario_sort_key(scenario: str) -> tuple[str, str, str, str]:
    time_horizon, regime, sampling_method, extension = parse_scenario(scenario)
    return time_horizon, sampling_method, extension or "", regime


def pair_scenarios(scenarios: list[str]) -> list[tuple[str, str]]:
    """
    Pair:
        on_demand <-> prophylaxis

    while preserving:
        - time horizon
        - sampling method
        - extension
    """

    grouped = {}

    for scenario in sorted(scenarios, key=_scenario_sort_key):
        (
            time_horizon,
            regime,
            sampling_method,
            extension,
        ) = parse_scenario(scenario)

        # IMPORTANT:
        # include extension in grouping key
        key = (
            time_horizon,
            sampling_method,
            extension,
        )

        if key not in grouped:
            grouped[key] = {}

        grouped[key][regime] = scenario

    paired_scenarios = []

    # Base scenarios use ``None`` for extension while extended scenarios use
    # strings. Python cannot directly order None and str, so normalize the
    # extension solely for sorting while retaining the original grouping key.
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2] or "")):
        regimes = grouped[key]
        if "on-demand" not in regimes:
            raise ValueError(f"Missing on-demand scenario for group: {key}")

        if "prophylaxis" not in regimes:
            raise ValueError(f"Missing prophylaxis scenario for group: {key}")

        paired_scenarios.append(
            (
                regimes["on-demand"],
                regimes["prophylaxis"],
            )
        )

    return paired_scenarios


def extend_scenario(
    scenario: Scenario,
    extension_name: str,
    parameter_name: str,
    distribution,
) -> Scenario:
    extended = scenario.model_copy(deep=True)
    extended.name += f" {extension_name}"
    if extended.overrides is None:
        extended.overrides = {}
    extended.overrides[parameter_name] = Parameter(distribution=distribution)
    return extended


def define_scenario_extension(
    scenarios: list[Scenario],
    extensions: dict[str, dict[str, Constant]],
) -> list[Scenario]:
    extended_scenarios = []

    for scenario in scenarios:
        for extension_name, extension in extensions.items():

            extended = scenario.model_copy(deep=True)
            extended.name += f" {extension_name}"

            if extended.overrides is None:
                extended.overrides = {}

            for parameter_name, distribution in extension.items():
                extended.overrides[parameter_name] = Parameter(
                    distribution=distribution
                )

            extended_scenarios.append(extended)

    return extended_scenarios


def split_tornado_extensions(pairs: list[tuple[str, str]]):
    """Splits the scenario pairs into two lists: one for low extensions and one for high extensions."""

    lows, highs = [], []
    for p in pairs:
        intervention = p[0]
        control = p[1]

        flag = intervention.split(sep=" ")[3].split(sep="_")[-1:]
        if flag[0] == "low":
            lows.append((intervention, control))
        elif flag[0] == "high":
            highs.append((intervention, control))

    return lows, highs


def get_tornado_ranges(
    base_scenario: ParameterSet, owsa_params: list, sample_size: int, seed: int
):
    out = {}
    for key, field in base_scenario.__dataclass_fields__.items():
        if isinstance(field, Field):
            if field.name in owsa_params:
                attr = base_scenario.__getattribute__(key)
                samples = attr.sample(n=sample_size, rng=np.random.default_rng(seed))
                out.update(
                    {
                        field.name: {
                            "mean": np.mean(samples),
                            "low": np.percentile(samples, 5),
                            "high": np.percentile(samples, 95),
                        }
                    }
                )
    return out


def get_parameter_label(extension: str | None) -> str:
    if not extension:
        return "base"
    if extension.endswith("_low") or extension.endswith("_high"):
        return extension.rsplit("_", 1)[0]
    return extension


def get_base_pair_key(scenario: str) -> tuple[str, str]:
    time_horizon, regime, sampling_method, _ = parse_scenario(scenario)
    return time_horizon, sampling_method


def ltb_mode_for_scenario(scenario: str) -> str:
    """Resolve the LTB specification mode for a scenario name.

    The thesis base case models life-threatening bleeding as a sampled
    fraction of ABR. Scenarios whose extension contains the explicit token
    ``ltb_absolute`` instead use evidence-based absolute annual incidence.
    """
    return "absolute" if "ltb_absolute" in scenario else "fraction"
