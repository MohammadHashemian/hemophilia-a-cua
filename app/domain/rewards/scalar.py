from scipy.stats import poisson

from app.domain.enums import HealthStates, Regime
from utils.math import build_zero_truncated_poisson_probs, cal_body_weight


def event_count(step: int, state: str, **kwargs) -> int:
    if state in {"intracranial_hemorrhage", "non_ich_major_bleeding"}:
        return 1

    if state not in ["bleeding", "hemarthrosis"]:
        return 0

    const = kwargs["const"]
    rng = kwargs["rng"]

    lam = const["lam_bleed"] if state == "bleeding" else const["lam_joint"]

    if lam <= 0.0:
        raise ValueError("not an acceptable lam value")

    # k_max = max(8, int(lam + 4.0 * np.sqrt(lam)) + 1)
    k_max = int(poisson.ppf(0.9999, lam))

    k_values, probs = build_zero_truncated_poisson_probs(lam, k_max)

    return int(rng.choice(k_values, p=probs))


def weight(step: int, state: str, **kwargs) -> float:
    const = kwargs["const"]
    inputs = kwargs["inputs"]

    return cal_body_weight(
        week=step,
        b=const["baseline_age_weeks"],
        weight_factor=inputs.weight_factor,
    )


def consumption(step: int, state: str, **kwargs) -> float:
    if state == "death":
        return 0.0

    inputs = kwargs["inputs"]
    regime = kwargs["regime"]
    weight_val = kwargs["weight"]

    dose = 0.0

    if regime == Regime.PROPHYLAXIS:
        dose += weight_val * inputs.prophylaxis_background_factor_consumption_per_kg

    k = kwargs.get("event_count", 0)

    if state == "bleeding":
        dose += weight_val * inputs.factor_consumption_per_spontaneous_bleeding_per_kg * k

    elif state == "hemarthrosis":
        dose += weight_val * inputs.factor_consumption_per_joint_bleeding_per_kg * k

    elif state == "intracranial_hemorrhage":
        dose += weight_val * inputs.factor_consumption_per_intracranial_hemorrhage_per_kg
    elif state == "non_ich_major_bleeding":
        dose += weight_val * inputs.factor_consumption_per_non_ich_major_bleeding_per_kg

    return dose


def utility(step: int, state: str, **kwargs) -> float:
    const = kwargs["const"]
    score = kwargs["pettersson_score"]

    utilities = const["utilities"]

    thresholds = const.get("utility_thresholds")
    if thresholds is None:
        # Backward-compatible adapter for callers using the former three-band
        # test fixture. Production workers always supply the JSON thresholds.
        thresholds = type(
            "UtilityThresholds",
            (),
            {
                "early_arthropathy": const["threshold_mild"],
                "moderate_arthropathy": const["threshold_moderate"],
                "severe_arthropathy": const["threshold_max"],
                "advanced_arthropathy": const["threshold_max"],
                "end_stage_arthropathy": const["threshold_max"],
            },
        )()
    if score < thresholds.early_arthropathy:
        arth = utilities.healthy
    elif score < thresholds.moderate_arthropathy:
        arth = utilities.mild_arthropathy
    elif score < thresholds.severe_arthropathy:
        arth = utilities.moderate_arthropathy
    elif score < thresholds.advanced_arthropathy:
        arth = utilities.severe_arthropathy
    elif score < thresholds.end_stage_arthropathy:
        arth = getattr(utilities, "advanced_arthropathy", utilities.severe_arthropathy)
    else:
        arth = getattr(utilities, "end_stage_arthropathy", utilities.severe_arthropathy)

    if state == HealthStates.HEALTHY.value:
        u = arth
    else:
        acute = getattr(utilities, state)
        u = arth if arth < acute else acute

    weekly = u / 52

    rate = const["weekly_discount"]
    if rate == 0:
        return weekly

    return weekly / ((1 + rate) ** step)


def make_pettersson_score(factor: float):
    count = 0

    def pettersson_score(step: int, state: str, **kwargs) -> int:
        nonlocal count
        event_count = kwargs["event_count"]

        if state == "hemarthrosis":
            count += event_count

        score = count / factor

        return 79 if score > 79 else int(score)

    return pettersson_score
