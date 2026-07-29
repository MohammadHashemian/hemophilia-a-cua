from app.domain.enums import ArthropathySeverity


def pettersson_to_severity(
    score: int,
    threshold_mild: int = 4,
    threshold_moderate: int = 27,
    threshold_max: int = 79,
) -> ArthropathySeverity:
    """Map a Pettersson score to an arthropathy severity band.

    Bands follow the same upper-bound thresholds used by the utility
    reward function (and ``clinical.json``):
        score < mild          -> healthy (no clinically relevant arthropathy)
        mild <= score < moderate -> mild arthropathy
        moderate <= score < max  -> moderate arthropathy
        score >= max          -> severe arthropathy
    """
    if not (0 <= score <= 79):
        raise ValueError("Pettersson score must be in range [0, 79]")

    if score < threshold_mild:
        return ArthropathySeverity.HEALTHY
    elif score < threshold_moderate:
        return ArthropathySeverity.MILD
    elif score < threshold_max:
        return ArthropathySeverity.MODERATE
    else:
        return ArthropathySeverity.SEVERE
