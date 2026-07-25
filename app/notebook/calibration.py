import numpy as np
import polars as pl
from scipy.stats import bootstrap, ks_2samp, pearsonr

# Calibration Diagnostics


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else np.nan


def _bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval.
    """

    values = np.asarray(values)

    if len(values) < 2:
        return (np.nan, np.nan)

    result = bootstrap(
        (values,),
        statistic,
        confidence_level=confidence_level,
        method="percentile",
        random_state=42,  # type: ignore
    )

    return (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def classify_calibration(
    oe_ratio: float,
    relative_error: float,
    distribution_similarity_index: float,
) -> str:
    """
    Qualitative calibration classification.
    """

    abs_rel_error = abs(relative_error)

    if (
        0.95 <= oe_ratio <= 1.05
        and abs_rel_error <= 0.05
        and distribution_similarity_index >= 0.90
    ):
        return "Excellent"

    if (
        0.85 <= oe_ratio <= 1.15
        and abs_rel_error <= 0.15
        and distribution_similarity_index >= 0.80
    ):
        return "Acceptable"

    return "Needs Investigation"


def build_calibration_report(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Build a per-(time_horizon, sampling_method, regime) calibration
    report from a long-form polars DataFrame of simulation results.
    """

    reports = []

    group_cols = [
        "time_horizon",
        "sampling_method",
        "regime",
    ]

    grouped = df.group_by(group_cols)

    for keys, sub in grouped:

        if sub.is_empty():
            continue

        sampled_abr = sub["sampled_abr"].to_numpy()
        annual_bleeding_rate = sub["annual_bleeding_rate"].to_numpy()

        # Input expectation
        expected_abr = float(sampled_abr.mean())
        expected_ci_low, expected_ci_high = _bootstrap_ci(sampled_abr)

        # Realized cohort ABR
        total_bleeds = float(sub["bleeding_events"].sum())
        total_person_years = float(sub["person_years"].sum())

        realized_abr = _safe_div(
            total_bleeds,
            total_person_years,
        )

        realized_ci_low, realized_ci_high = _bootstrap_ci(annual_bleeding_rate)

        # Calibration statistics

        bias = realized_abr - expected_abr
        relative_error = _safe_div(
            bias,
            expected_abr,
        )
        oe_ratio = _safe_div(
            realized_abr,
            expected_abr,
        )

        # Distribution similarity
        ks_stat, ks_p = ks_2samp(
            sampled_abr,
            annual_bleeding_rate,
        )

        distribution_similarity_index = 1 - float(ks_stat)  # type: ignore

        # Correlation

        try:
            pearson_r, pearson_p = pearsonr(
                sampled_abr,
                annual_bleeding_rate,
            )

        except Exception:

            pearson_r = np.nan
            pearson_p = np.nan

        # Prediction error
        from sklearn.metrics import mean_squared_error

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    sampled_abr,
                    annual_bleeding_rate,
                )
            )
        )

        # Survival diagnostics
        mortality_rate = float(sub["is_absorbed"].mean())
        mean_person_years = float(sub["person_years"].mean())

        # Calibration quality
        calibration_status = classify_calibration(
            oe_ratio=oe_ratio,
            relative_error=relative_error,
            distribution_similarity_index=distribution_similarity_index,
        )

        reports.append(
            {
                # Scenario
                "time_horizon": keys[0],
                "sampling_method": keys[1],
                "regime": keys[2],
                # Sample size
                "n_patients": sub.height,
                # Expected
                "expected_abr": expected_abr,
                "expected_abr_ci_low": expected_ci_low,
                "expected_abr_ci_high": expected_ci_high,
                # Realized
                "realized_abr": realized_abr,
                "realized_abr_ci_low": realized_ci_low,
                "realized_abr_ci_high": realized_ci_high,
                # Calibration
                "bias": bias,
                "relative_error": relative_error,
                "oe_ratio": oe_ratio,
                # Similarity
                "distribution_similarity_index": distribution_similarity_index,
                "ks_p_value": ks_p,
                # Correlation
                "pearson_r": pearson_r,
                "pearson_p_value": pearson_p,
                # Error
                "rmse": rmse,
                # Survival
                "mortality_rate": mortality_rate,
                "mean_person_years": mean_person_years,
                # Cohort totals
                "total_bleeds": total_bleeds,
                "total_person_years": total_person_years,
                # Overall interpretation
                "calibration_status": calibration_status,
            }
        )

    report_df = pl.DataFrame(reports)

    return report_df.sort(
        [
            "time_horizon",
            "sampling_method",
            "regime",
        ]
    )
