"""Reusable descriptive and clinical PSA tables."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import polars as pl

from app.notebook.calibration import build_calibration_report
from app.notebook.psa.economics import economic_summary, scenario_pairs


def mean_sd(series: pd.Series, digits: int = 2) -> str:
    return f"{series.mean():.{digits}f} ± {series.std():.{digits}f}"


def median_iqr(series: pd.Series, digits: int = 2) -> str:
    return (
        f"{series.median():.{digits}f} "
        f"[{series.quantile(0.25):.{digits}f}, "
        f"{series.quantile(0.75):.{digits}f}]"
    )


def percent(series: pd.Series, digits: int = 2) -> float:
    return round(float(series.mean() * 100), digits)


def grouped_callables(
    df: pl.DataFrame,
    *,
    by: str,
    aggregations: dict[str, tuple[str, str | Callable]],
) -> pl.DataFrame:
    pdf = df.to_pandas()
    result = pdf.groupby(by, as_index=False).agg(
        **{
            output: pd.NamedAgg(column=source, aggfunc=function)
            for output, (source, function) in aggregations.items()
        }
    )
    return pl.from_pandas(result)


def clinical_summary(df: pl.DataFrame) -> pl.DataFrame:
    return grouped_callables(
        df,
        by="scenario",
        aggregations={
            "simulations": ("scenario", "size"),
            "qaly_mean": ("total_qaly", "mean"),
            "qaly_mean_sd": ("total_qaly", mean_sd),
            "qaly_median_iqr": ("total_qaly", median_iqr),
            "pettersson_mean_sd": ("pettersson_score", mean_sd),
            "pettersson_median_iqr": ("pettersson_score", median_iqr),
        },
    )


def bleeding_summary(df: pl.DataFrame) -> pl.DataFrame:
    return grouped_callables(
        df,
        by="scenario",
        aggregations={
            "abr_mean_sd": ("annual_bleeding_rate", mean_sd),
            "abr_median_iqr": ("annual_bleeding_rate", median_iqr),
            "sbr_mean_sd": ("spontaneous_bleeding_rate", mean_sd),
            "sbr_median_iqr": ("spontaneous_bleeding_rate", median_iqr),
            "ajbr_mean_sd": ("annual_joint_bleeding_rate", mean_sd),
            "ajbr_median_iqr": ("annual_joint_bleeding_rate", median_iqr),
            "aich_mean_sd": ("annual_intracranial_hemorrhage_rate", mean_sd),
            "aich_median_iqr": ("annual_intracranial_hemorrhage_rate", median_iqr),
            "anon_ich_mean_sd": ("annual_non_ich_major_bleeding_rate", mean_sd),
            "anon_ich_median_iqr": ("annual_non_ich_major_bleeding_rate", median_iqr),
            "total_bleeding_events": ("bleeding_events", "sum"),
            "total_joint_bleeding_events": ("joint_bleeding_events", "sum"),
            "total_person_years": ("person_years", "sum"),
        },
    ).with_columns(
        (pl.col("total_bleeding_events") / pl.col("total_person_years"))
        .round(3)
        .alias("cohort_abr"),
        (pl.col("total_joint_bleeding_events") / pl.col("total_person_years"))
        .round(3)
        .alias("cohort_ajbr"),
    )


def mortality_summary(df: pl.DataFrame) -> pl.DataFrame:
    return grouped_callables(
        df,
        by="scenario",
        aggregations={
            "person_years_mean_sd": ("person_years", mean_sd),
            "person_years_median_iqr": ("person_years", median_iqr),
            "lost_person_years_mean_sd": ("lost_person_years", mean_sd),
            "lost_person_years_median_iqr": ("lost_person_years", median_iqr),
            "lost_life_expectancy_median_iqr": (
                "lost_person_year_life_expectancy",
                median_iqr,
            ),
            "absorbed_percent": ("is_absorbed", percent),
        },
    )


def resource_summary(df: pl.DataFrame) -> pl.DataFrame:
    return grouped_callables(
        df,
        by="scenario",
        aggregations={
            "total_factor_mean_sd": ("total_factor", mean_sd),
            "total_factor_median_iqr": ("total_factor", median_iqr),
            "min_total_factor": ("total_factor", "min"),
            "max_total_factor": ("total_factor", "max"),
            "total_cost_mean": ("total_cost", "mean"),
        },
    )


def state_occupation(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("scenario")
        .agg(
            healthy=pl.col("healthy_share").mean(),
            bleeding=pl.col("bleeding_share").mean(),
            hemarthrosis=pl.col("hemarthrosis_share").mean(),
            intracranial_hemorrhage=pl.col("intracranial_hemorrhage_share").mean(),
            non_ich_major_bleeding=pl.col("non_ich_major_bleeding_share").mean(),
            death=pl.col("death_share").mean(),
        )
        .with_columns(pl.all().exclude("scenario").round(3))
        .sort("scenario")
    )


def survival_efficiency(df: pl.DataFrame) -> pl.DataFrame:
    summary = df.group_by("scenario").agg(
        simulations=pl.len(),
        absorbed_rate=pl.col("is_absorbed").mean(),
        person_years=pl.col("person_years").sum(),
        lost_person_years=pl.col("lost_person_years").sum(),
    ).with_columns(
        (
            pl.col("person_years")
            / (pl.col("person_years") + pl.col("lost_person_years"))
        ).alias("survival_efficiency")
    )
    rows = []
    for base, comparison in scenario_pairs(df):
        base_row = summary.filter(pl.col("scenario") == base).row(0, named=True)
        comp_row = summary.filter(pl.col("scenario") == comparison).row(0, named=True)
        base_eff = base_row["survival_efficiency"]
        comp_eff = comp_row["survival_efficiency"]
        ratio = comp_eff / base_eff if base_eff else float("nan")
        rows.append(
            {
                "comparison": f"{comparison} vs {base}",
                "base_survival_efficiency": base_eff,
                "comparison_survival_efficiency": comp_eff,
                "absolute_survival_gain": comp_eff - base_eff,
                "relative_survival_ratio": ratio,
                "relative_survival_gain_percent": (ratio - 1) * 100,
                "absorbed_rate_base": base_row["absorbed_rate"],
                "absorbed_rate_comparison": comp_row["absorbed_rate"],
            }
        )
    return pl.DataFrame(rows)


def reduction_tables(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    bleeding = bleeding_summary(df)
    abr_rows, factor_rows = [], []
    for base, comparison in scenario_pairs(df):
        base_abr = bleeding.filter(pl.col("scenario") == base)["cohort_abr"].item()
        comp_abr = bleeding.filter(pl.col("scenario") == comparison)["cohort_abr"].item()
        base_factor = float(df.filter(pl.col("scenario") == base)["total_factor"].mean())
        comp_factor = float(
            df.filter(pl.col("scenario") == comparison)["total_factor"].mean()
        )
        abr_rows.append(
            {
                "base_scenario": base,
                "comparison_scenario": comparison,
                "base_cohort_abr": base_abr,
                "comparison_cohort_abr": comp_abr,
                "absolute_abr_reduction": base_abr - comp_abr,
                "relative_abr_reduction": (base_abr - comp_abr) / base_abr,
            }
        )
        factor_rows.append(
            {
                "base_scenario": base,
                "comparison_scenario": comparison,
                "base_factor_mean": base_factor,
                "comparison_factor_mean": comp_factor,
                "absolute_factor_reduction": base_factor - comp_factor,
                "relative_factor_reduction": (
                    (base_factor - comp_factor) / base_factor
                ),
            }
        )
    return pl.DataFrame(abr_rows), pl.DataFrame(factor_rows)


def calibration_tables(df_base: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    report = build_calibration_report(df_base)
    absorption = (
        df_base.group_by(["regime", "is_absorbed"])
        .agg(
            sampled_abr=pl.col("sampled_abr").mean(),
            annual_bleeding_rate=pl.col("annual_bleeding_rate").mean(),
            person_years=pl.col("person_years").mean(),
        )
        .sort(["regime", "is_absorbed"])
    )
    return report, absorption


def all_tables(df: pl.DataFrame, *, wtp: float) -> dict[str, pl.DataFrame]:
    base = df.filter(pl.col("extension").is_null())
    calibration, absorption = calibration_tables(base)
    abr, factor = reduction_tables(df)
    return {
        "Calibration": calibration,
        "Absorption diagnostics": absorption,
        "Clinical outcomes": clinical_summary(df),
        "Bleeding outcomes": bleeding_summary(df),
        "Mortality and life expectancy": mortality_summary(df),
        "Resource utilization": resource_summary(df),
        "Health-state occupation": state_occupation(df),
        "Survival efficiency": survival_efficiency(df),
        "ABR reduction": abr,
        "Factor-consumption reduction": factor,
        "ICER and NMB": economic_summary(df, wtp=wtp),
    }
