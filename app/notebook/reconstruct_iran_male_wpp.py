"""One-off script: reconstruct a complete Iran Male WPP mortality CSV
from the partial UN Data Portal export.

The user re-downloaded the WPP file but it only contains Male data for
age 0 (the portal got confused by mixing indicators). However, the
download DOES contain the full single-year (IndicatorId 80) data for
Both sexes and Female, plus the Iran_<year>.csv has the population by
sex. Because Both sexes = pop-weighted average of Male and Female, we
can solve for the Male rate per age:

    rate_M(age) = (rate_Both(age) * (pop_M + pop_F) - pop_F(age) * rate_F(age)) / pop_M(age)

The result is the same numbers the UN would have published, recovered
algebraically. Then we aggregate the 101 single-year ages into the 22
abridged 5-year buckets (IndicatorId 79) and write a clean WPP-shaped
CSV that the calculator can consume directly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
WPP_CSV = ROOT / "data" / "raw" / "population-un-data-portal-iran.csv"
POP_CSV = ROOT / "data" / "raw" / "Iran_2024.csv"
OUT_CSV = ROOT / "data" / "raw" / "population-un-data-portal-iran.csv"  # overwrite
YEAR = 2024

# Abridged 5-year buckets (AgeStart -> AgeEnd and human label)
ABRIDGED_BUCKETS = [
    (0, 1, "0"),
    (1, 5, "1-4"),
    (5, 10, "5-9"),
    (10, 15, "10-14"),
    (15, 20, "15-19"),
    (20, 25, "20-24"),
    (25, 30, "25-29"),
    (30, 35, "30-34"),
    (35, 40, "35-39"),
    (40, 45, "40-44"),
    (45, 50, "45-49"),
    (50, 55, "50-54"),
    (55, 60, "55-59"),
    (60, 65, "60-64"),
    (65, 70, "65-69"),
    (70, 75, "70-74"),
    (75, 80, "75-79"),
    (80, 85, "80-84"),
    (85, 90, "85-89"),
    (90, 95, "90-94"),
    (95, 100, "95-99"),
    (100, 100, "100+"),
]


def reconstruct_male_abridged() -> pl.DataFrame:
    """Recover Male single-year mortality rates from Both + Female + population.

    Returns a single-column polars DataFrame with columns
    ``AgeStart`` (Int64) and ``value`` (Float64), sorted by ``AgeStart``.
    """
    df = pl.read_csv(WPP_CSV)
    mask_80_year = (pl.col("IndicatorId") == 80) & (pl.col("Time") == YEAR)

    both = df.filter(mask_80_year & (pl.col("Sex") == "Both sexes")).select(
        pl.col("AgeStart").cast(pl.Int64),
        pl.col("Value").cast(pl.Float64).alias("value"),
    )
    female = df.filter(mask_80_year & (pl.col("Sex") == "Female")).select(
        pl.col("AgeStart").cast(pl.Int64),
        pl.col("Value").cast(pl.Float64).alias("value"),
    )

    pop = pl.read_csv(POP_CSV).with_columns(
        pl.col("Age").replace("100+", "100").cast(pl.Int64).alias("Age"),
    )
    pop_m = pop.select(pl.col("Age"), pl.col("M").cast(pl.Float64).alias("M"))
    pop_f = pop.select(pl.col("Age"), pl.col("F").cast(pl.Float64).alias("F"))

    # Inner join on age so we only consider ages where we have rates on
    # both sides plus population. Polars join keys are explicit.
    joined = (
        both.rename({"value": "r_b"})
        .join(female.rename({"value": "r_f"}), on="AgeStart", how="inner")
        .join(pop_m.rename({"M": "m"}), left_on="AgeStart", right_on="Age", how="inner")
        .join(pop_f.rename({"F": "f"}), left_on="AgeStart", right_on="Age", how="inner")
    )

    rate_m = joined.filter(pl.col("m") > 0).select(
        pl.col("AgeStart"),
        (
            (pl.col("r_b") * (pl.col("m") + pl.col("f")) - pl.col("f") * pl.col("r_f"))
            / pl.col("m")
        ).alias("value"),
    )

    return rate_m.sort("AgeStart")


def aggregate_to_abridged(single_year: pl.DataFrame) -> pl.DataFrame:
    """Average single-year m(x) into 5-year abridged buckets, weighted
    by Iran male population per age so the result is the rate actually
    experienced by the bucket.
    """
    pop = pl.read_csv(POP_CSV).with_columns(
        pl.col("Age").replace("100+", "100").cast(pl.Int64).alias("Age"),
    )
    m_pop = pop.select(pl.col("Age").alias("AgeStart"), pl.col("M").cast(pl.Float64))

    # Left-join single-year rates with population weights so every age
    # in the bucket inherits its weight (or 0 if population is missing).
    rates = single_year.join(m_pop, on="AgeStart", how="left").with_columns(
        pl.col("M").fill_null(0.0).alias("M"),
    )

    rows: list[dict] = []
    for age_start, age_end, label in ABRIDGED_BUCKETS:
        ages = list(range(age_start, age_end)) if age_start != 100 else [100]
        sub = rates.filter(pl.col("AgeStart").is_in(ages))
        valid = sub.filter(pl.col("value").is_not_null() & pl.col("M") > 0)

        if valid.height == 0:
            continue

        weight_sum = float(valid["M"].sum())
        if weight_sum <= 0:
            continue

        agg = float((valid["value"] * valid["M"]).sum() / weight_sum)
        rows.append(
            {
                "AgeStart": age_start,
                "AgeEnd": age_end if age_start != 100 else 100,
                "Age": label,
                "Value": agg,
                "pop_weight": weight_sum,
            }
        )
    return pl.DataFrame(rows)


def write_wpp_csv(abridged: pl.DataFrame) -> None:
    """Write a WPP-shaped CSV with just abridged Male 2024 Median rows."""
    fieldnames = [
        "IndicatorId", "IndicatorName", "IndicatorShortName", "Source",
        "SourceYear", "Author", "LocationId", "Location", "Iso2", "Iso3",
        "TimeId", "Time", "VariantId", "Variant", "SexId", "Sex", "AgeId",
        "AgeStart", "AgeEnd", "Age", "CategoryId", "Category",
        "EstimateTypeId", "EstimateType", "EstimateMethodId", "EstimateMethod",
        "Value",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in abridged.iter_rows(named=True):
            w.writerow(
                {
                    "IndicatorId": 79,
                    "IndicatorName": "Age specific mortality rate m(x,n) - abridged",
                    "IndicatorShortName": "Age-specific mortality rates by age groups and by sex",
                    "Source": "World Population Prospects",
                    "SourceYear": 2024,
                    "Author": "United Nations Population Division (recovered from Both sexes + Female)",
                    "LocationId": 364,
                    "Location": "Iran (Islamic Republic of)",
                    "Iso2": "IR",
                    "Iso3": "IRN",
                    "TimeId": 75,
                    "Time": YEAR,
                    "VariantId": 4,
                    "Variant": "Median",
                    "SexId": 1,
                    "Sex": "Male",
                    "AgeId": 42,
                    "AgeStart": int(r["AgeStart"]),
                    "AgeEnd": int(r["AgeEnd"]),
                    "Age": r["Age"],
                    "CategoryId": 0,
                    "Category": "Not applicable",
                    "EstimateTypeId": 1,
                    "EstimateType": "Model-based Estimates",
                    "EstimateMethodId": 2,
                    "EstimateMethod": "Interpolation (recovered)",
                    "Value": f"{r['Value']:.8f}",
                }
            )


def main() -> None:
    single_year = reconstruct_male_abridged()
    print(f"Reconstructed Male single-year m(x) for {single_year.height} ages")
    print("\nSample single-year Male rates (per 1000 person-years):")
    sample_ages = {0, 1, 5, 30, 50, 70, 90, 100}
    for r in single_year.iter_rows(named=True):
        if r["AgeStart"] in sample_ages:
            print(f"  age {r['AgeStart']:>3}: {r['value'] * 1000:.3f}")

    abridged = aggregate_to_abridged(single_year)
    print("\nAbridged 5-year Male rates (per 1000 person-years):")
    for r in abridged.iter_rows(named=True):
        print(f"  {r['Age']:>7}: {r['Value'] * 1000:.3f}")

    write_wpp_csv(abridged)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
