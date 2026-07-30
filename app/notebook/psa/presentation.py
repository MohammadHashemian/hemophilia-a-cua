"""Readable presentation helpers for PSA result tables.

This module changes only how calculated tables are displayed. It keeps related
measures together, repeats identifier columns where useful, and renders each
section without wrapping cell contents onto multiple lines.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from html import escape

import polars as pl


@dataclass(frozen=True)
class TableSection:
    """A narrow, labelled view of one calculated result table."""

    group: str
    title: str
    data: pl.DataFrame


SECTION_LAYOUTS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "Calibration": (
        (
            "Expected and observed bleeding",
            (
                "time_horizon",
                "sampling_method",
                "regime",
                "n_patients",
                "expected_abr",
                "realized_abr",
            ),
        ),
        (
            "Uncertainty intervals",
            (
                "sampling_method",
                "regime",
                "expected_abr_ci_low",
                "expected_abr_ci_high",
                "realized_abr_ci_low",
                "realized_abr_ci_high",
            ),
        ),
        (
            "Calibration error and fit",
            (
                "sampling_method",
                "regime",
                "bias",
                "relative_error",
                "oe_ratio",
                "distribution_similarity_index",
                "rmse",
            ),
        ),
        (
            "Statistical checks",
            (
                "sampling_method",
                "regime",
                "ks_p_value",
                "pearson_r",
                "pearson_p_value",
                "calibration_status",
            ),
        ),
        (
            "Population totals",
            (
                "sampling_method",
                "regime",
                "mortality_rate",
                "mean_person_years",
                "total_bleeds",
                "total_person_years",
            ),
        ),
    ),
    "Clinical outcomes": (
        (
            "Quality-adjusted life-years",
            (
                "scenario",
                "simulations",
                "qaly_mean",
                "qaly_mean_sd",
                "qaly_median_iqr",
            ),
        ),
        (
            "Pettersson score",
            (
                "scenario",
                "simulations",
                "pettersson_mean_sd",
                "pettersson_median_iqr",
            ),
        ),
    ),
    "Bleeding outcomes": (
        (
            "Annual and spontaneous bleeding rates",
            (
                "scenario",
                "abr_mean_sd",
                "abr_median_iqr",
                "sbr_mean_sd",
                "sbr_median_iqr",
            ),
        ),
        (
            "Joint and life-threatening bleeding rates",
            (
                "scenario",
                "ajbr_mean_sd",
                "ajbr_median_iqr",
                "altb_mean_sd",
                "altb_median_iqr",
            ),
        ),
        (
            "Cohort bleeding totals",
            (
                "scenario",
                "total_bleeding_events",
                "total_joint_bleeding_events",
                "total_person_years",
                "cohort_abr",
                "cohort_ajbr",
            ),
        ),
    ),
    "Mortality and life expectancy": (
        (
            "Observed person-years",
            (
                "scenario",
                "person_years_mean_sd",
                "person_years_median_iqr",
                "absorbed_percent",
            ),
        ),
        (
            "Lost life expectancy",
            (
                "scenario",
                "lost_person_years_mean_sd",
                "lost_person_years_median_iqr",
                "lost_life_expectancy_median_iqr",
            ),
        ),
    ),
    "Survival efficiency": (
        (
            "Survival comparison",
            (
                "comparison",
                "base_survival_efficiency",
                "comparison_survival_efficiency",
                "absolute_survival_gain",
            ),
        ),
        (
            "Relative survival and absorption",
            (
                "comparison",
                "relative_survival_ratio",
                "relative_survival_gain_percent",
                "absorbed_rate_base",
                "absorbed_rate_comparison",
            ),
        ),
    ),
    "ICER and NMB": (
        (
            "Incremental costs",
            (
                "comparison",
                "paired_iterations",
                "delta_cost",
                "delta_cost_ci_low",
                "delta_cost_ci_high",
            ),
        ),
        (
            "Incremental QALYs",
            (
                "comparison",
                "paired_iterations",
                "delta_qaly",
                "delta_qaly_ci_low",
                "delta_qaly_ci_high",
            ),
        ),
        (
            "Cost-effectiveness decision",
            (
                "comparison",
                "icer",
                "interpretation",
                "delta_nmb",
                "probability_cost_effective",
            ),
        ),
    ),
}


def table_sections(
    tables: Mapping[str, pl.DataFrame],
) -> list[TableSection]:
    """Split wide calculated tables into ordered, domain-focused views."""

    sections: list[TableSection] = []
    for group, frame in tables.items():
        layout = SECTION_LAYOUTS.get(group)
        if layout is None:
            sections.append(TableSection(group, group, frame))
            continue

        covered: set[str] = set()
        for title, requested_columns in layout:
            columns = [column for column in requested_columns if column in frame.columns]
            if not columns:
                continue
            covered.update(columns)
            sections.append(TableSection(group, title, frame.select(columns)))

        missing = [column for column in frame.columns if column not in covered]
        if missing:
            sections.append(TableSection(group, "Additional measures", frame.select(missing)))

    return sections


def _format_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_table(frame: pl.DataFrame) -> str:
    """Return notebook HTML with horizontal scrolling and no cell wrapping."""

    header = "".join(f"<th>{escape(column)}</th>" for column in frame.columns)
    rows = []
    for row in frame.iter_rows():
        cells = "".join(f"<td>{escape(_format_value(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")

    return f"""
<style>
.psa-table-scroll {{
  --psa-table-bg: #ffffff;
  --psa-table-header-bg: #eef2f6;
  --psa-table-text: #1f2328;
  --psa-table-muted-text: #424a53;
  --psa-table-border: #c7cdd4;
  --psa-table-hover: #f3f7fb;
  color-scheme: light dark;
  max-width: 100%;
  overflow-x: auto;
  margin: 0.35rem 0 1.25rem;
}}
@media (prefers-color-scheme: dark) {{
  .psa-table-scroll {{
    --psa-table-bg: #161b22;
    --psa-table-header-bg: #252c35;
    --psa-table-text: #e6edf3;
    --psa-table-muted-text: #c9d1d9;
    --psa-table-border: #48515c;
    --psa-table-hover: #222b36;
  }}
}}
.psa-readable-table {{
  width: max-content;
  min-width: 55%;
  border-collapse: collapse;
  font-size: 0.88rem;
  line-height: 1.25;
  color: var(--jp-ui-font-color1, var(--psa-table-text));
  background: var(--jp-layout-color1, var(--psa-table-bg));
}}
.psa-readable-table th,
.psa-readable-table td {{
  border: 1px solid var(--jp-border-color2, var(--psa-table-border));
  padding: 0.42rem 0.62rem;
  text-align: right;
  white-space: nowrap;
}}
.psa-readable-table th {{
  color: var(--jp-ui-font-color1, var(--psa-table-text));
  background: var(--jp-layout-color2, var(--psa-table-header-bg));
  font-weight: 600;
}}
.psa-readable-table tbody tr:hover td {{
  background: var(--jp-layout-color2, var(--psa-table-hover));
}}
.psa-readable-table th:first-child,
.psa-readable-table td:first-child {{
  position: sticky;
  left: 0;
  text-align: left;
  color: var(--jp-ui-font-color1, var(--psa-table-text));
  background: var(--jp-layout-color1, var(--psa-table-bg));
  border-right-width: 2px;
  z-index: 1;
}}
.psa-readable-table th:first-child {{
  background: var(--jp-layout-color2, var(--psa-table-header-bg));
  z-index: 2;
}}
.psa-readable-table tbody tr:hover td:first-child {{
  background: var(--jp-layout-color2, var(--psa-table-hover));
}}
</style>
<div class="psa-table-scroll">
  <table class="psa-readable-table">
    <thead><tr>{header}</tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>
""".strip()
