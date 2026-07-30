"""Data-driven narrative interpretations for PSA report outputs."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import polars as pl

from app.notebook.psa.presentation import TableSection


def _number(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(value))
    return float(match.group(0).replace(",", "")) if match else float("nan")


def _values(frame: pl.DataFrame, column: str) -> np.ndarray:
    return np.array([_number(value) for value in frame[column]], dtype=float)


def _range(frame: pl.DataFrame, column: str, fmt: str = ",.2f") -> str:
    values = _values(frame, column)
    values = values[np.isfinite(values)]
    if not values.size:
        return "not estimable"
    return f"{values.min():{fmt}} to {values.max():{fmt}}"


def _mean(frame: pl.DataFrame, column: str) -> float:
    values = _values(frame, column)
    return float(np.nanmean(values))


def _narrative(interpretation: str, conclusion: str) -> str:
    return (
        f"**Interpretation.** {interpretation}\n\n"
        f"**Conclusion.** {conclusion}"
    )


def table_interpretation(section: TableSection, *, wtp: float) -> str:
    """Interpret one displayed (possibly narrow) report table."""
    frame, group, title = section.data, section.group, section.title

    if group == "Calibration" and title == "Expected and observed bleeding":
        differences = np.abs(
            _values(frame, "realized_abr") - _values(frame, "expected_abr")
        )
        return _narrative(
            f"Across the four base arms, expected ABR is "
            f"{_range(frame, 'expected_abr')} and simulated ABR is "
            f"{_range(frame, 'realized_abr')}. The largest absolute difference "
            f"is {np.nanmax(differences):.2f} bleeds/person-year.",
            "This is the primary face-validity check that event generation reproduces "
            "the ABR inputs; differences should be judged together with the uncertainty "
            "and formal calibration diagnostics below.",
        )
    if group == "Calibration" and title == "Uncertainty intervals":
        overlap = (
            (_values(frame, "expected_abr_ci_low") <= _values(frame, "realized_abr_ci_high"))
            & (_values(frame, "realized_abr_ci_low") <= _values(frame, "expected_abr_ci_high"))
        )
        return _narrative(
            f"Expected and realized 95% intervals overlap in {overlap.sum()} of "
            f"{frame.height} base-arm comparisons.",
            "Interval overlap supports distribution-level agreement, while a non-overlap "
            "would identify an arm requiring calibration review.",
        )
    if group == "Calibration" and title == "Calibration error and fit":
        return _narrative(
            f"Relative error ranges from {_range(frame, 'relative_error', '.1%')}; "
            f"the observed/expected ratio ranges from {_range(frame, 'oe_ratio', '.3f')}, "
            f"and RMSE from {_range(frame, 'rmse', '.3f')}.",
            "Values closest to zero error, an observed/expected ratio of one, and lower "
            "RMSE indicate better reproduction of the sampled bleeding process.",
        )
    if group == "Calibration" and title == "Statistical checks":
        statuses = frame["calibration_status"].value_counts()
        status_text = ", ".join(
            f"{row['calibration_status']}: {row['count']}"
            for row in statuses.iter_rows(named=True)
        )
        return _narrative(
            f"Calibration classifications are {status_text}. Pearson correlation ranges "
            f"from {_range(frame, 'pearson_r', '.3f')}.",
            "These tests complement, rather than replace, graphical and clinical "
            "calibration because very large PSA samples can make small deviations "
            "statistically significant.",
        )
    if group == "Calibration" and title == "Population totals":
        return _narrative(
            f"Mean observed follow-up is {_range(frame, 'mean_person_years')} years and "
            f"mortality is {_range(frame, 'mortality_rate', '.2%')} across base arms.",
            "These denominators confirm that reported cohort event rates use accumulated "
            "person-time and account for early absorption.",
        )
    if group == "Absorption diagnostics":
        absorbed = frame.filter(pl.col("is_absorbed"))
        alive = frame.filter(~pl.col("is_absorbed"))
        return _narrative(
            f"Absorbed groups contribute {_range(absorbed, 'person_years')} mean "
            f"person-years versus {_range(alive, 'person_years')} among survivors.",
            "Shorter follow-up among absorbed patients is expected; this table is a "
            "diagnostic for informative truncation and should not be interpreted as a "
            "treatment-effect estimate.",
        )
    if group == "Clinical outcomes" and title == "Quality-adjusted life-years":
        return _narrative(
            f"Mean QALYs range from {_range(frame, 'qaly_mean')}; each scenario contains "
            f"{int(_values(frame, 'simulations').min()):,} simulations.",
            "Higher QALYs represent better combined length and quality of life, but the "
            "incremental paired QALY table is the appropriate basis for treatment comparison.",
        )
    if group == "Clinical outcomes" and title == "Pettersson score":
        return _narrative(
            f"Mean Pettersson scores range from "
            f"{_range(frame, 'pettersson_mean_sd')}.",
            "Lower scores indicate less modelled joint damage. Differences should be "
            "read alongside joint-bleeding rates because the score is a downstream outcome.",
        )
    if group == "Bleeding outcomes" and title == "Annual and spontaneous bleeding rates":
        return _narrative(
            f"Mean total ABR ranges from {_range(frame, 'abr_mean_sd')}, while mean "
            f"spontaneous bleeding rates range from {_range(frame, 'sbr_mean_sd')}.",
            "The separation between on-demand and prophylaxis scenarios describes the "
            "modelled preventive effect; paired reductions are quantified later.",
        )
    if group == "Bleeding outcomes" and title == "Joint and life-threatening bleeding rates":
        return _narrative(
            f"Mean annual joint-bleeding rates range from "
            f"{_range(frame, 'ajbr_mean_sd')}.",
            "Lower joint-bleeding frequency is clinically important because recurrent "
            "hemarthrosis drives accumulated arthropathy.",
        )
    if group == "Bleeding outcomes" and title == "Cohort bleeding totals":
        return _narrative(
            f"Person-time adjusted cohort ABR ranges from {_range(frame, 'cohort_abr')} "
            f"and cohort AJBR from {_range(frame, 'cohort_ajbr')}.",
            "These rates are preferred to raw event totals when mortality or follow-up "
            "differs because their denominator is observed person-years.",
        )
    if group == "Bleeding outcomes" and title == "Additional measures":
        return _narrative(
            f"Mean annual ICH rates range from {_range(frame, 'aich_mean_sd')} and "
            f"non-ICH major-bleeding rates from {_range(frame, 'anon_ich_mean_sd')}.",
            "These rare-event outputs should be interpreted with their wide stochastic "
            "uncertainty and compared primarily across explicitly defined bleeding scenarios.",
        )
    if group == "Mortality and life expectancy" and title == "Observed person-years":
        return _narrative(
            f"Mean observed person-years range from "
            f"{_range(frame, 'person_years_mean_sd')}; absorption ranges from "
            f"{_range(frame, 'absorbed_percent', '.2f')}%.",
            "For a childhood horizon, small absolute mortality differences are expected "
            "and should not be overstated.",
        )
    if group == "Mortality and life expectancy" and title == "Lost life expectancy":
        return _narrative(
            f"Mean lost person-years within the horizon range from "
            f"{_range(frame, 'lost_person_years_mean_sd')}.",
            "Lost life measures summarize premature absorption; causal treatment "
            "conclusions require the paired survival comparisons.",
        )
    if group == "Resource utilization":
        return _narrative(
            f"Mean factor consumption ranges from "
            f"{_range(frame, 'total_factor_mean_sd')} IU and mean total cost from "
            f"{_range(frame, 'total_cost_mean', ',.0f')} IRR.",
            "Factor exposure is the principal resource driver, but cost effectiveness "
            "depends on whether added costs are justified by paired health gains.",
        )
    if group == "Health-state occupation" and title == "State occupation":
        state_columns = [column for column in frame.columns if column != "scenario"]
        averages = {column: _mean(frame, column) for column in state_columns}
        largest = max(averages, key=averages.get)
        return _narrative(
            f"The largest average occupation component is {largest.replace('_', ' ')} "
            f"({averages[largest]:.1%} of observed weeks).",
            "Occupation shares describe where simulated time and rewards accumulate; "
            "they are useful for face validation but do not alone validate transition risks.",
        )
    if group == "Health-state occupation" and title == "Occupation validation":
        maximum_error = float(frame["state_share_abs_error"].max())
        return _narrative(
            f"State shares sum to one with a maximum absolute error of "
            f"{maximum_error:.2e}.",
            "The accounting identity passes for all scenarios, supporting internal "
            "consistency of state-time allocation.",
        )
    if group == "Survival efficiency" and title == "Survival comparison":
        return _narrative(
            f"Absolute paired survival-efficiency gains range from "
            f"{_range(frame, 'absolute_survival_gain', '.4%')}.",
            "The sign indicates the direction of the prophylaxis comparison; childhood "
            "differences should be interpreted as small short-horizon signals.",
        )
    if group == "Survival efficiency" and title == "Relative survival and absorption":
        return _narrative(
            f"Relative survival-efficiency gains range from "
            f"{_range(frame, 'relative_survival_gain_percent', '.3f')} percentage points.",
            "Relative measures can magnify small absolute changes, so absorbed rates and "
            "absolute gains should remain the primary interpretation.",
        )
    if group == "ABR reduction":
        return _narrative(
            f"Prophylaxis reduces cohort ABR by {_range(frame, 'absolute_abr_reduction')} "
            f"events/person-year, corresponding to "
            f"{_range(frame, 'relative_abr_reduction', '.1%')}.",
            "All reductions are paired within matching sampling methods and sensitivity "
            "extensions, making this the clearest comparative bleeding-effect table.",
        )
    if group == "Factor-consumption reduction":
        changes = -_values(frame, "absolute_factor_reduction")
        return _narrative(
            f"Compared with on-demand care, prophylaxis changes mean factor use by "
            f"{np.nanmin(changes):,.0f} to {np.nanmax(changes):,.0f} IU per patient.",
            "Positive values denote additional prophylaxis consumption. Weight-reduction "
            "scenarios test how this resource requirement changes with lower dosing weight.",
        )
    if group == "ICER and NMB" and title == "Incremental costs":
        return _narrative(
            f"Mean paired incremental cost ranges from "
            f"{_range(frame, 'delta_cost', ',.0f')} IRR across comparisons.",
            "Positive values mean prophylaxis costs more. The percentile intervals show "
            "patient-level PSA variation rather than uncertainty in an unpaired difference.",
        )
    if group == "ICER and NMB" and title == "Incremental QALYs":
        return _narrative(
            f"Mean paired incremental QALYs range from "
            f"{_range(frame, 'delta_qaly', '.4f')}.",
            "Positive values favour prophylaxis; whether their value justifies incremental "
            "cost is evaluated using NMB at the selected WTP.",
        )
    if group == "ICER and NMB" and title == "Cost-effectiveness decision":
        favourable = int((frame["probability_cost_effective"] >= 0.5).sum())
        return _narrative(
            f"At {wtp:,.0f} IRR/QALY, {favourable} of {frame.height} comparisons have "
            f"at least 50% probability of being cost-effective. Probabilities range from "
            f"{_range(frame, 'probability_cost_effective', '.1%')}.",
            "NMB and probability cost-effective are more stable decision summaries than "
            "iteration-level ICER ratios.",
        )
    if group == "ICER vs ABR threshold" and title == "Estimated decision threshold":
        found = int(frame["threshold_found"].sum())
        return _narrative(
            f"A cost-effective ABR crossing is found for {found} of {frame.height} "
            f"comparisons; estimated thresholds range from "
            f"{_range(frame, 'cost_effective_abr_threshold')}.",
            "The threshold estimates identify the baseline bleeding burden above which "
            "prophylaxis becomes economically favourable under the selected WTP.",
        )
    if group == "ICER vs ABR threshold":
        return _narrative(
            f"At the least restrictive baseline cutoff, probability cost-effective ranges "
            f"from {_range(frame, 'probability_ce_at_baseline_cutoff', '.1%')}.",
            "Threshold conclusions are supported only within the observed ABR range and "
            "should not be extrapolated beyond it.",
        )
    if group == "CEAC decision thresholds" and title == "Selected WTP decision":
        return _narrative(
            f"At the selected WTP of {wtp:,.0f} IRR/QALY, paired cost-effectiveness "
            f"probabilities range from "
            f"{_range(frame, 'probability_ce_at_selected_wtp', '.1%')}.",
            "Probabilities near 50% indicate decision uncertainty; values near zero or one "
            "indicate more stable conclusions under current PSA assumptions.",
        )
    if group == "CEAC decision thresholds":
        found = int(frame["crossing_found_in_range"].sum())
        return _narrative(
            f"The CEAC reaches 50% within the searched range for {found} of "
            f"{frame.height} comparisons. The corresponding WTP ranges from "
            f"{_range(frame, 'wtp_at_50_percent_ce', ',.0f')} IRR/QALY.",
            "This is the median decision threshold across paired iterations, not a clinical "
            "price recommendation or a conventional confidence limit.",
        )
    return _narrative(
        f"This table reports {frame.height} rows derived from the childhood PSA.",
        "Interpret the direction and uncertainty together with the paired incremental results.",
    )


def figure_interpretation(
    name: str,
    *,
    df: pl.DataFrame,
    tables: Mapping[str, pl.DataFrame],
    wtp: float,
) -> str:
    """Interpret one report figure using its underlying numerical results."""
    bleeding = tables["Bleeding outcomes"]
    economic = tables["ICER and NMB"]
    narratives = {
        "abr_distribution": _narrative(
            f"The plotted sampled and simulated bleeding distributions correspond to "
            f"cohort ABRs ranging from {_range(bleeding, 'cohort_abr')}.",
            "Close alignment supports event-rate calibration; visible shifts between "
            "regimes represent the modelled prophylaxis effect.",
        ),
        "survival_curve": _narrative(
            f"Childhood absorption ranges from "
            f"{_range(tables['Mortality and life expectancy'], 'absorbed_percent', '.2f')}%. "
            f"The vertical axis is deliberately zoomed because survival remains high.",
            "The empirical steps are Monte Carlo and weekly-event variation, not evidence "
            "of rapidly changing biological hazards; no smoothing is used.",
        ),
        "health_state_distribution": _narrative(
            f"All plotted occupation vectors sum to one; maximum accounting error is "
            f"{tables['Health-state occupation']['state_share_abs_error'].max():.2e}.",
            "The chart validates allocation of simulated time across states, but transition "
            "probabilities require separate calibration and face-validity checks.",
        ),
        "qaly_distribution": _narrative(
            f"Scenario mean QALYs range from "
            f"{_range(tables['Clinical outcomes'], 'qaly_mean')}.",
            "Distributional overlap shows individual uncertainty; paired incremental QALYs "
            "determine comparative effectiveness.",
        ),
        "cost_distribution": _narrative(
            f"Scenario mean costs range from "
            f"{_range(tables['Resource utilization'], 'total_cost_mean', ',.0f')} IRR.",
            "Separation largely reflects factor consumption and should be considered with "
            "health gains rather than interpreted alone.",
        ),
        "joint_cost_qaly": _narrative(
            "Points show individual PSA outcomes and contours identify their highest-density "
            "regions without replacing the underlying observations.",
            "A favourable intervention shifts QALYs rightward; its vertical cost shift must "
            "then be assessed against the WTP boundary in the incremental plane.",
        ),
        "cost_effectiveness_planes": _narrative(
            f"At {wtp:,.0f} IRR/QALY, probabilities cost-effective range from "
            f"{_range(economic, 'probability_cost_effective', '.1%')}. Points below the "
            "WTP line have positive incremental NMB.",
            "The plane preserves joint cost–effect uncertainty and is more informative than "
            "considering the marginal cost or QALY distributions separately.",
        ),
        "icer_distributions": _narrative(
            "The upper strips show individual paired ICERs on the same x-scale as the "
            "histograms below; the displayed range excludes the outer 1% tails on each side.",
            "Long or irregular tails are expected when incremental QALYs approach zero. "
            "Therefore, ICER histograms are descriptive and NMB is preferred for decisions.",
        ),
        "ceac": _narrative(
            f"The selected WTP is {wtp:,.0f} IRR/QALY. Fifty-percent CEAC thresholds range "
            f"from {_range(tables['CEAC decision thresholds'], 'wtp_at_50_percent_ce', ',.0f')} "
            "IRR/QALY.",
            "The CEAC expresses decision uncertainty over alternative WTP values; it does "
            "not identify a universally correct threshold.",
        ),
        "incremental_nmb_distribution": _narrative(
            f"Mean incremental NMB ranges from "
            f"{_range(economic, 'delta_nmb', ',.0f')} IRR at the selected WTP.",
            "Values above zero favour prophylaxis. This is the primary probabilistic "
            "cost-effectiveness display because it remains interpretable across quadrants.",
        ),
        "icer_vs_abr_threshold": _narrative(
            f"Estimated cost-effective baseline ABR thresholds range from "
            f"{_range(tables['ICER vs ABR threshold'], 'cost_effective_abr_threshold')}.",
            "The curves indicate how conclusions depend on baseline bleeding burden; only "
            "crossings inside the observed range should be interpreted.",
        ),
    }
    return narratives.get(
        name,
        _narrative(
            f"This figure summarizes {df.height:,} childhood PSA rows.",
            "Interpret it together with its numerical table and model assumptions.",
        ),
    )
