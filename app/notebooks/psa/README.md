# PSA notebook workflows

The PSA workflow is separated by patient age horizon.

## Childhood horizon (ages 1–15)

Run in order:

1. `childhood_age_1_15/01_scenario_definitions.ipynb`
2. `childhood_age_1_15/02_simulation.ipynb`
3. `childhood_age_1_15/03_analysis.ipynb`

Results are written to:

`app/cache/psa/childhood_age_1_15/parquet/all_results_combined.parquet`

## Lifetime horizon (ages 1–100)

Run in order:

1. `lifetime_age_1_100/01_scenario_definitions.ipynb`
2. `lifetime_age_1_100/02_simulation.ipynb`
3. `lifetime_age_1_100/03_analysis.ipynb`

Results are written to:

`app/cache/psa/lifetime_age_1_100/parquet/all_results_combined.parquet`

The old childhood cache covered ages 2–12 and the old lifetime cache covered
ages 2–100. Both are intentionally rejected. Run the current horizon's
simulation before its analysis notebook so age-2 results cannot be mistaken for
the current age-1 cohorts.

The legacy combined notebooks are retained for reference under:

`app/notebooks/deprecated/psa_combined/`

New PSA work should use the horizon-specific notebooks above.

## Reusable analysis architecture

Analysis notebooks are display layers. Shared calculations live under
`app/notebook/psa/`:

- `economics.py`: explicit iteration pairing, incremental outcomes, ICER,
  NMB, confidence intervals, and CEAC data.
- `tables.py`: calibration, clinical, bleeding, mortality, resource,
  occupation, survival, and reduction tables.
- `report_plots.py`: horizon-agnostic PSA figures.
- `report.py`: the `PSAReport` facade used by notebooks.

For a future horizon, define a `HorizonSpec`, run its simulation, then use:

```python
report = PSAReport.load(horizon)
tables = report.tables()
figures = report.figures()
```
