# PSA notebook workflows

The PSA workflow is separated by patient age horizon.

## Childhood horizon (ages 2–12)

Run in order:

1. `childhood_age_2_12/01_scenario_definitions.ipynb`
2. `childhood_age_2_12/02_simulation.ipynb`
3. `childhood_age_2_12/03_analysis.ipynb`

Results are written to:

`app/cache/psa/childhood_age_2_12/parquet/all_results_combined.parquet`

## Lifetime horizon (ages 2–100)

Run in order:

1. `lifetime_age_2_100/01_scenario_definitions.ipynb`
2. `lifetime_age_2_100/02_simulation.ipynb`
3. `lifetime_age_2_100/03_analysis.ipynb`

Results are written to:

`app/cache/psa/lifetime_age_2_100/parquet/all_results_combined.parquet`

The analysis notebooks temporarily support the previous mixed PSA cache as a
read-only migration fallback. Once a separated simulation is run, its dedicated
cache takes precedence.

The legacy combined notebooks are retained for reference under:

`app/notebooks/deprecated/psa_combined/`

New PSA work should use the horizon-specific notebooks above.
