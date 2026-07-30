# Composable OWSA

The OWSA workflow mirrors the horizon-specific PSA structure without placing
calculation logic in notebooks.

- `scenarios.py` defines deterministic base, low, and high scenarios for any
  `HorizonSpec`. Low/high values are the 2.5th and 97.5th percentiles of the
  corresponding PSA input distribution. Fixed inputs with no range are
  excluded. Arthropathy utilities are constrained between their adjacent
  severity utilities so one-way changes cannot make a mild state worse than
  a more severe state.
- `workflow.py` builds paired model inputs, runs the vectorized simulation,
  and stores results under `app/cache/owsa/<horizon-directory>/`.
  OWSA inputs are deterministic point values. The default 1,000 replications
  per scenario estimate stochastic microsimulation outcomes; they are not PSA
  draws. Common random numbers are shared across base/low/high scenarios to
  reduce noise in incremental differences.
- `analysis.py` calculates paired incremental cost, QALY, ICER, and NMB for
  each one-way change.
- `plots.py` renders NMB and ICER tornado diagrams. NMB is the primary
  decision statistic because it remains interpretable when incremental QALY
  approaches zero or results cross cost-effectiveness quadrants.
  The ICER tornado includes only parameters whose low, base, and high results
  all remain in the conventional more-costly/more-effective quadrant.
- `report.py` is the display-oriented facade used by analysis notebooks.

Childhood notebooks are under
`app/notebooks/owsa/childhood_age_1_15/`. Another horizon can reuse the same
modules by passing a different `HorizonSpec`; no calculation code needs to be
copied into the notebook.
