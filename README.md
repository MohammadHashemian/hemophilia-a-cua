# Hemophilia A State-Transition Cost-Utility Model

Installable individual-level state-transition microsimulation for comparing prophylactic and
on-demand factor VIII treatment in patients with severe hemophilia A without inhibitors.

This is not a cohort Markov model. Chronic arthropathy states are updated from each patient's
accumulated joint-bleed history, while acute bleeds are recurrent within-cycle events. The
former Markov engines remain available under `modular_models.markov` only for reproducing legacy
analyses.

## Model at a glance

- Economic evaluation: cost-utility analysis from the payer perspective.
- Population: patients with severe hemophilia A without inhibitors, entering at age 1.
- Exclusive horizon: first birthday to immediately before the 12th birthday.
- Follow-up: 11 years, represented by exactly 572 weekly cycles.
- Strategies: prophylaxis and on-demand human plasma-derived SHL FVIII.
- Chronic states: no/minimal, mild, moderate and severe arthropathy, plus absorbing death.
- Acute events: joint bleed, non-major non-joint bleed, non-ICH major bleed and ICH.
- Patient memory: cumulative joint bleeds, Pettersson score, acute carry-over and persistent
  post-ICH flag.
- Mortality: age-specific background mortality and ICH case fatality as competing causes.
- Outcomes: clinical events, survival, cause-specific deaths, post-ICH entries, cost in Iranian
  rial (IRR), QALY, ICER and incremental net monetary benefit.
- FVIII price: 58,000 IRR for each IU.

```mermaid
flowchart LR
    JSON["Reference-based JSON data"] --> LOAD["Loader and Pydantic validation"]
    LOAD --> CTX["Immutable StudyContext"]
    CTX --> RESOLVE["Base / scenario / OWSA / PSA resolver"]
    RESOLVE --> ENGINE["Vectorized patient-level state-transition engine"]
    ENGINE --> EVENTS["Weekly events and competing mortality"]
    EVENTS --> MEMORY["Patient memory and chronic-state update"]
    MEMORY --> REWARDS["IRR costs, utility and discounting"]
    REWARDS --> RESULT["Polars-ready clinical and economic results"]
```

## Repository layout

| Path | Purpose |
|---|---|
| `modular_models/state_transition/` | Production state-transition engine and analyses |
| `modular_models/markov/` | Compatibility facade for legacy Markov engines |
| `app/data/state_transition/model.json` | Values, units, ranges, distributions and assumptions |
| `app/data/state_transition/scenarios.json` | Pre-specified structural scenarios |
| `app/data/state_transition/references.json` | Parameter reference catalog |
| `docs/state_transition/` | Architecture, model traceability, data dictionary and validation |
| `tests/state_transition/` | Backbone verification and regression tests |
| `outputs/state_transition/` | Generated summaries and run-audit artifacts |

The data dependency is one-directional: validated JSON feeds `StudyContext`, the context feeds
the parameter resolver and engine, and the engine returns immutable result objects. Model code
does not read ad-hoc spreadsheet cells or global variables.

## Installation

Python 3.11 or newer is required.

```bash
python -m pip install -e .
```

For development and validation:

```bash
python -m pip install -e ".[dev]"
```

The installed command is `hemophilia-stm`. Running the module directly is equivalent:

```bash
python -m modular_models.state_transition --help
```

## Reproducible workflows

Validate JSON schemas and cross-references:

```bash
python -m modular_models.state_transition validate
```

Run the paired base case and generate the full clinical/economic summary, machine-readable run
trace and PNG audit diagram:

```bash
python -m modular_models.state_transition base \
  --patients 100000 \
  --output-dir outputs/state_transition
```

Run the Monte Carlo population-size convergence sequence:

```bash
python -m modular_models.state_transition convergence \
  --sizes 1000 5000 10000 25000 50000 100000
```

Run every registered structural scenario:

```bash
python -m modular_models.state_transition scenarios --patients 100000
```

Run or resume checkpointed production analyses:

```bash
python -m modular_models.state_transition psa \
  --iterations 2500 --patients 5000 --batch-size 24 --jobs 0 --backend cpu \
  --output-dir outputs/state_transition/final_analysis/psa_production

python -m modular_models.state_transition owsa \
  --patients 50000 --jobs 0 --backend cpu \
  --output-dir outputs/state_transition/final_analysis/owsa_production
```

`--jobs 0` uses every logical CPU core. The numerical kernels are JIT-compiled and each worker
is limited to one native numerical thread, avoiding nested thread oversubscription.

CUDA is optional and is installed separately:

```bash
python -m pip install -e ".[gpu]"
python -m modular_models.state_transition psa \
  --iterations 24 --patients 5000 --jobs 8 --backend cuda \
  --output-dir outputs/state_transition/cuda_benchmark
```

The CUDA backend batches the within-cycle FP64 QALY interval calculation on the GPU. NumPy RNG,
Poisson events, competing mortality, state transitions and factor dosing remain on CPU, so the
clinical event history is unchanged. Backend selection must be benchmark-driven; CUDA is not
assumed to be faster for every GPU.

Before a final PSA, the first-order population within each second-order draw can be audited with
common parameter draws and common seeds:

```bash
python -m modular_models.state_transition psa-precision \
  --iterations 40 --sizes 1000 2500 5000 10000 --jobs 0 --batch-size 24 \
  --output-dir outputs/state_transition/psa_inner_precision_final
```

The production pipelines write immutable parameter draws, atomic Parquet checkpoints, a model
input fingerprint and a resumable manifest. An interrupted run continues only missing iterations
or endpoints and rejects changed inputs or incompatible configuration.

The deterministic base case uses 50,000 patients per strategy: this is the first tested size for
which relative changes in incremental cost and incremental QALY were both below 1%. These
patients represent first-order variation in individual event histories while all input parameters
are fixed. PSA is a separate, nested uncertainty analysis: every PSA iteration draws a new set of
uncertain input parameters and then simulates a patient population under both strategies.

The common-draw inner-loop diagnostic supports 5,000 patients per strategy for each PSA
iteration. Relative mean differences versus 10,000 patients were 0.0032% for cost and 0.6139%
for QALY; paired QALY noise was 8.14% of between-parameter variation. The configured strict PSA
run uses 2,500 second-order iterations under the documented one-hour compute limit. This provides
an approximately ±1.96 percentage-point worst-case 95% Monte Carlo margin for CEAC probabilities;
10,000 draws remain the stricter ±0.98-point alternative. Each PSA row preserves outcomes
together with arm-specific survival, background deaths, ICH deaths, mortality probability and
Post-ICH entries.

The executed analysis notebook is `notebooks/state_transition_final_analysis.ipynb`. It contains
the locked input configuration, Polars tables, deterministic results, scenario analysis,
validation/calibration, sample patient/cycle traces, PSA/OWSA controls and Matplotlib figures.

## Notebook use

```python
from modular_models.state_transition import StudyContext, StudyRunner
from modular_models.state_transition.currency import DisplayCurrency, convert_from_irr

context = StudyContext.load()
comparison = StudyRunner(context).compare(
    scenario_id="base_case",
    n_patients=100_000,
    retain_patient_level=True,
)

economic_table = comparison.to_polars()
patient_table = comparison.prophylaxis.to_polars(patient_level=True)

# Display conversion only. Internal calculations and stored outputs remain IRR.
cost_toman = convert_from_irr(
    comparison.prophylaxis.summary["mean_cost_irr"],
    DisplayCurrency.TOMAN,
)
```

USD display requires an explicitly supplied, dated exchange rate. The package never embeds a
silent exchange-rate assumption.

## Exclusive childhood horizon and body weight

The cohort enters at exact age 1 and exits immediately before age 12:

```text
n_cycles = (12 - 1) * 52 = 572
age_at_cycle_start = 1 + cycle / 52
last_cycle_start_age = 11 + 51/52
```

There is no completed year at age 12. Body weight is linearly interpolated at every weekly cycle,
so the final simulated weight is evaluated below age 12; the age-12 reference value is only the
upper interpolation endpoint.

## Mortality and Post-ICH audit

Background annual hazards are converted to weekly probabilities:

```text
p_background_week(age_band) = 1 - exp(-annual_hazard(age_band) / 52)
```

For every ICH event, case fatality is sampled independently. If background death and fatal ICH
occur in the same weekly cycle, the earlier sampled time defines the single cause of death. Death
is absorbing and no subsequent event, FVIII cost or QALY is accrued.

Each arm reports:

- initial cohort, alive at end and total deaths;
- background deaths and ICH deaths separately;
- all-cause, background and ICH mortality probabilities;
- annual hazard, weekly probability, full-band probability, exposure and observed deaths for
  ages 1 to <5, 5 to <10 and 10 to <12;
- ICH events, input case-fatality probability and observed deaths per ICH;
- count and probability of patients who ever entered the persistent post-ICH flag.

Patient-level output includes `death_cycle`, `death_age_years`, `death_cause` and
`ever_post_ich` when retention is requested.

## Input and provenance contract

Every numerical input has a stable ID, value, unit, reference or explicit assumption, OWSA range
where applicable, and PSA distribution where parameter uncertainty is modeled.
`StudyContext.load()` rejects missing references, unknown scenario parameters, malformed
distributions, inverted OWSA ranges and missing required inputs.

The current factor-price record is:

```text
factor_price_irr_per_iu = 58,000 IRR/IU
product = human plasma-derived standard-half-life FVIII
```

Costs remain in IRR throughout the engine. Toman and USD are display conversions only.

## Core calculations

```text
lambda_weekly = lambda_annual / 52
N[event, patient, cycle] ~ Poisson(lambda_weekly)
lambda_non_major_non_joint = ABR - AJBR - lambda_ICH - lambda_non_ICH_major
Pettersson = min(78, floor(cumulative_joint_bleeds / joint_bleeds_per_point))
cost_cycle = factor_VIII_IU_cycle * 58,000 IRR/IU
discount(t) = 1 / (1 + annual_rate) ** (t / 52)
```

The residual bleeding rate must be non-negative. In PSA, inconsistent draws are rejected and
resampled. Chronic progression is one-directional because cumulative joint bleeds cannot fall.
Continuing acute effects carry into later cycles and overlapping utility effects use the lowest
applicable value.

ABR is the total annual bleed rate and AJBR is a subset of it. Therefore an AJBR endpoint cannot
be interpreted independently if it leaves a negative residual for non-joint bleeding. When the
high on-demand AJBR endpoint (14.2/year) is infeasible with the base ABR (13.8/year), OWSA pairs
it with the documented high ABR (15.6/year). The output labels this explicitly as a
`linked_endpoint`, stores both overrides, and does not misrepresent it as a pure one-way result.

## Structural scenarios

The registry contains direct AJBR/direct ICH base inputs, fraction-based AJBR, fraction-based ICH,
no mild-arthropathy decrement, less-severe post-ICH utility mapping and no discounting. Structural
scenarios remain separate from parameter uncertainty.

## Validation

Backbone tests cover typed loading, reference integrity, event-rate accounting, PSA constraints,
the exclusive 1-to-<12 horizon, interpolated weight at exit, Pettersson thresholds, absorbing
death, cause-specific death reconciliation, age-specific probability conversion, Post-ICH entry,
paired reproducibility, ABR recovery, currency conversion and JSON/PNG trace generation.
Production tests also verify process-parallel equivalence, linked AJBR/ABR endpoint auditing,
common-draw inner-loop diagnostics and checkpoint-based PSA iteration convergence.

```bash
python -m pytest -p no:cacheprovider --basetemp .tmp/pytest-run
python -m ruff check modular_models tests/state_transition
python -m mypy modular_models/state_transition
```

## Reporting cautions

- Base ABR and AJBR values are midpoints of reported ranges, not pooled means.
- The WBDR CNS fraction is a proxy for ICH only in its structural scenario.
- Acute ICH and GI-bleed utilities use indirect preference-based proxies.
- The 0.35 post-ICH utility cap is conservative and is tested structurally at 0.78.
- Only FVIII acquisition costs are included in the base analysis.
- The ten-times-GDP WTP value is a policy scenario, not a universal official threshold.
- Price-year and GDP inputs should be paired with dated sources for a time-specific analysis.

## License

MIT. Scientific use should cite the parameter sources in `references.json` and the exact code
version or commit used for the run.
