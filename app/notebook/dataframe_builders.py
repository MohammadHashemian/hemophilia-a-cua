
from collections import Counter

import numpy as np
import polars as pl

from app.domain.enums import HealthStates
from app.domain.inputs import ModelInput
from app.domain.worker import ModelOutput
from app.notebook.scenario_helpers import parse_scenario
from app.persistence.context import ModelContext
from engine.runners import SimulationResult
from utils.logging import setup_root_logger

columns = [
    "scenario",
    "time_horizon",
    "regime",
    "extension",
    "sampling_method",
    "sampled_abr",
    "cycles",
    "observed_cycles",
    "person_years",
    "lost_person_years",
    "lost_person_year_life_expectancy",
    "total_factor",
    "total_qaly",
    "total_cost",
    "annual_cost",
    "pettersson_score",
    "absorbed_at",
    "is_absorbed",
    "bleeding_events",
    "spontaneous_bleeding_events",
    "joint_bleeding_events",
    "life_threatening_events",
    "annual_bleeding_rate",
    "spontaneous_bleeding_rate",
    "annual_joint_bleeding_rate",
    "annual_life_threatening_rate",
]

occupation_columns = [
    "healthy_weeks",
    "healthy_share",
    "bleeding_weeks",
    "bleeding_share",
    "hemarthrosis_weeks",
    "hemarthrosis_share",
    "lt_bleeding_weeks",
    "lt_bleeding_share",
    "death_weeks",
    "death_share",
]


def require(object, typo):
    if not object or not isinstance(object, typo):
        raise ValueError("required object is not provided")
    return object


def calculate_state_occupation(seq, states: list["str"]) -> dict:
    """
    Calculate state occupation weeks and shares.

    Parameters
    ----------
    seq : list[str]
        State sequence truncated to observed cycles.
    states : list[str]
        List of states to calculate occupation for.

    Returns
    -------
    dict
    """

    counter = Counter(seq)
    total_weeks = len(seq)

    occupation = {}

    for state in states:
        weeks = counter.get(state, 0)

        occupation[f"{state}_weeks"] = weeks
        occupation[f"{state}_share"] = weeks / total_weeks if total_weeks > 0 else 0.0

    return occupation


def build_df(
    results: list["SimulationResult"],
    context: ModelContext,
    options: dict | None = {},
) -> pl.DataFrame:
    """Build a polars DataFrame from a batch of simulation results.

    Each row corresponds to a single ``SimulationResult`` and contains
    the model inputs, summary outputs, and state-occupation aggregates
    listed in ``columns`` + ``occupation_columns``.
    """
    logger = setup_root_logger()

    cost_unit = context.costs.currencies[0].code
    per_unit_cost = context.costs.costs[0].pricing.per_unit[cost_unit]

    data: list[dict] = []

    for result in results:
        inputs = require(result.input_data, ModelInput)
        output = require(result.output, ModelOutput)

        if output is None:
            continue

        cycles = int(output.cycles)
        end = int(output.absorbed_at) if output.absorbed_at is not None else cycles
        end = min(end, cycles)

        seq = output.sequence[:end]
        event_seq = output.event_count[:end]

        if len(seq) != len(event_seq):
            logger.warning(
                f"Sequence length mismatch in scenario={result.scenario}: "
                f"len(seq)={len(seq)}, len(event_seq)={len(event_seq)}"
            )
            continue

        discount_rate = inputs.costs_discount_rate
        weekly_discount = (1 + discount_rate) ** (1 / 52) - 1 if discount_rate else 0

        factor_seq = output.factor_consumption[:end]

        if weekly_discount:
            discounted_costs = [
                (factor * per_unit_cost) / ((1 + weekly_discount) ** step)
                for step, factor in enumerate(factor_seq)
            ]
            total_cost = sum(discounted_costs)
        else:
            total_cost = sum(factor_seq) * per_unit_cost

        total_cost = float(total_cost)

        annual_cost = (total_cost / end * 52) if end > 0 else 0
        person_years = end / 52

        if person_years <= 0:
            continue

        lost_person_years = (
            ((cycles - end) / 52) if output.absorbed_at is not None else 0
        )

        life_expectancy_in_week = 72 * 52
        lost_person_year_life_expectancy = (
            (life_expectancy_in_week - end) / 52
            if (output.absorbed_at is not None and end <= life_expectancy_in_week)
            else 0
        )

        bleeding_events = int(np.sum(event_seq))

        spontaneous_bleeding_events = int(
            sum(ec for s, ec in zip(seq, event_seq) if s == "bleeding")
        )

        joint_bleeding_events = int(
            sum(ec for s, ec in zip(seq, event_seq) if s == "hemarthrosis")
        )

        life_threatening_events = int(
            sum(ec for s, ec in zip(seq, event_seq) if s == "lt_bleeding")
        )

        annual_bleeding_rate = bleeding_events / person_years
        spontaneous_bleeding_rate = spontaneous_bleeding_events / person_years
        annual_joint_bleeding_rate = joint_bleeding_events / person_years
        life_threatening_rate = life_threatening_events / person_years

        parts = parse_scenario(result.scenario)
        time_horizon, regime, sampling_method, extension = parts

        if len(parts) == 3:
            extension = None

        row = {
            "scenario": result.scenario,
            "time_horizon": time_horizon,
            "regime": regime,
            "extension": extension,
            "sampling_method": sampling_method,
            "sampled_abr": inputs.bleeding_rate,
            "cycles": cycles,
            "observed_cycles": end,
            "person_years": person_years,
            "lost_person_years": lost_person_years,
            "lost_person_year_life_expectancy": lost_person_year_life_expectancy,
            "total_factor": int(output.total_factor),
            "total_qaly": float(output.total_qaly),
            "total_cost": total_cost,
            "annual_cost": annual_cost,
            "pettersson_score": output.pettersson_score,
            "absorbed_at": output.absorbed_at,
            "is_absorbed": output.absorbed_at is not None,
            "bleeding_events": bleeding_events,
            "spontaneous_bleeding_events": spontaneous_bleeding_events,
            "joint_bleeding_events": joint_bleeding_events,
            "life_threatening_events": life_threatening_events,
            "annual_bleeding_rate": annual_bleeding_rate,
            "spontaneous_bleeding_rate": spontaneous_bleeding_rate,
            "annual_joint_bleeding_rate": annual_joint_bleeding_rate,
            "annual_life_threatening_rate": life_threatening_rate,
        }
        state_occupation = calculate_state_occupation(
            seq,
            states=[state for state in HealthStates],
        )
        row.update(state_occupation)

        data.append(row)

    df = pl.DataFrame(data, infer_schema_length=10000)

    # Polars infers nullable columns as ``Null`` when a batch happens to
    # have only nulls in that column (e.g. ``extension`` is None for
    # every "base" scenario in a batch). When a later batch carries
    # any string value, the same column is inferred as ``String`` and
    # ``pl.concat(..., how="vertical")`` refuses to vstack the two
    # frames. Force a stable schema by casting each known nullable
    # column to its non-null dtype; nulls are valid String / Float64
    # values.
    nullable_casts: dict[str, pl.DataType] = {
        "extension": pl.String,
        "absorbed_at": pl.Float64,
    }
    for col, dtype in nullable_casts.items():
        if col in df.columns and df.schema[col] != dtype:
            df = df.with_columns(pl.col(col).cast(dtype))

    return df
