from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.domain.enums import Regime
from app.domain.inputs import ModelInput

# from app.visualization.matrices import visualize_matrix
from app.domain.rewards.scalar import (
    consumption,
    event_count,
    make_pettersson_score,
    utility,
    weight,
)
from app.domain.rewards.vectorized import (
    VECTORIZED_REWARD_FUNCS,
    VECTORIZED_STORE_FUNCS,
    register_state_index,
)
from app.domain.scenario import Scenario
from app.domain.transition_builder import AgeBasedMortalityModifier, build_transition_matrix
from app.persistence.context import ModelContext
from app.persistence.schemas.utilities import StateUtilities
from engine.chains import Chain, MarkovChains
from engine.vectorized import BatchMarkovChain, BatchResult


def setup_rewards(
    markov: MarkovChains, inputs: ModelInput, regime: Regime, factor: float
):
    """_summary_

    Args:
        chain (MarkovChains): _description_
        inputs (ModelInput): _description_
    """
    # required arguments passes to store and reward functions
    markov.worker_kwargs.update({"regime": regime, "inputs": inputs})
    # requirements for reward functions
    markov.add_store_function("weight", weight)
    markov.add_store_function("event_count", event_count)
    markov.add_store_function("pettersson_score", make_pettersson_score(factor=factor))
    # reward functions
    markov.add_reward_function(consumption)
    markov.add_reward_function(utility)


@dataclass(frozen=True)
class ModelOutput:
    cycles: int
    regime: Regime
    total_factor: float
    factor_consumption: list[float]
    total_qaly: float
    qalys: list[float]
    mean_weight: float
    pettersson_score: int
    absorbed_at: int | None
    sequence: list[str]
    event_count: list[int]


def build_output(markov: MarkovChains, sequences, rewards, absorbed_at):
    cycles = markov.steps

    factor_seq = rewards[consumption.__name__][:]
    utility_seq = rewards[utility.__name__][:]
    weight_seq = rewards[weight.__name__][:]
    event_seq = rewards[event_count.__name__][:]
    pettersson_seq = rewards["pettersson_score"][:]

    regime = markov.worker_kwargs.get("regime", None)
    if not regime or not isinstance(regime, Regime):
        raise ValueError("regime is not defined")

    factor_sum = np.sum(factor_seq)
    qaly_sum = np.sum(utility_seq)
    mean_weight = np.mean(weight_seq)
    pettersson_score = int(pettersson_seq[-1])

    return ModelOutput(
        cycles=cycles,
        regime=regime,
        total_factor=factor_sum,
        factor_consumption=factor_seq,
        total_qaly=qaly_sum,
        qalys=utility_seq,
        mean_weight=mean_weight,
        pettersson_score=pettersson_score,
        absorbed_at=absorbed_at,
        event_count=event_seq,
        sequence=sequences,
    )


def _init_worker(inputs: ModelInput, context: ModelContext):
    global _reward_constants
    ctx = context
    inp = inputs

    utils = StateUtilities(
        healthy=inp.healthy_utility,
        bleeding=inp.spontaneous_bleeding_utility,
        hemarthrosis=inp.joint_bleeding_utility,
        lt_bleeding=inp.life_threatening_bleeding_utility,
        death=inp.death_utility,
        mild_arthropathy=inp.mild_arthropathy_utility,
        moderate_arthropathy=inp.moderate_arthropathy_utility,
        severe_arthropathy=inp.severe_arthropathy_utility,
    )

    _reward_constants = {
        "lam_bleed": inp.spontaneous_bleeding_rate / 52,
        "lam_joint": inp.joint_bleeding_rate / 52,
        "weekly_discount": (
            (1 + inp.benefits_discount_rate) ** (1 / 52) - 1
            if inp.benefits_discount_rate
            else 0
        ),
        # Coerce to int so the scalar ``weight`` reward (which calls
        # ``cal_body_weight``) is not poisoned by the float annotation on
        # ``ModelInput.baseline_age``. The vectorized path explicitly casts
        # to int as well, so both paths agree.
        "baseline_age_weeks": int(inp.baseline_age * 52),
        # flatten deep structures
        "utilities": utils,
        "threshold_mild": ctx.clinical.clinical_scoring.pettersson_score.thresholds.mild,
        "threshold_moderate": ctx.clinical.clinical_scoring.pettersson_score.thresholds.moderate,
        "threshold_max": ctx.clinical.clinical_scoring.pettersson_score.thresholds.max,
        "conversion_factor": ctx.clinical.clinical_scoring.pettersson_score.conversion_factor,
    }


def worker_function(
    chain: Chain,
    inputs: ModelInput,
    scenario: Scenario,
    context: ModelContext,
    run_id: int = 0,
    worker_id: int = 0,
):
    """
    Single markov model initializer function

    - Build transition matrix
    - Configure markov chain
    - Execute simulation
    - Aggregate outputs
    """
    # 0. Precompute constants
    _init_worker(inputs, context)

    # 1. Build transitions
    matrix = build_transition_matrix(inputs, chain.states)
    chain.update(matrix)

    # # DEBUG: Visualization
    # visualize_matrix(
    #     matrix=matrix,
    #     states=chain.states,
    #     inputs=inputs,
    #     sub=str(run_id),
    #     filename=f"matrix_{scenario.regime}_{worker_id}",
    # )

    # 2. Configure chain
    regime = scenario.regime
    chains = [chain]
    markov = MarkovChains(
        chains=chains,
        entrance="healthy",
        conditions=None,
        entrance_chain="main",
        steps=int(inputs.cycle),
        transition_modifier=AgeBasedMortalityModifier(
            mortality_file=context.mortality, baseline_age=inputs.baseline_age
        ),
        worker_kwargs={},
        absorbing_states={"death"},
    )
    markov.worker_kwargs.update(
        {
            "const": _reward_constants,
            "rng": np.random.default_rng(
                context.simulation.environment.seed + worker_id
            ),  # Required for AgeBaseMortalityModifier
        }
    )
    setup_rewards(
        markov=markov,
        inputs=inputs,
        regime=regime,
        factor=_reward_constants["conversion_factor"],
    )

    # 3. Run simulation
    sequences = markov.run()
    absorbed_at = markov.absorbed_at
    rewards = markov.collect_rewards()

    # # 4. Build outputs
    outputs = build_output(markov, sequences, rewards, absorbed_at)

    return outputs


# =============================================================================
# Vectorized batch worker: runs N independent simulations in one numpy sweep.
# =============================================================================


def _build_shared_per_iter(inputs: list[ModelInput], states: list[str]) -> dict:
    """Stack per-iter constants into (n_iters,) arrays for vectorized reward funcs."""
    per_iter: dict[str, np.ndarray] = {
        "lam_bleed": np.array(
            [inp.spontaneous_bleeding_rate / 52 for inp in inputs], dtype=np.float64
        ),
        "lam_joint": np.array(
            [inp.joint_bleeding_rate / 52 for inp in inputs], dtype=np.float64
        ),
        "weight_factor": np.array(
            [inp.weight_factor for inp in inputs], dtype=np.float64
        ),
        "healthy_utility": np.array(
            [inp.healthy_utility for inp in inputs], dtype=np.float64
        ),
        "mild_arthropathy_utility": np.array(
            [inp.mild_arthropathy_utility for inp in inputs], dtype=np.float64
        ),
        "moderate_arthropathy_utility": np.array(
            [inp.moderate_arthropathy_utility for inp in inputs], dtype=np.float64
        ),
        "severe_arthropathy_utility": np.array(
            [inp.severe_arthropathy_utility for inp in inputs], dtype=np.float64
        ),
        "spontaneous_bleeding_utility": np.array(
            [inp.spontaneous_bleeding_utility for inp in inputs], dtype=np.float64
        ),
        "joint_bleeding_utility": np.array(
            [inp.joint_bleeding_utility for inp in inputs], dtype=np.float64
        ),
        "life_threatening_bleeding_utility": np.array(
            [inp.life_threatening_bleeding_utility for inp in inputs], dtype=np.float64
        ),
        "death_utility": np.array(
            [inp.death_utility for inp in inputs], dtype=np.float64
        ),
        "prophylaxis_background_factor_consumption_per_kg": np.array(
            [
                inp.prophylaxis_background_factor_consumption_per_kg
                for inp in inputs
            ],
            dtype=np.float64,
        ),
        "factor_consumption_per_spontaneous_bleeding_per_kg": np.array(
            [
                inp.factor_consumption_per_spontaneous_bleeding_per_kg
                for inp in inputs
            ],
            dtype=np.float64,
        ),
        "factor_consumption_per_joint_bleeding_per_kg": np.array(
            [inp.factor_consumption_per_joint_bleeding_per_kg for inp in inputs],
            dtype=np.float64,
        ),
        "factor_consumption_per_life_threatening_bleeding_per_kg": np.array(
            [inp.factor_consumption_per_life_threatening_bleeding_per_kg for inp in inputs],
            dtype=np.float64,
        ),
    }
    return per_iter


def _build_all_matrices(
    inputs: list[ModelInput], states: list[str]
) -> np.ndarray:
    """Build per-iter transition matrices and stack into (n_iters, n, n)."""
    matrices = np.empty((len(inputs), len(states), len(states)), dtype=np.float64)
    for i, inp in enumerate(inputs):
        matrices[i] = build_transition_matrix(inp, states)
    return matrices


def _aggregate_vectorized_output(
    inputs: list[ModelInput],
    batch: BatchResult,
    states: list[str],
    regime: Regime,
    idx: int,
) -> ModelOutput:
    """Build a single ModelOutput from the (n_iters, ...) batch arrays at index idx."""
    sequences = batch.sequences[idx]
    # ``walk_batch`` records ``steps + 1`` for iters that never reach an
    # absorbing state; map that sentinel to ``None`` so downstream
    # ``is_absorbed`` semantics match the scalar engine exactly.
    steps = sequences.shape[0] - 1
    raw_absorbed = int(batch.absorbed_at[idx])
    absorbed_at = None if raw_absorbed > steps else raw_absorbed
    rewards = batch.rewards

    factor_seq = rewards["consumption"][idx]
    utility_seq = rewards["utility"][idx]
    weight_seq = rewards["weight"][idx]
    event_seq = rewards["event_count"][idx]
    pettersson_seq = rewards["pettersson_score"][idx]

    return ModelOutput(
        cycles=int(inputs[idx].cycle),
        regime=regime,
        total_factor=float(np.sum(factor_seq)),
        factor_consumption=factor_seq.tolist(),
        total_qaly=float(np.sum(utility_seq)),
        qalys=utility_seq.tolist(),
        mean_weight=float(np.mean(weight_seq)),
        pettersson_score=int(pettersson_seq[-1]),
        absorbed_at=absorbed_at,
        sequence=[states[s] for s in sequences.tolist()],
        event_count=event_seq.astype(int).tolist(),
    )


def worker_function_batch(
    chain: Chain,
    inputs: list[ModelInput],
    scenario: Scenario,
    context: ModelContext,
    run_id: int = 0,
    worker_id: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_every: int = 52,
) -> list[ModelOutput]:
    """Vectorized batch worker: runs all `inputs` in a single numpy sweep.

    Replaces 1-at-a-time calls to `worker_function` for the same scenario
    with a single call that processes every iter in parallel via vectorized
    numpy ops. Output ordering matches input ordering.

    `progress_callback(step, total_steps)` is invoked every `progress_every`
    steps when provided. Each call represents `n_iters` iters advancing
    one step in lockstep, so the caller can convert to iter-progress by
    multiplying by len(inputs).
    """
    if not inputs:
        return []

    states = list(chain.states)

    # Ensure state index lookup for vectorized reward funcs is registered.
    for i, s in enumerate(states):
        register_state_index(s, i)

    # 1. Build all transition matrices
    matrices = _build_all_matrices(inputs, states)

    # 2. Build vectorized chain
    batch_chain = BatchMarkovChain.from_chain(
        matrices=matrices,
        states=states,
        absorbing_states={"death"},
        mortality_file=context.mortality,
        baseline_age=inputs[0].baseline_age,
    )

    # 3. Build per-iter constants
    per_iter = _build_shared_per_iter(inputs, states)
    steps = int(inputs[0].cycle)
    weekly_discount = (
        (1 + inputs[0].benefits_discount_rate) ** (1 / 52) - 1
        if inputs[0].benefits_discount_rate
        else 0.0
    )

    shared_kwargs: dict = {
        "regime": scenario.regime,
        "per_iter": per_iter,
        "thresholds": {
            "mild": context.clinical.clinical_scoring.pettersson_score.thresholds.mild,
            "moderate": context.clinical.clinical_scoring.pettersson_score.thresholds.moderate,
            "max": context.clinical.clinical_scoring.pettersson_score.thresholds.max,
        },
        "conversion_factor": context.clinical.clinical_scoring.pettersson_score.conversion_factor,
        "weekly_discount": weekly_discount,
        "baseline_age_weeks": float(inputs[0].baseline_age) * 52,
    }

    # 4. Run vectorized walk
    rng = np.random.default_rng(
        context.simulation.environment.seed + worker_id
    )
    entrance_idx = states.index("healthy") if "healthy" in states else 0
    batch_result = batch_chain.walk_batch(
        steps=steps,
        entrance_idx=entrance_idx,
        rng=rng,
        store_funcs=VECTORIZED_STORE_FUNCS,
        reward_funcs=VECTORIZED_REWARD_FUNCS,
        shared_kwargs=shared_kwargs,
        progress_callback=progress_callback,
        progress_every=progress_every,
    )

    # 5. Aggregate per-iter outputs
    outputs = [
        _aggregate_vectorized_output(inputs, batch_result, states, scenario.regime, i)
        for i in range(len(inputs))
    ]
    return outputs
