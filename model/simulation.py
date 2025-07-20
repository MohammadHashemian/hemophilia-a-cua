from enlighten import Counter
from src.utils.logger import get_logger
from src.data.loaders import PROJECT_ROOT
from model.treatments import initialize_treatments
from model.transition_matrix import initialize_transition_matrix
from model.analysis import calculate_average_abr_ajbr, visualize_results
from model.markov import MarkovChain
from model.schemas import (
    Regimes,
    BaseStates,
    EARLY_MODEL,
    INTERMEDIATE_MODEL,
    END_MODEL,
)
from model.constants import (
    NUMBER_OF_CYCLES,
    HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL,
)
import numpy as np
import asyncio
import multiprocessing
from multiprocessing import Pool
import enlighten
import time

logger = get_logger()


def run_simulation(args):
    """Run a single Markov chain simulation (for multiprocessing)."""
    markov_chain, num_steps = args
    markov_chain: MarkovChain
    result = markov_chain.simulate(num_steps)
    return result


async def run_simulations(markov_chain: MarkovChain, num_steps: int, num_runs: int):
    """Run multiple simulations using multiprocessing with a single progress bar."""
    results = []
    start_total = time.time()
    num_processes = min(multiprocessing.cpu_count(), 4)
    logger.info(
        f"Using {num_processes} processes for {markov_chain.regime.value} {markov_chain.model.stage} simulations"
    )

    # Initialize Enlighten progress bar
    manager = enlighten.get_manager()
    pbar: Counter = manager.counter(
        total=num_runs,
        desc=f"Simulating {markov_chain.regime.value} {markov_chain.model.stage}",
        leave=True,
    )

    def update_pbar(_):
        pbar.update(1)

    with Pool(processes=num_processes) as pool:
        tasks = [
            pool.apply_async(
                run_simulation,
                args=((markov_chain, num_steps),),
                callback=update_pbar,
            )
            for _ in range(num_runs)
        ]
        for task in tasks:
            result = task.get()
            results.append(result)
            await asyncio.sleep(0.01)

    pbar.close()
    manager.stop()

    logger.info(
        f"Total simulation time for {markov_chain.regime.value} {markov_chain.model.stage}: {time.time() - start_total:.2f} seconds"
    )
    return results


def run():
    """Main function to run the Markov chain simulation for hemophilia A across all models and regimes."""
    # Define paths for transition matrices
    matrix_paths = {
        (Regimes.ON_DEMAND, "early"): PROJECT_ROOT
        / "data"
        / "processed"
        / "od_early_transition_matrix.csv",
        (Regimes.ON_DEMAND, "intermediate"): PROJECT_ROOT
        / "data"
        / "processed"
        / "od_intermediate_transition_matrix.csv",
        (Regimes.ON_DEMAND, "end"): PROJECT_ROOT
        / "data"
        / "processed"
        / "od_end_transition_matrix.csv",
        (Regimes.PROPHYLAXIS, "early"): PROJECT_ROOT
        / "data"
        / "processed"
        / "pro_early_transition_matrix.csv",
        (Regimes.PROPHYLAXIS, "intermediate"): PROJECT_ROOT
        / "data"
        / "processed"
        / "pro_intermediate_transition_matrix.csv",
        (Regimes.PROPHYLAXIS, "end"): PROJECT_ROOT
        / "data"
        / "processed"
        / "pro_end_transition_matrix.csv",
    }

    # Initialize treatments
    treatments = initialize_treatments()

    # Define models
    models = {
        "early": EARLY_MODEL,
        "intermediate": INTERMEDIATE_MODEL,
        "end": END_MODEL,
    }

    # Generate transition matrices and run simulations
    results = {}
    num_steps = NUMBER_OF_CYCLES
    num_runs = 100
    num_years = num_steps / 52

    for regime in [Regimes.ON_DEMAND, Regimes.PROPHYLAXIS]:
        for model_name, model in models.items():
            # Generate transition matrix
            matrix_path = matrix_paths[(regime, model_name)]
            transition_matrix = initialize_transition_matrix(
                matrix_path, regime, model, override=True
            )
            states = model.states_value
            transition_matrix_np = transition_matrix.to_numpy()

            # Set initial state probabilities (start in NO_BLEEDING)
            initial_state_probs = np.zeros(len(states))
            no_bleeding_idx = (
                states.index(BaseStates.NO_BLEEDING.value)
                if BaseStates.NO_BLEEDING.value in states
                else 0
            )
            initial_state_probs[no_bleeding_idx] = 1.0

            # Initialize MarkovChain with model
            markov_chain = MarkovChain(
                states=states,
                transition_matrix=transition_matrix_np.tolist(),
                initial_state_probs=initial_state_probs.tolist(),
                treatment=treatments[regime],
                price_per_unit=HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL,
                regime=regime,
                model=model,  # Pass model for logging
            )

            logger.info(
                f"Starting Markov chain simulation for {regime.value} {model_name}"
            )
            sim_results = asyncio.run(
                run_simulations(markov_chain, num_steps, num_runs)
            )
            results[(regime, model_name)] = sim_results

            # Calculate ABR and AJBR
            avg_abr, avg_ajbr = calculate_average_abr_ajbr(
                sim_results, num_years, regime
            )
            logger.info(
                f"{regime.value} {model_name}: Average ABR over {num_years:.2f} years: {avg_abr:.2f} bleeds/year, "
                f"Average AJBR: {avg_ajbr:.2f} joint bleeds/year"
            )

            # Calculate total dose and cost
            total_dose = np.mean([sum(r["weekly_doses"]) for r in sim_results])
            total_cost = np.mean([sum(r["weekly_costs"]) for r in sim_results])
            logger.info(
                f"{regime.value} {model_name}: Average total factor VIII dose over {num_years:.2f} years: {total_dose:.2f} IU, "
                f"Average total cost: {total_cost:.2f} Rial"
            )

            # Log first few weeks
            for week in range(min(num_steps, 5)):
                logger.debug(
                    f"{regime.value} {model_name} Week {week}, Age {week/52:.2f} years, "
                    f"State: {sim_results[0]['state_path'][week]}, "
                    f"Dose: {sim_results[0]['weekly_doses'][week]:.2f} IU, "
                    f"Cost: {sim_results[0]['weekly_costs'][week]:.2f} Rial"
                )

    # Visualize results for each model
    for model_name, model in models.items():
        od_results = results.get((Regimes.ON_DEMAND, model_name), [])
        pro_results = results.get((Regimes.PROPHYLAXIS, model_name), [])
        if od_results and pro_results:
            visualize_results(od_results, pro_results, model.states_value, model_name)


if __name__ == "__main__":
    run()
