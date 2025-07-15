from enlighten import Counter
from src.utils.logger import get_logger
from src.data.loaders import PROJECT_ROOT
from model.treatments import initialize_treatments
from model.transition_matrix import initialize_transition_matrix
from model.analysis import calculate_average_abr_ajbr, visualize_results
from model.markov import MarkovChain
from model.schemas import Regimes, Status
from model.constants import (
    NUMBER_OF_CYCLES,
    HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL,
)
import numpy as np
import asyncio
import multiprocessing
from multiprocessing import Pool
from tqdm.asyncio import tqdm
import time

logger = get_logger()


import enlighten
import multiprocessing
import asyncio
from multiprocessing import Pool
import time
import logging

logger = logging.getLogger(__name__)


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
        f"Using {num_processes} processes for {markov_chain.regime.value} simulations"
    )

    # Initialize Enlighten progress bar
    manager = enlighten.get_manager()
    pbar: Counter = manager.counter(
        total=num_runs, desc=f"Simulating {markov_chain.regime.value}", leave=True
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
        f"Total simulation time for {markov_chain.regime.value}: {time.time() - start_total:.2f} seconds"
    )
    return results


def run():
    """Main function to run the Markov chain simulation for hemophilia A."""
    od_matrix_path = PROJECT_ROOT / "data" / "processed" / "od_transition_matrix.csv"
    pro_matrix_path = PROJECT_ROOT / "data" / "processed" / "pro_transition_matrix.csv"
    treatments = initialize_treatments()
    states = [state.value for state in Status]
    od_matrix = initialize_transition_matrix(od_matrix_path, Regimes.ON_DEMAND)
    pro_matrix = initialize_transition_matrix(pro_matrix_path, Regimes.PROPHYLAXIS)
    od_matrix = od_matrix.to_numpy()
    pro_matrix = pro_matrix.to_numpy()

    initial_state_probs = np.zeros(len(states))
    no_bleeding_idx = states.index(Status.NO_BLEEDING.value)
    initial_state_probs[no_bleeding_idx] = 1.0

    od_markov_chain = MarkovChain(
        states=states,
        transition_matrix=od_matrix.tolist(),
        initial_state_probs=initial_state_probs.tolist(),
        treatment=treatments[Regimes.ON_DEMAND],
        price_per_unit=HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL,
        regime=Regimes.ON_DEMAND,
    )
    pro_markov_chain = MarkovChain(
        states=states,
        transition_matrix=pro_matrix.tolist(),
        initial_state_probs=initial_state_probs.tolist(),
        treatment=treatments[Regimes.PROPHYLAXIS],
        price_per_unit=HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL,
        regime=Regimes.PROPHYLAXIS,
    )

    logger.info("Starting Markov chain simulation with dose and cost calculations")
    num_steps = NUMBER_OF_CYCLES
    num_runs = 100
    num_years = num_steps / 52

    od_results = asyncio.run(run_simulations(od_markov_chain, num_steps, num_runs))
    pro_results = asyncio.run(run_simulations(pro_markov_chain, num_steps, num_runs))

    od_avg_abr, od_avg_ajbr = calculate_average_abr_ajbr(
        od_results, num_years, Regimes.ON_DEMAND
    )
    pro_avg_abr, pro_avg_ajbr = calculate_average_abr_ajbr(
        pro_results, num_years, Regimes.PROPHYLAXIS
    )
    logger.info(
        f"Average ABR over {num_years:.2f} years: "
        f"On-demand = {od_avg_abr:.2f} bleeds/year, Prophylaxis = {pro_avg_abr:.2f} bleeds/year"
    )
    logger.info(
        f"Average AJBR over {num_years:.2f} years: "
        f"On-demand = {od_avg_ajbr:.2f} joint bleeds/year, Prophylaxis = {pro_avg_ajbr:.2f} joint bleeds/year"
    )

    total_dose_od = np.mean([sum(r["weekly_doses"]) for r in od_results])
    total_cost_od = np.mean([sum(r["weekly_costs"]) for r in od_results])
    total_dose_pro = np.mean([sum(r["weekly_doses"]) for r in pro_results])
    total_cost_pro = np.mean([sum(r["weekly_costs"]) for r in pro_results])
    logger.info(
        f"Average total factor VIII dose over {num_years:.2f} years: "
        f"On-demand = {total_dose_od:.2f} IU, Prophylaxis = {total_dose_pro:.2f} IU"
    )
    logger.info(
        f"Average total cost over {num_years:.2f} years: "
        f"On-demand = {total_cost_od:.2f} Rial, Prophylaxis = {total_cost_pro:.2f} Rial"
    )

    for week in range(min(num_steps, 5)):
        logger.debug(
            f"Week {week}, Age {week/52:.2f} years, "
            f"OD State: {od_results[0]['state_path'][week]}, OD Dose: {od_results[0]['weekly_doses'][week]} IU, "
            f"OD Cost: {od_results[0]['weekly_costs'][week]} Rial, "
            f"PRO State: {pro_results[0]['state_path'][week]}, PRO Dose: {pro_results[0]['weekly_doses'][week]} IU, "
            f"PRO Cost: {pro_results[0]['weekly_costs'][week]} Rial"
        )

    visualize_results(od_results, pro_results, states)
