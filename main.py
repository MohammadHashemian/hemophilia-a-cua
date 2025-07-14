from src.utils.logger import get_logger, suppress_matplotlib_debug
from src.data.scarper import fetch_irc_factors
from src.data.loaders import (
    load_global_hemophilia_data,
    process_amarnameh,
    merge_and_save,
    load_irc_data,
)
from src.processing.distribution_adjuster import adjust_age_distribution
from model import simulation
from pathlib import Path
import numpy as np
import typer

PROJECT_ROOT = Path(__file__).parents[0]
logger = get_logger()

app = typer.Typer()


@app.command(help="Runs loaders and scarpers, then clean and store the results.")
async def process(execute: bool):
    if execute:
        # Loading & Cleaning Amarnameh files
        (df_agg_1400, df_agg_1399), df_recombinant_1400 = process_amarnameh()
        logger.info("-" * 64)
        # Storing cleaned dataframes
        output_path = PROJECT_ROOT / "data" / "processed" / "factor_viii_analysis.xlsx"
        output_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
        merge_and_save(output_path, df_agg_1400, df_agg_1399, df_recombinant_1400)
        logger.info("-" * 64)
        # Loading global hemophilia age distribution data
        df_known_a, df_known_b = load_global_hemophilia_data()
        logger.info("-" * 64)
        known_ha = np.array([0.01, 0.12, 0.07, 0.37, 0.19])
        known_hb = np.array([0.02, 0.10, 0.06, 0.39, 0.19])
        # Add the unknown portion of patients age to iran known distribution records
        logger.info("[Hemophilia A][2023]")
        POPULATION_PROBABILITIES_HA = adjust_age_distribution(df_known_a, known_ha)
        logger.info("-" * 64)
        logger.info("[Hemophilia B][2023]")
        POPULATION_PROBABILITIES_HB = adjust_age_distribution(df_known_b, known_hb)
        logger.info("-" * 64)
        logger.warning(
            "Probably instead of adjusting to mean, should use population pyramid of iran."
        )
        logger.info("-" * 64)
        logger.info("[IRC_FDA][Pricing 2025]")
        await fetch_irc_factors()
        load_irc_data(override=False)
        return True
    else:
        logger.warning("Main runner is disable")
        return False


@app.command(help="Runs markov model simulation.")
def markov():
    suppress_matplotlib_debug()
    simulation.run()


if __name__ == "__main__":
    app()
