from model_dep.schemas import Regimes, Treatment
from src.utils.logger import get_logger
from model_dep.constants import (
    ON_DEMAND_ANNUAL_ABR,
    ON_DEMAND_ANNUAL_AJBR,
    ON_DEMAND_ANNUAL_ALBR,
    ON_DEMAND_ANNUAL_EABR,
    PROPHYLAXIS_ANNUAL_ABR,
    PROPHYLAXIS_ANNUAL_AJBR,
    PROPHYLAXIS_ANNUAL_ALBR,
    PROPHYLAXIS_ANNUAL_EABR,
    AVG_DOSE_PER_BLEED,
)

logger = get_logger()


def initialize_treatments() -> dict[Regimes, Treatment]:
    """Initialize treatment regimes with their respective parameters, converting daily durations to weekly."""
    logger.debug(
        f"OD Bleedings ABR: {ON_DEMAND_ANNUAL_ABR}, and AJBR + ALBR + EABR sums to: {ON_DEMAND_ANNUAL_AJBR + ON_DEMAND_ANNUAL_ALBR + ON_DEMAND_ANNUAL_EABR}"
    )
    logger.debug(
        f"Prophylaxis Bleedings ABR: {PROPHYLAXIS_ANNUAL_ABR}, and AJBR + ALBR + EABR sums to: {PROPHYLAXIS_ANNUAL_AJBR + PROPHYLAXIS_ANNUAL_ALBR + PROPHYLAXIS_ANNUAL_EABR}"
    )
    return {
        Regimes.ON_DEMAND: Treatment(
            name=Regimes.ON_DEMAND.value,
            dose_joint=25,
            dose_muscle=31.5,
            dose_mucous=25,
            dose_intracranial=32,
            dose_neck_throat=35,
            dose_gastro=45,
            duration_joint=2 / 7,  # 2 days to weeks
            duration_muscle=5 / 7,  # 5 days to weeks
            duration_mucous=3 / 7,  # 3 days to weeks
            duration_intracranial=21 / 7,  # 21 days to weeks
            duration_neck_throat=14 / 7,  # 14 days to weeks
            duration_gastro=11 / 7,  # 11 days to weeks
            avg_dose_per_bleed=AVG_DOSE_PER_BLEED,
            abr=ON_DEMAND_ANNUAL_ABR,
            ajbr=ON_DEMAND_ANNUAL_AJBR,
            albr=ON_DEMAND_ANNUAL_ALBR,
            eabr=ON_DEMAND_ANNUAL_EABR,
        ),
        Regimes.PROPHYLAXIS: Treatment(
            name=Regimes.PROPHYLAXIS.value,
            dose_joint=25,
            dose_muscle=31.5,
            dose_mucous=25,
            dose_intracranial=32,
            dose_neck_throat=35,
            dose_gastro=45,
            duration_joint=2 / 7,  # 2 days to weeks
            duration_muscle=5 / 7,  # 5 days to weeks
            duration_mucous=3 / 7,  # 3 days to weeks
            duration_intracranial=21 / 7,  # 21 days to weeks
            duration_neck_throat=14 / 7,  # 14 days to weeks
            duration_gastro=11 / 7,  # 11 days to weeks
            avg_dose_per_bleed=AVG_DOSE_PER_BLEED,
            abr=PROPHYLAXIS_ANNUAL_ABR,
            ajbr=PROPHYLAXIS_ANNUAL_AJBR,
            albr=PROPHYLAXIS_ANNUAL_ALBR,
            eabr=PROPHYLAXIS_ANNUAL_EABR,
        ),
    }
