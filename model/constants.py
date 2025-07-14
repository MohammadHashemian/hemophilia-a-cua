# Constants for simulation parameters
NUMBER_OF_CYCLES = 73 * 52  # Total weeks in 73 years for Markov chain cycles

# Clinical parameters (events per year)
ON_DEMAND_ANNUAL_ABR = 44  # Annual bleeding rate for on-demand treatment
ON_DEMAND_ANNUAL_AJBR = 34  # Annual joint bleeding rate for on-demand treatment
PROPHYLAXIS_ANNUAL_ABR = 3.76  # Annual bleeding rate for prophylaxis
PROPHYLAXIS_ANNUAL_AJBR = 3.66  # Annual joint bleeding rate

# Life-threatening bleeding rates
LIFE_THREATENING_BLEEDING_FRACTION = 0.045
ON_DEMAND_ANNUAL_ALBR = (
    ON_DEMAND_ANNUAL_ABR * LIFE_THREATENING_BLEEDING_FRACTION
)  # 1.98 events/year
PROPHYLAXIS_ANNUAL_ALBR = (
    PROPHYLAXIS_ANNUAL_ABR * LIFE_THREATENING_BLEEDING_FRACTION
)  # 0.1692 events/year

# Extra-articular bleeding rates
EXTRA_ARTICULAR_BLEEDING_FRACTION = 0.15
ON_DEMAND_ANNUAL_EABR = (
    ON_DEMAND_ANNUAL_ABR * EXTRA_ARTICULAR_BLEEDING_FRACTION
)  # 6.6 events/year
PROPHYLAXIS_ANNUAL_EABR = (
    PROPHYLAXIS_ANNUAL_ABR * EXTRA_ARTICULAR_BLEEDING_FRACTION
)  # 0.564 events/year

# ITI-specific bleeding rates (set to 0 during ITI to focus on success/failure)
ITI_ANNUAL_ABR = 0.0
ITI_ANNUAL_AJBR = 0.0
ITI_ANNUAL_ALBR = 0.0
ITI_ANNUAL_EABR = 0.0

# Cost parameters
HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL = (
    58_000  # Cost per unit of factor VIII (Rial)
)
AVG_DOSE_PER_BLEED = 25.805  # Weighted average: 0.8*25 + 0.05*31.5 + 0.1*25 + 0.015*32 + 0.01*35 + 0.02*45

# Transition probabilities
BLEED_RESOLUTION_PROB = 0.8  # Reduced to allow longer bleeding states
CRITICAL_BLEED_RESOLUTION_PROB = 0.6  # Reduced to allow longer critical bleeds
ON_DEMAND_ARTHROPATHY_PROGRESSION_PROB = (
    0.01  # Reduced to balance higher bleeding rates
)
PROPHYLAXIS_ARTHROPATHY_PROGRESSION_PROB = (
    0.002  # Reduced to balance higher bleeding rates
)
CRITICAL_BLEED_DEATH_PROB = (
    2.4e-5  # Weekly probability: 0.5% annual ICH rate * 25% fatality / 52
)
COMPLICATION_PROB = 0.001  # Probability of complications (e.g., surgery)
ITI_COMPLICATION_PROB = COMPLICATION_PROB * 0.5  # Reduced for ITI
CHRONIC_ARTHROPATHY_STAY_PROB = 0.85  # Reduced to allow more bleeding transitions
SEVERE_ARTHROPATHY_STAY_PROB = 0.90  # Reduced to allow more bleeding transitions
ON_DEMAND_INHIBITOR_PROB = 7.94e-5  # Realistic 30% lifetime risk
PROPHYLAXIS_INHIBITOR_PROB = 4.29e-5  # Realistic 15% lifetime risk
