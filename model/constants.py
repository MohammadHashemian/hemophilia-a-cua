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

# ITI-specific bleeding rates
ITI_ANNUAL_ABR = 0.0
ITI_ANNUAL_AJBR = 0.0
ITI_ANNUAL_ALBR = 0.0
ITI_ANNUAL_EABR = 0.0

# Cost parameters
HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL = (
    58_000  # Cost per unit of factor VIII (Rial)
)
AVG_DOSE_PER_BLEED = 25.805  # Weighted average dose

# Transition probabilities
BLEED_RESOLUTION_PROB = 0.95  # High to maximize bleeding episodes
CRITICAL_BLEED_RESOLUTION_PROB = 0.95  # High for critical bleeds
ON_DEMAND_ARTHROPATHY_PROGRESSION_PROB = 0.0005  # Tuned for ~50% chronic, 30% severe
PROPHYLAXIS_ARTHROPATHY_PROGRESSION_PROB = 0.0001  # Tuned for ~30% chronic, 10% severe
CRITICAL_BLEED_DEATH_PROB = 2.4e-5  # 0.5% annual ICH rate * 25% fatality / 52
ON_DEMAND_COMPLICATION_PROB = 0.0001  # Lower for on-demand
PROPHYLAXIS_COMPLICATION_PROB = 0.0005  # Higher for prophylaxis (infections)
ITI_COMPLICATION_PROB = 0.00005  # Reduced for ITI
CHRONIC_ARTHROPATHY_STAY_PROB = 0.5  # Reduced to allow bleeding transitions
SEVERE_ARTHROPATHY_STAY_PROB = 0.55  # Reduced to allow bleeding transitions
ON_DEMAND_INHIBITOR_PROB = 0.0002  # ~20% lifetime risk
PROPHYLAXIS_INHIBITOR_PROB = 0.0003  # ~25% lifetime risk

# Utility values
UTILITY_NO_BLEEDING = 0.9  # Baseline, no joint damage
UTILITY_CHRONIC_ARTHROPATHY = 0.7  # Moderate joint damage
UTILITY_SEVERE_ARTHROPATHY = 0.5  # Severe joint damage
UTILITY_MINOR_BLEEDING_DECREMENT = 0.05  # Temporary decrement
UTILITY_MAJOR_BLEEDING_DECREMENT = 0.15  # Joint bleed decrement
UTILITY_CRITICAL_BLEEDING_DECREMENT = 0.3  # Life-threatening bleed
UTILITY_COMPLICATION_DECREMENT = 0.1  # Infection decrement
UTILITY_INHIBITOR_DECREMENT = 0.2  # Inhibitor treatment challenges
