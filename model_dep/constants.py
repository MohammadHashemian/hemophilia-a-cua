# Constants for simulation parameters
NUMBER_OF_CYCLES = 73 * 52  # Total weeks in 73 years for Markov chain cycles

# Clinical parameters (events per year)
ON_DEMAND_ANNUAL_ABR = 44  # Annual bleeding rate for on-demand treatment
ON_DEMAND_ANNUAL_AJBR = 34  # Annual joint bleeding rate for on-demand treatment
PROPHYLAXIS_ANNUAL_ABR = 3.76  # Annual bleeding rate for prophylaxis
PROPHYLAXIS_ANNUAL_AJBR = 3.66  # Annual joint bleeding rate

# Life-threatening bleeding rates (LTB)
LIFE_THREATENING_BLEEDING_FRACTION = 0.045
ON_DEMAND_ANNUAL_ALBR = round(
    ON_DEMAND_ANNUAL_ABR * LIFE_THREATENING_BLEEDING_FRACTION
)  # 1.98 events/year ~ rounded to 2
PROPHYLAXIS_ANNUAL_ALBR = (
    PROPHYLAXIS_ANNUAL_ABR * LIFE_THREATENING_BLEEDING_FRACTION
)  # 0.1692 events/year

# Extra-articular bleeding rates (Minor)
EXTRA_ARTICULAR_BLEEDING_FRACTION = 0.15
ON_DEMAND_ANNUAL_EABR = (
    round(ON_DEMAND_ANNUAL_ABR * EXTRA_ARTICULAR_BLEEDING_FRACTION) + 1
)  # 6.6 events/year ~ rounded to 8 to scale od probs
PROPHYLAXIS_ANNUAL_EABR = (
    PROPHYLAXIS_ANNUAL_ABR * EXTRA_ARTICULAR_BLEEDING_FRACTION
)  # 0.564 events/year

# Cost parameters
HUMAN_DERIVED_FACTOR_VIII_PER_UNIT_PRICE_RIAL = (
    58_000  # Cost per unit of factor VIII (Rial)
)
AVG_DOSE_PER_BLEED = 25.805  # Weighted average dose

# Transition probabilities
ITI_RESOLUTION_PROB = 0.0085  # Weekly Success Probability over 26 weak therapy
BLEED_RESOLUTION_PROB = 0.95  # High to maximize bleeding episodes
CRITICAL_BLEED_RESOLUTION_PROB = 0.95  # High for critical bleeds
CRITICAL_BLEED_DEATH_PROB = 2.4e-5  # 0.5% annual ICH rate * 25% fatality / 52
# Inhibitor probabilities for first 104 weeks (2 years)
ON_DEMAND_INHIBITOR_PROB_EARLY = 0.00414  # ~35% risk over first 2 years
PROPHYLAXIS_INHIBITOR_PROB_EARLY = 0.00215  # ~20% risk over first 2 years
# Disease progression probs for weakly intervals
ON_DEMAND_ARTHROPATHY_PROGRESSION_PROB = 0.000435  # ~80% lifetime risk
PROPHYLAXIS_ARTHROPATHY_PROGRESSION_PROB = 0.0000939  # ~30% lifetime risk

# Utility values
UTILITY_NO_BLEEDING = 0.9  # Baseline, no joint damage
UTILITY_CHRONIC_ARTHROPATHY = 0.8  # Moderate joint damage
UTILITY_SEVERE_ARTHROPATHY = 0.7  # Severe joint damage
UTILITY_MINOR_BLEEDING_DECREMENT = 0.65  # Temporary decrement
UTILITY_MAJOR_BLEEDING_DECREMENT = 0.4  # Joint bleed decrement
UTILITY_CRITICAL_BLEEDING_DECREMENT = 0.3  # Life-threatening bleed
UTILITY_INHIBITOR_DECREMENT = 0.4  # Inhibitor treatment challenges
