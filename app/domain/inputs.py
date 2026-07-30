from dataclasses import dataclass


# ------ Transformation Layer ------
@dataclass(frozen=True)
class ModelInput:
    # Time
    cycle: float

    # Clinical
    bleeding_rate: float
    spontaneous_bleeding_rate: float  # computed property
    joint_bleeding_rate: float
    intracranial_hemorrhage_rate: float
    non_ich_major_bleeding_rate: float
    ich_case_fatality: float
    non_ich_case_fatality: float

    # Demographics
    baseline_age: float
    weight_factor: float

    # Utilities
    benefits_discount_rate: float
    healthy_utility: float
    mild_arthropathy_utility: float
    moderate_arthropathy_utility: float
    severe_arthropathy_utility: float
    spontaneous_bleeding_utility: float
    joint_bleeding_utility: float
    intracranial_hemorrhage_utility: float
    non_ich_major_bleeding_utility: float
    death_utility: float

    # Costs
    per_unit_price: float
    costs_discount_rate: float

    prophylaxis_background_factor_consumption_per_kg: float
    factor_consumption_per_spontaneous_bleeding_per_kg: float
    factor_consumption_per_joint_bleeding_per_kg: float
    factor_consumption_per_intracranial_hemorrhage_per_kg: float
    factor_consumption_per_non_ich_major_bleeding_per_kg: float
