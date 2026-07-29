from dataclasses import dataclass

from app.analysis.psa.parameters import Parameter


@dataclass
class ParameterSet:
    cycles: Parameter

    # Transition builder inputs
    bleeding_rate: Parameter
    joint_bleeding_fraction: Parameter
    # LTB, modeled as a fraction of ABR ("fraction" mode)
    life_threatening_bleeding_fraction: Parameter
    # LTB, modeled as an absolute annual rate ("absolute" mode)
    life_threatening_bleeding_rate: Parameter
    # Conditional case-fatality probability per LTB episode
    ltb_case_fatality: Parameter

    # weight control
    baseline_age: Parameter
    weight_factor: Parameter

    # age_specific_mortality_rate

    # Benefit calculations
    benefits_discount_rate: Parameter
    healthy_utility: Parameter
    mild_arthropathy_utility: Parameter
    moderate_arthropathy_utility: Parameter
    severe_arthropathy_utility: Parameter
    spontaneous_bleeding_utility: Parameter
    joint_bleeding_utility: Parameter
    life_threatening_bleeding_utility: Parameter
    death_utility: Parameter

    # ---- Costs calculation ----
    per_unit_price: Parameter
    costs_discount_rate: Parameter
    # weekly factor unit usage
    prophylaxis_background_factor_consumption_per_kg: Parameter
    factor_consumption_per_spontaneous_bleeding_per_kg: Parameter
    factor_consumption_per_joint_bleeding_per_kg: Parameter
    factor_consumption_per_life_threatening_bleeding_per_kg: Parameter
