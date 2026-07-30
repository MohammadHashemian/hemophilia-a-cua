from dataclasses import dataclass

from app.analysis.psa.parameters import Parameter


@dataclass
class ParameterSet:
    cycles: Parameter

    # Transition builder inputs
    bleeding_rate: Parameter
    joint_bleeding_fraction: Parameter
    intracranial_hemorrhage_rate: Parameter
    gi_neck_bleeding_fraction: Parameter
    iliopsoas_bleeding_fraction: Parameter
    ich_case_fatality: Parameter
    non_ich_case_fatality: Parameter

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
    advanced_arthropathy_utility: Parameter
    end_stage_arthropathy_utility: Parameter
    spontaneous_bleeding_utility: Parameter
    joint_bleeding_utility: Parameter
    intracranial_hemorrhage_utility: Parameter
    non_ich_major_bleeding_utility: Parameter
    death_utility: Parameter

    # ---- Costs calculation ----
    per_unit_price: Parameter
    costs_discount_rate: Parameter
    # weekly factor unit usage
    prophylaxis_background_factor_consumption_per_kg: Parameter
    factor_consumption_per_spontaneous_bleeding_per_kg: Parameter
    factor_consumption_per_joint_bleeding_per_kg: Parameter
    factor_consumption_per_intracranial_hemorrhage_per_kg: Parameter
    factor_consumption_per_non_ich_major_bleeding_per_kg: Parameter
