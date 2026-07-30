import numpy as np

from app.domain.inputs import ModelInput


class ParameterResolver:
    """Resolve dependent clinical parameters for the redesigned bleed model."""

    @staticmethod
    def resolve_samples(
        samples: dict[str, np.ndarray], ltb_mode: str | None = None
    ) -> dict[str, np.ndarray]:
        bleeding = np.maximum(samples["bleeding_rate"], 0.0)
        joint_fraction = np.clip(samples["joint_bleeding_fraction"], 0.0, 1.0)
        non_ich_fraction = np.clip(
            samples["gi_neck_bleeding_fraction"]
            + samples["iliopsoas_bleeding_fraction"],
            0.0,
            1.0,
        )

        # ICH is an absolute incidence and is therefore not removed from ABR.
        # Major non-ICH bleeds are a subset of ABR and must be removed before
        # the residual ABR is split between joint and spontaneous bleeds.
        non_ich_rate = bleeding * non_ich_fraction
        routine_bleeding = np.maximum(bleeding - non_ich_rate, 0.0)
        joint_rate = routine_bleeding * joint_fraction
        spontaneous_rate = routine_bleeding - joint_rate

        return {
            **samples,
            "spontaneous_bleeding_rate": spontaneous_rate,
            "joint_bleeding_rate": joint_rate,
            "non_ich_major_bleeding_rate": non_ich_rate,
            "non_ich_major_bleeding_fraction": non_ich_fraction,
        }

    @staticmethod
    def build_single(res: dict[str, np.ndarray], i: int) -> ModelInput:
        return ModelInput(
            cycle=res["cycles"][i],
            bleeding_rate=res["bleeding_rate"][i],
            spontaneous_bleeding_rate=res["spontaneous_bleeding_rate"][i],
            joint_bleeding_rate=res["joint_bleeding_rate"][i],
            intracranial_hemorrhage_rate=res["intracranial_hemorrhage_rate"][i],
            non_ich_major_bleeding_rate=res["non_ich_major_bleeding_rate"][i],
            ich_case_fatality=res["ich_case_fatality"][i],
            non_ich_case_fatality=res["non_ich_case_fatality"][i],
            baseline_age=res["baseline_age"][i],
            weight_factor=res["weight_factor"][i],
            benefits_discount_rate=res["benefits_discount_rate"][i],
            healthy_utility=res["healthy_utility"][i],
            mild_arthropathy_utility=res["mild_arthropathy_utility"][i],
            moderate_arthropathy_utility=res["moderate_arthropathy_utility"][i],
            severe_arthropathy_utility=res["severe_arthropathy_utility"][i],
            advanced_arthropathy_utility=res["advanced_arthropathy_utility"][i],
            end_stage_arthropathy_utility=res["end_stage_arthropathy_utility"][i],
            spontaneous_bleeding_utility=res["spontaneous_bleeding_utility"][i],
            joint_bleeding_utility=res["joint_bleeding_utility"][i],
            intracranial_hemorrhage_utility=res[
                "intracranial_hemorrhage_utility"
            ][i],
            non_ich_major_bleeding_utility=res[
                "non_ich_major_bleeding_utility"
            ][i],
            death_utility=res["death_utility"][i],
            per_unit_price=res["per_unit_price"][i],
            costs_discount_rate=res["costs_discount_rate"][i],
            prophylaxis_background_factor_consumption_per_kg=res[
                "prophylaxis_background_factor_consumption_per_kg"
            ][i],
            factor_consumption_per_spontaneous_bleeding_per_kg=res[
                "factor_consumption_per_spontaneous_bleeding_per_kg"
            ][i],
            factor_consumption_per_joint_bleeding_per_kg=res[
                "factor_consumption_per_joint_bleeding_per_kg"
            ][i],
            factor_consumption_per_intracranial_hemorrhage_per_kg=res[
                "factor_consumption_per_intracranial_hemorrhage_per_kg"
            ][i],
            factor_consumption_per_non_ich_major_bleeding_per_kg=res[
                "factor_consumption_per_non_ich_major_bleeding_per_kg"
            ][i],
        )
