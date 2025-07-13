import numpy as np
from typing import List, Union
from model.schemas import Regimes, Status, Treatment
import math


class MarkovChain:
    """A Markov chain model for simulating state transitions with dose and cost calculations."""

    def __init__(
        self,
        states: List[str],
        transition_matrix: List[List[float]],
        initial_state_probs: List[float],
        treatment: Treatment,
        price_per_unit: float = 58_000,  # Cost per IU in Rial
    ):
        """
        Initialize the Markov chain with states, transition matrix, initial probabilities, and treatment.

        Args:
            states: List of state names.
            transition_matrix: 2D list representing the transition probabilities.
            initial_state_probs: Initial state probability distribution.
            treatment: Treatment object containing dosing and bleeding rates.
            price_per_unit: Cost per unit of factor VIII (Rial/IU).
        """
        self.states = states
        self.transition_matrix = np.array(transition_matrix)
        self.initial_state_probs = np.array(initial_state_probs)
        self.treatment = treatment
        self.price_per_unit = price_per_unit
        # Validate inputs
        if len(states) != len(transition_matrix):
            raise ValueError(
                "Number of states must match transition matrix dimensions."
            )
        if not np.isclose(sum(initial_state_probs), 1.0):
            raise ValueError("Initial state probabilities must sum to 1.")
        if not all(np.isclose(np.sum(row), 1.0) for row in transition_matrix):
            raise ValueError("Each row in transition matrix must sum to 1.")

    def get_body_weight(self, age_weeks: int) -> float:
        """
        Calculate body weight (kg) based on age in weeks, using a non-linear growth model.

        Args:
            age_weeks: Age in weeks (0 to 73*52 = 3796).

        Returns:
            Body weight in kg.
        """
        if age_weeks < 0 or age_weeks > 73 * 52:
            print(f"Age {age_weeks} weeks out of range, clamping to 0–3796 weeks")
            age_weeks = max(0, min(age_weeks, 73 * 52))

        if age_weeks <= 104:  # 0–2 years
            L, k, x0 = 12, 0.05, 52
            weight = L / (1 + math.exp(-k * (age_weeks - x0)))
            weight = max(3.5, weight)  # Ensure at least birth weight
        elif age_weeks <= 624:  # 2–12 years
            weight = 12 + (40 - 12) / (624 - 104) * (age_weeks - 104)
        elif age_weeks <= 936:  # 12–18 years
            weight = 40 + (70 - 40) / (936 - 624) * (age_weeks - 624)
        elif age_weeks <= 3380:  # 18–65 years
            weight = 70 + 5 * math.sin(age_weeks / 520)
        else:  # 65–73 years
            weight = 70 - (age_weeks - 3380) / (3796 - 3380) * 5

        return round(weight, 2)

    def calculate_weekly_dose_and_cost(
        self, state: str, age_weeks: int
    ) -> tuple[float, float]:
        """
        Calculate total factor VIII dose (IU) and cost (Rial) for a week based on state and age.

        Args:
            state: Current state (e.g., articular_bleeding, minor_bleeding, surgery_or_injury_or_infection).
            age_weeks: Age in weeks for body weight calculation.

        Returns:
            Tuple of (total_dose in IU, total_cost in Rial).
        """
        body_weight = self.get_body_weight(age_weeks)

        if state == Status.MAJOR_BLEEDING.value:  # articular_bleeding
            dose_per_kg = self.treatment.dose_joint
            duration = self.treatment.duration_joint
        elif state == Status.MINOR_BLEEDING.value:  # minor_bleeding
            total_freq = 0.05 + 0.1
            dose_per_kg = (
                0.05 * self.treatment.dose_muscle + 0.1 * self.treatment.dose_mucous
            ) / total_freq  # 27.167 IU/kg
            duration = (
                0.05 * self.treatment.duration_muscle
                + 0.1 * self.treatment.duration_mucous
            ) / total_freq  # 3.667 days
        elif state == Status.CRITICAL_BLEEDING.value:  # surgery_or_injury_or_infection
            total_freq = 0.015 + 0.01 + 0.02
            dose_per_kg = (
                0.015 * self.treatment.dose_intracranial
                + 0.01 * self.treatment.dose_neck_throat
                + 0.02 * self.treatment.dose_gastro
            ) / total_freq  # 37.111 IU/kg
            duration = (
                0.015 * self.treatment.duration_intracranial
                + 0.01 * self.treatment.duration_neck_throat
                + 0.02 * self.treatment.duration_gastro
            ) / total_freq  # 15.222 days
        elif state == Status.INHIBITOR.value:  # inhibitor
            total_freq = 0.05 + 0.1
            dose_per_kg = (
                0.05 * self.treatment.dose_muscle + 0.1 * self.treatment.dose_mucous
            ) / total_freq  # 27.167 IU/kg
            duration = (
                0.05 * self.treatment.duration_muscle
                + 0.1 * self.treatment.duration_mucous
            ) / total_freq  # 3.667 days
        else:
            return (
                0.0,
                0.0,
            )  # No injections for non-bleed states (e.g., alive_wo_arthropathy, chronic_arthropathy, etc.)

        num_injections = math.ceil(duration)  # One injection per day, rounded up
        total_dose = dose_per_kg * body_weight * num_injections
        total_cost = total_dose * self.price_per_unit
        return round(total_dose, 2), round(total_cost, 2)

    def simulate(self, num_steps: int) -> dict:
        """
        Simulate the Markov chain for a specified number of steps, including dose and cost calculations.

        Args:
            num_steps: Number of time steps (weeks) to simulate.

        Returns:
            Dictionary containing state path, weekly doses, and weekly costs.
        """
        state_path = []
        weekly_doses = []
        weekly_costs = []
        current_state = np.random.choice(self.states, p=self.initial_state_probs)

        for week in range(num_steps):
            state_path.append(current_state)
            dose, cost = self.calculate_weekly_dose_and_cost(current_state, week)
            weekly_doses.append(dose)
            weekly_costs.append(cost)

            # Transition to next state
            current_state_idx = self.states.index(current_state)
            current_state = np.random.choice(
                self.states, p=self.transition_matrix[current_state_idx]
            )

        return {
            "state_path": state_path,
            "weekly_doses": weekly_doses,
            "weekly_costs": weekly_costs,
        }
