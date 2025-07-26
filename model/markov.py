from typing import List, Union, Generator, Optional, Callable, Dict
from model.transition import ProbabilityBuilder
from model.utils import cal_body_weight
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Constants
NUM_CYCLES = 73 * 52  # 73 years in weeks


class MarkovChain:
    """A Markov chain implementation that generates state transitions with reward tracking."""

    def __init__(
        self,
        states: List[str],
        transitions: Union[List[List[float]], np.ndarray],
        start_state: str,
        steps: int,
    ) -> None:
        """
        Initialize Markov chain with states, transitions, and optional reward function.

        Args:
            states: List of possible states
            transitions: Transition probability matrix
            start_state: Initial state
            steps: Number of steps to simulate
        """
        self.transitions = np.array(transitions, dtype=float)
        self.states = states
        self.steps = steps
        self.reward_functions = []
        self.rewards: Dict[str, List[float | int]] = {}

        # Validate inputs
        if self.transitions.ndim != 2:
            raise ValueError("Transition matrix must be 2D")
        if self.transitions.shape != (len(states), len(states)):
            raise ValueError(
                f"Expected {len(states)}x{len(states)} transition matrix, got {self.transitions.shape}"
            )
        if not np.allclose(self.transitions.sum(axis=1), 1, rtol=1e-5):
            raise ValueError("Each row in the transition matrix must sum to 1")
        if start_state not in states:
            raise ValueError(f"Start state '{start_state}' not in states list")

        self.current_state_idx = states.index(start_state)
        self.num_states = len(states)

    def add_reward_function(self, func: Callable) -> None:
        """Add a reward function to be calculated at each step."""
        self.reward_functions.append(func)
        self.rewards[func.__name__] = []

    def walk(self, steps: Optional[int] = None) -> Generator[str, None, None]:
        """Generate a sequence of states for the specified number of steps."""
        current_state = self.current_state_idx
        if steps:
            self.steps = steps

        # Calculate reward for initial state (step 0)
        if self.reward_functions:
            for func in self.reward_functions:
                r = func(step=0, state=self.states[current_state])
                self.rewards[func.__name__].append(r)

        for step in range(self.steps):
            # Yield current state
            yield self.states[current_state]

            # Transition to next state
            probs = self.transitions[current_state]
            current_state = np.random.choice(self.num_states, p=probs)

            # Calculate rewards for the new state
            if self.reward_functions:
                for func in self.reward_functions:
                    r = func(step=step, state=self.states[current_state])
                    self.rewards[func.__name__].append(r)

        # Yield the final state
        yield self.states[current_state]

    def collect_rewards(self) -> dict:
        """Return all collected rewards for each reward function."""
        return self.rewards

    def run(self) -> List[str]:
        """Run the Markov chain and return the complete sequence of states."""
        return list(self.walk())


def load_markov_chain(
    io: Path,
    sheet_name: str,
    steps: int = NUM_CYCLES,
) -> MarkovChain:
    """
    Load Markov chain from Excel file.

    Args:
        io: Path to Excel file
        sheet_name: Sheet containing transition matrix
        steps: Number of steps to simulate
        reward_function: Optional function to call at each step

    Returns:
        Initialized MarkovChain instance
    """
    df = pd.read_excel(io, sheet_name=sheet_name)
    states = list(df.columns[1:-1])  # Exclude 'States' and 'SUM' columns
    start_state = states[0]
    transitions = df.drop(columns=["States", "SUM"]).to_numpy()

    return MarkovChain(
        states=states,
        transitions=transitions,
        start_state=start_state,
        steps=steps,
    )


def on_demand_factor_consumption(step: int, state: str):
    """Example reward function that calculates and prints body weight."""
    weight = cal_body_weight(step)
    # print(f"Week {step}: Weight = {weight} kg")
    # print(f"Current state: {state}")
    injected_dose = 0
    if state.lower() == "minor":
        injected_dose = weight * 25  # (20 * 1.25)
    elif state.lower() == "major":
        injected_dose = weight * 90  # 30 * 3
    elif state.lower() == "lt_bleeding":
        injected_dose = weight * 250
    # print(f"Weekly injected dose: {injected_dose} unit")
    # print(64 * "-")
    return injected_dose


def prophylaxis_factor_consumption(step: int, state: str):
    """Example reward function that calculates and prints body weight."""
    weight = cal_body_weight(step)
    # print(f"Week {step}: Weight = {weight} kg")
    # print(f"Current state: {state}")
    injected_dose = weight * 25 * 2
    if state.lower() == "minor":
        injected_dose += weight * 25  # (20 * 1.25)
    elif state.lower() == "major":
        injected_dose += weight * 90  # 30 * 3
    elif state.lower() == "lt_bleeding":
        injected_dose += weight * 250
    # print(f"Weekly injected dose: {injected_dose} unit")
    # print(64 * "-")
    return injected_dose
