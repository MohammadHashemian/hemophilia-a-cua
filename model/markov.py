from typing import List, Union, Generator, Optional, Callable
from model.utils import cal_body_weight
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Constants
CYCLES = 73 * 52  # 73 years in weeks


class MarkovChain:
    """A Markov chain implementation that generates state transitions with reward tracking."""

    def __init__(
        self,
        states: List[str],
        transitions: Union[List[List[float]], np.ndarray],
        start_state: str,
        steps: int,
        reward_function: Optional[Callable] = None,
    ) -> None:
        """
        Initialize Markov chain with states, transitions, and optional reward function.

        Args:
            states: List of possible states
            transitions: Transition probability matrix
            start_state: Initial state
            steps: Number of steps to simulate
            reward_function: Optional function to call at each step
        """
        self.transitions = np.array(transitions, dtype=float)
        self.states = states
        self.steps = steps
        self.reward_function = reward_function

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

    def walk(self) -> Generator[str, None, None]:
        """Generate a sequence of states for the specified number of steps."""
        current_state = self.current_state_idx
        for step in range(self.steps):
            # Yield current state before transition
            yield self.states[current_state]

            # Call reward function if provided
            if self.reward_function is not None:
                self.reward_function(step=step, state=self.states[current_state])

            # Transition to next state
            probs = self.transitions[current_state]
            current_state = np.random.choice(self.num_states, p=probs)

        # Yield the final state
        yield self.states[current_state]

    def run(self) -> List[str]:
        """Run the Markov chain and return the complete sequence of states."""
        return list(self.walk())


def reward_function(step: int, state: str) -> None:
    """Example reward function that calculates and prints body weight."""
    weight = cal_body_weight(step)
    print(f"Current state: {state}")
    print(f"Week {step}: Weight = {weight} kg")
    # TODO:
    # Calculate Dose and Costs
    # Calculate Utilities


def load_markov_chain(
    io: Path,
    sheet_name: str,
    steps: int = CYCLES,
    reward_function: Optional[Callable] = None,
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
        reward_function=reward_function,
    )
