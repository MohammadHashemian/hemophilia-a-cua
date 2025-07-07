import numpy as np
import random


class MarkovChain:
    # Mentioning that if a Markov chain has N Possible states,
    # the matrix will be an N*N matrix and the transition matrix must be
    # stochastic and must add up to exactly 1.
    def __init__(
        self,
        states: list[str],
        transition_matrix: list[list[float]],
        initial_state_probs: list[float] | None = None,
    ):
        self.states = states
        self.transition_matrix = np.array(transition_matrix)

        # Validate transition matrix dimensions
        if self.transition_matrix.shape != (len(states), len(states)):
            raise ValueError(
                "Transition matrix must be square and match the number of states."
            )

        # Validate rows sum to 1
        if not np.allclose(self.transition_matrix.sum(axis=1), 1):
            raise ValueError("Rows in transition matrix must sum to 1.")

        if initial_state_probs:
            self.initial_state_probs = np.array(initial_state_probs)
            if not np.isclose(self.initial_state_probs.sum(), 1):
                raise ValueError("Initial state probabilities must sum to 1.")
            if len(self.initial_state_probs) != len(states):
                raise ValueError(
                    "Initial state probabilities must match the number of states."
                )
        else:
            # Default to uniform initial distribution if not provided
            self.initial_state_probs = np.full(len(states), 1 / len(states))

    def get_next_state(self, current_state_index: int):
        """Determines the next state based on transition probabilities."""
        return random.choices(
            self.states, weights=self.transition_matrix[current_state_index], k=1
        )[0]

    def simulate(self, num_steps, start_state=None):
        """Simulates the Markov chain for a given number of steps."""
        if start_state is None:
            # Choose initial state based on initial probabilities
            current_state = random.choices(
                self.states, weights=self.initial_state_probs.tolist(), k=1
            )[0]
        else:
            if start_state not in self.states:
                raise ValueError("Start state not in defined states.")
            current_state = start_state

        path = [current_state]

        for _ in range(num_steps - 1):
            current_state_index = self.states.index(current_state)
            current_state = self.get_next_state(current_state_index)
            path.append(current_state)
        return path
