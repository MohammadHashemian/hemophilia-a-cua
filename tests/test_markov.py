from hemophilia.model.markov import MarkovChain


# Example Usage:
states = ["Sunny", "Cloudy", "Rainy"]
transition_matrix = [
    [0.7, 0.2, 0.1],  # Sunny -> Sunny, Cloudy, Rainy
    [0.4, 0.4, 0.2],  # Cloudy -> Sunny, Cloudy, Rainy
    [0.2, 0.4, 0.4],  # Rainy -> Sunny, Cloudy, Rainy
]
initial_probs = [0.7, 0.2, 0.1]  # Initial probability of being Sunny, Cloudy, Rainy

weather_chain = MarkovChain(
    states, transition_matrix, initial_state_probs=initial_probs
)
simulation_results = weather_chain.simulate(num_steps=10)
print(f"Weather simulation over 10 days: {simulation_results}")
