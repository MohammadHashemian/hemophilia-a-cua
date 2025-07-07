import numpy as np


# Life expectancy converted into array, be careful
class Population:
    """
    Parameters
    -------

    **total_patients**: sample size

    **max_age**: life expectancy

    **age_probabilities**: probability distribution of ages in population
    """

    def __init__(self, total_patients, max_age, age_probabilities):
        self.populationSize = total_patients
        self.lifeExpectancy = np.arange(max_age)
        self.probabilities = age_probabilities

    def generate(self, is_shuffled: None | bool = None):
        if len(self.lifeExpectancy) != len(self.probabilities):
            raise TypeError(
                f"life expectancy {len(self.lifeExpectancy)} and age distribution len {len(self.probabilities)} must be same size"
            )
        rng = np.random.default_rng()
        samples = rng.choice(
            self.lifeExpectancy, self.populationSize, True, self.probabilities
        )
        if not is_shuffled:
            samples = np.sort(samples)
        return np.array(samples)
