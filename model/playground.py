from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np


def test_sample():
    # Define problem for Ishigami function (3 variables)
    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [
            [-np.pi, np.pi],  # Bounds for x1
            [-np.pi, np.pi],  # Bounds for x2
            [-np.pi, np.pi],
        ],  # Bounds for x3
    }

    try:
        # Generate samples using Saltelli's method
        param_values = saltelli.sample(problem, 1024)
        print(
            f"Generated {param_values.shape[0]} samples with shape {param_values.shape}"
        )

        # Evaluate the Ishigami function
        Y = Ishigami.evaluate(param_values)

        # Perform Sobol analysis
        Si = sobol.analyze(problem, Y)

        # Print first-order and total-order sensitivity indices
        print("First-order sensitivity indices (S1):")
        for name, s1 in zip(problem["names"], Si["S1"]):
            print(f"{name}: {s1:.4f}")
        print("\nTotal-order sensitivity indices (ST):")
        for name, st in zip(problem["names"], Si["ST"]):
            print(f"{name}: {st:.4f}")

        # Optionally print some samples for verification
        print("\nFirst 5 samples:")
        for i, x in enumerate(param_values[:5]):
            print(f"Sample {i}: {x}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_sample()
