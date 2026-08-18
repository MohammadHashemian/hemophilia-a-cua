"""Compatibility facade for the original cohort/individual Markov engines.

New production analyses should use :mod:`modular_models.state_transition`. These
exports remain available so earlier notebooks can be reproduced without
mixing the two model structures.
"""

from engine.chains import Chain, MarkovChains
from engine.vectorized import BatchMarkovChain, BatchResult

__all__ = ["BatchMarkovChain", "BatchResult", "Chain", "MarkovChains"]
