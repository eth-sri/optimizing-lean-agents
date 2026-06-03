"""Oracle probability model using true empirical success rates."""

import math
from typing import List, Optional
import numpy as np

from .base import ProbabilityModel
from ...actions import Action
from ...state import SimulationState
from ...problem import SimulatedProblem


class OracleNoiseProbabilityModel(ProbabilityModel):
    """Uses true empirical success rates from the data.

    Args:
        problems: List of simulated problems for oracle lookup.
        sigma: Std-dev of Gaussian noise added in logit space. 0 = deterministic.
        noise_per: When to sample noise.
            "step" (default) - fresh noise on every predict() call.
            "seed" - sample once per (problem, action) per seed, cached.
    """

    def __init__(self, problems: List[SimulatedProblem], sigma: float = 0.0, noise_per: str = "step"):
        if noise_per not in ("step", "seed"):
            raise ValueError(f"noise_per must be 'step' or 'seed', got '{noise_per}'")
        self._problems = {p.problem_id: p for p in problems}
        self.sigma = sigma
        self.noise_per = noise_per
        self._rng: Optional[np.random.Generator] = None
        self._noise_cache: dict = {}  # (problem_id, action) -> noise value

    def seed(self, seed: int):
        """Set the RNG seed for reproducible noise."""
        self._rng = np.random.default_rng(seed)
        self._noise_cache = {}

    def predict(self, state: SimulationState, action: Action, problem: SimulatedProblem, tracked_state: dict = None) -> float:
        rates = problem.get_empirical_success_rates()

        rates_prob = rates.get(action, 0.0)

        if self.sigma > 0.0:
            rng = self._rng or np.random.default_rng()
            if self.noise_per == "seed":
                cache_key = (problem.problem_id, action)
                if cache_key not in self._noise_cache:
                    self._noise_cache[cache_key] = rng.normal(0, self.sigma)
                noise = self._noise_cache[cache_key]
            else:
                noise = rng.normal(0, self.sigma)

            rates_prob += noise

        return rates_prob
