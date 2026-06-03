"""Oracle and noisy oracle state features."""

from typing import Dict, Optional

import numpy as np

from ..actions import ActionType
from ..problem import SimulatedProblem
from ..state import SimulationState
from .base import StateFeature


class NoisyOracleFeature(StateFeature):
    """Noisy oracle success probabilities.

    For each PROVE action in the problem's empirical success rates,
    computes clamp(p + N(0, sigma^2), 0, 1). Cached per (problem_id, model)
    within a seed.
    """

    def __init__(self, sigma: float = 0.1, resample_per_step: bool = False):
        self.sigma = sigma
        self.resample_per_step = resample_per_step
        self._rng: Optional[np.random.Generator] = None
        self._cache: Dict[tuple, float] = {}

    def name(self) -> str:
        return "noisy_p"

    def seed(self, seed: int):
        self._rng = np.random.default_rng(seed)
        self._cache = {}

    def reset(self):
        pass  # cache persists across problems within a seed

    def compute(self, state: SimulationState, problem: SimulatedProblem) -> Dict[str, float]:
        rates = problem.get_empirical_success_rates()
        result = {}
        for action, p in rates.items():
            if action.type != ActionType.PROVE or action.model is None:
                continue
            cache_key = (problem.problem_id, state.target_id, action.model)
            if self.resample_per_step or cache_key not in self._cache:
                rng = self._rng or np.random.default_rng()
                if self.sigma < 0:
                    # sigma < 0 means uniform on [0, 1] (ignore oracle)
                    self._cache[cache_key] = float(rng.uniform(0.0, 1.0))
                else:
                    noise = rng.normal(0, self.sigma)
                    self._cache[cache_key] = float(np.clip(p + noise, 0.0, 1.0))
            result[action.model] = self._cache[cache_key]
        return result


class OracleFeature(StateFeature):
    """True oracle success probabilities (no noise)."""

    def name(self) -> str:
        return "oracle_p"

    def compute(self, state: SimulationState, problem: SimulatedProblem) -> Dict[str, float]:
        rates = problem.get_empirical_success_rates()
        result = {}
        for action, p in rates.items():
            if action.type != ActionType.PROVE or action.model is None:
                continue
            result[action.model] = float(p)
        return result
