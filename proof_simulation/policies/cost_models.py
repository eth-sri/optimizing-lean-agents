"""Cost models for proof simulation policies."""

from abc import ABC, abstractmethod
from typing import Dict

from ..actions import Action, DetailedCost
from ..state import SimulationState
from ..problem import SimulatedProblem


class CostModel(ABC):
    """Predicts expected cost of an action."""

    @abstractmethod
    def predict(self, state: SimulationState, action: Action, problem: SimulatedProblem) -> float:
        """Return estimated cost (in total_sflops) for this action."""
        pass


class RunningAverageCostModel(CostModel):
    """Tracks observed costs during simulation and returns running averages."""

    def __init__(self, default_cost: float = 0.0):
        self.default_cost = default_cost
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def reset(self):
        self._totals.clear()
        self._counts.clear()

    def observe(self, action: Action, cost: DetailedCost, target_id: str = ""):
        key = self._action_key(action, target_id)
        self._totals[key] = self._totals.get(key, 0.0) + cost.total_sflops
        self._counts[key] = self._counts.get(key, 0) + 1

    def predict(self, state: SimulationState, action: Action, problem: SimulatedProblem) -> float:
        key = self._action_key(action, state.target_id)
        if key in self._counts:
            return self._totals[key] / self._counts[key]
        # Fall back to global average across targets
        global_key = self._action_key(action)
        if global_key in self._counts:
            return self._totals[global_key] / self._counts[global_key]
        return self.default_cost

    def _action_key(self, action: Action, target_id: str = "") -> str:
        base = f"{action.type.value}_{action.model}" if action.model else action.type.value
        if target_id:
            return f"{target_id}:{base}"
        return base
