"""Base probability model interface."""

from abc import ABC, abstractmethod

from ...actions import Action
from ...state import SimulationState
from ...problem import SimulatedProblem


class ProbabilityModel(ABC):
    """Predicts P(success | state, action)."""

    @abstractmethod
    def predict(self, state: SimulationState, action: Action, problem: SimulatedProblem, tracked_state: dict = None) -> float:
        """Return estimated probability of success for this action."""
        pass
