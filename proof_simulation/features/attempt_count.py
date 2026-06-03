"""Attempt count state feature."""

from typing import Dict

from ..problem import SimulatedProblem
from ..state import SimulationState
from .base import StateFeature


class AttemptCountFeature(StateFeature):
    """Current number of prove attempts used per model."""

    def name(self) -> str:
        return "num_attempts"

    def compute(self, state: SimulationState, problem: SimulatedProblem) -> Dict[str, int]:
        return dict(state.prove_attempts_used)
