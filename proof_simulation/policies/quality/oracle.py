"""Oracle probability model using true empirical success rates."""

from typing import List

from .base import ProbabilityModel
from ...actions import Action
from ...state import SimulationState
from ...problem import SimulatedProblem


class OracleProbabilityModel(ProbabilityModel):
    """Uses true empirical success rates from the data."""

    def __init__(self, problems: List[SimulatedProblem]):
        self._problems = {p.problem_id: p for p in problems}

    def predict(self, state: SimulationState, action: Action, problem: SimulatedProblem, tracked_state: dict = None) -> float:
        rates = problem.get_empirical_success_rates()
        return rates.get(action, 0.0)
