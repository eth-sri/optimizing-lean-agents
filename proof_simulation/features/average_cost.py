"""Average cost state feature tracking running average cost per model."""

from typing import Dict

from ..actions import Action, ActionResult, ActionType
from ..problem import SimulatedProblem
from ..state import SimulationState
from .base import StateFeature


class AverageCostFeature(StateFeature):
    """Running average of a cost field per model.

    Tracks the mean of a specified cost field (e.g. output_sflops) across
    all PROVE actions observed so far, per model. Returns {} at cold start.
    """

    def __init__(self, cost_field: str = "output_sflops"):
        self._cost_field = cost_field
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def name(self) -> str:
        return "avg_cost"

    def reset(self):
        self._totals.clear()
        self._counts.clear()

    def observe(self, state: SimulationState, action: Action, result: ActionResult, problem: SimulatedProblem):
        if action.type != ActionType.PROVE or action.model is None:
            return
        model_name = action.model
        value = float(getattr(result.cost, self._cost_field))
        self._totals[model_name] = self._totals.get(model_name, 0.0) + value
        self._counts[model_name] = self._counts.get(model_name, 0) + 1

    def compute(self, state: SimulationState, problem: SimulatedProblem) -> Dict[str, float]:
        return {
            model: self._totals[model] / self._counts[model]
            for model in self._counts
        }
