"""Subgoal repetition: ratio of redundant have statements (incremental)."""

from typing import Dict

from feature_engineering.features.have_repetition import _count_redundant_haves

from ..actions import Action, ActionResult, ActionType
from ..problem import SimulatedProblem
from ..state import SimulationState
from .base import StateFeature


class SubgoalRepetitionFeature(StateFeature):
    """Running ratio of redundant have statements across proof attempts."""

    def __init__(self, threshold: float = 90.0):
        self.threshold = threshold
        self._total_redundant: Dict[str, int] = {}
        self._total_haves: Dict[str, int] = {}

    def name(self) -> str:
        return "subgoal_repetition"

    def reset(self):
        self._total_redundant.clear()
        self._total_haves.clear()

    def observe(self, state: SimulationState, action: Action, result: ActionResult, problem: SimulatedProblem):
        if action.type not in (ActionType.PROVE, ActionType.CORRECT):
            return
        if result.code is None:
            return

        r, t = _count_redundant_haves(result.code, self.threshold)
        tid = state.target_id
        self._total_redundant[tid] = self._total_redundant.get(tid, 0) + r
        self._total_haves[tid] = self._total_haves.get(tid, 0) + t

    def compute(self, state: SimulationState, problem: SimulatedProblem) -> float:
        tid = state.target_id
        total = self._total_haves.get(tid, 0)
        if total == 0:
            return 0.0
        return self._total_redundant.get(tid, 0) / total
