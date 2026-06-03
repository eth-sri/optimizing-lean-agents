"""Error diversity: ratio of unique error messages to total errors (incremental)."""

from typing import Dict, List

from ..actions import Action, ActionResult, ActionType
from ..problem import SimulatedProblem
from ..state import SimulationState
from .base import StateFeature


class ErrorDiversityFeature(StateFeature):
    """Tracks ratio of unique error messages to total errors per target.

    Observes errors from failed PROVE/CORRECT actions incrementally.
    """

    def __init__(self):
        self._all_errors: Dict[str, List[str]] = {}
        self._failed_targets: set = set()

    def name(self) -> str:
        return "error_diversity"

    def reset(self):
        self._all_errors.clear()
        self._failed_targets.clear()

    def observe(self, state: SimulationState, action: Action, result: ActionResult, problem: SimulatedProblem):
        if action.type not in (ActionType.PROVE, ActionType.CORRECT):
            return
        if result.success:
            return
        tid = state.target_id
        self._failed_targets.add(tid)
        if not result.error_messages:
            return
        self._all_errors.setdefault(tid, []).extend(result.error_messages)

    def compute(self, state: SimulationState, problem: SimulatedProblem) -> float:
        tid = state.target_id
        errors = self._all_errors.get(tid)
        if not errors:
            # No diversity to compute. Silent for targets that simply never failed
            # (the common case); warn only if a failure was seen but carried no error
            # message (a genuine data-loading gap worth surfacing).
            if tid in self._failed_targets:
                print(f"error_diversity: target {tid} had failed attempt(s) but no error messages recorded, returning 0.0")
            return 0.0
        return len(set(errors)) / len(errors)
