"""Unique proofs ratio: unique failed proof attempts / total, with code-as-proxy fallback."""

from typing import Dict, List

from ..actions import Action, ActionResult, ActionType
from ..problem import SimulatedProblem
from ..state import SimulationState
from .base import StateFeature


class UniqueProofsFeature(StateFeature):
    """Ratio of unique failed proof submissions to total failures per target.

    Uses error messages as the fingerprint when available; falls back to the
    proof code itself when error messages were not loaded. Returns 0.0 if no
    failed attempts have been observed yet.
    """

    def __init__(self):
        self._all_fingerprints: Dict[str, List[str]] = {}

    def name(self) -> str:
        return "unique_proofs"

    def reset(self):
        self._all_fingerprints.clear()

    def observe(self, state: SimulationState, action: Action, result: ActionResult, problem: SimulatedProblem):
        if action.type not in (ActionType.PROVE, ActionType.CORRECT):
            return
        if result.success:
            return
        tid = state.target_id
        # if result.error_messages:
        #     self._all_fingerprints.setdefault(tid, []).extend(result.error_messages)
        if result.code:
            self._all_fingerprints.setdefault(tid, []).append(result.code)


    def compute(self, state: SimulationState, problem: SimulatedProblem) -> float:
        tid = state.target_id
        fps = self._all_fingerprints.get(tid)
        if not fps:
            return 0.0
        return len(set(fps)) / len(fps)
