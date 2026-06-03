"""Debug feature: first N characters of the last proof code."""

from typing import Optional

from ..actions import Action, ActionResult, ActionType
from ..problem import SimulatedProblem
from ..state import SimulationState
from .base import StateFeature


class CodePreviewFeature(StateFeature):
    """Tracks the first N chars of the last proof code per target (for debugging)."""

    def __init__(self, n: int = 16):
        self.n = n
        self._last_code: dict[str, Optional[str]] = {}

    def name(self) -> str:
        return "code_preview"

    def reset(self):
        self._last_code.clear()

    def observe(self, state: SimulationState, action: Action, result: ActionResult, problem: SimulatedProblem):
        if action.type not in (ActionType.PROVE, ActionType.CORRECT):
            return
        if result.code is None:
            self._last_code[state.target_id] = "CODE_IS_NONE"
        elif not result.code.strip():
            self._last_code[state.target_id] = f"EMPTY_len={len(result.code)}_repr={repr(result.code[:20])}"
        else:
            self._last_code[state.target_id] = result.code.strip()[:self.n]

    def compute(self, state: SimulationState, problem: SimulatedProblem) -> Optional[str]:
        return self._last_code.get(state.target_id)
