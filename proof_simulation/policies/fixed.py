"""Fixed staged waterfall policy."""

from typing import Dict, List, Optional

from .base import Policy
from ..actions import Action, ActionType
from ..state import SimulationState


class FixedPolicy(Policy):
    """Staged waterfall policy with per-target attempt budgets.

    Stages (in order):
    1. Full proof with each model (up to max_attempts per model per problem)
       - After each failed PROVE, try CORRECT up to max_corrections times
    2. Decompose -> create breakdowns (up to max_breakdowns)
    3. Within each breakdown: prove each target (up to max_attempts per model)
       - After each failed PROVE, try CORRECT up to max_corrections times
    4. If breakdown fails, try next breakdown
    5. TERMINATE when all budgets exhausted

    Example:
        policy = FixedPolicy(
            full_proof_budget={"8b": 32, "32b": 16},
            max_breakdowns=3,
            breakdown_proof_budget={"8b": 8},
            max_corrections=1,
        )
    """

    def __init__(
        self,
        full_proof_budget: Dict[str, int],
        max_breakdowns: int = 3,
        breakdown_proof_budget: Optional[Dict[str, int]] = None,
        max_corrections: int = 1,
    ):
        self.full_proof_budget = full_proof_budget
        self.max_breakdowns = max_breakdowns
        self.breakdown_proof_budget = breakdown_proof_budget or {}
        self.max_corrections = max_corrections

    def choose_action(self, state: SimulationState, valid_actions: List[Action], tracked_state: dict = None) -> Action:
        valid_set = {(a.type, a.model): a for a in valid_actions}
        valid_types = {a.type: a for a in valid_actions}

        if state.decomposition_depth == 0 and ActionType.PROVE in valid_types:
            # Stage 1: Full proof attempts with budget per model
            return self._stage_full_proof(state, valid_set, valid_types)
        elif state.decomposition_depth == 0:
            # Stage 2: Breakdown management at top level (post-decompose)
            return self._stage_breakdown_management(state, valid_types)
        else:
            # Stage 3: Inside a breakdown — prove targets
            return self._stage_breakdown_proving(state, valid_set, valid_types)

    def _stage_full_proof(self, state, valid_set, valid_types):
        """Stage 1: Try full proof with each model in budget order."""
        # CORRECT first if last prove failed and under correction budget
        if (state.last_prove_success is False
                and ActionType.CORRECT in valid_types
                and state.corrections_used < self.max_corrections):
            return valid_types[ActionType.CORRECT]

        # Try PROVE for each model in order, respecting budget
        for model, budget in self.full_proof_budget.items():
            used = state.prove_attempts_used.get(model, 0)
            if used < budget and (ActionType.PROVE, model) in valid_set:
                return valid_set[(ActionType.PROVE, model)]

        # Move to decomposition if breakdowns are configured
        if self.max_breakdowns > 0 and ActionType.DECOMPOSE in valid_types:
            return valid_types[ActionType.DECOMPOSE]

        # Nothing left
        if ActionType.TERMINATE in valid_types:
            return valid_types[ActionType.TERMINATE]
        return list(valid_set.values())[-1]

    def _stage_breakdown_management(self, state, valid_types):
        """Stage 2: Decompose then create breakdowns up to max_breakdowns."""
        if self.max_breakdowns > 0 and ActionType.DECOMPOSE in valid_types:
            return valid_types[ActionType.DECOMPOSE]

        if (state.breakdowns_created < self.max_breakdowns
                and ActionType.CREATE_BREAKDOWN in valid_types):
            return valid_types[ActionType.CREATE_BREAKDOWN]

        if ActionType.TERMINATE in valid_types:
            return valid_types[ActionType.TERMINATE]
        return list(valid_types.values())[-1]

    def _stage_breakdown_proving(self, state, valid_set, valid_types):
        """Stage 3: Inside a breakdown — prove targets with budget."""
        # CORRECT first if last prove failed and under correction budget
        if (state.last_prove_success is False
                and ActionType.CORRECT in valid_types
                and state.corrections_used < self.max_corrections):
            return valid_types[ActionType.CORRECT]

        # Try PROVE for each model in breakdown budget order
        for model, budget in self.breakdown_proof_budget.items():
            used = state.prove_attempts_used.get(model, 0)
            if used < budget and (ActionType.PROVE, model) in valid_set:
                return valid_set[(ActionType.PROVE, model)]

        # This target is exhausted — the framework handles round-robin via
        # BreakdownState.on_exhausted(). But if we're still getting called,
        # try CREATE_BREAKDOWN (abandon current breakdown for a new one)
        if (ActionType.CREATE_BREAKDOWN in valid_types
                and state.breakdowns_created < self.max_breakdowns):
            return valid_types[ActionType.CREATE_BREAKDOWN]

        if ActionType.TERMINATE in valid_types:
            return valid_types[ActionType.TERMINATE]
        return list(valid_set.values())[-1]
