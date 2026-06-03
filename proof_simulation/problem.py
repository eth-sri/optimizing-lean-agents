"""SimulatedProblem: top-level wrapper around a TargetNode tree."""

from typing import Dict, List, Optional

from .actions import Action, ActionType, ActionResult, DetailedCost
from .state import SimulationState
from .target import TargetNode


class SimulatedProblem:
    """Thin wrapper around the root TargetNode. Manages top-level concerns."""

    def __init__(self, problem_id: str, root: TargetNode):
        self.problem_id = problem_id
        self.root = root
        self.total_cost = DetailedCost()
        self._terminated = False
        self._last_action: Optional[Action] = None
        self._last_result: Optional[ActionResult] = None

    def get_valid_actions(self) -> List[Action]:
        """Get valid actions at the current focus point."""
        if self._terminated or self.root.is_solved():
            return []
        return self.root.get_valid_actions()

    def simulate_action(self, action: Action) -> ActionResult:
        """Execute an action and return the result."""
        if action.type == ActionType.TERMINATE:
            self._terminated = True
            result = ActionResult(success=False, cost=DetailedCost())
            self._last_action = action
            self._last_result = result
            return result

        focus = self.root.get_current_focus()         # capture for breakdown update
        result = self.root.execute_action(action)      # route through root
        self.total_cost = self.total_cost + result.cost

        # After a PROVE/CORRECT succeeds/fails inside a breakdown, notify the breakdown
        if action.type in (ActionType.PROVE, ActionType.CORRECT):
            self._handle_breakdown_update(focus, result)

        self._last_action = action
        self._last_result = result
        return result

    def _handle_breakdown_update(self, focus: TargetNode, result: ActionResult):
        """After a proof action in a breakdown, update breakdown state."""
        # Walk up the decomposition chain to find the parent breakdown
        self._update_breakdown_chain(self.root, focus, result)

    def _update_breakdown_chain(self, node: TargetNode, focus: TargetNode, result: ActionResult):
        """Recursively find and update the breakdown containing the focused target."""
        if not node.decomposed or node.active_breakdown is None:
            return

        bs = node.active_breakdown
        # Check if the focus is a direct child of this breakdown
        for target_id, child_node in bs.target_nodes.items():
            if child_node is focus:
                # This breakdown directly contains the focus
                if result.success:
                    bs.on_action_complete(target_id, result)
                else:
                    # Check if the target is exhausted (no more actions besides TERMINATE/DECOMPOSE/CREATE_BREAKDOWN)
                    remaining = focus.get_valid_actions()
                    proof_actions = [a for a in remaining
                                     if a.type in (ActionType.PROVE, ActionType.CORRECT)]
                    if not proof_actions and not focus.decomposed:
                        bs.on_exhausted(target_id)
                    else:
                        bs.on_action_complete(target_id, result)
                return

            # Check recursively into decomposed children
            if child_node.decomposed:
                self._update_breakdown_chain(child_node, focus, result)

    def is_done(self) -> bool:
        """Is the simulation done? (solved or terminated)"""
        return self._terminated or self.root.is_solved()

    def is_solved(self) -> bool:
        """Is the problem fully solved?"""
        return self.root.is_solved()

    def get_state(self) -> SimulationState:
        """Snapshot for policy."""
        focus = self.root.get_current_focus()
        snap = focus.get_state_snapshot()
        root_snap = self.root.get_state_snapshot()

        return SimulationState(
            problem_id=self.problem_id,
            total_cost=self.total_cost,
            decomposition_depth=self._calculate_depth(),
            target_type=snap['target_type'],
            target_id=snap['target_id'],
            target_metadata={
                'origin_problem_id': self.problem_id,
                'breakdown_id': snap.get('breakdown_id', -1),
                'lemma_id': snap.get('lemma_id', -1),
            },
            prove_attempts_used=snap['prove_attempts_used'],
            prove_attempts_available=snap['prove_attempts_available'],
            last_prove_model=snap['last_prove_model'],
            last_prove_success=snap['last_prove_success'],
            corrections_used=snap['corrections_used'],
            corrections_available=snap['corrections_available'],
            breakdowns_created=root_snap['breakdowns_created'],
            breakdowns_available=root_snap['breakdowns_available'],
            current_breakdown_targets_proven=snap['active_breakdown_targets_proven'],
            current_breakdown_targets_remaining=snap['active_breakdown_targets_remaining'],
            last_action=self._last_action,
            last_result=self._last_result,
        )

    def _calculate_depth(self) -> int:
        """Count number of decomposition hops to current focus."""
        depth = 0
        node = self.root
        while node.decomposed and node.active_breakdown and not node.active_breakdown.failed:
            if node.active_breakdown.is_complete():
                break
            depth += 1
            node = node.active_breakdown.current_target_node
            if not node.decomposed:
                break
        return depth

    def reset(self, seed: int, breakdown_model_filter=None, max_breakdowns=None, max_corrections=None):
        """Reset the problem with a new seed."""
        self.root.reset(seed, breakdown_model_filter=breakdown_model_filter, max_breakdowns=max_breakdowns, max_corrections=max_corrections)
        self.total_cost = DetailedCost()
        self._terminated = False
        self._last_action = None
        self._last_result = None

    def get_empirical_success_rates(self) -> Dict[Action, float]:
        """Rates at current focus (for oracle)."""
        focus = self.root.get_current_focus()
        return focus.get_empirical_rates()
