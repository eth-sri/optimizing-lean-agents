"""BreakdownState: tracks state within one breakdown attempt."""

from typing import Dict, List, Optional, Set, TYPE_CHECKING

from .actions import ActionResult, DetailedCost

if TYPE_CHECKING:
    from .target import TargetNode


class BreakdownState:
    """Manages target queue and round-robin advancement within a breakdown.

    Starts with the theorem (target_id=-1) as the current target.
    When the theorem succeeds, its used_lemma_ids are queued.
    Cycles through targets in round-robin order.
    """

    def __init__(
        self,
        breakdown_idx: int,
        cost: DetailedCost,
        target_nodes: Dict[int, 'TargetNode'],
    ):
        self.breakdown_idx = breakdown_idx
        self.cost = cost
        self.target_nodes = target_nodes

        # Start with theorem as the current target
        self.current_target_id: int = -1
        self.target_queue: List[int] = []
        self.proven_targets: Set[int] = set()
        self.failed: bool = False

        # Track which lemmas have been discovered (queued at least once)
        self._discovered: Set[int] = set()

    @property
    def current_target_node(self) -> 'TargetNode':
        return self.target_nodes[self.current_target_id]

    def on_action_complete(self, target_id: int, result: ActionResult):
        """Called after PROVE/CORRECT on a target. Handles success/failure and round-robin."""
        if result.success:
            self.proven_targets.add(target_id)

            # Queue newly discovered lemma dependencies
            if result.used_lemma_ids:
                for lid in result.used_lemma_ids:
                    if lid not in self.proven_targets and lid not in self._discovered:
                        if lid in self.target_nodes:
                            self._discovered.add(lid)
                            self.target_queue.append(lid)
                        else:
                            # Theorem uses a lemma that has no proof data in this breakdown
                            self.failed = True
                            return

            # Remove proven target from queue
            if target_id in self.target_queue:
                self.target_queue.remove(target_id)

            self._advance()
        else:
            # Failed but not exhausted — round-robin to next target
            if target_id in self.target_queue:
                self.target_queue.remove(target_id)
                self.target_queue.append(target_id)
            self._advance()

    def on_exhausted(self, target_id: int):
        """Current target has no more attempts — breakdown is dead."""
        self.failed = True

    def _advance(self):
        """Advance to next target in round-robin queue."""
        if self.target_queue:
            self.current_target_id = self.target_queue[0]
        # else: queue empty, check is_complete() externally

    def is_complete(self) -> bool:
        """Theorem proven AND all discovered lemma dependencies proven."""
        if self.failed:
            return False
        if -1 not in self.proven_targets:
            return False
        # All queued targets must be proven
        return len(self.target_queue) == 0

    def targets_remaining(self) -> int:
        """Number of unproven targets in the queue."""
        return len(self.target_queue)

    def targets_proven(self) -> int:
        """Number of proven targets."""
        return len(self.proven_targets)
