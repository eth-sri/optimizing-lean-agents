"""
Exceptions and helper classes for the data models.
"""

from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING
from seed_data_models.formalization import Formalization
from seed_data_models.proof_attempt import ProofAttempt

if TYPE_CHECKING:
    from seed_prover.simulations.attempt_tracker import ProofAttemptTracker

class SimulationFailure(Exception):
    """Raised when simulation fails at any level."""
    pass


class ProverPathNode:
    """Represents a node in the prover path tree (for visualization)."""
    _id_counter = 0  # Class variable for unique IDs

    def __init__(self, label: str, color: str):
        self.label = label  # e.g., "T" for theorem, "L2" for lemma 2
        self.color = color  # "blue", "green", or "red"
        self.children: List['ProverPathNode'] = []
        self._id = ProverPathNode._id_counter  # Unique ID for this node
        ProverPathNode._id_counter += 1

    def add_child(self, child: 'ProverPathNode') -> None:
        """Add a child node to this node."""
        self.children.append(child)

    def __repr__(self) -> str:
        """String representation showing label and color."""
        return f"ProverPathNode({self.label}, {self.color})"

    def __hash__(self) -> int:
        """Make node hashable using its unique ID."""
        return hash(self._id)

    def __eq__(self, other) -> bool:
        """Equality based on unique ID."""
        if not isinstance(other, ProverPathNode):
            return False
        return self._id == other._id

def _try_formalization(
    formalization: Formalization,
    seed: int,
    max_depth: Optional[int],
    search_policy: str,  # DEPRECATED: kept for API compatibility but ignored
    strategy_template: Any,
    tracker: Optional["ProofAttemptTracker"] = None,
    breakdown_id: int = 0,
    lemma_id: int = -1,
    formalization_id: int = 0,
    iteration_id: int = 0,
    origin_problem_id: Optional[str] = None,
) -> Optional[List[ProofAttempt]]:
    """
    Try proof attempts using Strategy with progressive model complexity.

    The strategy owns all search logic including 8b/32b progression.

    Args:
        formalization: Formalization to try
        seed: Random seed (passed to strategy)
        max_depth: Maximum total attempts (None = unlimited)
        search_policy: DEPRECATED - kept for API compatibility, ignored
        strategy_template: Strategy instance to clone with fresh context
        tracker: Optional ProofAttemptTracker for recording attempts
        breakdown_id: Breakdown ID for tracking
        lemma_id: Lemma ID for tracking (-1 for theorem)
        formalization_id: Formalization ID for tracking
        iteration_id: Iteration depth (0=theorem, 1=direct lemmas, 2=lemma's lemmas, etc.)
        origin_problem_id: Origin problem ID for tracking

    Returns:
        List of attempts tried up to first passing attempt, or None if all failed
    """
    # Build metadata for strategy
    metadata = {
        "origin_problem_id": origin_problem_id,
        "breakdown_id": breakdown_id,
        "lemma_id": lemma_id,
        "formalization_id": formalization_id,
        "iteration_id": iteration_id,
    }

    # Clone strategy with fresh all_attempts, seed, and metadata
    strategy_class = type(strategy_template)
    strategy_config = strategy_template.get_config()

    strategy = strategy_class(
        all_attempts=formalization.proof_attempts,
        seed=seed,
        metadata=metadata,
        **strategy_config
    )

    attempts_tried = []
    attempt_number = 0

    while True:
        # Get next attempt from strategy
        result = strategy.next_attempt()

        if result is None:
            # Strategy says we're done
            break

        attempt, metadata = result
        attempts_tried.append(attempt)
        attempt_number += 1

        is_passing = attempt.is_passing()
        is_complete = attempt.compilation_result.complete if attempt.compilation_result else False

        # Capture strategy state BEFORE recording result (so we get the mode used for THIS attempt)
        strategy_snapshot_before = strategy.summarize_params()

        # Record result with strategy
        strategy.record_result(
            reasoning_summary=attempt.reasoning_summary,
            compilation_summary=attempt.compilation_summary
        )

        # Record attempt if tracker provided
        if tracker is not None:
            tracker.record_attempt(
                breakdown_id=breakdown_id,
                lemma_id=lemma_id,
                formalization_id=formalization_id,
                depth_level=0,  # Not used anymore
                attempt_number=attempt_number,
                is_passing=is_passing,
                is_complete=is_complete,
                reasoning_summary=attempt.reasoning_summary,
                compilation_summary=attempt.compilation_summary,
                should_stop=None,  # Strategy doesn't expose this anymore
                strategy_params_snapshot=strategy_snapshot_before,
                uid=attempt.uid if hasattr(attempt, "uid") else "unknown",
                compilation_result={
                    "passed": attempt.compilation_result.passed if attempt.compilation_result else False,
                    "complete": is_complete,
                    "error_count": len(attempt.compilation_result.errors) if attempt.compilation_result else 0,
                },
                model_config_path=attempt.model_config_path,
                detailed_cost=attempt.detailed_cost,
                iteration_id=attempt.iteration_id,
                correction_round_id=attempt.correction_round_id,
                used_lemma_ids=attempt.used_lemma_ids,
            )

        # Check max_depth limit
        if max_depth is not None and attempt_number >= max_depth:
            break

        # Check if passing
        if is_passing:
            return attempts_tried  # Success!

    return attempts_tried if attempts_tried else None