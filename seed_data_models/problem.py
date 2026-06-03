"""
Problem data model.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, TYPE_CHECKING, Any
import random


from .breakdown_models import Breakdown
from .exceptions_and_helpers import ProverPathNode, SimulationFailure

if TYPE_CHECKING:
    from .simulations.attempt_tracker import ProofAttemptTracker

@dataclass
class Problem:
    """Represents a problem with multiple breakdowns.

    For round 0 problems:
    - breakdowns: Dict of round 0 breakdowns only (not filtered during load, stored directly)
    - recursive_attempts: List of separate Problem objects for round 1+ recursive proving
    - full_proof_attempts: Dict with "8b" and "32b" keys containing full proof attempts

    The recursive_attempts are completely separate Problem objects where:
    - Each represents a failed lemma from round 0 being recursively broken down
    - Their origin_problem_id is the lemma's UID
    - They can themselves have recursive_attempts (multi-level recursion)

    For round 1+ problems (in recursive_attempts):
    - origin_problem_id: The UID of the failed lemma from the parent round
    - breakdowns: The round 1+ (or higher) breakdowns for this recursive attempt
    - recursive_attempts: Further recursive attempts if this attempt also failed
    """
    origin_problem_id: str
    breakdowns: Dict[tuple, Breakdown] = field(default_factory=dict)  # Key: (parent_problem_id, round_id, breakdown_id)
    recursive_attempts: List['Problem'] = field(default_factory=list)  # Separate Problems for round 1+ recursive proving
    difficulty: Optional[str] = None  # Difficulty tag from dataset.json
    full_proof_attempts: Dict[str, List] = field(default_factory=dict)  # Keys: "8b", "32b"

    def is_solved(self) -> bool:
        """Returns True if any main breakdown of this problem is solved.

        Note: Only checks main breakdowns, not recursive attempts.
        For overall status including recursive problems, use get_solved_breakdowns().
        """
        # Check only this problem's main breakdowns
        for breakdown in self.breakdowns.values():
            if breakdown.is_solved():
                return True
        return False

    def get_solved_breakdowns(self) -> List[Breakdown]:
        """Returns list of all solved breakdowns for this problem (not including recursive attempts)."""
        return [bd for bd in self.breakdowns.values() if bd.is_solved()]

    def get_total_cost(self, cost_type="cost", exclude_prover_calls: bool = False) -> float:
        """Returns the total cost for this problem across all breakdowns and recursive attempts.

        Args:
            cost_type: Type of cost to return (e.g., 'cost', 'output_sflops', 'input_tokens')
            exclude_prover_calls: If True, excludes proof attempt costs (useful for seeing just breakdown/formalization costs)
        """
        cost = sum(breakdown.get_total_cost(cost_type, exclude_prover_calls=exclude_prover_calls) for breakdown in self.breakdowns.values())
        # Add costs from recursive attempts
        for recursive_problem in self.recursive_attempts:
            cost += recursive_problem.get_total_cost(cost_type, exclude_prover_calls=exclude_prover_calls)
        return cost

    def count_rounds(self) -> int:
        """Returns the number of breakdown rounds (based on round_id)."""
        if not self.breakdowns:
            return 0
        round_ids = set(bd.round_id for bd in self.breakdowns.values())
        return len(round_ids)

    def get_cost_by_round(self, cost_type="cost") -> Dict[int, float]:
        """Returns a dict with cost per round."""
        cost_by_round = {}
        for breakdown in self.breakdowns.values():
            round_id = breakdown.round_id
            if round_id not in cost_by_round:
                cost_by_round[round_id] = 0.0
            cost_by_round[round_id] += breakdown.get_cost(cost_type)
        return cost_by_round

    def get_average_component_costs(self) -> Dict[str, Dict[str, float]]:
        """
        Returns average component costs across all breakdowns in this problem.

        Aggregates component costs from all breakdowns and returns the average.

        Returns:
        {
            "breakdown": {"input_tokens": avg, "output_tokens": avg},
            "breakdown_parser": {"input_tokens": avg, "output_tokens": avg},
            "formalization": {"input_tokens": avg, "output_tokens": avg},
            "prover": {"input_tokens": avg, "output_tokens": avg}
        }
        """
        if not self.breakdowns:
            return {
                "breakdown": {"input_tokens": 0, "output_tokens": 0},
                "breakdown_parser": {"input_tokens": 0, "output_tokens": 0},
                "formalization": {"input_tokens": 0, "output_tokens": 0},
                "prover": {"input_tokens": 0, "output_tokens": 0}
            }

        component_totals = {
            "breakdown": {"input_tokens": 0, "output_tokens": 0},
            "breakdown_parser": {"input_tokens": 0, "output_tokens": 0},
            "formalization": {"input_tokens": 0, "output_tokens": 0},
            "prover": {"input_tokens": 0, "output_tokens": 0}
        }

        # Sum component costs across all breakdowns
        for breakdown in self.breakdowns.values():
            costs = breakdown.get_component_costs()
            for component, tokens in costs.items():
                component_totals[component]["input_tokens"] += tokens.get("input_tokens", 0)
                component_totals[component]["output_tokens"] += tokens.get("output_tokens", 0)

        # Calculate averages
        num_breakdowns = len(self.breakdowns)
        averages = {}
        for component, tokens in component_totals.items():
            averages[component] = {
                "input_tokens": tokens["input_tokens"] / num_breakdowns if num_breakdowns > 0 else 0,
                "output_tokens": tokens["output_tokens"] / num_breakdowns if num_breakdowns > 0 else 0
            }

        return averages

    def get_component_costs_by_round(self) -> Dict[int, Dict[str, Dict[str, int]]]:
        """
        Returns component costs broken down by round.

        Returns:
        {
            0: {"breakdown": {...}, "parser": {...}, ...},
            1: {"breakdown": {...}, "parser": {...}, ...},
            ...
        }
        """
        costs_by_round = {}
        for breakdown in self.breakdowns.values():
            round_id = breakdown.round_id
            if round_id not in costs_by_round:
                costs_by_round[round_id] = {
                    "breakdown": {"input_tokens": 0, "output_tokens": 0},
                    "breakdown_parser": {"input_tokens": 0, "output_tokens": 0},
                    "formalization": {"input_tokens": 0, "output_tokens": 0},
                    "prover": {"input_tokens": 0, "output_tokens": 0}
                }

            # Add this breakdown's component costs to the round
            costs = breakdown.get_component_costs()
            for component, tokens in costs.items():
                costs_by_round[round_id][component]["input_tokens"] += tokens.get("input_tokens", 0)
                costs_by_round[round_id][component]["output_tokens"] += tokens.get("output_tokens", 0)

        return costs_by_round

    def generate_proof(self) -> str:
        """
        Generate a complete proof for this problem by finding the best solved breakdown.

        This is used for recursive attempts where a Problem object represents a failed lemma
        that was recursively broken down. It finds the best (first) solved breakdown and
        generates its proof.

        Returns:
            Complete Lean code string, or empty string if no solved breakdown found
        """
        # Find the first solved breakdown
        for breakdown in self.breakdowns.values():
            if breakdown.is_solved():
                proof_code, _ = breakdown.generate_proof()
                if proof_code:
                    return proof_code

        # No solved breakdown found
        return ""

    def get_prover_path(self, recursive: bool = True) -> Optional[ProverPathNode]:
        """
        Build a prover path tree for this problem.

        Finds a solved breakdown (or the best attempt), and returns its prover path.

        Args:
            recursive: If True, follow recursive_attempts for unsolved lemmas

        Returns:
            ProverPathNode tree, or None if no breakdown found
        """
        # Find the first solved breakdown
        for breakdown in self.breakdowns.values():
            if breakdown.is_solved():
                path = breakdown.get_prover_path(recursive=recursive)
                if path:
                    return path

        # No solved breakdown found, try the first breakdown (best attempt)
        if self.breakdowns:
            first_breakdown = next(iter(self.breakdowns.values()))
            return first_breakdown.get_prover_path(recursive=recursive)

        return None

    def create_fresh_copy(self) -> "Problem":
        """
        Create a fresh copy of this problem with empty breakdowns.

        Used to start a clean simulation where breakdowns are added one at a time.

        Returns:
            New Problem with same metadata but no breakdowns
        """
        return Problem(
            origin_problem_id=self.origin_problem_id,
            breakdowns={},
            recursive_attempts=[],
            difficulty=self.difficulty,
        )

    def simulate(
        self,
        seed: int,
        max_depth: Optional[int] = None,
        search_policy: str = "sequential",
        strategy: Optional[Any] = None,
        tracker: Optional["ProofAttemptTracker"] = None,
        source_problem: Optional["Problem"] = None,
    ) -> "Problem":
        """
        Simulate problem solving by trying multiple breakdowns one at a time.

        If source_problem is provided:
        - Uses source_problem as the source of breakdowns to attempt
        - self is the result container (must be empty)
        - Only adds successfully-simulated breakdowns to self
        - Stops when one is solved

        If source_problem is not provided:
        - self contains all available breakdowns
        - Attempts them in random order
        - Returns a new Problem with only attempted breakdowns

        Args:
            seed: Random seed for reproducibility
            max_depth: Maximum depth/attempts to try (None = unlimited)
            search_policy: Either "exponential" or "sequential"
            strategy: Optional strategy object to guide simulation decisions
            tracker: Optional ProofAttemptTracker for recording attempts
            source_problem: If provided, read breakdowns from here instead of self

        Returns:
            If source_problem provided: returns self (modified in-place)
            Otherwise: New Problem with only attempted breakdowns (stopping after first solved)
        """
        # Determine where to read breakdowns from
        source = source_problem if source_problem is not None else self
        result = self if source_problem is not None else self.create_fresh_copy()

        # Shuffle breakdown keys to try them in random order
        rng = random.Random(seed)
        breakdown_keys = list(source.breakdowns.keys())
        rng.shuffle(breakdown_keys)

        # Try each breakdown in shuffled order
        for breakdown_key in breakdown_keys:
            breakdown = source.breakdowns[breakdown_key]
            bd_seed = rng.randint(0, 2**31 - 1)

            # Simulate this breakdown (always returns a breakdown, may be failing)
            new_breakdown = breakdown.simulate(
                bd_seed,
                max_depth,
                search_policy,
                strategy=strategy,
                tracker=tracker,
            )

            # Add the simulated breakdown to result (successful or not)
            result.breakdowns[breakdown_key] = new_breakdown

            # Check if this breakdown is solved
            if new_breakdown.is_solved():
                # Solved! Stop here - don't try more breakdowns
                break

        # Return the result problem with only attempted breakdowns
        return result

    def simulate_action(
        self,
        action: Any,
        tracker: Optional["ProofAttemptTracker"] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Simulate a single RL action on this problem.

        Supported actions:
        - CREATE_BREAKDOWN: Create a new breakdown attempt
        - FULL_PROOF_8B/32B: Direct full proof (non-agentic)
        - ATTEMPT_8B/32B: Agentic attempt on current breakdown
        - CORRECTION_8B/32B: Correction on last agentic attempt
        - TERMINATE: Stop search

        Args:
            action: ProofSearchAction from RL framework
            tracker: Attempt tracker to record actions
            seed: Random seed for this action

        Returns:
            Dict with:
            - "cost": float (SFLOPs)
            - "success": bool
            - "state": updated state dict
            - "breakdown": Breakdown object if applicable
            - "attempt": ProofAttempt object if applicable
        """
        from seed_prover.simulations.rl.actions import ActionType

        # Get action type
        if hasattr(action, 'action_type'):
            action_type = action.action_type
            attempt_index = action.attempt_index or 0
        else:
            raise ValueError("Invalid action: must have action_type attribute")

        # TERMINATE
        if action_type == ActionType.TERMINATE:
            return {
                "cost": 0.0,
                "success": False,
                "state": {"terminated": True},
            }

        # CREATE_BREAKDOWN
        if action_type == ActionType.CREATE_BREAKDOWN:
            return self._simulate_create_breakdown(attempt_index, tracker, seed)

        # FULL_PROOF attempts (delegated to Breakdown)
        if action_type in (ActionType.FULL_PROOF_8B, ActionType.FULL_PROOF_32B):
            return self._simulate_full_proof(action_type, attempt_index, tracker, seed)

        # AGENTIC attempts (require active breakdown)
        if action_type in (ActionType.ATTEMPT_8B, ActionType.ATTEMPT_32B, ActionType.CORRECTION_8B, ActionType.CORRECTION_32B):
            return self._simulate_agentic_attempt(action_type, attempt_index, tracker, seed)

        raise ValueError(f"Unsupported action type: {action_type}")

    def _get_shuffled_full_proof_attempts(self, model_type: str, seed: Optional[int]) -> list:
        """
        Get shuffled full proof attempts for given model type and seed.

        Caches results per (model_type, seed) combination.

        Args:
            model_type: "8b" or "32b"
            seed: Random seed for shuffling (None uses seed=0)

        Returns:
            List of shuffled attempt dicts
        """
        import random

        # Use seed=0 if not provided
        if seed is None:
            seed = 0

        # Cache key based on model_type and seed
        cache_key = f"_full_proof_{model_type}_seed_{seed}"
        if hasattr(self, cache_key):
            return getattr(self, cache_key)

        # Get original attempts
        attempts = self.full_proof_attempts.get(model_type, [])

        # Shuffle with seed
        shuffled = attempts.copy()
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        # Cache and return
        setattr(self, cache_key, shuffled)
        return shuffled

    def _simulate_create_breakdown(
        self,
        breakdown_index: int,
        tracker: Optional["ProofAttemptTracker"],
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Create a new breakdown (overhead cost only)."""
        from seed_prover.simulations.rl.models import ActionCostModel

        # Get breakdown overhead cost
        cost_model = ActionCostModel()
        from seed_prover.simulations.rl.actions import ActionType, ProofSearchAction
        breakdown_action = ProofSearchAction.create_breakdown(breakdown_index)
        cost = cost_model(None, breakdown_action)

        # Record in tracker
        if tracker:
            tracker.create_breakdown(breakdown_index, cost)

        return {
            "cost": cost,
            "success": False,  # Doesn't solve problem, just creates overhead
            "state": {
                "current_breakdown_id": breakdown_index,
                "breakdown_created": True,
            },
            "breakdown_index": breakdown_index,
        }

    def _simulate_full_proof(
        self,
        action_type: Any,
        attempt_index: int,
        tracker: Optional["ProofAttemptTracker"],
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Simulate full proof attempt (non-agentic)."""
        from seed_prover.simulations.rl.actions import ActionType as AT
        from seed_data_models.model_config import get_effective_parameters

        if action_type == AT.FULL_PROOF_8B:
            model_type = "8b"
        else:
            model_type = "32b"

        # Get shuffled attempts based on seed
        attempts = self._get_shuffled_full_proof_attempts(model_type, seed)
        if attempt_index < len(attempts):
            attempt_data = attempts[attempt_index]
            success = attempt_data.get("pass", False) and attempt_data.get("complete", False)

            # Get cost from detailed_cost.output_sflops
            detailed_cost = attempt_data.get("detailed_cost", {})
            cost = detailed_cost.get("output_sflops", 0.0)

            # Get proof_length (in lines) from reasoning_summary or main attempt_data
            # For full_proof attempts, proof_length can be in the main dict
            proof_length = 0
            reasoning_summary = attempt_data.get("reasoning_summary")
            if reasoning_summary:
                proof_length = reasoning_summary.get("proof_length", 0)

            # If not found in reasoning_summary, check main attempt_data
            if proof_length == 0:
                proof_length = attempt_data.get("proof_length", 0)

            # Get num_errors from attempt_data (already extracted in minified.json)
            # Default to None if not available to distinguish from "0 errors"
            num_errors = attempt_data.get("num_errors")

            # Record in tracker
            if tracker and hasattr(tracker, 'add_full_proof_attempt'):
                tracker.add_full_proof_attempt(
                    action_type=action_type,
                    attempt_index=attempt_index,
                    cost=cost,
                    success=success,
                    proof_attempt=attempt_data,
                )

            return {
                "cost": cost,
                "success": success,
                "attempt": attempt_data,
                "state": {
                    "proof_length": proof_length,
                    "num_errors": num_errors,
                },
            }

        # No attempt available
        return {
            "cost": 0.0,
            "success": False,
            "state": {"error": "No full proof attempt available at this index"},
        }

    def _simulate_agentic_attempt(
        self,
        action_type: Any,
        attempt_index: int,
        tracker: Optional["ProofAttemptTracker"],
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Simulate agentic attempt on breakdown."""
        from seed_prover.simulations.rl.actions import ProofSearchAction, ActionType as AT

        # Need to have an active breakdown
        # Tracker should track current_breakdown_id
        if not tracker or not hasattr(tracker, 'current_breakdown_id'):
            raise ValueError("Agentic attempts require active breakdown via tracker")

        current_breakdown_id = tracker.current_breakdown_id
        if current_breakdown_id < 0:
            raise ValueError("No active breakdown for agentic attempt")

        # Get the breakdown
        # Find it in self.breakdowns by breakdown_id
        breakdown = None
        for key, bd in self.breakdowns.items():
            if bd.breakdown_id == current_breakdown_id:
                breakdown = bd
                break

        if not breakdown:
            raise ValueError(f"Breakdown {current_breakdown_id} not found in problem")

        # Create action
        if action_type == AT.ATTEMPT_8B:
            action = ProofSearchAction.attempt_8b(attempt_index)
        elif action_type == AT.ATTEMPT_32B:
            action = ProofSearchAction.attempt_32b(attempt_index)
        elif action_type == AT.CORRECTION_8B:
            action = ProofSearchAction.correction_8b(attempt_index)
        elif action_type == AT.CORRECTION_32B:
            action = ProofSearchAction.correction_32b(attempt_index)
        else:
            raise ValueError(f"Unsupported agentic action type: {action_type}")

        # Simulate via breakdown
        result = breakdown.simulate_action(
            action=action,
            tracker=tracker,
            seed=seed,
            max_depth=1,
            search_policy="sequential",
        )

        # Record in tracker
        if tracker and hasattr(tracker, 'add_agentic_attempt'):
            # Get actual target lemma_id from result state
            state_dict = result.get("state", {})
            target_lemma_id = state_dict.get("target_lemma_id", -1)

            tracker.add_agentic_attempt(
                action_type=action_type,
                lemma_id=target_lemma_id,
                attempt_index=attempt_index,
                cost=result["cost"],
                success=result["success"],
                proof_attempt=result.get("attempt"),
                metadata=result.get("state"),
            )

        return result

    def to_dict(self):
        """Convert to dictionary representation (flat, no nested breakdowns/recursive_attempts)."""
        return {
            "origin_problem_id": self.origin_problem_id,
            "difficulty": self.difficulty
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Problem':
        """Reconstruct Problem from dictionary representation (without breakdowns)."""
        return cls(
            origin_problem_id=data["origin_problem_id"],
            breakdowns={},  # Loaded separately
            recursive_attempts=[],  # Loaded separately
            difficulty=data.get("difficulty")
        )

    def load_full_proof_attempts(
        self,
        path_8b: str = "",
        path_32b: str = ""
    ):
        """
        Load full proof attempts for this problem from minified.json files.

        Args:
            path_8b: Path to 8b full proof attempts
            path_32b: Path to 32b full proof attempts

        Updates:
            self.full_proof_attempts with "8b" and "32b" lists
        """
        import json
        from pathlib import Path

        self.full_proof_attempts = {"8b": [], "32b": []}

        # Load 8b attempts
        if Path(path_8b).exists():
            with open(path_8b) as f:
                attempts_8b = json.load(f)

            # Filter for this problem
            for attempt_data in attempts_8b:
                if attempt_data.get("origin_problem_id") == self.origin_problem_id:
                    self.full_proof_attempts["8b"].append(attempt_data)

        # Load 32b attempts
        if Path(path_32b).exists():
            with open(path_32b) as f:
                attempts_32b = json.load(f)

            # Filter for this problem
            for attempt_data in attempts_32b:
                if attempt_data.get("origin_problem_id") == self.origin_problem_id:
                    self.full_proof_attempts["32b"].append(attempt_data)


def load_full_proof_attempts_for_problems(
    problems: Dict[str, 'Problem'],
    path_8b: str = "",
    path_32b: str = ""
) -> None:
    """
    Load full proof attempts for multiple problems.

    Args:
        problems: Dict of problem_id -> Problem
        path_8b: Path to 8b full proof attempts
        path_32b: Path to 32b full proof attempts

    Updates problems in-place with full_proof_attempts.
    """
    import json
    from pathlib import Path
    from collections import defaultdict

    # Group attempts by origin_problem_id
    attempts_by_problem = defaultdict(lambda: {"8b": [], "32b": []})

    # Load and group 8b attempts
    if Path(path_8b).exists():
        with open(path_8b) as f:
            attempts_8b = json.load(f)

        for attempt_data in attempts_8b:
            origin_problem_id = attempt_data.get("origin_problem_id")
            if origin_problem_id:
                attempts_by_problem[origin_problem_id]["8b"].append(attempt_data)

    # Load and group 32b attempts
    if Path(path_32b).exists():
        with open(path_32b) as f:
            attempts_32b = json.load(f)

        for attempt_data in attempts_32b:
            origin_problem_id = attempt_data.get("origin_problem_id")
            if origin_problem_id:
                attempts_by_problem[origin_problem_id]["32b"].append(attempt_data)

    # Assign to problems
    for problem_id, problem in problems.items():
        if problem.origin_problem_id in attempts_by_problem:
            problem.full_proof_attempts = attempts_by_problem[problem.origin_problem_id]
