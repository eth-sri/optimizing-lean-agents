"""TargetNode: the core recursive proof target abstraction."""

import random
from typing import Dict, List, Optional, Set, Tuple

from .actions import Action, ActionType, ActionResult, DetailedCost
from .breakdown_state import BreakdownState
from .data.types import AttemptPair, BreakdownTemplate


class TargetNode:
    """A provable entity at any depth. Uniform interface.

    Every provable entity (problem, theorem, lemma) is a TargetNode.
    The action space is identical at every depth.
    """

    def __init__(
        self,
        target_id: str,
        target_type: str,
        proof_data: Dict[str, List[AttemptPair]],
        breakdown_templates: List[BreakdownTemplate],
        seed: int,
        breakdown_id: int = -1,
        lemma_id: int = -1,
    ):
        """Initialize a TargetNode.

        Args:
            target_id: Unique identifier (e.g., "putnam_1962_a5" or "r0_b0_l3")
            target_type: "problem", "theorem", or "lemma"
            proof_data: {model_name: [AttemptPair]} — direct proof attempts
            breakdown_templates: Available decompositions
            seed: Random seed for shuffling
            breakdown_id: Original breakdown_id from agent data (-1 for root)
            lemma_id: Original lemma_id (-1=root/theorem, 0+=lemma)
        """
        self.target_id = target_id
        self.target_type = target_type
        self.breakdown_id = breakdown_id
        self.lemma_id = lemma_id

        # Immutable data
        self._original_proof_data = proof_data
        self._original_breakdown_templates = breakdown_templates

        # Initialize mutable state with shuffling
        self._seed = seed
        self._breakdown_model_filter: Optional[set] = None
        self._max_breakdowns: Optional[int] = None
        self._max_corrections: Optional[int] = None
        self._init_state(seed)

    def _init_state(self, seed: int):
        """Initialize/reset all mutable state with given seed."""
        rng = random.Random(seed)

        # Shuffle proof attempts per model
        self.proof_attempts: Dict[str, List[AttemptPair]] = {}
        for model_name, pairs in self._original_proof_data.items():
            shuffled = list(pairs)
            rng.shuffle(shuffled)
            self.proof_attempts[model_name] = shuffled

        # Shuffle breakdown templates (filtered by active models if set)
        if self._breakdown_model_filter:
            self.breakdown_templates: List[BreakdownTemplate] = [
                t for t in self._original_breakdown_templates
                if any(m in self._breakdown_model_filter
                       for td in t.target_proof_data.values()
                       for m in td.keys())
            ]
        else:
            self.breakdown_templates: List[BreakdownTemplate] = list(self._original_breakdown_templates)
        rng.shuffle(self.breakdown_templates)

        # Limit number of available breakdowns
        if self._max_breakdowns is not None:
            self.breakdown_templates = self.breakdown_templates[:self._max_breakdowns]

        # Decomposition state
        self.decomposed: bool = False
        self.breakdowns: List[BreakdownState] = []
        self.active_breakdown: Optional[BreakdownState] = None
        self.next_breakdown_idx: int = 0

        # Direct-proof tracking (when NOT decomposed)
        self.prove_indices: Dict[str, int] = {m: 0 for m in self.proof_attempts}
        self.last_attempt_pair: Optional[AttemptPair] = None
        self.correction_index: int = 0
        self.last_prove_model: Optional[str] = None
        self.last_prove_success: Optional[bool] = None

        # Solved flag
        self._solved: bool = False

        # Store rng state for deriving child seeds
        self._child_seed_counter = rng.randint(0, 2**31 - 1)

    def _derive_child_seed(self) -> int:
        """Derive a new seed for child TargetNodes."""
        self._child_seed_counter += 1
        return self._child_seed_counter

    # ───────── Core interface ─────────

    def get_valid_actions(self) -> List[Action]:
        """Valid actions for THIS target node."""
        if self._solved:
            return []

        if not self.decomposed:
            return self._get_direct_actions()
        else:
            return self._get_decomposed_actions()

    def execute_action(self, action: Action) -> ActionResult:
        """Execute action, update state, return result."""
        # When decomposed, delegate proof actions to the inner focus
        if (self.decomposed
                and self.active_breakdown is not None
                and not self.active_breakdown.failed
                and not self.active_breakdown.is_complete()
                and action.type in (ActionType.PROVE, ActionType.CORRECT)):
            focus = self.active_breakdown.current_target_node.get_current_focus()
            return focus.execute_action(action)

        # Handle locally (non-decomposed, or structural actions like CREATE_BREAKDOWN)
        if action.type == ActionType.PROVE:
            return self._execute_prove(action.model)
        elif action.type == ActionType.CORRECT:
            return self._execute_correct()
        elif action.type == ActionType.DECOMPOSE:
            return self._execute_decompose()
        elif action.type == ActionType.CREATE_BREAKDOWN:
            return self._execute_create_breakdown()
        elif action.type == ActionType.TERMINATE:
            return ActionResult(success=False, cost=DetailedCost())
        else:
            raise ValueError(f"Unknown action type: {action.type}")

    def get_current_focus(self) -> 'TargetNode':
        """Follow decomposition chain to the innermost active target."""
        if not self.decomposed:
            return self
        if self.active_breakdown is None:
            return self
        if self.active_breakdown.failed:
            return self
        if self.active_breakdown.is_complete():
            return self  # solved, should be caught by is_solved()

        # Recurse into the current target of the active breakdown
        current = self.active_breakdown.current_target_node
        return current.get_current_focus()

    def is_solved(self) -> bool:
        """Is this target fully solved?"""
        if self._solved:
            return True
        # Check if any active breakdown is complete
        if self.decomposed:
            for bs in self.breakdowns:
                if bs.is_complete():
                    self._solved = True
                    return True
        return False

    def reset(self, seed: int, breakdown_model_filter: Optional[set] = None, max_breakdowns: Optional[int] = None, max_corrections: Optional[int] = None):
        """Reshuffle and reset all state recursively."""
        if breakdown_model_filter is not None:
            self._breakdown_model_filter = breakdown_model_filter
        if max_breakdowns is not None:
            self._max_breakdowns = max_breakdowns
        if max_corrections is not None:
            self._max_corrections = max_corrections
        self._seed = seed
        self._init_state(seed)

    # ───────── Direct proof actions (not decomposed) ─────────

    def _get_direct_actions(self) -> List[Action]:
        """Actions when target is NOT decomposed."""
        actions = []

        # PROVE for each model with remaining attempts
        for model_name, pairs in self.proof_attempts.items():
            idx = self.prove_indices.get(model_name, 0)
            if idx < len(pairs):
                actions.append(Action(type=ActionType.PROVE, model=model_name))

        # CORRECT if last PROVE failed and corrections exist
        max_corr = len(self.last_attempt_pair.corrections) if self.last_attempt_pair is not None else 0
        if self._max_corrections is not None:
            max_corr = min(max_corr, self._max_corrections)
        if (self.last_prove_success is False
                and self.last_attempt_pair is not None
                and self.correction_index < max_corr):
            actions.append(Action(type=ActionType.CORRECT))

        # DECOMPOSE if breakdown templates exist
        if self.breakdown_templates:
            actions.append(Action(type=ActionType.DECOMPOSE))

        # TERMINATE always available
        actions.append(Action(type=ActionType.TERMINATE))

        return actions

    def _execute_prove(self, model_name: str) -> ActionResult:
        """Draw next shuffled proof attempt for given model."""
        pairs = self.proof_attempts.get(model_name, [])
        idx = self.prove_indices.get(model_name, 0)

        if idx >= len(pairs):
            return ActionResult(success=False, cost=DetailedCost())

        pair = pairs[idx]
        self.prove_indices[model_name] = idx + 1

        # Update tracking
        self.last_attempt_pair = pair
        self.correction_index = 0
        self.last_prove_model = model_name
        self.last_prove_success = pair.initial.success

        if pair.initial.success:
            self._solved = True

        return ActionResult(
            success=pair.initial.success,
            cost=pair.initial.cost,
            proof_length=pair.initial.proof_length,
            num_errors=pair.initial.num_errors,
            used_lemma_ids=pair.initial.used_lemma_ids,
            attempt_id=pair.initial.attempt_id,
            code=pair.initial.code,
            error_messages=pair.initial.error_messages,
        )

    def _execute_correct(self) -> ActionResult:
        """Draw next correction for the last failed attempt."""
        if self.last_attempt_pair is None:
            return ActionResult(success=False, cost=DetailedCost())

        corrections = self.last_attempt_pair.corrections
        if self.correction_index >= len(corrections):
            return ActionResult(success=False, cost=DetailedCost())

        correction = corrections[self.correction_index]
        self.correction_index += 1
        self.last_prove_success = correction.success

        if correction.success:
            self._solved = True

        return ActionResult(
            success=correction.success,
            cost=correction.cost,
            proof_length=correction.proof_length,
            num_errors=correction.num_errors,
            used_lemma_ids=correction.used_lemma_ids,
            attempt_id=correction.attempt_id,
            code=correction.code,
            error_messages=correction.error_messages,
        )

    # ───────── Decomposed actions ─────────

    def _execute_decompose(self) -> ActionResult:
        """Free meta-action: commit to breakdown approach (irreversible)."""
        self.decomposed = True
        # Reset direct-proof tracking (no longer relevant)
        self.last_attempt_pair = None
        self.last_prove_model = None
        self.last_prove_success = None
        return ActionResult(success=True, cost=DetailedCost())

    def _get_decomposed_actions(self) -> List[Action]:
        """Actions when target IS decomposed."""
        actions = []

        if self.active_breakdown is None or self.active_breakdown.failed:
            # No active breakdown or current one failed
            if self.next_breakdown_idx < len(self.breakdown_templates):
                actions.append(Action(type=ActionType.CREATE_BREAKDOWN))
            actions.append(Action(type=ActionType.TERMINATE))
            return actions

        if self.active_breakdown.is_complete():
            # Breakdown complete — target is solved
            self._solved = True
            return []

        # Active breakdown in progress — get actions from current target
        current = self.active_breakdown.current_target_node
        focus = current.get_current_focus()
        inner_actions = focus.get_valid_actions()

        # Filter out TERMINATE from inner actions (we add our own at this level)
        actions = [a for a in inner_actions if a.type != ActionType.TERMINATE]

        # Can always create another breakdown (abandon current) if available
        if self.next_breakdown_idx < len(self.breakdown_templates):
            actions.append(Action(type=ActionType.CREATE_BREAKDOWN))

        # TERMINATE at this level
        actions.append(Action(type=ActionType.TERMINATE))

        return actions

    def _execute_create_breakdown(self) -> ActionResult:
        """Create one more breakdown from the shuffled templates."""
        if self.next_breakdown_idx >= len(self.breakdown_templates):
            return ActionResult(success=False, cost=DetailedCost())

        template = self.breakdown_templates[self.next_breakdown_idx]
        self.next_breakdown_idx += 1

        # Build TargetNodes for each target in the breakdown
        target_nodes: Dict[int, TargetNode] = {}
        for target_id, proof_data in template.target_proof_data.items():
            node_id = f"{self.target_id}_b{template.breakdown_idx}_{'theorem' if target_id == -1 else f'l{target_id}'}"
            node_type = "theorem" if target_id == -1 else "lemma"

            child = TargetNode(
                target_id=node_id,
                target_type=node_type,
                proof_data=proof_data,
                breakdown_templates=[],  # No recursive decomposition data yet
                seed=self._derive_child_seed(),
                breakdown_id=template.breakdown_idx,
                lemma_id=target_id,
            )
            if self._max_corrections is not None:
                child._max_corrections = self._max_corrections
            target_nodes[target_id] = child

        # If no theorem node exists, the breakdown is invalid
        if -1 not in target_nodes:
            return ActionResult(success=False, cost=template.cost)

        bs = BreakdownState(
            breakdown_idx=template.breakdown_idx,
            cost=template.cost,
            target_nodes=target_nodes,
        )
        self.breakdowns.append(bs)
        self.active_breakdown = bs

        return ActionResult(success=True, cost=template.cost)

    # ───────── Empirical rates (for oracle policy) ─────────

    def get_empirical_rates(self) -> Dict[Action, float]:
        """Compute empirical success rates from the full data."""
        rates: Dict[Action, float] = {}

        if not self.decomposed:
            # Direct proof rates per model
            for model_name, pairs in self._original_proof_data.items():
                if not pairs:
                    continue
                total = len(pairs)
                successes = sum(1 for p in pairs if p.initial.success)
                # Include correction successes
                correction_successes = sum(
                    1 for p in pairs
                    for c in p.corrections if c.success
                )
                rate = (successes + correction_successes) / max(total, 1)
                rates[Action(type=ActionType.PROVE, model=model_name)] = rate

            # Correction rate (conditional on having a failed attempt)
            if self.last_attempt_pair and not self.last_prove_success:
                corr = self.last_attempt_pair.corrections
                if corr:
                    remaining = corr[self.correction_index:]
                    if remaining:
                        rate = sum(1 for c in remaining if c.success) / len(remaining)
                        rates[Action(type=ActionType.CORRECT)] = rate

            # Decompose rate: fraction of breakdowns that lead to solutions
            if self._original_breakdown_templates:
                solved = sum(1 for t in self._original_breakdown_templates
                             if _template_is_solved(t))
                rates[Action(type=ActionType.DECOMPOSE)] = solved / len(self._original_breakdown_templates)

        else:
            # CREATE_BREAKDOWN rate
            remaining_templates = self.breakdown_templates[self.next_breakdown_idx:]
            if remaining_templates:
                solved = sum(1 for t in remaining_templates if _template_is_solved(t))
                rates[Action(type=ActionType.CREATE_BREAKDOWN)] = solved / len(remaining_templates)

        rates[Action(type=ActionType.TERMINATE)] = 0.0
        return rates

    def get_average_action_cost(self, action: Action) -> DetailedCost:
        """Average cost for an action from the full data."""
        if action.type == ActionType.PROVE and action.model:
            pairs = self._original_proof_data.get(action.model, [])
            if not pairs:
                return DetailedCost()
            total = DetailedCost()
            for p in pairs:
                total += p.initial.cost
            return DetailedCost(
                input_sflops=total.input_sflops // len(pairs),
                output_sflops=total.output_sflops // len(pairs),
                input_tokens=total.input_tokens // len(pairs),
                output_tokens=total.output_tokens // len(pairs),
            )

        if action.type == ActionType.CORRECT:
            if self.last_attempt_pair and self.last_attempt_pair.corrections:
                corr = self.last_attempt_pair.corrections
                total = DetailedCost()
                for c in corr:
                    total += c.cost
                n = len(corr)
                return DetailedCost(
                    input_sflops=total.input_sflops // n,
                    output_sflops=total.output_sflops // n,
                    input_tokens=total.input_tokens // n,
                    output_tokens=total.output_tokens // n,
                )

        if action.type == ActionType.CREATE_BREAKDOWN:
            remaining = self.breakdown_templates[self.next_breakdown_idx:]
            if remaining:
                total = DetailedCost()
                for t in remaining:
                    total += t.cost
                n = len(remaining)
                return DetailedCost(
                    input_sflops=total.input_sflops // n,
                    output_sflops=total.output_sflops // n,
                    input_tokens=total.input_tokens // n,
                    output_tokens=total.output_tokens // n,
                )

        return DetailedCost()

    # ───────── State snapshot ─────────

    def get_depth(self) -> int:
        """Get the depth of this node in the decomposition chain (0=root)."""
        return 0  # depth is tracked externally by get_current_focus traversal

    def get_state_snapshot(self) -> dict:
        """Return a dict summarizing this node's state."""
        return {
            'target_id': self.target_id,
            'target_type': self.target_type,
            'breakdown_id': self.breakdown_id,
            'lemma_id': self.lemma_id,
            'decomposed': self.decomposed,
            'solved': self._solved,
            'prove_attempts_used': dict(self.prove_indices),
            'prove_attempts_available': {
                m: len(p) for m, p in self.proof_attempts.items()
            },
            'last_prove_model': self.last_prove_model,
            'last_prove_success': self.last_prove_success,
            'breakdowns_created': len(self.breakdowns),
            'breakdowns_available': len(self.breakdown_templates),
            'corrections_used': self.correction_index,
            'corrections_available': len(self.last_attempt_pair.corrections) if self.last_attempt_pair else 0,
            'active_breakdown_idx': self.active_breakdown.breakdown_idx if self.active_breakdown else None,
            'active_breakdown_failed': self.active_breakdown.failed if self.active_breakdown else None,
            'active_breakdown_targets_proven': self.active_breakdown.targets_proven() if self.active_breakdown else 0,
            'active_breakdown_targets_remaining': self.active_breakdown.targets_remaining() if self.active_breakdown else 0,
        }


def _template_is_solved(template: BreakdownTemplate) -> bool:
    """Check if a breakdown template leads to a full solution in the data.

    A template is 'solved' if:
    1. The theorem (-1) has at least one successful proof attempt
    2. All lemmas used by the theorem's best proof also have successful proofs
    """
    theorem_data = template.target_proof_data.get(-1, {})

    # Check if theorem has any successful proof
    theorem_solved = False
    best_used_lemmas: Optional[Set[int]] = None

    for model, pairs in theorem_data.items():
        for pair in pairs:
            if pair.initial.success:
                theorem_solved = True
                best_used_lemmas = pair.initial.used_lemma_ids
                break
            for c in pair.corrections:
                if c.success:
                    theorem_solved = True
                    best_used_lemmas = c.used_lemma_ids
                    break
            if theorem_solved:
                break
        if theorem_solved:
            break

    if not theorem_solved:
        return False

    # Check if all used lemmas have proofs
    if best_used_lemmas:
        for lid in best_used_lemmas:
            lemma_data = template.target_proof_data.get(lid, {})
            lemma_solved = False
            for model, pairs in lemma_data.items():
                for pair in pairs:
                    if pair.initial.success:
                        lemma_solved = True
                        break
                    for c in pair.corrections:
                        if c.success:
                            lemma_solved = True
                            break
                    if lemma_solved:
                        break
                if lemma_solved:
                    break
            if not lemma_solved:
                return False

    return True
