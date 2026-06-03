"""Cost-quality tradeoff policy with pluggable probability and cost models.

Uses three-tier action scoring:
  1. Real actions (PROVE, CORRECT): scored by p - lambda*c
  2. DECOMPOSE / CREATE_BREAKDOWN: neutral baseline (score=0), wins only when
     all real actions < 0
  3. TERMINATE: pure fallback, chosen only when no positive-score action exists
     and DECOMPOSE is unavailable

Optional hot start: forces an initial sequence (decompose → prove cheap model)
before cost-quality scoring kicks in.
"""

import math
from typing import Dict, List, Optional


from .base import Policy
from .cost_models import CostModel
from .quality import ProbabilityModel
from ..actions import Action, ActionType
from ..state import SimulationState
from ..problem import SimulatedProblem

_MIN_PROB = 1e-8


class CostQualityPolicy(Policy):
    """Three-tier action scoring: real actions > DECOMPOSE/CREATE_BREAKDOWN > TERMINATE.

    Real actions (PROVE, CORRECT) are scored by p - lambda*c.
    DECOMPOSE and CREATE_BREAKDOWN are neutral baselines at score=0 — they win
    only when every real action scores negative.  TERMINATE is a pure fallback
    chosen only when no positive-scoring action exists and DECOMPOSE is unavailable.

    - lambda=0: picks highest-probability action (32b typically wins)
    - high lambda: picks cheapest action (8b typically wins)
    - very high lambda: DECOMPOSE wins, then TERMINATE if DECOMPOSE unavailable

    Hot start (optional):
        When hot_start is set (e.g. {"8b": 1}), the policy forces an initial
        sequence before cost-quality scoring kicks in:
        1. Force DECOMPOSE at depth 0 (if available)
        2. Force CREATE_BREAKDOWN at depth 0 (if available)
        3. Force N prove attempts of the specified model(s) per target
        This gives the cost model a meaningful baseline to compare against.
    """

    def __init__(
        self,
        prob_model: ProbabilityModel,
        cost_model: CostModel,
        lambda_val: float = 0.0,
        allowed_actions: Optional[List[str]] = None,
        hot_start: Optional[Dict[str, int]] = None,
    ):
        self.prob_model = prob_model
        self.cost_model = cost_model
        self.lambda_val = lambda_val
        self._allowed_actions = allowed_actions
        self.hot_start = hot_start
        self._problems: Dict[str, SimulatedProblem] = {}
        self._decision_metadata: Optional[dict] = None

    def set_problems(self, problems: List[SimulatedProblem]):
        """Store problem references for lookup during choose_action."""
        self._problems = {p.problem_id: p for p in problems}

    def _check_hot_start(self, state: SimulationState, filtered: List[Action]) -> Optional[Action]:
        """Check hot-start conditions and return forced action, or None to fall through.

        Order:
        1. Force DECOMPOSE at depth 0
        2. Force CREATE_BREAKDOWN at depth 0
        3. Force prove attempts for each (model, n) in hot_start
           - After each failed prove, also force any available CORRECT actions
        """
        action_by_type = {}
        for a in filtered:
            if a.type == ActionType.DECOMPOSE:
                action_by_type[ActionType.DECOMPOSE] = a
            elif a.type == ActionType.CREATE_BREAKDOWN:
                action_by_type[ActionType.CREATE_BREAKDOWN] = a
            elif a.type == ActionType.PROVE:
                action_by_type[("prove", a.model)] = a
            elif a.type == ActionType.CORRECT:
                action_by_type[ActionType.CORRECT] = a

        # 1. Force decompose at depth 0
        if state.decomposition_depth == 0 and ActionType.DECOMPOSE in action_by_type:
            return action_by_type[ActionType.DECOMPOSE]

        # 2. Force create_breakdown at depth 0
        if state.decomposition_depth == 0 and ActionType.CREATE_BREAKDOWN in action_by_type:
            return action_by_type[ActionType.CREATE_BREAKDOWN]

        # 3. Force CORRECT if available during hot start (doesn't count toward prove budget)
        if ActionType.CORRECT in action_by_type:
            return action_by_type[ActionType.CORRECT]

        # 4. Force prove attempts per hot_start spec
        for model, n in self.hot_start.items():
            if state.prove_attempts_used.get(model, 0) < n:
                key = ("prove", model)
                if key in action_by_type:
                    return action_by_type[key]

        return None

    def _is_action_allowed(self, action: Action) -> bool:
        """Check if action passes the allowed_actions filter."""
        if self._allowed_actions is None:
            return True
        if action.type == ActionType.TERMINATE:
            return True
        if action.type == ActionType.PROVE:
            return f"prove_{action.model}" in self._allowed_actions
        if action.type == ActionType.CORRECT:
            return "correct" in self._allowed_actions
        if action.type in (ActionType.DECOMPOSE, ActionType.CREATE_BREAKDOWN):
            return "decompose" in self._allowed_actions
        return True

    def choose_action(self, state: SimulationState, valid_actions: List[Action], tracked_state: dict = None) -> Action:
        # Filter actions based on allowed_actions config
        filtered = [a for a in valid_actions if self._is_action_allowed(a)]
        if not filtered:
            filtered = [a for a in valid_actions if a.type == ActionType.TERMINATE]

        if len(filtered) == 1:
            self._decision_metadata = None
            return filtered[0]

        problem = self._problems.get(state.problem_id)
        if not problem:
            self._decision_metadata = None
            return filtered[0]

        # --- Hot start: force initial sequence before scoring ---
        if self.hot_start is not None:
            hot_action = self._check_hot_start(state, filtered)
            if hot_action is not None:
                oracle_rates = problem.get_empirical_success_rates()
                self._decision_metadata = {
                    "hot_start": True,
                    "action": hot_action.to_dict(),
                    "oracle_p": oracle_rates.get(hot_action, 0.0),
                }
                return hot_action

        # --- Partition actions into three tiers ---
        real_actions: List[Action] = []
        decompose_action: Optional[Action] = None
        terminate_action: Optional[Action] = None

        for action in filtered:
            if action.type == ActionType.TERMINATE:
                terminate_action = action
            elif action.type in (ActionType.DECOMPOSE, ActionType.CREATE_BREAKDOWN):
                decompose_action = action
            else:
                real_actions.append(action)

        # --- Score real actions ---
        best_real_action: Optional[Action] = None
        best_real_score = float('-inf')
        action_scores = []
        oracle_rates = problem.get_empirical_success_rates()

        for action in real_actions:
            p = max(min(self.prob_model.predict(state, action, problem, tracked_state=tracked_state), 1.0), 0.0)
            c = self.cost_model.predict(state, action, problem)
            score = p - self.lambda_val * c

            # Energy E = c / -log(1-p) for diagnostics
            if p >= 1.0 or p <= 0.0:
                energy = 0.0
            else:
                energy = c / -math.log(1.0 - p)

            action_scores.append({
                "action": action.to_dict(),
                "oracle_p": oracle_rates.get(action, 0.0),
                "p": p,
                "c": c,
                "score": score,
                "E": energy,
            })

            if score > best_real_score:
                best_real_score = score
                best_real_action = action

        # Record DECOMPOSE and TERMINATE in metadata (no p/c/E)
        if decompose_action is not None:
            action_scores.append({
                "action": decompose_action.to_dict(),
                "p": None, "c": None, "score": 0.0, "E": None,
            })
        if terminate_action is not None:
            action_scores.append({
                "action": terminate_action.to_dict(),
                "p": None, "c": None, "score": None, "E": None,
            })

        # --- Three-tier selection ---
        # Real action wins if score >= 0 (ties at 0 beat DECOMPOSE)
        if best_real_action is not None and best_real_score >= 0:
            chosen = best_real_action
            chosen_score = best_real_score
        elif decompose_action is not None:
            chosen = decompose_action
            chosen_score = 0.0
        elif terminate_action is not None:
            chosen = terminate_action
            chosen_score = None
        else:
            # All real actions negative and no DECOMPOSE/TERMINATE — pick best real
            chosen = best_real_action or filtered[0]
            chosen_score = best_real_score

        self._decision_metadata = {
            "lambda": self.lambda_val,
            "action_scores": action_scores,
            "chosen_score": chosen_score,
        }

        return chosen

    def get_decision_metadata(self) -> Optional[dict]:
        return self._decision_metadata
