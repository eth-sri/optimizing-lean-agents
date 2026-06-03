"""Base classes for composable state features and the StateTracker compositor."""

from abc import ABC, abstractmethod
from typing import Any, List

from ..actions import Action, ActionResult, ActionType
from ..problem import SimulatedProblem
from ..state import SimulationState


class StateFeature(ABC):
    """Base class for composable state features."""

    @abstractmethod
    def name(self) -> str:
        """Key used in the output dict."""
        ...

    def seed(self, seed: int):
        """Called once per seed. Override for stochastic features."""
        pass

    def reset(self):
        """Called per problem. Override to clear per-problem state."""
        pass

    @abstractmethod
    def compute(self, state: SimulationState, problem: SimulatedProblem) -> Any:
        """Compute feature value before the action is taken."""
        ...

    def observe(self, state: SimulationState, action: Action, result: ActionResult, problem: SimulatedProblem):
        """Called after action execution. Override for online features."""
        pass


class ComputedFeature(ABC):
    """Base class for computed values that consume the feature state dict.

    Unlike StateFeature, compute() receives the already-populated feature dict
    from phase 1 as its first argument.
    """

    @abstractmethod
    def name(self) -> str:
        """Key used in the output dict."""
        ...

    def seed(self, seed: int):
        """Called once per seed. Override for stochastic computed features."""
        pass

    def reset(self):
        """Called per problem. Override to clear per-problem state."""
        pass

    @abstractmethod
    def compute(self, tracked_state: dict, state: SimulationState, problem: SimulatedProblem) -> Any:
        """Compute value from the feature state dict."""
        ...

    def observe(self, state: SimulationState, action: Action, result: ActionResult, problem: SimulatedProblem):
        """Called after action execution. Override for online computed features."""
        pass


class StateTracker:
    """Composes multiple StateFeature and ComputedFeature instances.

    Two-phase get_tracked_state:
      Phase 1: raw features (StateFeature.compute)
      Phase 2: computed values that read from phase 1 (ComputedFeature.compute)
    """

    def __init__(self, features: List[StateFeature], computed: List[ComputedFeature] = None):
        self.features = features
        self.computed = computed or []

    def seed(self, seed: int):
        for f in self.features:
            f.seed(seed)
        for c in self.computed:
            c.seed(seed)

    def reset(self):
        for f in self.features:
            f.reset()
        for c in self.computed:
            c.reset()

    def get_tracked_state(self, state: SimulationState, problem: SimulatedProblem) -> dict:
        result = {}
        # Phase 1: raw features
        for f in self.features:
            result[f.name()] = f.compute(state, problem)
        # Phase 2: computed values that read from phase 1
        # Disambiguate duplicate names (e.g. two predicted_prob → predicted_prob, predicted_prob2)
        seen_names: dict[str, int] = {}
        for c in self.computed:
            base = c.name()
            count = seen_names.get(base, 0)
            seen_names[base] = count + 1
            key = base if count == 0 else f"{base}{count + 1}"
            result[key] = c.compute(result, state, problem)
        return result

    def observe(self, state: SimulationState, action: Action, result: ActionResult, problem: SimulatedProblem):
        for f in self.features:
            f.observe(state, action, result, problem)
        for c in self.computed:
            c.observe(state, action, result, problem)
