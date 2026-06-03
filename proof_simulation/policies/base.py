"""Base policy interface."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..actions import Action
from ..state import SimulationState


class Policy(ABC):
    """Abstract base class for proof search policies."""

    @abstractmethod
    def choose_action(self, state: SimulationState, valid_actions: List[Action], tracked_state: dict = None) -> Action:
        """Choose an action given state and valid actions.

        Args:
            state: Current simulation state snapshot
            valid_actions: List of valid actions to choose from
            tracked_state: Optional dict of tracked features from StateTracker

        Returns:
            The chosen action (must be from valid_actions)
        """
        pass

    def get_decision_metadata(self) -> Optional[dict]:
        """Return metadata from the most recent choose_action() call.

        Override in subclasses to provide decision reasoning (scores, probabilities, etc.).
        Returns None by default.
        """
        return None
