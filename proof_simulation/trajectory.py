"""Trajectory recording for simulation runs."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .actions import Action, ActionResult, DetailedCost
from .state import SimulationState


@dataclass
class TrajectoryStep:
    """A single step in a simulation trajectory."""
    state: SimulationState
    action: Action
    result: ActionResult
    decision_metadata: Optional[dict] = None
    tracked_state: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "state": self.state.to_dict(),
            "action": self.action.to_dict(),
            "result": self.result.to_dict(),
        }
        if self.decision_metadata is not None:
            d["decision_metadata"] = self.decision_metadata
        if self.tracked_state is not None:
            d["tracked_state"] = self.tracked_state
        return d

    @staticmethod
    def from_dict(d: dict) -> 'TrajectoryStep':
        return TrajectoryStep(
            state=SimulationState.from_dict(d["state"]),
            action=Action.from_dict(d["action"]),
            result=ActionResult.from_dict(d["result"]),
            decision_metadata=d.get("decision_metadata"),
            tracked_state=d.get("tracked_state"),
        )


@dataclass
class Trajectory:
    """Complete trajectory for a single problem simulation."""
    problem_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    solved: bool = False
    total_cost: DetailedCost = field(default_factory=DetailedCost)
    seed: Optional[int] = None

    def add_step(self, state: SimulationState, action: Action, result: ActionResult, decision_metadata: Optional[dict] = None, tracked_state: Optional[dict] = None):
        self.steps.append(TrajectoryStep(state=state, action=action, result=result, decision_metadata=decision_metadata, tracked_state=tracked_state))

    def to_dict(self) -> dict:
        """Full serialization of the trajectory."""
        return {
            "problem_id": self.problem_id,
            "solved": self.solved,
            "seed": self.seed,
            "total_cost": self.total_cost.to_dict(),
            "num_steps": len(self.steps),
            "steps": [
                {"step": i, **step.to_dict()}
                for i, step in enumerate(self.steps)
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> 'Trajectory':
        """Deserialize a trajectory from a dict."""
        traj = Trajectory(
            problem_id=d["problem_id"],
            solved=d["solved"],
            total_cost=DetailedCost.from_dict(d["total_cost"]),
            seed=d.get("seed"),
        )
        for step_d in d["steps"]:
            traj.steps.append(TrajectoryStep.from_dict(step_d))
        return traj
