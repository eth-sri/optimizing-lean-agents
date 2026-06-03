"""SimulationState: read-only snapshot for policies."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .actions import Action, ActionResult, DetailedCost


@dataclass
class SimulationState:
    """Read-only snapshot of simulation state for policy decisions.

    Uniform regardless of depth — the policy doesn't need to know
    what level it's at.
    """
    problem_id: str
    total_cost: DetailedCost
    decomposition_depth: int  # 0=top level (not decomposed), 1+=inside breakdown
    target_type: str  # "problem", "theorem", "lemma"
    target_id: str
    target_metadata: Dict = field(default_factory=dict)  # {breakdown_id, lemma_id}; -1 at root

    # Direct proving (when not decomposed)
    prove_attempts_used: Dict[str, int] = field(default_factory=dict)
    prove_attempts_available: Dict[str, int] = field(default_factory=dict)
    last_prove_model: Optional[str] = None
    last_prove_success: Optional[bool] = None

    # Correction tracking
    corrections_used: int = 0
    corrections_available: int = 0

    # Breakdown tracking (when decomposed)
    breakdowns_created: int = 0
    breakdowns_available: int = 0
    current_breakdown_targets_proven: int = 0
    current_breakdown_targets_remaining: int = 0

    # Last action info
    last_action: Optional[Action] = None
    last_result: Optional[ActionResult] = None

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "total_cost": self.total_cost.to_dict(),
            "decomposition_depth": self.decomposition_depth,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "target_metadata": self.target_metadata,
            "prove_attempts_used": dict(self.prove_attempts_used),
            "prove_attempts_available": dict(self.prove_attempts_available),
            "last_prove_model": self.last_prove_model,
            "last_prove_success": self.last_prove_success,
            "corrections_used": self.corrections_used,
            "corrections_available": self.corrections_available,
            "breakdowns_created": self.breakdowns_created,
            "breakdowns_available": self.breakdowns_available,
            "current_breakdown_targets_proven": self.current_breakdown_targets_proven,
            "current_breakdown_targets_remaining": self.current_breakdown_targets_remaining,
        }

    @staticmethod
    def from_dict(d: dict) -> 'SimulationState':
        return SimulationState(
            problem_id=d["problem_id"],
            total_cost=DetailedCost.from_dict(d["total_cost"]),
            decomposition_depth=d["decomposition_depth"],
            target_type=d["target_type"],
            target_id=d["target_id"],
            target_metadata=d.get("target_metadata", {}),
            prove_attempts_used=d.get("prove_attempts_used", {}),
            prove_attempts_available=d.get("prove_attempts_available", {}),
            last_prove_model=d.get("last_prove_model"),
            last_prove_success=d.get("last_prove_success"),
            corrections_used=d.get("corrections_used", 0),
            corrections_available=d.get("corrections_available", 0),
            breakdowns_created=d.get("breakdowns_created", 0),
            breakdowns_available=d.get("breakdowns_available", 0),
            current_breakdown_targets_proven=d.get("current_breakdown_targets_proven", 0),
            current_breakdown_targets_remaining=d.get("current_breakdown_targets_remaining", 0),
        )
