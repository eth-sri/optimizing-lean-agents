"""Action types, actions, costs, and results for the proof simulation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set, Union


class ActionType(Enum):
    PROVE = "prove"
    CORRECT = "correct"
    DECOMPOSE = "decompose"
    CREATE_BREAKDOWN = "create_breakdown"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class Action:
    """An action to take in the simulation.

    Only PROVE is parameterized by model. CORRECT uses whatever model
    the correction data was generated with.
    """
    type: ActionType
    model: Optional[str] = None  # model name, only set for PROVE

    def __repr__(self) -> str:
        if self.model:
            return f"Action({self.type.value}, {self.model})"
        return f"Action({self.type.value})"

    def to_dict(self) -> dict:
        d = {"type": self.type.value}
        if self.model is not None:
            d["model"] = self.model
        return d

    @staticmethod
    def from_dict(d: dict) -> 'Action':
        return Action(
            type=ActionType(d["type"]),
            model=d.get("model"),
        )


@dataclass
class DetailedCost:
    """Tracks both SFLOPs and tokens for cost accounting."""
    input_sflops: int = 0
    output_sflops: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_sflops(self) -> int:
        return self.input_sflops + self.output_sflops

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: 'DetailedCost') -> 'DetailedCost':
        return DetailedCost(
            input_sflops=self.input_sflops + other.input_sflops,
            output_sflops=self.output_sflops + other.output_sflops,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def __iadd__(self, other: 'DetailedCost') -> 'DetailedCost':
        self.input_sflops += other.input_sflops
        self.output_sflops += other.output_sflops
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self

    def to_dict(self) -> dict:
        return {
            "input_sflops": self.input_sflops,
            "output_sflops": self.output_sflops,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @staticmethod
    def from_dict(d: dict) -> 'DetailedCost':
        return DetailedCost(
            input_sflops=d.get("input_sflops", 0),
            output_sflops=d.get("output_sflops", 0),
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
        )


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool  # pass AND complete
    cost: DetailedCost
    proof_length: Optional[int] = None
    num_errors: Optional[int] = None
    used_lemma_ids: Optional[Set[int]] = None
    attempt_id: Optional[Union[int, str]] = None  # original attempt_id/proof_id for tracing
    code: Optional[str] = None  # proof code (not serialized to avoid bloating trajectories)
    error_messages: Optional[list] = None  # error message strings for diversity tracking

    def to_dict(self) -> dict:
        d = {
            "success": self.success,
            "cost": self.cost.to_dict(),
        }
        if self.proof_length is not None:
            d["proof_length"] = self.proof_length
        if self.num_errors is not None:
            d["num_errors"] = self.num_errors
        if self.used_lemma_ids is not None:
            d["used_lemma_ids"] = sorted(self.used_lemma_ids)
        if self.attempt_id is not None:
            d["attempt_id"] = self.attempt_id
        return d

    @staticmethod
    def from_dict(d: dict) -> 'ActionResult':
        return ActionResult(
            success=d["success"],
            cost=DetailedCost.from_dict(d["cost"]),
            proof_length=d.get("proof_length"),
            num_errors=d.get("num_errors"),
            used_lemma_ids=set(d["used_lemma_ids"]) if d.get("used_lemma_ids") is not None else None,
            attempt_id=d.get("attempt_id"),
        )
