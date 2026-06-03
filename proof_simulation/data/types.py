"""Internal data types used during loading and by TargetNode."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..actions import DetailedCost


@dataclass
class AttemptData:
    """A single proof attempt's outcome data (no full code needed)."""
    success: bool  # pass AND complete
    cost: DetailedCost
    proof_length: Optional[int] = None
    num_errors: Optional[int] = None
    used_lemma_ids: Optional[Set[int]] = None
    correction_round_id: int = 0
    attempt_id: Optional[int] = None  # original attempt_id (agent) or proof_id (full proof)
    code: Optional[str] = None
    error_messages: Optional[List[str]] = None


@dataclass
class AttemptPair:
    """An initial proof attempt paired with its sequential corrections."""
    initial: AttemptData
    corrections: List[AttemptData] = field(default_factory=list)


@dataclass
class BreakdownTemplate:
    """Template for a breakdown: the cost to create it and per-target proof data.

    target_proof_data maps target_id (-1=theorem, 0+=lemma) to
    {model_name: [AttemptPair]}.
    """
    breakdown_idx: int
    cost: DetailedCost
    target_proof_data: Dict[int, Dict[str, List[AttemptPair]]] = field(default_factory=dict)
    breakdown_key: Optional[Tuple[int, int]] = None  # (round_id, breakdown_id) for cross-source matching
