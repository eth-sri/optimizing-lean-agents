"""Normalized similarity: average pairwise similarity across a target's
attempts, normalizing identifiers before comparing."""

import random
from typing import Dict, List, Optional

from rapidfuzz import fuzz

from feature_engineering.features.lean_similarity import _extract_tactics
from feature_engineering.features.lean_similarity_normalized import _normalize_lean

from ..actions import Action, ActionResult, ActionType
from ..problem import SimulatedProblem
from ..state import SimulationState
from .base import StateFeature


class NormalizedSimilarityFeature(StateFeature):
    """Average pairwise similarity after normalizing Lean identifiers."""

    def __init__(self, seed: int = 42, max_pairs: Optional[int] = None, **kwargs):
        self.max_pairs = max_pairs
        self._items: Dict[str, List[str]] = {}
        self._pairwise_sims: Dict[str, List[float]] = {}
        self._last_batch_size: Dict[str, int] = {}
        self._rng = random.Random(seed)

    def name(self) -> str:
        return "normalized_similarity"

    def seed(self, seed: int):
        self._rng = random.Random(seed)

    def reset(self):
        self._items.clear()
        self._pairwise_sims.clear()
        self._last_batch_size.clear()

    def observe(self, state: SimulationState, action: Action, result: ActionResult, problem: SimulatedProblem):
        if action.type not in (ActionType.PROVE, ActionType.CORRECT):
            return
        if result.code is None:
            return

        tactics = _extract_tactics(result.code)
        if not tactics:
            return
        item = _normalize_lean(tactics)
        if not item:
            return

        tid = state.target_id

        if action.type == ActionType.CORRECT and tid in self._items and self._items[tid]:
            self._items[tid].pop()
            last_batch = self._last_batch_size.get(tid, 0)
            if last_batch > 0:
                self._pairwise_sims[tid] = self._pairwise_sims[tid][:-last_batch]

        prev_items = self._items.get(tid, [])
        if prev_items:
            new_sims = [fuzz.ratio(item, prev) for prev in prev_items]
            self._pairwise_sims.setdefault(tid, []).extend(new_sims)
            self._last_batch_size[tid] = len(new_sims)
        else:
            self._last_batch_size[tid] = 0

        self._items.setdefault(tid, []).append(item)

    def compute(self, state: SimulationState, problem: SimulatedProblem) -> float:
        tid = state.target_id
        all_sims = self._pairwise_sims.get(tid)
        if not all_sims:
            return 1.0
        if self.max_pairs is None or len(all_sims) <= self.max_pairs:
            selected = all_sims
        else:
            selected = self._rng.sample(all_sims, self.max_pairs)
        return sum(selected) / (len(selected) * 100.0)
