"""Have-repetition feature: measures how often `have` statements in a proof
share the same type (subgoal).

Any two `have` lines with the same (or very similar) type within one proof
indicate redundancy — the model is restating a subgoal instead of making progress.

Computes the ratio of redundant `have` lines to total `have` lines,
averaged across all attempts per problem.
"""

import re
from multiprocessing import Pool, cpu_count

import pandas as pd
from rapidfuzz import fuzz

from .base import BaseFeature

# Match `have <name> : <type> := ...` and capture the type part
_HAVE_TYPE = re.compile(
    r"^\s*(?:have|let)\s+\S+\s*:\s*(.*?)\s*:=.*$"
)


def _extract_have_types(code: str) -> list[str]:
    """Extract normalized types from all have/let lines."""
    types = []
    for line in code.split("\n"):
        m = _HAVE_TYPE.match(line)
        if m:
            type_str = re.sub(r"\s+", " ", m.group(1)).strip()
            if type_str:
                types.append(type_str)
    return types


def _count_redundant_haves(code: str, threshold: float = 90.0) -> tuple[int, int]:
    """Count redundant have statements and total have statements.

    A have is redundant if its type is very similar (fuzz.ratio >= threshold)
    to any earlier have in the same proof.

    Returns (redundant_count, total_count).
    """
    types = _extract_have_types(code)
    if not types:
        return 0, 0

    total = len(types)
    redundant = 0
    seen: list[str] = []

    for t in types:
        for s in seen:
            if fuzz.ratio(t, s) >= threshold:
                redundant += 1
                break
        seen.append(t)

    return redundant, total


def _compute_problem_repetition(args: tuple) -> tuple[str, float]:
    """Worker: compute have-repetition ratio for one problem."""
    pid, chains, threshold = args
    total_redundant = 0
    total_haves = 0
    for chain in chains:
        code = chain["rounds"][-1]["full_code"]
        if not code:
            continue
        r, t = _count_redundant_haves(code, threshold)
        total_redundant += r
        total_haves += t

    if total_haves > 0:
        ratio = total_redundant / total_haves
    else:
        ratio = float("nan")
    return pid, ratio


class HaveRepetition(BaseFeature):
    """Average ratio of redundant have statements per problem.

    A have is redundant if any earlier have in the same proof has the same type.
    """

    name = "subgoal_repetition"

    def __init__(self, threshold: float = 90.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def compute(self, data: dict) -> pd.DataFrame:
        work_items = [(pid, data[pid], self.threshold) for pid in sorted(data.keys())]

        n_workers = min(cpu_count(), max(1, len(work_items)))
        with Pool(n_workers) as pool:
            results = pool.map(_compute_problem_repetition, work_items)

        rows = [{"problem_id": pid, self.name: ratio} for pid, ratio in results]
        return pd.DataFrame(rows).set_index("problem_id")
