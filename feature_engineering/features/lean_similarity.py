import random
import re
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .base import BaseFeature

METHODS = {
    "ratio": fuzz.ratio,
}


TACTIC_PATTERN = re.compile(
    r":=\s*by\b(.*?)(?=\n(?:theorem|lemma|def|axiom|noncomputable|section|end|#)|$)",
    re.DOTALL,
)



def _extract_tactics(code: str) -> str:
    """Extract tactic blocks (everything after ':= by') from Lean code."""
    blocks = TACTIC_PATTERN.findall(code)
    return "\n".join(block.strip() for block in blocks)



def _compute_similarity(args: tuple) -> tuple[str, float, float]:
    """Worker: compute average pairwise similarity for one problem."""
    problem_id, codes, n_pairs, seed, method_name, use_tactics = args
    rng = random.Random(seed)

    if use_tactics:
        items = [_extract_tactics(c) for c in codes]
        items = [t for t in items if t]
    else:
        items = codes

    if len(items) < 2:
        return problem_id, 1.0, 0.0

    pairs = set()
    max_possible = len(items) * (len(items) - 1) // 2
    n_pairs = min(n_pairs, max_possible)

    while len(pairs) < n_pairs:
        i, j = rng.sample(range(len(items)), 2)
        pairs.add((min(i, j), max(i, j)))

    sim_func = METHODS[method_name]
    sims = [sim_func(items[i], items[j]) / 100 for i, j in pairs]
    return problem_id, sum(sims) / len(sims), float(np.std(sims))


class LeanSimilarity(BaseFeature):
    """Average pairwise Lean code similarity per problem.

    Uses the last correction round's full_code for each attempt.
    """

    def __init__(self, method: str = "ratio", num_pairs: int = 32, seed: int = 42, use_tactics: bool = True, **kwargs):
        super().__init__(**kwargs)
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(METHODS.keys())}")
        self.method = method
        self.name = "lean_similarity"
        self.num_pairs = num_pairs
        self.seed = seed
        self.use_tactics = use_tactics

    def compute(self, data: dict) -> pd.DataFrame:
        codes_by_problem: dict[str, list[str]] = {}
        for pid in sorted(data.keys()):
            codes = []
            for chain in data[pid]:
                code = chain["rounds"][-1]["full_code"]
                if code:
                    codes.append(code)
            if codes:
                codes_by_problem[pid] = codes

        rng = random.Random(self.seed)
        work_items = [
            (pid, codes, self.num_pairs, rng.randint(0, 2**32), self.method, self.use_tactics)
            for pid, codes in sorted(codes_by_problem.items())
        ]

        n_workers = min(cpu_count(), max(1, len(work_items)))
        with Pool(n_workers) as pool:
            results = pool.map(_compute_similarity, work_items)

        rows = [{"problem_id": pid, self.name: sim, f"{self.name}_std": std} for pid, sim, std in results]
        return pd.DataFrame(rows).set_index("problem_id")


