"""Normalized Lean code similarity.

Before computing pairwise similarity, normalizes Lean code by replacing
variable names, hypothesis names, and theorem/lemma references with
canonical placeholders. This way two proofs using the same strategy but
different naming are recognized as similar.
"""

import random
import re
from multiprocessing import Pool, cpu_count

import pandas as pd
from rapidfuzz import fuzz

from .base import BaseFeature

TACTIC_PATTERN = re.compile(
    r":=\s*by\b(.*?)(?=\n(?:theorem|lemma|def|axiom|noncomputable|section|end|#)|$)",
    re.DOTALL,
)

# Matches hypothesis-style names: h, h₁, h₂₃, ih, hx, hxy, h_imp, h_pos, ...
_HYPS = re.compile(r"\b(h[₀₁₂₃₄₅₆₇₈₉]*(?:_\w+)?|ih\w*)\b")

# Matches theorem/lemma/def declarations and references with problem-id-style names
# e.g. putnam_1990_a5_lemma2_f0, mathd_algebra_209_lemma1_f0
_DECL_NAMES = re.compile(r"\b(\w+(?:_lemma\d+|_theorem)(?:_f\d+)?)\b")

# Matches `let X := ...` or `let x := ...` single-letter bindings
_LET_BINDINGS = re.compile(r"\blet\s+([A-Za-z_]\w*)\b")

# Matches `intro x y z` or `intro h₁ h₂` — captures the identifiers after intro/intros
_INTRO_ARGS = re.compile(r"\b(?:intro|intros)\s+((?:[A-Za-z_][A-Za-z₀₁₂₃₄₅₆₇₈₉_\w]*\s*)+)")

# Matches `have name :` or `have name :`
_HAVE_NAMES = re.compile(r"\b(?:have|let)\s+([A-Za-z_][A-Za-z₀₁₂₃₄₅₆₇₈₉_\w]*)\s*(?::|:=)")

# Single-line comments
_COMMENTS = re.compile(r"--.*$", re.MULTILINE)

# Consecutive whitespace
_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n\s*\n")


def _normalize_lean(code: str) -> str:
    """Normalize Lean code by replacing names with canonical placeholders."""
    # Strip comments
    code = _COMMENTS.sub("", code)

    # Collect names introduced by have/let so we can replace them consistently
    local_names: dict[str, str] = {}
    counter = [0]

    def _get_placeholder(name: str, prefix: str = "V") -> str:
        if name not in local_names:
            local_names[name] = f"{prefix}{counter[0]}"
            counter[0] += 1
        return local_names[name]

    # Replace declaration names (problem-specific theorem/lemma refs)
    def _repl_decl(m: re.Match) -> str:
        return _get_placeholder(m.group(1), "DECL")

    code = _DECL_NAMES.sub(_repl_decl, code)

    # Collect names from have/let bindings
    for m in _HAVE_NAMES.finditer(code):
        _get_placeholder(m.group(1), "L")

    # Collect names from intro
    for m in _INTRO_ARGS.finditer(code):
        for name in m.group(1).split():
            name = name.strip()
            if name:
                _get_placeholder(name, "I")

    # Collect let binding names
    for m in _LET_BINDINGS.finditer(code):
        _get_placeholder(m.group(1), "L")

    # Replace all collected local names (longest first to avoid partial matches)
    for name in sorted(local_names, key=len, reverse=True):
        code = re.sub(r"\b" + re.escape(name) + r"\b", local_names[name], code)

    # Replace remaining hypothesis-style names not yet captured
    code = _HYPS.sub("H", code)

    # Normalize whitespace
    code = _WHITESPACE.sub(" ", code)
    code = _BLANK_LINES.sub("\n", code)

    return code.strip()


def _extract_tactics(code: str) -> str:
    blocks = TACTIC_PATTERN.findall(code)
    return "\n".join(block.strip() for block in blocks)


def _compute_normalized_similarity(args: tuple) -> tuple[str, float]:
    """Worker: compute average pairwise similarity on normalized code."""
    problem_id, codes, n_pairs, seed, use_tactics = args
    rng = random.Random(seed)

    items = []
    for c in codes:
        text = _extract_tactics(c) if use_tactics else c
        if text:
            items.append(_normalize_lean(text))

    if len(items) < 2:
        return problem_id, 1.0

    pairs = set()
    max_possible = len(items) * (len(items) - 1) // 2
    n_pairs = min(n_pairs, max_possible)

    while len(pairs) < n_pairs:
        i, j = rng.sample(range(len(items)), 2)
        pairs.add((min(i, j), max(i, j)))

    total = sum(fuzz.ratio(items[i], items[j]) for i, j in pairs)
    return problem_id, total / (len(pairs) * 100)


class NormalizedLeanSimilarity(BaseFeature):
    """Pairwise similarity of Lean code after normalizing identifiers.

    Replaces variable names, hypothesis names, and lemma references with
    canonical placeholders before computing fuzzy string similarity.
    """

    def __init__(self, num_pairs: int = 32, seed: int = 42, use_tactics: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.num_pairs = num_pairs
        self.seed = seed
        self.use_tactics = use_tactics
        self.name = "normalized_similarity"

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
            (pid, codes, self.num_pairs, rng.randint(0, 2**32), self.use_tactics)
            for pid, codes in sorted(codes_by_problem.items())
        ]

        n_workers = min(cpu_count(), max(1, len(work_items)))
        with Pool(n_workers) as pool:
            results = pool.map(_compute_normalized_similarity, work_items)

        rows = [{"problem_id": pid, self.name: sim} for pid, sim in results]
        return pd.DataFrame(rows).set_index("problem_id")
