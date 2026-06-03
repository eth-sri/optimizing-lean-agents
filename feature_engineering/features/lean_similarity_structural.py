"""Structural fingerprint similarity: compares the nesting structure of proofs.

Encodes each proof as a fingerprint capturing the shape of have/let nesting
and the top-level tactics used at each level, ignoring arguments and details.

Example:
    have h1 : ... := by        →  have(simp,linarith)
      simp [h₁]
      linarith
    have h2 : ... := by        →  have(have(omega),exact)
      have h3 : ... := by
        omega
      exact h3
    exact h1.trans h2           →  exact

    Fingerprint: "have(simp,linarith);have(have(omega),exact);exact"
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

# Match have/let lines
_HAVE_BY = re.compile(r"^(\s*)(?:have|let)\s+.*:=\s*by\s*$")

# Extract leading tactic name from a line (first word that looks like a tactic)
_TACTIC_NAME = re.compile(r"^\s*(?:<;>\s*)?(?:\(try\s+)?(\w+)")

# Tactics to ignore (pure control flow / noise)
_IGNORE_TACTICS = {"try", "do", "first"}


def _extract_tactic_name(line: str) -> str | None:
    """Extract the leading tactic name from a line, ignoring combinators."""
    stripped = line.strip()
    if not stripped:
        return None
    # Strip leading <;>, (try, {, }
    cleaned = re.sub(r"^(?:<;>\s*)*(?:\(try\s+)?(?:\{\s*)?", "", stripped)
    m = _TACTIC_NAME.match(cleaned)
    if m:
        name = m.group(1)
        if name in _IGNORE_TACTICS:
            return None
        return name
    return None


def _build_fingerprint(tactic_block: str) -> str:
    """Build a structural fingerprint from a tactic block.

    Recursively encodes have/let blocks as `have(...)` with their
    child tactics, and keeps top-level tactic names.
    """
    lines = tactic_block.split("\n")
    if not lines:
        return ""

    def _indent(line: str) -> int:
        return len(line) - len(line.lstrip())

    # Parse into a flat list of (indent, is_have, tactic_name)
    entries: list[tuple[int, bool, str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped in ("{", "}"):
            continue
        ind = _indent(line)
        if _HAVE_BY.match(line):
            entries.append((ind, True, "have"))
        else:
            tac = _extract_tactic_name(line)
            if tac:
                entries.append((ind, False, tac))

    if not entries:
        return ""

    # Build tree structure recursively
    def _build(idx: int, parent_indent: int) -> tuple[list[str], int]:
        parts = []
        while idx < len(entries):
            ind, is_have, name = entries[idx]
            if ind <= parent_indent:
                break
            if is_have:
                # Recurse into children
                children, idx = _build(idx + 1, ind)
                if children:
                    parts.append(f"have({','.join(children)})")
                else:
                    parts.append("have()")
            else:
                parts.append(name)
                idx += 1
        return parts, idx

    # Find the base indentation
    base_indent = entries[0][0] - 1
    parts, _ = _build(0, base_indent)
    return ";".join(parts)


def _extract_fingerprint(code: str) -> str:
    """Extract structural fingerprints from all tactic blocks in code."""
    blocks = TACTIC_PATTERN.findall(code)
    fingerprints = [_build_fingerprint(block) for block in blocks]
    return " | ".join(f for f in fingerprints if f)


def _compute_structural_similarity(args: tuple) -> tuple[str, float]:
    """Worker: compute average pairwise similarity of structural fingerprints."""
    problem_id, codes, n_pairs, seed = args
    rng = random.Random(seed)

    items = [_extract_fingerprint(c) for c in codes]
    items = [f for f in items if f]

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


class StructuralLeanSimilarity(BaseFeature):
    """Pairwise similarity of proof structural fingerprints."""

    def __init__(self, num_pairs: int = 32, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.num_pairs = num_pairs
        self.seed = seed
        self.name = "structural_similarity"

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
            (pid, codes, self.num_pairs, rng.randint(0, 2**32))
            for pid, codes in sorted(codes_by_problem.items())
        ]

        n_workers = min(cpu_count(), max(1, len(work_items)))
        with Pool(n_workers) as pool:
            results = pool.map(_compute_structural_similarity, work_items)

        rows = [{"problem_id": pid, self.name: sim} for pid, sim in results]
        return pd.DataFrame(rows).set_index("problem_id")
