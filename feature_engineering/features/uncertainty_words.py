"""Uncertainty words feature: normalized frequency of uncertainty signals
in the model's reasoning trace (excluding code blocks).

Three categories:
1. Simple hedging words: but, however, wait
2. Skipping/giving-up patterns: "let's assume ... and proceed", "use sorry", etc.
3. Blaming the problem: "not correctly formulated", "statement is false", etc.

Higher ratio = model is more uncertain / giving up more during reasoning.
"""

import re
from multiprocessing import Pool, cpu_count

import pandas as pd

from .base import BaseFeature

# --- Simple hedging words ---
HEDGE_WORDS = {"but", "however", "wait"}

# --- Category 2: Skipping / giving up / assuming without proof ---
SKIP_PATTERNS = [
    # "let's assume X and proceed/move on/go on"
    re.compile(r"let'?s\s+assume\b.*?\b(?:proceed|move\s+on|go\s+on|hold)", re.IGNORECASE),
    # "let's just proceed/skip/move on"
    re.compile(r"let'?s\s+(?:just\s+)?(?:proceed|skip|move\s+on)", re.IGNORECASE),
    # "to move forward"
    re.compile(r"to\s+move\s+forward", re.IGNORECASE),
    # "given time constraints"
    re.compile(r"given\s+(?:time\s+)?constraints", re.IGNORECASE),
    # sorry-related: "use sorry", "leave as sorry", "skip with sorry"
    re.compile(r"(?:use|leave\b.*?as|skip\b.*?with|proceed\s+with)\s+[`]?sorry[`]?", re.IGNORECASE),
    # "we cannot provide a proof"
    re.compile(r"(?:we\s+)?cannot\s+provide\s+a\s+proof", re.IGNORECASE),
    # "we must leave the proof as sorry"
    re.compile(r"must\s+leave\s+(?:the\s+)?proof\s+as\s+[`]?sorry", re.IGNORECASE),
]

# --- Category 3: Blaming the problem / formalization ---
BLAME_PATTERNS = [
    # "not correctly formulated/formalized"
    re.compile(r"not\s+correctly\s+(?:formulated|formalized)", re.IGNORECASE),
    # "problem is incorrectly stated"
    re.compile(r"(?:problem|theorem|statement)\s+(?:is\s+)?(?:incorrectly|not\s+correctly)\s+(?:stated|formulated|formalized)", re.IGNORECASE),
    # "problem statement might be wrong/incorrect"
    re.compile(r"(?:problem\s+)?statement\s+(?:might|may|could)\s+be\s+(?:wrong|incorrect)", re.IGNORECASE),
    # "the statement is false"
    re.compile(r"(?:the\s+)?statement\s+is\s+false", re.IGNORECASE),
    # "theorem is false"
    re.compile(r"(?:the\s+)?theorem\s+is\s+false", re.IGNORECASE),
    # "formalization issues" / "issue with the formalization"
    re.compile(r"(?:formalization|formulating)\s+issue", re.IGNORECASE),
    re.compile(r"issue\s+with\s+(?:the\s+)?(?:formalization|formulation)", re.IGNORECASE),
    # "Lean code/statement is not ... valid/correct"
    re.compile(r"lean\s+(?:code|statement|formalization)\s+(?:is\s+)?(?:not\s+)?(?:syntactically\s+)?(?:valid|correct|wrong|incorrect)", re.IGNORECASE),
    # "the statement claims ... but" (identifying discrepancy)
    re.compile(r"(?:the\s+)?(?:original\s+)?(?:statement|theorem)\s+(?:claims|assumes)\b.*?\bbut\b", re.IGNORECASE),
    # "too strong/broad" (overly general formalization)
    re.compile(r"(?:is\s+)?too\s+(?:strong|broad|general)", re.IGNORECASE),
    # "this is false when/for/unless"
    re.compile(r"this\s+is\s+false\s+(?:when|for|unless)", re.IGNORECASE),
    # "no such ... exists"
    re.compile(r"no\s+such\s+\S+\s+exists", re.IGNORECASE),
]

# Regex to strip code blocks (```...```)
_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_WORD_TOKENIZE = re.compile(r"[a-zA-Z']+")


def _strip_code_blocks(text: str) -> str:
    """Remove markdown code blocks from reasoning trace."""
    return _CODE_BLOCK.sub("", text)


def _count_uncertainty(text: str) -> tuple[int, int]:
    """Count uncertainty signals and total words in text.

    Returns (uncertainty_count, total_words).
    """
    text_lower = text.lower()
    words = _WORD_TOKENIZE.findall(text_lower)
    total_words = len(words)

    count = 0

    # Simple hedge words
    count += sum(1 for w in words if w in HEDGE_WORDS)

    # Skip/give-up patterns
    for pattern in SKIP_PATTERNS:
        count += len(pattern.findall(text))

    # Blame-the-problem patterns
    for pattern in BLAME_PATTERNS:
        count += len(pattern.findall(text))

    return count, total_words


def _compute_problem_uncertainty(args: tuple) -> tuple[str, float]:
    """Worker: compute uncertainty ratio for one problem."""
    pid, chains = args
    total_uncertainty = 0
    total_words = 0
    for chain in chains:
        for round_data in chain["rounds"]:
            output = round_data.get("output", "")
            if not output:
                continue
            reasoning = _strip_code_blocks(output)
            u, w = _count_uncertainty(reasoning)
            total_uncertainty += u
            total_words += w

    if total_words > 0:
        ratio = total_uncertainty / total_words
    else:
        ratio = float("nan")
    return pid, ratio


class UncertaintyWords(BaseFeature):
    """Average ratio of uncertainty signals to total words in reasoning traces."""

    name = "reasoning_uncertainty"

    def compute(self, data: dict) -> pd.DataFrame:
        work_items = [(pid, data[pid]) for pid in sorted(data.keys())]

        n_workers = min(cpu_count(), max(1, len(work_items)))
        with Pool(n_workers) as pool:
            results = pool.map(_compute_problem_uncertainty, work_items)

        rows = [{"problem_id": pid, self.name: ratio} for pid, ratio in results]
        return pd.DataFrame(rows).set_index("problem_id")
