"""
Seed data models and loader.

This package provides the OOP data model hierarchy for organizing proof data:
Session → Problem → Breakdown → ParsedBreakdown → Theorem/Lemmas → ProofAttempts

And the DataLoader for populating these models from filesystem results.
"""

from .models import (
    CompilationResult,
    ProverPathNode,
    ProofAttempt,
    Formalization,
    Theorem,
    Lemma,
    ParsedBreakdown,
    Breakdown,
    Problem,
    Session,
)
from .model_loader import DataLoader

__all__ = [
    "CompilationResult",
    "ProverPathNode",
    "ProofAttempt",
    "Formalization",
    "Theorem",
    "Lemma",
    "ParsedBreakdown",
    "Breakdown",
    "Problem",
    "Session",
    "DataLoader",
]
