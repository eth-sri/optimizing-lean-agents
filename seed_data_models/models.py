"""
Data models for the Analysis GUI.

Provides OOP structure for organizing proof data:
Session → Problem → Breakdown → ParsedBreakdown → Theorem/Lemmas → ProofAttempts

This module re-exports all data classes from their individual files for backwards compatibility.
The actual implementations are in separate modules:
- exceptions_and_helpers: SimulationFailure, ProverPathNode
- compilation: CompilationResult
- proof_attempt: ProofAttempt
- formalization: Formalization
- theorem: Theorem
- lemma: Lemma
- breakdown_models: ParsedBreakdown, Breakdown
- problem: Problem
- session: Session
"""

# Import and re-export all classes
from .exceptions_and_helpers import (
    SimulationFailure,
    ProverPathNode,
)
from .compilation import CompilationResult
from .proof_attempt import ProofAttempt
from .formalization import Formalization
from .theorem import Theorem
from .lemma import Lemma
from .breakdown_models import ParsedBreakdown, Breakdown
from .problem import Problem
from .session import Session

# Public API - all classes are available from this module
__all__ = [
    'SimulationFailure',
    'ProverPathNode',
    'get_attempt_ranges',
    'CompilationResult',
    'ProofAttempt',
    'Formalization',
    'Theorem',
    'Lemma',
    'ParsedBreakdown',
    'Breakdown',
    'Problem',
    'Session',
]
