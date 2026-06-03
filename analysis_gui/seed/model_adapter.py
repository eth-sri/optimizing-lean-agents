"""
Compatibility adapter to convert new OOP models to old data structures.

This allows existing UI components to work with the new model system
without immediate refactoring. Over time, components can be migrated
to use the new models directly.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add root directory to path to import seed_data_models
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from seed_data_models import Session, Problem, Breakdown, ParsedBreakdown, Theorem, Lemma, ProofAttempt


def convert_session_to_old_problems(session: Session) -> List[Dict[str, Any]]:
    """
    Convert a Session object to the old ProblemSummary-like structure.

    Returns a list of problem dicts compatible with old render functions.
    """
    problems = []
    for problem in session.problems.values():
        problem_dict = convert_problem_to_old_format(problem)
        problems.append(problem_dict)
    return problems


def convert_problem_to_old_format(problem: Problem) -> Dict[str, Any]:
    """
    Convert a Problem object to an old ProblemSummary-like dict.
    """
    breakdowns = []
    for breakdown in problem.breakdowns.values():
        breakdown_dict = convert_breakdown_to_old_format(breakdown)
        breakdowns.append(breakdown_dict)

    return {
        'origin_problem_id': problem.origin_problem_id,
        'name': problem.origin_problem_id,
        'num_breakdowns': len(problem.breakdowns),
        'breakdowns': breakdowns,
        'tags': [],  # Not available in new model
        'solved': problem.is_solved(),
        'analysis': None,  # Not available in new model
        '_problem_obj': problem  # Keep reference to original object
    }


def convert_breakdown_to_old_format(breakdown: Breakdown) -> Dict[str, Any]:
    """
    Convert a Breakdown object to an old Breakdown-like dict.
    """
    parsed_bd = breakdown.parsed_breakdown

    # Build theorem info
    theorem_info = None
    if parsed_bd and parsed_bd.theorem:
        theorem = parsed_bd.theorem
        best_attempt = theorem.get_best_attempt()
        theorem_info = {
            'statement': theorem.statement,
            'formal_statement': theorem.formal_statement,
            'proof_idea': theorem.proof_idea,
            'passed': best_attempt.is_passing() if best_attempt else False,
            'attempts': len(theorem.proof_attempts)
        }

    # Build lemma info
    lemmas_info = {}
    if parsed_bd:
        for lemma_id, lemma in parsed_bd.lemmas.items():
            best_attempt = lemma.get_best_attempt()
            lemmas_info[lemma_id] = {
                'lemma_id': lemma_id,
                'statement': lemma.statement,
                'formal_statement': lemma.formal_statement,
                'assumptions': lemma.assumptions,
                'passed': best_attempt.is_passing() if best_attempt else False,
                'attempts': len(lemma.proof_attempts)
            }

    # Build theorem_prover_results structure
    theorem_prover_results = None
    if parsed_bd and parsed_bd.theorem:
        theorem = parsed_bd.theorem
        if theorem.proof_attempts:
            theorem_prover_results = {
                'breakdown_id': breakdown.problem_id,
                'attempts': [
                    {
                        'data': _proof_attempt_to_dict(attempt),
                        'correction_round': attempt.correction_round_id
                    }
                    for attempt in theorem.proof_attempts
                ]
            }

    # Build lemma_prover_results structure
    lemma_prover_results = None
    if parsed_bd and parsed_bd.lemmas:
        lemma_prover_results = {
            'breakdown_id': breakdown.problem_id,
            'all_attempts': []
        }
        for lemma_id, lemma in parsed_bd.lemmas.items():
            for attempt in lemma.proof_attempts:
                lemma_prover_results['all_attempts'].append({
                    'data': _proof_attempt_to_dict(attempt),
                    'correction_round': attempt.correction_round_id
                })

    return {
        'problem_id': breakdown.problem_id,
        'origin_problem_id': breakdown.origin_problem_id,
        'name': breakdown.origin_problem_id,
        'informal_breakdown': breakdown.informal_breakdown,
        'parsed_breakdown': parsed_bd,
        'theorem_prover_results': theorem_prover_results,
        'lemma_prover_results': lemma_prover_results,
        'parsed_breakdown': None,  # TODO: Add this to models if needed
        'parse_failure': None,  # TODO: Add this to models if needed
        'theorem_full_records': None,  # TODO: Add this to models if needed
        'lemma_full_records': None,  # TODO: Add this to models if needed
        'formal_statement': None,
        'informal_prefix': None,
        'informal_solution': None,
        'tags': [],
        'detailed_cost': None,
        'lean4_code': None,
        '_breakdown_obj': breakdown  # Keep reference to original object
    }


def _proof_attempt_to_dict(attempt: ProofAttempt) -> Dict[str, Any]:
    """Convert a ProofAttempt to old format dict."""
    return {
        'uid': f"{attempt.origin_problem_id}_r{attempt.round_id}_b{attempt.breakdown_id}_{'theorem' if attempt.lemma_id == -1 else f'l{attempt.lemma_id}'}_a{attempt.attempt_id}",
        'name': attempt.origin_problem_id,
        'metadata': {
            'origin_problem_id': attempt.origin_problem_id,
            'round_id': attempt.round_id,
            'breakdown_id': attempt.breakdown_id,
            'lemma_id': attempt.lemma_id,
            'attempt_id': attempt.attempt_id,
            'iteration_id': attempt.iteration_id,
            'correction_round_id': attempt.correction_round_id
        },
        'model_reasoning': attempt.model_reasoning,
        'full_code': attempt.code,
        'formal_statement': attempt.formal_statement,
        'compilation_result': {
            'pass': attempt.compilation_result.passed,
            'complete': attempt.compilation_result.complete,
            'errors': attempt.compilation_result.errors,
            'warnings': attempt.compilation_result.warnings
        }
    }
