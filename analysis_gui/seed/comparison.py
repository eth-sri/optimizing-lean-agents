"""
Utility module for comparing data across multiple runs.

Provides functionality to load and compare problems across different pipeline runs.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from seed_data_models import DataLoader


def _get_compiled_lemmas(breakdown) -> Set[int]:
    """Get set of lemma IDs that compiled successfully in a breakdown."""
    compiled_lemmas = set()
    if breakdown.lemma_prover_results:
        all_attempts = breakdown.lemma_prover_results.get('all_attempts', [])
        for attempt in all_attempts:
            problem_id = attempt.get('data', {}).get('problem_id', '')
            if '_l' in problem_id:
                try:
                    parts = problem_id.split('_l')
                    if len(parts) >= 2:
                        lemma_part = parts[1]
                        lemma_num = lemma_part.split('_')[0]
                        if lemma_num.isdigit():
                            comp_result = attempt.get('data', {}).get('compilation_result', {})
                            if comp_result.get('pass', False):
                                compiled_lemmas.add(int(lemma_num))
                except (IndexError, ValueError):
                    pass
    return compiled_lemmas


def _get_all_lemmas(breakdown) -> Set[int]:
    """Get set of all lemma IDs in a breakdown."""
    all_lemmas = set()
    if breakdown.lemma_prover_results:
        all_attempts = breakdown.lemma_prover_results.get('all_attempts', [])
        for attempt in all_attempts:
            problem_id = attempt.get('data', {}).get('problem_id', '')
            if '_l' in problem_id:
                try:
                    parts = problem_id.split('_l')
                    if len(parts) >= 2:
                        lemma_part = parts[1]
                        lemma_num = lemma_part.split('_')[0]
                        if lemma_num.isdigit():
                            all_lemmas.add(int(lemma_num))
                except (IndexError, ValueError):
                    pass
    return all_lemmas


def load_run(run_path: str, round_num: int = 0) -> Optional[Tuple[DataLoader, List['ProblemSummary']]]:
    """
    Load a run and get its problem summaries.

    Args:
        run_path: Path to the run directory
        round_num: Round number to load

    Returns:
        Tuple of (loader, problems) or None if loading fails
    """
    try:
        loader = DataLoader(run_path)
        session = loader.load_session()
        problems = list(session.problems.values())
        return loader, problems
    except Exception as e:
        return None


def count_fully_validated_breakdowns(problem: 'ProblemSummary', validation_results: dict = None) -> int:
    """
    Count breakdowns where all lemmas have at least one sample that was validated with 'yes' verdict.
    A fully formalized breakdown is one where every lemma has at least one successful validation.

    Args:
        problem: The problem to check
        validation_results: Dict mapping problem_id to validation verdicts (optional)

    Returns:
        Number of breakdowns that are fully formalized
    """
    count = 0
    for breakdown in problem.breakdowns.values():
        # Check if this breakdown is formalized
        if breakdown.parsed_breakdown is None:
            continue

        # If we have validation results, check if all lemmas validated successfully
        if validation_results:
            problem_validations = validation_results.get(breakdown.problem_id, [])
            if not problem_validations:
                continue

            # Group by lemma and check if each has at least one 'yes'
            from collections import defaultdict
            by_lemma = defaultdict(list)
            for v in problem_validations:
                by_lemma[v.get('lemma_id')].append(v.get('verdict'))

            # All lemmas must have at least one 'yes' verdict
            if by_lemma and all(verdicts.count('yes') > 0 for verdicts in by_lemma.values()):
                count += 1
        else:
            # Fallback: just count if formalized (when validation data not available)
            count += 1

    return count


def _count_theorem_proven(breakdowns):
    """Helper to count theorem proven breakdowns (pass AND complete)."""
    count = 0
    # Handle both dict and list - if dict, get values
    breakdowns_list = breakdowns.values() if isinstance(breakdowns, dict) else breakdowns
    for b in breakdowns_list:
        if b.theorem_prover_results:
            attempts = b.theorem_prover_results.get('attempts', [])
            for attempt in attempts:
                comp_result = attempt.get('data', {}).get('compilation_result', {})
                if comp_result.get('pass', False) and comp_result.get('complete', False):
                    count += 1
                    break
    return count


def _count_solved_breakdowns(problem):
    """
    Helper to count solved breakdowns for a problem.

    Uses breakdown.is_solved() which checks if the theorem is proven and all used lemmas are proven.

    Args:
        problem: Problem object

    Returns:
        Count of solved breakdowns
    """
    solved_count = 0
    for breakdown in problem.breakdowns.values():
        if breakdown.is_solved():
            solved_count += 1
    return solved_count


def _check_combined_proof(problem, run_dir: Optional[Path] = None) -> str:
    """
    Check if problem has a combined proof file and return status.

    Args:
        problem: Problem object
        run_dir: Optional run directory to search for combined proofs

    Returns:
        Status string: "✅" if has combined proof, "❌" if not, "?" if unknown
    """
    if not run_dir:
        return "?"

    from pathlib import Path

    run_path = Path(run_dir)

    # Check for combined proof in either location
    # Handle both direct paths and minified subdirectory paths
    lean_file_dirs = [
        run_path / "combined" / "lean_files",
        run_path.parent / "combined" / "lean_files" if run_path.name == "minified" else None
    ]

    # Look for any lean file matching the problem's origin_problem_id
    for lean_dir in filter(None, lean_file_dirs):
        for subdir in ["complete", "incomplete"]:
            dir_path = lean_dir / subdir
            if dir_path.exists():
                # Look for any file matching the origin problem pattern
                matches = list(dir_path.glob(f"{problem.origin_problem_id}_r*_b*.lean"))
                if matches:
                    return "✅"

    return "❌"


def get_problem_comparison_data(problem1: 'ProblemSummary', problem2: Optional['ProblemSummary'], validation_results1=None, validation_results2=None, run_dir1: Optional[Path] = None, run_dir2: Optional[Path] = None) -> Dict:
    """
    Extract comparison data for a problem across runs.

    Args:
        problem1: Problem from first run
        problem2: Problem from second run (if exists)
        validation_results1: Validation results for run 1
        validation_results2: Validation results for run 2

    Returns:
        Dictionary with comparison data
    """
    # Count formalized breakdowns for problem1
    # Formalized = has theorem prover attempts
    formalized1 = 0
    for breakdown in problem1.breakdowns.values():
        if breakdown.theorem_prover_results and breakdown.theorem_prover_results.get('attempts', []):
            formalized1 += 1

    theorem_proven1 = _count_theorem_proven(problem1.breakdowns)
    solved1_count = _count_solved_breakdowns(problem1)

    # Theorem proven out of formalized (or "0/0" if no formalized)
    theorem_ratio1 = f"{theorem_proven1}/{formalized1}" if formalized1 > 0 else "0/0"
    theorem_display1 = f"{theorem_ratio1} ({('✅' if theorem_proven1 > 0 else '❌')})"

    # Solved out of theorem proven (or "N/A" if no theorems proven)
    if theorem_proven1 > 0:
        solved_ratio1 = f"{solved1_count}/{theorem_proven1}"
        solved_display1 = f"{solved_ratio1} ({('✅' if solved1_count > 0 else '❌')})"
    else:
        solved_display1 = "N/A"

    data = {
        "Problem ID": problem1.origin_problem_id,
        "Run 1 Formalized": f"{formalized1}/{len(problem1.breakdowns)}",
        "Run 1 Theorem Proven": theorem_display1,
        "Run 1 Solved": solved_display1,
    }

    if problem2:
        # Count formalized breakdowns for problem2
        # Formalized = has theorem prover attempts
        formalized2 = 0
        for breakdown in problem2.breakdowns.values():
            if breakdown.theorem_prover_results and breakdown.theorem_prover_results.get('attempts', []):
                formalized2 += 1

        theorem_proven2 = _count_theorem_proven(problem2.breakdowns)
        solved2_count = _count_solved_breakdowns(problem2)

        # Theorem proven out of formalized (or "0/0" if no formalized)
        theorem_ratio2 = f"{theorem_proven2}/{formalized2}" if formalized2 > 0 else "0/0"
        theorem_display2 = f"{theorem_ratio2} ({('✅' if theorem_proven2 > 0 else '❌')})"

        # Solved out of theorem proven (or "N/A" if no theorems proven)
        if theorem_proven2 > 0:
            solved_ratio2 = f"{solved2_count}/{theorem_proven2}"
            solved_display2 = f"{solved_ratio2} ({('✅' if solved2_count > 0 else '❌')})"
        else:
            solved_display2 = "N/A"

        data["Run 2 Formalized"] = f"{formalized2}/{len(problem2.breakdowns)}"
        data["Run 2 Theorem Proven"] = theorem_display2
        data["Run 2 Solved"] = solved_display2
    else:
        data["Run 2 Formalized"] = "—"
        data["Run 2 Theorem Proven"] = "—"
        data["Run 2 Solved"] = "—"

    return data


def create_comparison_table(problems1: List['ProblemSummary'], problems2: Optional[List['ProblemSummary']] = None, validation_results1=None, validation_results2=None) -> List[Dict]:
    """
    Create a comparison table across two runs.

    Args:
        problems1: Problems from first run
        problems2: Problems from second run (optional)
        validation_results1: Validation results for run 1
        validation_results2: Validation results for run 2

    Returns:
        List of dictionaries with comparison data
    """
    comparison_data = []
    problem2_map = {p.origin_problem_id: p for p in problems2} if problems2 else {}

    # Get all unique problem IDs
    all_problem_ids = set(p.origin_problem_id for p in problems1)
    if problems2:
        all_problem_ids.update(p.origin_problem_id for p in problems2)

    # Build comparison rows
    for problem_id in sorted(all_problem_ids):
        problem1 = next((p for p in problems1 if p.origin_problem_id == problem_id), None)
        problem2 = problem2_map.get(problem_id)

        if problem1:
            row = get_problem_comparison_data(problem1, problem2, validation_results1, validation_results2)
            comparison_data.append(row)
        elif problem2:
            # Problem only in run2
            # Formalized = has theorem prover attempts
            formalized2 = 0
            for breakdown in problem2.breakdowns.values():
                if breakdown.theorem_prover_results and breakdown.theorem_prover_results.get('attempts', []):
                    formalized2 += 1

            theorem_proven2 = _count_theorem_proven(problem2.breakdowns)
            solved2_count = _count_solved_breakdowns(problem2)

            # Theorem proven out of formalized (or "0/0" if no formalized)
            theorem_ratio2 = f"{theorem_proven2}/{formalized2}" if formalized2 > 0 else "0/0"
            theorem_display2 = f"{theorem_ratio2} ({('✅' if theorem_proven2 > 0 else '❌')})"

            # Solved out of theorem proven (or "N/A" if no theorems proven)
            if theorem_proven2 > 0:
                solved_ratio2 = f"{solved2_count}/{theorem_proven2}"
                solved_display2 = f"{solved_ratio2} ({('✅' if solved2_count > 0 else '❌')})"
            else:
                solved_display2 = "N/A"

            row = {
                "Problem ID": problem_id,
                "Run 1 Formalized": "—",
                "Run 1 Theorem Proven": "—",
                "Run 1 Solved": "—",
                "Run 2 Formalized": f"{formalized2}/{len(problem2.breakdowns)}",
                "Run 2 Theorem Proven": theorem_display2,
                "Run 2 Solved": solved_display2,
            }
            comparison_data.append(row)

    return comparison_data
