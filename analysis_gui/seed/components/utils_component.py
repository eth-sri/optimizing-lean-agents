"""Utility functions for file path finding
"""
"""
Enhanced breakdown details component with nested views for parsing, formalization, and proofs.
"""
from typing import Optional

# Try to import id_utils if available
try:
    from id_utils import get_lemma_id, get_lemma_component
except ImportError:
    import re

    def get_lemma_id(problem_id: str) -> str:
        """Fallback implementation of get_lemma_id - matches id_utils.py."""
        canonical = str(problem_id)

        # Remove correction suffix (_corr<N>)
        if "_corr" in canonical:
            canonical = canonical.split("_corr")[0]

        # Remove proof attempt suffix (_p<N>)
        if "_p" in canonical:
            parts = canonical.rsplit("_p", 1)
            if len(parts) == 2 and parts[1].isdigit():
                canonical = parts[0]

        # Remove proof retry suffix (_r<N>) that comes before lemma/theorem
        if "_r" in canonical:
            if "_l" in canonical or "_theorem" in canonical:
                lemma_match = None
                if "_l" in canonical:
                    lemma_match = canonical.rfind("_l")
                if "_theorem" in canonical:
                    theorem_match = canonical.rfind("_theorem")
                    if lemma_match is None or theorem_match > lemma_match:
                        lemma_match = theorem_match

                if lemma_match is not None:
                    after_lemma = canonical[lemma_match:]
                    r_match = re.search(r'_r(\d+)', after_lemma)
                    if r_match:
                        end_pos = lemma_match + r_match.start()
                        canonical = canonical[:end_pos] + canonical[lemma_match + r_match.end():]

        # Remove formalization sample suffixes
        if "_sample_" in canonical:
            canonical = re.sub(r'_sample_\d+', '', canonical)

        # Remove breakdown sampling _s<N> suffix that appears BEFORE lemma/theorem markers
        canonical = re.sub(r'_s\d+(?=_[lt])', '', canonical)

        return canonical

    def get_lemma_component(problem_id: str):
        """Fallback implementation of get_lemma_component."""
        # Use get_lemma_id to normalize first
        normalized = get_lemma_id(problem_id)

        if "_theorem" in normalized:
            return "theorem"
        elif "_l" in normalized:
            parts = normalized.rsplit("_l", 1)
            if len(parts) == 2 and parts[1].split("_")[0].isdigit():
                return f"l{parts[1].split('_')[0]}"
        return None



def _find_lean_file_path(problem_id: str, run_path: "Path") -> Optional["Path"]:
    """
    Locate the compiled Lean proof file for a given problem.

    Searches the expected round first, then falls back to searching all rounds.
    Checks both 'complete' and 'incomplete' subdirectories.

    Uses breakdown_id (without lemma suffix) to find the file, as lean files are stored
    by breakdown with formalization sample index (e.g., putnam_1975_a1_r0_b0_s1.lean).

    Args:
        problem_id: The problem identifier (e.g., 'putnam_2001_b2_r0_b0' or 'putnam_2001_b2_r0_b0_l1')
        run_path: Path to the run directory

    Returns:
        Path to the lean file if found, None otherwise
    """
    from pathlib import Path
    import sys

    # Try to import get_breakdown_id for accurate ID extraction
    try:
        from id_utils import get_breakdown_id
    except ImportError:
        def get_breakdown_id(pid: str) -> str:
            """Fallback breakdown ID extraction."""
            canonical = str(pid)
            # Remove lemma suffix
            if "_l" in canonical:
                parts = canonical.rsplit("_l", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    canonical = parts[0]
            # Remove theorem suffix
            if "_theorem" in canonical:
                canonical = canonical.replace("_theorem", "")
            return canonical

    # Use breakdown_id (without lemma suffix) to find the file
    breakdown_id = get_breakdown_id(problem_id)

    # The lean files are stored with their breakdown_id + formalization sample as filename
    # (e.g., putnam_1975_a1_r0_b0_s1.lean)
    # First try exact match with breakdown_id
    lean_filename = f"{breakdown_id}.lean"

    # Extract round number from problem_id
    round_match = None
    for part in problem_id.split('_'):
        if part.startswith('r'):
            round_match = part
            break

    # Try top-level combined folder first (new location after refactor)
    # Handle both direct paths and minified subdirectory paths
    paths_to_check = [run_path / "combined" / "lean_files"]

    # If we're in a minified directory, also check parent (the dump root)
    if run_path.name == "minified":
        paths_to_check.insert(0, run_path.parent / "combined" / "lean_files")

    for base_lean_dir in paths_to_check:
        for subdir in ["complete", "incomplete"]:
            possible_path = base_lean_dir / subdir / lean_filename
            if possible_path.exists():
                return possible_path

            # If exact match not found, search for files with matching breakdown_id
            lean_dir = base_lean_dir / subdir
            if lean_dir.exists():
                for file_path in lean_dir.glob(f"{breakdown_id}_s*.lean"):
                    return file_path

    # Try the expected round next (old location)
    if round_match:
        round_num = round_match[1:]
        for subdir in ["complete", "incomplete"]:
            possible_path = run_path / f"round{round_num}" / "combined" / "lean_files" / subdir / lean_filename
            if possible_path.exists():
                return possible_path

            # If exact match not found, search for files with matching breakdown_id
            # (in case the file has a _s<N> suffix for formalization sample)
            lean_dir = run_path / f"round{round_num}" / "combined" / "lean_files" / subdir
            if lean_dir.exists():
                for file_path in lean_dir.glob(f"{breakdown_id}_s*.lean"):
                    return file_path

    # Fallback: search all rounds
    for round_dir in run_path.glob("round*"):
        for subdir in ["complete", "incomplete"]:
            possible_path = round_dir / "combined" / "lean_files" / subdir / lean_filename
            if possible_path.exists():
                return possible_path

            # If exact match not found, search for files with matching breakdown_id
            lean_dir = round_dir / "combined" / "lean_files" / subdir
            if lean_dir.exists():
                for file_path in lean_dir.glob(f"{breakdown_id}_s*.lean"):
                    return file_path

    # Final fallback: Search for any lean file that matches the origin problem
    # (e.g., search for putnam_1979_b6_r0_b* when given putnam_1979_b6_r0_b0)
    # Extract the origin problem from breakdown_id
    origin_problem = '_'.join(breakdown_id.split('_')[:3])  # e.g., putnam_1979_b6

    # Try top-level combined folder first (new location)
    # Handle both direct paths and minified subdirectory paths
    top_level_paths = [run_path / "combined" / "lean_files"]
    if run_path.name == "minified":
        top_level_paths.insert(0, run_path.parent / "combined" / "lean_files")

    for base_dir in top_level_paths:
        for subdir in ["complete", "incomplete"]:
            lean_dir = base_dir / subdir
            if lean_dir.exists():
                # Look for any file matching the origin problem pattern (new format with _s<N>)
                matches = list(lean_dir.glob(f"{origin_problem}_r*_b*_s*.lean"))
                if matches:
                    return matches[0]

                # Also try without the sample suffix (old format)
                matches = list(lean_dir.glob(f"{origin_problem}_r*_b*.lean"))
                if matches:
                    return matches[0]

    # Prefer complete over incomplete in round directories
    for round_dir in sorted(run_path.glob("round*")):
        for subdir in ["complete", "incomplete"]:
            lean_dir = round_dir / "combined" / "lean_files" / subdir
            if lean_dir.exists():
                # Look for any file matching the origin problem pattern (new format with _s<N>)
                matches = list(lean_dir.glob(f"{origin_problem}_r*_b*_s*.lean"))
                if matches:
                    return matches[0]  # Return first match (complete dir is preferred)

                # Also try without the sample suffix (old format)
                matches = list(lean_dir.glob(f"{origin_problem}_r*_b*.lean"))
                if matches:
                    return matches[0]

    return None



