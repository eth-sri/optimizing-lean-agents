"""
Centralized utilities for handling problem ID hierarchies.

ID Hierarchy:
-------------
1. origin_problem_id: The base problem name from the dataset
   Example: "imo_p2_1969"

2. breakdown_id: The first layer of divergence (after breakdown sampling)
   Example: "imo_p2_1969_b0"

3. lemma_id: The full identifier including lemma/theorem
   Example: "imo_p2_1969_b0_l1" or "imo_p2_1969_b0_theorem"

4. Full problem_id: May include correction rounds (_corr<N>) and proof attempts (_p<N>)
   Example: "imo_p2_1969_b0_l1_p2_corr3"

Suffix Patterns:
---------------
- _b<N>: Breakdown sample number (e.g., _b0, _b1, ...)
- _l<N>: Lemma number (e.g., _l0, _l1, ...)
- _lemma_<N>: Alternative lemma format (e.g., _lemma_0, _lemma_1, ...)
- _sample_<N>: Old formalization sample number (e.g., _sample_0, _sample_1, ...)
- _s<N>: New formalization sample number (e.g., _s0, _s1, ...) - SHORT FORM
- _theorem: Indicates the main theorem
- _r<N>: Round number (e.g., _r0, _r1, ...)
- _p<N>: Proof attempt number (e.g., _p0, _p1, ...)
- _corr<N>: Correction round number (e.g., _corr1, _corr2, ...)

NOTE: _s<N> is the short form for formalization sample that became prominent in the
      c1/2025/10/28/010649 run. It represents which formalization sample variant was used.
"""
from typing import Optional


def get_origin_problem_id(problem_id: str) -> str:
    """
    Extract the origin problem ID by removing all suffixes.

    This returns just the base problem name from the dataset.

    Args:
        problem_id: Any problem ID variant

    Returns:
        The origin problem ID (e.g., "imo_p2_1969")

    Examples:
        >>> get_origin_problem_id("imo_p2_1969_b0_l1_p2_corr3")
        "imo_p2_1969"
        >>> get_origin_problem_id("imo_p2_1969_b1_theorem_corr1")
        "imo_p2_1969"
        >>> get_origin_problem_id("imo_p2_1969_b0_s2")
        "imo_p2_1969"
        >>> get_origin_problem_id("imo_p2_1969")
        "imo_p2_1969"
    """
    canonical = str(problem_id)

    # Remove suffixes from right to left
    # Order: _corr -> _p -> _s/_sample -> _lemma/_l/_theorem -> _b -> _r
    # Note: _b must come before _r because names are like: name_r0_b0
    #       _s must be checked after _corr/_p but we need to be careful with underscore parsing

    # Remove correction suffix (_corr<N>)
    if "_corr" in canonical:
        canonical = canonical.split("_corr")[0]

    # Remove proof attempt suffix (_p<N>)
    if "_p" in canonical:
        parts = canonical.rsplit("_p", 1)
        if len(parts) == 2 and parts[1].isdigit():
            canonical = parts[0]

    # Remove formalization sample suffixes (both old and new formats)
    # New format: _s<N> where N is a digit (e.g., _s0, _s1, _s2)
    # Old format: _sample_<N> (e.g., _sample_0, _sample_1)
    if "_sample_" in canonical:
        canonical = canonical.rsplit("_sample_", 1)[0]

    # Handle _s<N> format: must be careful to match digit-only suffix at the end
    import re
    canonical = re.sub(r'_s\d+$', '', canonical)

    # Remove lemma/theorem suffix
    if "_theorem" in canonical:
        canonical = canonical.replace("_theorem", "")

    # Remove _lemma_<N> format
    if "_lemma_" in canonical:
        canonical = canonical.rsplit("_lemma_", 1)[0]

    if "_l" in canonical:
        parts = canonical.rsplit("_l", 1)
        if len(parts) == 2 and parts[1].isdigit():
            canonical = parts[0]

    # Remove breakdown suffix (_b<N>) before round suffix
    if "_b" in canonical:
        parts = canonical.rsplit("_b", 1)
        if len(parts) == 2 and parts[1].isdigit():
            canonical = parts[0]

    # Remove round suffix (_r<N>)
    if "_r" in canonical:
        parts = canonical.rsplit("_r", 1)
        if len(parts) == 2 and parts[1].isdigit():
            canonical = parts[0]

    return canonical


def get_breakdown_id(problem_id: str) -> str:
    """
    Extract the breakdown ID by removing lemma/theorem, round, correction, proof attempt, and sample suffixes.

    This returns the first layer of divergence (breakdown number only, no samples).

    Handles multiple formats:
    - Old format: putnam_1971_a2_r0_b1_l5_corr2 -> putnam_1971_a2_r0_b1
    - New format with step suffix: putnam_1971_a2_r0_b1_s3_l5_r0_p0 -> putnam_1971_a2_r0_b1
    - Breakdown sampling: putnam_1971_a2_r0_b1_sample_0_l5_corr2 -> putnam_1971_a2_r0_b1
    - End-of-ID step suffix: putnam_1971_a2_r0_b1_s2 -> putnam_1971_a2_r0_b1

    Args:
        problem_id: Any problem ID variant

    Returns:
        The breakdown ID (e.g., "imo_p2_1969_b0")

    Examples:
        >>> get_breakdown_id("imo_p2_1969_b0_l1_p2_corr3")
        "imo_p2_1969_b0"
        >>> get_breakdown_id("imo_p2_1969_b1_theorem_corr1")
        "imo_p2_1969_b1"
        >>> get_breakdown_id("imo_p2_1969_b0_l2")
        "imo_p2_1969_b0"
        >>> get_breakdown_id("imo_p2_1969_b3_s2")
        "imo_p2_1969_b3"
        >>> get_breakdown_id("imo_p2_1969_b3_s2_theorem_r0_p0")
        "imo_p2_1969_b3"
        >>> get_breakdown_id("putnam_1971_a2_r0_b1_s3_l5_r0_p0")
        "putnam_1971_a2_r0_b1"
    """
    import re
    canonical = str(problem_id)

    # Remove correction suffix (_corr<N>)
    if "_corr" in canonical:
        canonical = canonical.split("_corr")[0]

    # Remove proof attempt suffix (_p<N>)
    if "_p" in canonical:
        parts = canonical.rsplit("_p", 1)
        if len(parts) == 2 and parts[1].isdigit():
            canonical = parts[0]

    # Remove proof retry suffix (_r<N>) that comes AFTER lemma/theorem
    # We need to be careful because we also have breakdown _r<N>
    if "_r" in canonical:
        # Check if there's a lemma/theorem marker
        if "_l" in canonical or "_theorem" in canonical:
            # Find the last lemma/theorem marker
            lemma_match = None
            if "_l" in canonical:
                lemma_match = canonical.rfind("_l")
            if "_theorem" in canonical:
                theorem_match = canonical.rfind("_theorem")
                if lemma_match is None or theorem_match > lemma_match:
                    lemma_match = theorem_match

            if lemma_match is not None:
                # Look for _r<N> after the lemma/theorem marker
                after_lemma = canonical[lemma_match:]
                r_match = re.search(r'_r(\d+)', after_lemma)
                if r_match:
                    # Remove this specific _r<N>
                    end_pos = lemma_match + r_match.start()
                    canonical = canonical[:end_pos] + canonical[lemma_match + r_match.end():]

    # Remove lemma/theorem suffix BEFORE removing sample suffixes
    # because theorem IDs look like: name_s2_theorem_r0_p0
    if "_theorem" in canonical:
        canonical = canonical.replace("_theorem", "")

    if "_l" in canonical:
        parts = canonical.rsplit("_l", 1)
        if len(parts) == 2 and parts[1].isdigit():
            canonical = parts[0]

    # Remove formalization sample suffixes (both old and new formats)
    if "_sample_" in canonical:
        canonical = re.sub(r'_sample_\d+', '', canonical)

    # Remove breakdown sampling _s<N> suffix (both at end and in middle)
    # At end: _s<N>$ (e.g., putnam_1971_a2_r0_b1_s2)
    # In middle: _s<N>_ (e.g., putnam_1971_a2_r0_b1_s3_l5 after lemma removal)
    canonical = re.sub(r'_s\d+(?:_|$)', '', canonical)

    return canonical


def get_lemma_id(problem_id: str) -> str:
    """
    Extract the lemma ID by removing only round, correction, proof attempt, and sample suffixes.

    This returns the full identifier including breakdown and lemma/theorem parts,
    but without the transient round, correction, proof attempt and sample suffixes.

    Handles multiple formats:
    - Old format: putnam_1971_a2_r0_b1_l5_corr2
    - New format with step suffix: putnam_1971_a2_r0_b1_s3_l5_r0_p0
    - Breakdown sampling: putnam_1971_a2_r0_b1_sample_0_l5_corr2

    Args:
        problem_id: Any problem ID variant

    Returns:
        The lemma ID (e.g., "imo_p2_1969_b0_l1" or "imo_p2_1969_b0_theorem")

    Examples:
        >>> get_lemma_id("imo_p2_1969_b0_l1_p2_corr3")
        "imo_p2_1969_b0_l1"
        >>> get_lemma_id("imo_p2_1969_b1_theorem_corr1")
        "imo_p2_1969_b1_theorem"
        >>> get_lemma_id("imo_p2_1969_b0_l2")
        "imo_p2_1969_b0_l2"
        >>> get_lemma_id("putnam_1971_a2_r0_b1_s3_l5_r0_p0")
        "putnam_1971_a2_r0_b1_l5"
        >>> get_lemma_id("putnam_1971_a2_r0_b1_sample_0_l5_corr2")
        "putnam_1971_a2_r0_b1_l5"
    """
    import re
    canonical = str(problem_id)

    # Remove correction suffix (_corr<N>)
    if "_corr" in canonical:
        canonical = canonical.split("_corr")[0]

    # Remove proof attempt suffix (_p<N>) - must come after _r removal
    if "_p" in canonical:
        parts = canonical.rsplit("_p", 1)
        if len(parts) == 2 and parts[1].isdigit():
            canonical = parts[0]

    # Remove proof retry suffix (_r<N>) that comes before lemma/theorem
    # This is tricky because we also have breakdown _r<N>
    # Strategy: Only remove the LAST _r<N> if it comes after a lemma/theorem marker
    if "_r" in canonical:
        # Check if there's a lemma/theorem marker after any _r
        if "_l" in canonical or "_theorem" in canonical:
            # Find the last _r<N> that comes AFTER the lemma/theorem marker
            lemma_match = None
            if "_l" in canonical:
                lemma_match = canonical.rfind("_l")
            if "_theorem" in canonical:
                theorem_match = canonical.rfind("_theorem")
                if lemma_match is None or theorem_match > lemma_match:
                    lemma_match = theorem_match

            if lemma_match is not None:
                # Look for _r<N> after the lemma/theorem marker
                after_lemma = canonical[lemma_match:]
                r_match = re.search(r'_r(\d+)', after_lemma)
                if r_match:
                    # Remove this specific _r<N>
                    end_pos = lemma_match + r_match.start()
                    canonical = canonical[:end_pos] + canonical[lemma_match + r_match.end():]

    # Remove formalization sample suffixes (both old and new formats)
    if "_sample_" in canonical:
        canonical = re.sub(r'_sample_\d+', '', canonical)

    # Remove breakdown sampling _s<N> suffix that appears BEFORE lemma/theorem markers
    # Pattern: _s<N>_ (has underscore after it, indicating it's not the final suffix)
    canonical = re.sub(r'_s\d+(?=_[lt])', '', canonical)

    return canonical


def get_lemma_component(problem_id: str) -> Optional[str]:
    """
    Extract just the lemma/theorem component from a problem ID.

    Args:
        problem_id: Any problem ID variant

    Returns:
        The lemma component (e.g., "l1", "theorem") or None if not present

    Examples:
        >>> get_lemma_component("imo_p2_1969_b0_l1_p2_corr3")
        "l1"
        >>> get_lemma_component("imo_p2_1969_b1_theorem_corr1")
        "theorem"
        >>> get_lemma_component("imo_p2_1969_b0")
        None
    """
    canonical = get_lemma_id(problem_id)

    if "_theorem" in canonical:
        return "theorem"
    elif "_l" in canonical:
        parts = canonical.rsplit("_l", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return f"l{parts[1]}"

    return None


def get_breakdown_number(problem_id: str) -> Optional[int]:
    """
    Extract the breakdown number from a problem ID.

    Args:
        problem_id: Any problem ID variant

    Returns:
        The breakdown number or None if not present

    Examples:
        >>> get_breakdown_number("imo_p2_1969_b0_l1")
        0
        >>> get_breakdown_number("imo_p2_1969_b3_theorem")
        3
        >>> get_breakdown_number("imo_p2_1969")
        None
    """
    breakdown_id = get_breakdown_id(problem_id)

    if "_b" in breakdown_id:
        parts = breakdown_id.rsplit("_b", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])

    return None


def get_proof_id(problem_id: str) -> str:
    """
    Get the full proof ID including all suffixes (corrections and proof attempts).

    This is essentially the identity function but can be useful for clarity
    when you want to explicitly reference a specific proof attempt.

    Args:
        problem_id: Any problem ID variant

    Returns:
        The complete problem_id unchanged

    Examples:
        >>> get_proof_id("imo_p2_1969_b0_l1_p2_corr3")
        "imo_p2_1969_b0_l1_p2_corr3"
        >>> get_proof_id("imo_p2_1969_b0_theorem")
        "imo_p2_1969_b0_theorem"
    """
    return str(problem_id)


def get_correction_round(problem_id: str) -> Optional[int]:
    """
    Extract the correction round number from a problem ID.

    Args:
        problem_id: Any problem ID variant

    Returns:
        The correction round number or None if not present (meaning round 0)

    Examples:
        >>> get_correction_round("imo_p2_1969_b0_l1_corr3")
        3
        >>> get_correction_round("imo_p2_1969_b0_l1_p2_corr1")
        1
        >>> get_correction_round("imo_p2_1969_b0_l1")
        None
    """
    if "_corr" not in problem_id:
        return None

    parts = problem_id.split("_corr")
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])

    return None


def get_proof_attempt(problem_id: str) -> Optional[int]:
    """
    Extract the proof attempt number from a problem ID.

    Args:
        problem_id: Any problem ID variant

    Returns:
        The proof attempt number or None if not present

    Examples:
        >>> get_proof_attempt("imo_p2_1969_b0_l1_p2")
        2
        >>> get_proof_attempt("imo_p2_1969_b0_l1_p3_corr1")
        3
        >>> get_proof_attempt("imo_p2_1969_b0_l1")
        None
    """
    canonical = str(problem_id)

    # Remove correction suffix first to isolate the proof attempt
    if "_corr" in canonical:
        canonical = canonical.split("_corr")[0]

    if "_p" not in canonical:
        return None

    parts = canonical.rsplit("_p", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])

    return None


def get_sample_number(problem_id: str) -> Optional[int]:
    """
    Extract the formalization sample number from a problem ID.

    Handles both old format (_sample_<N>) and new format (_s<N>).

    Args:
        problem_id: Any problem ID variant

    Returns:
        The sample number or None if not present

    Examples:
        >>> get_sample_number("imo_p2_1969_b0_lemma_1_sample_0")
        0
        >>> get_sample_number("imo_p2_1969_b0_lemma_1_sample_3")
        3
        >>> get_sample_number("imo_p2_1969_b0_l1_s2")
        2
        >>> get_sample_number("imo_p2_1969_b0_l1")
        None
    """
    import re

    # Check for new format: _s<N> at the end
    match = re.search(r'_s(\d+)$', problem_id)
    if match:
        return int(match.group(1))

    # Check for old format: _sample_<N>
    if "_sample_" not in problem_id:
        return None

    parts = problem_id.rsplit("_sample_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])

    return None


def remove_sample_suffix(problem_id: str) -> str:
    """
    Remove the formalization sample suffix from a problem ID.

    Handles both old format (_sample_<N>) and new format (_s<N>).

    Args:
        problem_id: Any problem ID variant

    Returns:
        The problem ID without the sample suffix

    Examples:
        >>> remove_sample_suffix("imo_p2_1969_b0_lemma_1_sample_0")
        "imo_p2_1969_b0_lemma_1"
        >>> remove_sample_suffix("imo_p2_1969_b0_l1_s2")
        "imo_p2_1969_b0_l1"
        >>> remove_sample_suffix("imo_p2_1969_b0_l1")
        "imo_p2_1969_b0_l1"
    """
    import re

    # Remove new format: _s<N>
    result = re.sub(r'_s\d+$', '', problem_id)

    # Remove old format: _sample_<N> (only if new format wasn't found)
    if result == problem_id and "_sample_" in problem_id:
        result = problem_id.rsplit("_sample_", 1)[0]

    return result


def normalize_lemma_format(problem_id: str) -> str:
    """
    Convert _lemma_<N> format to _l<N> format.

    Args:
        problem_id: Any problem ID variant

    Returns:
        The problem ID with _l<N> format

    Examples:
        >>> normalize_lemma_format("imo_p2_1969_b0_lemma_1")
        "imo_p2_1969_b0_l1"
        >>> normalize_lemma_format("imo_p2_1969_b0_l1")
        "imo_p2_1969_b0_l1"
    """
    import re
    # Replace _lemma_<N> with _l<N>
    return re.sub(r'_lemma_(\d+)', r'_l\1', problem_id)


if __name__ == "__main__":
    # Test examples
    test_cases = [
        "imo_p2_1969",
        "imo_p2_1969_b0",
        "imo_p2_1969_b0_l1",
        "imo_p2_1969_b0_theorem",
        "imo_p2_1969_b0_l1_p2",
        "imo_p2_1969_b0_l1_p2_corr3",
        "imo_p2_1969_b1_theorem_corr1",
    ]

    print("Testing ID extraction functions:\n")
    for test_id in test_cases:
        print(f"Input: {test_id}")
        print(f"  origin_problem_id: {get_origin_problem_id(test_id)}")
        print(f"  breakdown_id:      {get_breakdown_id(test_id)}")
        print(f"  lemma_id:          {get_lemma_id(test_id)}")
        print(f"  lemma_component:   {get_lemma_component(test_id)}")
        print(f"  breakdown_number:  {get_breakdown_number(test_id)}")
        print(f"  correction_round:  {get_correction_round(test_id)}")
        print(f"  proof_attempt:     {get_proof_attempt(test_id)}")
        print(f"  proof_id:          {get_proof_id(test_id)}")
        print()
