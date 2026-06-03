"""Formalization and compilation results
"""
import streamlit as st
from typing import Dict, Any, Optional
from seed_data_models import Breakdown

# Note: render_theorem_formalization is imported locally in render_theorem_from_parsed
# to avoid circular imports (formalization_component <-> theorem_viewer_component)

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



def render_compilation_result(compilation_result: Dict[str, Any], lemma_id: int):
    """
    Render compilation result with errors, warnings, and proof state.

    Args:
        compilation_result: The compilation result from Lean
        lemma_id: The lemma ID for display
    """
    st.markdown("**Compilation Result:**")

    # Overall status
    passed = compilation_result.get('pass', False)
    complete = compilation_result.get('complete', False)

    col1, col2 = st.columns(2)
    with col1:
        status = "✅ Compiled" if passed else "❌ Failed"
        st.markdown(f"**Status:** {status}")
    with col2:
        proof_status = "✅ Complete" if complete else "⏳ Incomplete (has 'sorry')"
        st.markdown(f"**Proof:** {proof_status}")

    # Errors
    errors = compilation_result.get('errors', [])
    if errors:
        st.error("**Compilation Errors:**")
        for error in errors:
            if isinstance(error, dict):
                st.code(error.get('data', str(error)))
            else:
                st.code(str(error))

    # Warnings
    warnings = compilation_result.get('warnings', [])
    if warnings:
        st.warning("**Warnings:**")
        for warning in warnings:
            if isinstance(warning, dict):
                st.text(warning.get('data', str(warning)))
            else:
                st.text(str(warning))

    # Proof state (from sorries)
    sorries = compilation_result.get('sorries', [])
    if sorries and not complete:
        st.markdown("**Remaining Proof Obligations:**")
        for sorry in sorries:
            if isinstance(sorry, dict) and 'goal' in sorry:
                st.code(sorry['goal'], language="lean")



def render_parse_failure(parse_failure: Dict[str, Any]):
    """
    Render parse failure information.

    Args:
        parse_failure: Dictionary containing parse failure information (can be parse_failure dict or full entry)
    """
    st.error("❌ **Parse Failed**")
    st.markdown("This breakdown failed to parse into structured lemmas and theorem.")

    # Check if this is a NEW diagnostic error format from data loader
    error_type = parse_failure.get('error')
    if error_type in ['file_missing', 'no_matching_record', 'no_valid_parsed_breakdown', 'json_parse_error', 'exception', 'unknown_parse_failure', 'parsing_failed']:
        # Display diagnostic error information
        st.markdown("### Diagnostic Error Information")

        if error_type == 'file_missing':
            st.markdown("**Error Type:** File Missing")
            st.markdown("The parsed_breakdown.json file could not be found at expected locations:")
            if parse_failure.get('paths_tried'):
                for path in parse_failure['paths_tried']:
                    st.code(path, language="text")

        elif error_type == 'no_matching_record':
            st.markdown("**Error Type:** Problem Not Found in File")
            st.markdown(f"Could not find problem `{parse_failure.get('problem_id')}` in parsed_breakdown.json")
            if parse_failure.get('found_problem_ids'):
                st.markdown("**Problems that exist in the file:**")
                for pid in parse_failure['found_problem_ids']:
                    st.code(pid, language="text")

        elif error_type == 'no_valid_parsed_breakdown':
            st.markdown("**Error Type:** Invalid Parsed Breakdown")
            st.markdown(parse_failure.get('message', 'The parsed breakdown was invalid or missing'))

        elif error_type == 'json_parse_error':
            st.markdown("**Error Type:** JSON Parse Error")
            st.code(parse_failure.get('message', 'Unknown JSON error'), language="text")

        elif error_type == 'exception':
            st.markdown("**Error Type:** Exception During Parsing")
            if parse_failure.get('type'):
                st.markdown(f"**Exception Type:** `{parse_failure['type']}`")
            st.code(parse_failure.get('message', 'Unknown exception'), language="text")

        elif error_type == 'parsing_failed':
            st.markdown("**Error Type:** Breakdown Parsing Failed")
            error_msg = parse_failure.get('message', 'Unknown parsing error')
            st.code(error_msg, language="text")

        elif error_type == 'unknown_parse_failure':
            st.markdown("**Error Type:** Unknown Parse Failure")
            st.markdown(parse_failure.get('message', 'Parsed breakdown could not be loaded'))

        # Display context information
        st.markdown("### Context")
        context_cols = st.columns(3)
        with context_cols[0]:
            st.metric("Problem", parse_failure.get('origin_problem_id', 'N/A'))
        with context_cols[1]:
            st.metric("Round", parse_failure.get('round_id', 'N/A'))
        with context_cols[2]:
            st.metric("Breakdown", parse_failure.get('breakdown_id', 'N/A'))

        # Show full diagnostic JSON in expandable section
        with st.expander("📋 Full Diagnostic Information (JSON)"):
            st.json(parse_failure)

        return

    # Handle old format (direct error) and new format (nested in parsed_breakdown)
    error_msg = None
    model_output = None
    json_error = None
    solution_output = None

    # Check if this is a parse_failure dict with 'error' field
    if parse_failure.get('error'):
        error_msg = parse_failure['error']

    # Check if this is a full failed_parse entry with parsed_breakdown containing error
    if parse_failure.get('parsed_breakdown'):
        parsed_bd = parse_failure['parsed_breakdown']
        if isinstance(parsed_bd, dict):
            if parsed_bd.get('error'):
                json_error = parsed_bd['error']
            # Extract solution output from full_breakdown
            if parsed_bd.get('full_breakdown'):
                solution_output = parsed_bd['full_breakdown']

    # Get model output (LLM's raw output from structured_breakdown or fallback)
    if parse_failure.get('structured_breakdown'):
        model_output = parse_failure['structured_breakdown']

    # Display solution output first (the raw LLM response)
    if solution_output:
        with st.expander("📋 Solution Output (<solution> tag)", expanded=True):
            st.code(solution_output, language="json")

    # Display model output if different from solution
    if model_output and model_output != solution_output:
        with st.expander("📋 Model Output"):
            st.text(model_output)

    # Display JSON error
    if json_error:
        st.markdown("**JSON Error:**")
        st.code(json_error)

    # Display error message
    if error_msg:
        st.markdown("**Error Message:**")
        st.code(error_msg)

    # Legacy support for faulty_output field
    if parse_failure.get('faulty_output'):
        st.markdown("**Faulty Output:**")
        st.text_area(
            "Parsing output that couldn't be structured",
            value=parse_failure['faulty_output'],
            disabled=True,
            height=150
        )

    if parse_failure.get('timestamp'):
        st.caption(f"Failed at: {parse_failure['timestamp']}")



def render_theorem_from_parsed(
    theorem_data: Dict[str, Any],
    breakdown: Optional[Breakdown] = None,
    formalized_data: Optional[Dict[str, Any]] = None
):
    """
    Render theorem information from parsed breakdown with formalization.

    Args:
        theorem_data: Dictionary containing theorem information
        breakdown: Breakdown object (for formalized data)
        formalized_data: The formalized breakdown data
    """
    st.markdown("### 🎯 Theorem")

    if theorem_data.get('statement'):
        st.markdown("**Statement:**")
        st.text(theorem_data['statement'])

    if theorem_data.get('proof'):
        st.markdown("**Proof Strategy:**")
        st.text(theorem_data['proof'])

    # Show the problem's formal statement (Lean formalization of the original problem)
    if breakdown and breakdown.formal_statement:
        st.markdown("**Problem Formalization (Lean 4):**")
        st.code(breakdown.formal_statement.strip(), language="lean")

    # Show theorem formalizations if available
    if formalized_data:
        # Import here to avoid circular imports
        from .theorem_viewer_component import render_theorem_formalization
        render_theorem_formalization(breakdown, formalized_data)



def _get_compilation_result_for_problem(
    problem_id: str, compilation_results: list
) -> Optional[dict]:
    """
    Extract the compilation result entry for a specific problem.

    Uses a multi-level matching strategy:
    1. Exact match (old format)
    2. Breakdown ID match (new format with _s<N> suffix)
    3. Origin problem fallback (find any result for the same origin problem)

    Args:
        problem_id: The problem identifier (e.g., 'putnam_2001_b2_r0_b0' or 'putnam_2001_b2_r0_b0_l1')
        compilation_results: List of compilation result objects

    Returns:
        The compilation result dict if found, None otherwise
    """
    # Try to import get_breakdown_id for accurate ID extraction
    try:
        from id_utils import get_breakdown_id
    except ImportError:
        def get_breakdown_id(pid: str) -> str:
            """Fallback breakdown ID extraction - matches id_utils.py."""
            import re
            canonical = str(pid)
            # Remove correction suffix (_corr<N>)
            if "_corr" in canonical:
                canonical = canonical.split("_corr")[0]
            # Remove proof attempt suffix (_p<N>)
            if "_p" in canonical:
                parts = canonical.rsplit("_p", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    canonical = parts[0]
            # Remove proof retry suffix (_r<N>) that comes AFTER lemma/theorem
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
            # Remove lemma/theorem suffix BEFORE removing sample suffixes
            if "_theorem" in canonical:
                canonical = canonical.replace("_theorem", "")
            if "_l" in canonical:
                parts = canonical.rsplit("_l", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    canonical = parts[0]
            # Remove formalization sample suffixes
            if "_sample_" in canonical:
                canonical = re.sub(r'_sample_\d+', '', canonical)
            # Remove breakdown sampling _s<N> suffix (both at end and in middle)
            canonical = re.sub(r'_s\d+(?:_|$)', '', canonical)
            return canonical

    # Get breakdown_id for matching
    target_breakdown_id = get_breakdown_id(problem_id)

    # Level 1: Try exact match first (old format)
    for item in compilation_results:
        if item.get('name') == problem_id:
            return item

    # Level 2: Try breakdown_id match (handles new format with _s<N> suffix)
    for item in compilation_results:
        item_name = item.get('name', '')
        if item_name and get_breakdown_id(item_name) == target_breakdown_id:
            return item

    # Level 3: Fallback to any result matching the origin problem
    # (e.g., if looking for putnam_1979_b6_r0_b0, find putnam_1979_b6_r0_b2_s1)
    origin_problem = '_'.join(target_breakdown_id.split('_')[:3])
    for item in compilation_results:
        item_name = item.get('name', '')
        if item_name and origin_problem in item_name and '_r' in item_name:
            return item

    return None



def _render_compilation_metrics(comp_result: dict, verify_time: float, is_timeout: bool):
    """
    Display key compilation metrics in a three-column layout.

    Args:
        comp_result: The compilation result dictionary
        verify_time: Time taken to verify in seconds
        is_timeout: Whether compilation timed out
    """
    col1, col2, col3 = st.columns(3)
    with col1:
        complete = comp_result.get('complete', False)
        st.metric("Complete", "✅ Yes" if complete else "❌ No")
    with col2:
        st.metric("Verify Time", f"{verify_time:.2f}s")
    with col3:
        if is_timeout:
            st.metric("Status", "🔴 Timeout")
        else:
            st.metric("Status", "🟢 OK" if comp_result.get('pass', False) else "🔴 Failed")



def _render_compilation_issues(comp_result: dict):
    """
    Display errors, warnings, and unproven goals (sorries).

    Args:
        comp_result: The compilation result dictionary
    """
    # Display errors
    errors = comp_result.get('errors', [])
    if errors:
        st.markdown("**Errors:**")
        for error in errors:
            st.error(error)

    # Display warnings
    warnings = comp_result.get('warnings', [])
    if warnings:
        st.markdown("**Warnings:**")
        for warning in warnings:
            msg = warning.get('data', str(warning))
            st.warning(msg)

    # Display unproven goals
    sorries = comp_result.get('sorries', [])
    if sorries:
        st.markdown(f"**Sorries:** {len(sorries)} unproven parts")
        with st.expander("View Sorries"):
            for i, sorry in enumerate(sorries, 1):
                st.markdown(f"**Sorry {i}:**")
                st.code(sorry.get('goal', ''), language="lean")



