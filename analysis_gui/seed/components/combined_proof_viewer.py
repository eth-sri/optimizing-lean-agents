"""Combined proof viewing
"""
import streamlit as st
from typing import Optional

# Import utilities from other components
from .utils_component import _find_lean_file_path
from .formalization_component import _get_compilation_result_for_problem, _render_compilation_metrics, _render_compilation_issues
from seed_data_models import Breakdown

# Try to import id_utils if available
try:
    from id_utils import get_lemma_id
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



def render_combined_proof(breakdown: Breakdown, run_dir: Optional[str] = None):
    """
    Display the full combined proof from the combined/lean_files directory.
    Shows the proof code and compilation results in expandable sections.

    Args:
        breakdown: Breakdown object
        run_dir: Optional run directory path. If not provided, uses session state or searches.
    """
    import json
    from pathlib import Path

    st.markdown("---")
    st.header("🎯 Combined Proof")

    try:
        # Resolve run directory
        if run_dir is None:
            if 'run_dir' in st.session_state:
                run_dir = st.session_state.run_dir
            else:
                st.info("Run directory not available.")
                return

        run_path = Path(run_dir)
        problem_id = breakdown.problem_id

        # Locate the lean file
        lean_file_path = _find_lean_file_path(problem_id, run_path)
        if not lean_file_path:
            st.info(f"Combined proof file not found. Looking for: {problem_id}.lean")
            return

        # Read and display the proof code
        with open(lean_file_path, 'r') as f:
            lean_code = f.read()

        with st.expander("📄 Combined Proof Code", expanded=False):
            st.code(lean_code, language="lean")

        # Load compilation results
        compilation_results_path = lean_file_path.parent.parent.parent / "compilation_results.json"
        if not compilation_results_path.exists():
            st.info("Compilation results not found.")
            return

        with open(compilation_results_path, 'r') as f:
            compilation_results = json.load(f)

        result = _get_compilation_result_for_problem(problem_id, compilation_results)
        if not result or 'compilation_result' not in result:
            st.info("No compilation result for this problem.")
            return

        comp_result = result['compilation_result']
        verify_time = result.get('verify_time', 0)
        is_timeout = verify_time > 300

        # Determine expander title and initial state
        if comp_result.get('pass', False):
            expander_title = "✅ Compilation Results"
            should_expand = False
        elif is_timeout:
            expander_title = "⏱️ Compilation Results (Timeout)"
            should_expand = True
        else:
            expander_title = "❌ Compilation Results"
            should_expand = True

        # Display compilation results in expandable section
        with st.expander(expander_title, expanded=should_expand):
            # Overall status
            if comp_result.get('pass', False):
                st.success("✅ Compilation Passed")
            elif is_timeout:
                st.error("⏱️ Compilation Timeout (> 300s)")
            else:
                st.error("❌ Compilation Failed")

            # Key metrics
            _render_compilation_metrics(comp_result, verify_time, is_timeout)

            # Failure details
            if not comp_result.get('pass', False):
                st.markdown("**Failure Reason:**")
                if is_timeout:
                    st.warning(f"Compilation exceeded 300 second timeout (actual: {verify_time:.2f}s)")
                else:
                    st.info("Check errors and warnings below for details")

            # Detailed issues
            _render_compilation_issues(comp_result)

    except Exception as e:
        st.error(f"Error loading combined proof: {str(e)}")


