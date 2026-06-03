"""
Enhanced breakdown details component with nested views for parsing, formalization, and proofs.

This module orchestrates the display of breakdown details by importing and coordinating
multiple specialized components for different aspects of the breakdown analysis.
"""
import streamlit as st
from typing import List, Dict, Any, Optional

# Import all component rendering functions
from .proof_status_component import render_proof_status_summary
from .breakdown_tree_component import render_parsed_lemma_tree, render_used_lemmas_tree
from .lemma_details_component import render_lemma_list
from .theorem_viewer_component import render_theorem_prover_code_results, render_theorem_attempts_from_parsed_breakdown
from .formalization_component import render_parse_failure, render_theorem_from_parsed
from .combined_proof_viewer import render_combined_proof
from seed_data_models import Breakdown

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


def render_enhanced_breakdown_details(
    breakdown: Breakdown,
    analysis: Optional[Dict[str, Any]] = None
):
    """
    Render enhanced breakdown details with nested structure.
    This is kept for backward compatibility but now calls the inner function.

    Args:
        breakdown: Breakdown object
        analysis: ProblemAnalysis object (optional, for proof attempts)
    """
    render_enhanced_breakdown_details_inner(breakdown, analysis)


def render_enhanced_breakdown_details_inner(
    breakdown: Breakdown,
    analysis: Optional[Dict[str, Any]] = None
):
    """
    Render the inner breakdown details content (without outer basic info).
    This is the core content displayed in the Breakdown Details tab.

    Args:
        breakdown: Breakdown object
        analysis: ProblemAnalysis object (optional, for proof attempts)
    """
    # Display parsed breakdown (renamed from "Parsed Breakdown" to "Breakdown Details")
    render_parsed_breakdown_tab(breakdown, analysis)


def render_parsed_breakdown_tab(
    breakdown: Breakdown,
    analysis: Optional[Dict[str, Any]] = None
):
    """
    Render the parsed breakdown tab with lemmas and proof attempts.

    This orchestrates the display of all breakdown components in a logical order:
    1. Proof status summary
    2. Theorem details from parsed breakdown
    3. Theorem prover results
    4. Parsed lemma dependency tree
    5. Used lemmas tree visualization
    6. Lemma list with details
    7. Combined proof

    Args:
        breakdown: Breakdown object
        analysis: ProblemAnalysis object (optional)
    """
    st.subheader("Breakdown Details")

    # Check if parsing failed (either parse_failure is set or parsed_breakdown has error)
    if breakdown.parse_failure:
        render_parse_failure(breakdown.parse_failure)
        return

    # Check if parsed_breakdown exists but has an error
    if breakdown.parsed_breakdown:
        parsed_data = breakdown.parsed_breakdown
        # If it has an error, render as parse failure
        if hasattr(parsed_data, 'error') and parsed_data.error:
            render_parse_failure(breakdown.parsed_breakdown)
            return

    # If no parsed breakdown, show original data or error details
    if not breakdown.parsed_breakdown:
        # Always show parse failure prominently
        st.error("❌ Breakdown parsing failed")

        # Display error details if available
        if breakdown.parse_failure:
            st.json(breakdown.parse_failure)
        else:
            st.info("No parse_failure details recorded. The parsed_breakdown could not be loaded.")

        st.markdown("---")
        st.markdown("**Original breakdown structure (pre-parsing):**")
        parse_tab1, parse_tab2 = st.tabs(["Informal", "Formal"])

        with parse_tab1:
            st.markdown("**Informal Prefix:**")
            if breakdown.informal_prefix:
                st.markdown(breakdown.informal_prefix)
            else:
                st.info("No informal prefix available.")

        with parse_tab2:
            st.markdown("**Formal Statement:**")
            if breakdown.formal_statement:
                st.code(breakdown.formal_statement, language="lean")
            else:
                st.info("No formal statement available.")
        return

    # Display proof status summary
    render_proof_status_summary(breakdown)

    st.markdown("---")

    # Display theorem details (theorem from parsed breakdown + theorem prover results)
    parsed_data = breakdown.parsed_breakdown

    # Get theorem from OOP structure
    theorem_obj = parsed_data.theorem

    if theorem_obj:
        # Get parsed data for theorem formalization display
        parsed_data = breakdown.parsed_breakdown

        # Convert Theorem object to dict for render function
        theorem_data = {
            'statement': theorem_obj.statement,
            'proof': theorem_obj.proof_idea,
        }
        render_theorem_from_parsed(theorem_data, breakdown, parsed_data)

    # Display theorem prover results
    # First try the new parsed_breakdown structure (minified data)
    if hasattr(breakdown, 'parsed_breakdown') and breakdown.parsed_breakdown:
        st.markdown("---")

        # Load attempts button for theorem
        col1, col2, col3 = st.columns([2, 2, 6])
        with col1:
            if st.button("📥 Load Attempts (Theorem)", key=f"load_attempts_theorem_{breakdown.origin_problem_id}_{breakdown.round_id}_{breakdown.breakdown_id}", help="Load full proof attempts for the theorem"):
                # Lazy import to avoid circular dependencies
                from .breakdown_viewer import _handle_load_attempts
                _handle_load_attempts(breakdown, lemma_id=-1)

        render_theorem_attempts_from_parsed_breakdown(breakdown)
    # Fall back to legacy theorem_prover_results structure
    elif breakdown.theorem_prover_results:
        st.markdown("---")
        render_theorem_prover_code_results(breakdown)

    st.markdown("---")

    # Display used lemmas from proof (if available in parsed_breakdown)
    if hasattr(breakdown, 'parsed_breakdown') and breakdown.parsed_breakdown:
        render_used_lemmas_tree(breakdown)
        st.markdown("---")

    # Display parsed lemma dependency tree
    render_parsed_lemma_tree(breakdown)

    st.markdown("---")

    # Try to display prover path tree using new data model first, then fall back to legacy
    if hasattr(breakdown, 'parsed_breakdown') and breakdown.parsed_breakdown:
        # New OOP data model - use plot_prover_path (works with minified data now)
        st.markdown("### 🌳 Prover Path Tree (Current Round)")
        st.markdown("*🔵 Blue=Theorem | 🟢 Green=Directly Solved | 🟡 Gold=Solved via Recursion | 🔴 Red=Unsolved*")
        try:
            # Check if theorem has a best attempt before trying to plot
            pb = breakdown.parsed_breakdown
            if pb and pb.theorem:
                best_attempt = pb.theorem.get_best_attempt()
                if not best_attempt:
                    st.info("⚠️ Theorem has no proof attempts")
                else:
                    fig = breakdown.plot_prover_path(recursive=False)
                    if fig:
                        st.pyplot(fig, use_container_width=False)
                    else:
                        st.info("⚠️ Could not generate prover path figure")
            else:
                st.info("⚠️ No parsed breakdown or theorem")
        except Exception as e:
            st.error(f"❌ Error generating prover path: {str(e)}")
    elif breakdown.theorem_prover_results:
        # Legacy structure - use plot_prover_path
        st.markdown("### 🌳 Prover Path Tree (Current Round)")
        st.markdown("*🔵 Blue=Theorem | 🟢 Green=Directly Solved | 🟡 Gold=Solved via Recursion | 🔴 Red=Unsolved*")
        fig = breakdown.plot_prover_path(recursive=False)
        if fig:
            st.pyplot(fig, use_container_width=False)

    st.markdown("---")

    # Display lemma list with details
    render_lemma_list(breakdown, analysis)

    st.markdown("---")

    # Display combined proof at the end
    render_combined_proof(breakdown)
