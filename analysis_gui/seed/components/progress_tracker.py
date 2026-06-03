"""
Progress tracker component - visualizes pipeline progress for breakdowns.
"""
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import get_status_emoji, get_breakdown_stage_status


def render_progress_tracker(
    breakdowns: List[BreakdownInfo],
    analysis: Optional[Dict[str, Any]]
):
    """
    Render progress tracking visualization for breakdowns.

    Args:
        breakdowns: List of BreakdownInfo objects
        analysis: ProblemAnalysis object containing pipeline results
    """
    if not breakdowns:
        st.warning("No breakdowns to track.")
        return

    st.header("Pipeline Progress Tracker")

    if not analysis:
        st.warning("No analysis data available for progress tracking.")
        return

    # Overall statistics
    render_overall_stats(analysis)

    st.markdown("---")

    # Per-breakdown progress
    st.subheader("Breakdown-by-Breakdown Progress")

    # Create progress visualization
    render_progress_chart(breakdowns, analysis)

    st.markdown("---")

    # Detailed breakdown status
    render_detailed_status(breakdowns, analysis)


def render_overall_stats(analysis: Optional[Dict[str, Any]]):
    """
    Render overall pipeline statistics.

    Args:
        analysis: ProblemAnalysis object
    """
    st.subheader("Overall Pipeline Statistics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total = analysis.breakdown_stats.get('total_breakdowns', 0)
        st.metric("Total Breakdowns", total)

    with col2:
        parsed = analysis.breakdown_parser_stats.get('total_parsed', 0)
        st.metric("Parsed", parsed)

    with col3:
        formalized = analysis.formalizer_stats.get('total_formalized', 0)
        st.metric("Formalized", formalized)

    with col4:
        lemmas_proven = analysis.lemma_prover_stats.get('breakdowns_all_lemmas_proven', 0)
        st.metric("Lemmas Proven", lemmas_proven)

    with col5:
        theorems_proven = analysis.theorem_prover_stats.get('breakdowns_proven', 0)
        st.metric("Theorems Proven", theorems_proven)

    # Success rates
    total = analysis.breakdown_stats.get('total_breakdowns', 0)
    if total > 0:
        st.markdown("### Success Rates")
        rate_col1, rate_col2, rate_col3, rate_col4 = st.columns(4)

        with rate_col1:
            parsed = analysis.breakdown_parser_stats.get('total_parsed', 0)
            rate = (parsed / total) * 100 if total > 0 else 0
            st.metric("Parsing Rate", f"{rate:.1f}%")

        with rate_col2:
            formalized = analysis.formalizer_stats.get('total_formalized', 0)
            rate = (formalized / total) * 100 if total > 0 else 0
            st.metric("Formalization Rate", f"{rate:.1f}%")

        with rate_col3:
            lemmas_proven = analysis.lemma_prover_stats.get('breakdowns_all_lemmas_proven', 0)
            rate = (lemmas_proven / total) * 100 if total > 0 else 0
            st.metric("Lemma Success Rate", f"{rate:.1f}%")

        with rate_col4:
            theorems_proven = analysis.theorem_prover_stats.get('breakdowns_proven', 0)
            rate = (theorems_proven / total) * 100 if total > 0 else 0
            st.metric("Theorem Success Rate", f"{rate:.1f}%")


def render_progress_chart(breakdowns: List[BreakdownInfo], analysis: Optional[Dict[str, Any]]):
    """
    Render a visual chart showing progress for each breakdown.

    Args:
        breakdowns: List of BreakdownInfo objects
        analysis: ProblemAnalysis object
    """
    # Prepare data for visualization
    breakdown_ids = [bd.problem_id for bd in breakdowns]
    stages = ["Breakdown", "Parsed", "Formalized", "Lemmas", "Theorem"]

    # Create matrix of completion status
    completion_matrix = []
    for breakdown in breakdowns:
        status = get_breakdown_stage_status(analysis, breakdown.problem_id)
        row = [
            1 if status["breakdown"] else 0,
            1 if status["parsed"] else 0,
            1 if status["formalized"] else 0,
            1 if status["lemmas_proven"] else 0,
            1 if status["theorem_proven"] else 0,
        ]
        completion_matrix.append(row)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=completion_matrix,
        x=stages,
        y=breakdown_ids,
        colorscale=[[0, '#ffcccc'], [1, '#90EE90']],
        showscale=False,
        text=[[get_status_emoji(bool(val)) for val in row] for row in completion_matrix],
        texttemplate="%{text}",
        textfont={"size": 20},
        hovertemplate='Breakdown: %{y}<br>Stage: %{x}<br>Status: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title="Pipeline Stage Completion",
        xaxis_title="Pipeline Stage",
        yaxis_title="Breakdown ID",
        height=max(400, len(breakdowns) * 40),
        yaxis={'side': 'left'},
    )

    st.plotly_chart(fig, width="content")


def render_detailed_status(breakdowns: List[BreakdownInfo], analysis: Optional[Dict[str, Any]]):
    """
    Render detailed status for each breakdown.

    Args:
        breakdowns: List of BreakdownInfo objects
        analysis: ProblemAnalysis object
    """
    st.subheader("Detailed Status by Breakdown")

    for idx, breakdown in enumerate(breakdowns):
        with st.expander(f"{breakdown.problem_id}", expanded=(idx == 0)):
            status = get_breakdown_stage_status(analysis, breakdown.problem_id)

            # Display status for each stage
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.markdown(f"**Breakdown**")
                st.markdown(get_status_emoji(status["breakdown"]))

            with col2:
                st.markdown(f"**Parsed**")
                st.markdown(get_status_emoji(status["parsed"]))

            with col3:
                st.markdown(f"**Formalized**")
                st.markdown(get_status_emoji(status["formalized"]))

            with col4:
                st.markdown(f"**Lemmas**")
                st.markdown(get_status_emoji(status["lemmas_proven"]))

            with col5:
                st.markdown(f"**Theorem**")
                st.markdown(get_status_emoji(status["theorem_proven"]))

            # Show details from analysis
            render_breakdown_analysis_details(breakdown.problem_id, analysis)


def render_breakdown_analysis_details(breakdown_id: str, analysis: Optional[Dict[str, Any]]):
    """
    Render detailed analysis information for a specific breakdown.

    Args:
        breakdown_id: The breakdown ID
        analysis: ProblemAnalysis object
    """
    st.markdown("---")
    st.markdown("**Pipeline Details:**")

    # Theorem prover details
    theorem_results = [
        bd for bd in analysis.theorem_prover_stats.get('breakdown_results', [])
        if bd.get('breakdown_id') == breakdown_id
    ]

    if theorem_results:
        result = theorem_results[0]
        st.markdown("**Theorem Prover:**")
        st.markdown(f"- Proven: {get_status_emoji(result.get('proven', False))}")
        st.markdown(f"- Attempts: {result.get('attempts', 0)}")
        if 'error' in result:
            st.error(f"Error: {result['error']}")

    # Lemma prover details
    lemma_results = [
        bd for bd in analysis.lemma_prover_stats.get('breakdown_results', [])
        if bd.get('breakdown_id') == breakdown_id
    ]

    if lemma_results:
        result = lemma_results[0]
        st.markdown("**Lemma Prover:**")
        st.markdown(f"- All lemmas proven: {get_status_emoji(result.get('all_lemmas_proven', False))}")
        st.markdown(f"- Total lemmas: {result.get('total_lemmas', 0)}")
        st.markdown(f"- Proven lemmas: {result.get('proven_lemmas', 0)}")

    # Formalizer details
    formalized = [
        bd for bd in analysis.formalizer_stats.get('parsed_breakdowns', [])
        if bd.get('breakdown_id') == breakdown_id
    ]

    if formalized:
        st.markdown("**Formalizer:**")
        st.markdown(f"- Formalized: {get_status_emoji(True)}")

    # Failed formalizations
    failed = [
        bd for bd in analysis.formalizer_stats.get('failed_breakdowns', [])
        if bd.get('breakdown_id') == breakdown_id
    ]

    if failed:
        st.markdown("**Formalizer:**")
        st.markdown(f"- Failed: {get_status_emoji(False)}")
        if 'error' in failed[0]:
            st.error(f"Error: {failed[0]['error']}")
