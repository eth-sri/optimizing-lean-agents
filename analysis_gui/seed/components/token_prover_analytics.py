"""
Token and prover attempt analytics component.

Displays average reasoning tokens and prover attempt counts broken down by:
- Solved breakdowns
- Unsolved breakdowns
- Overall (all breakdowns)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Union, Dict, Tuple, Any
import sys
from pathlib import Path

# Add root directory to path to import seed_data_models
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

try:
    from seed_data_models import Session, Problem, Breakdown
    HAS_NEW_MODELS = True
except ImportError:
    HAS_NEW_MODELS = False
    Session = None
    Problem = None
    Breakdown = None



def render_token_prover_analytics(problems: Union['Session', List[Any]]):
    """
    Render token and prover attempt analytics.

    Displays metrics for reasoning tokens and prover attempts, broken down by:
    - Solved vs unsolved breakdowns (breakdown level)
    - Solved vs unsolved problems (problem level)

    Args:
        problems: List of Problem or ProblemSummary objects
    """
    st.subheader("🔬 Reasoning Tokens & Prover Attempts Analytics")

    # Get all breakdowns from the problems list
    breakdowns = _get_all_breakdowns_from_problems(problems)

    if not breakdowns:
        st.warning("No breakdown data available for token/prover analytics.")
        return

    # Calculate breakdown-level metrics
    metrics = _calculate_metrics(breakdowns)

    # Render breakdown-level metrics
    render_output_tokens_metrics(metrics)

    st.markdown("---")

    render_prover_attempts_metrics(metrics)

    st.markdown("---")

    # Render comparison visualizations
    render_comparison_charts(metrics)

    st.markdown("---")

    # Render problem-level analytics
    render_problem_level_analytics(problems)


def _get_all_breakdowns_from_problems(problems: List) -> List:
    """
    Extract all breakdowns from a list of problems.

    Handles both Problem (with dict of breakdowns) and ProblemSummary (with list) objects.

    Args:
        problems: List of Problem or ProblemSummary objects

    Returns:
        List of all Breakdown or BreakdownInfo objects
    """
    breakdowns = []
    for problem in problems:
        if hasattr(problem, 'breakdowns'):
            if isinstance(problem.breakdowns, dict):
                # Problem object with dict of breakdowns
                breakdowns.extend(problem.breakdowns.values())
            else:
                # ProblemSummary with list of breakdowns
                breakdowns.extend(problem.breakdowns)
    return breakdowns


def _calculate_metrics(breakdowns: List) -> Dict:
    """
    Calculate all metrics for solved/unsolved/overall breakdowns.

    Args:
        breakdowns: List of Breakdown or BreakdownInfo objects

    Returns:
        Dictionary with metrics for solved, unsolved, and overall
    """
    solved_breakdowns = []
    unsolved_breakdowns = []

    for bd in breakdowns:
        if bd.is_solved():
            solved_breakdowns.append(bd)
        else:
            unsolved_breakdowns.append(bd)

    metrics = {
        'solved': _get_breakdown_metrics(solved_breakdowns),
        'unsolved': _get_breakdown_metrics(unsolved_breakdowns),
        'overall': _get_breakdown_metrics(breakdowns),
        'solved_count': len(solved_breakdowns),
        'unsolved_count': len(unsolved_breakdowns),
        'total_count': len(breakdowns),
    }

    return metrics


def _get_breakdown_metrics(breakdowns: List) -> Dict:
    """
    Get metrics for a list of breakdowns.

    Args:
        breakdowns: List of Breakdown or BreakdownInfo objects

    Returns:
        Dictionary with output_tokens and prover_attempts metrics
    """
    if not breakdowns:
        return {
            'output_tokens': 0,
            'avg_output_tokens': 0.0,
            'total_output_tokens': 0,
            'prover_attempts': 0,
            'avg_prover_attempts': 0.0,
            'total_prover_attempts': 0,
        }

    total_output_tokens = sum(bd.get_total_cost('output_tokens') for bd in breakdowns)
    total_prover_attempts = sum(bd.get_total_cost('cost') for bd in breakdowns)

    return {
        'output_tokens': total_output_tokens,
        'avg_output_tokens': total_output_tokens / len(breakdowns),
        'total_output_tokens': total_output_tokens,
        'prover_attempts': total_prover_attempts,
        'avg_prover_attempts': total_prover_attempts / len(breakdowns),
        'total_prover_attempts': total_prover_attempts,
    }


def render_output_tokens_metrics(metrics: Dict):
    """
    Render output tokens metrics in a card layout.

    Args:
        metrics: Dictionary from _calculate_metrics()
    """
    st.markdown("### Output Tokens Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Avg Output Tokens (Solved)",
            f"{metrics['solved']['avg_output_tokens']:.0f}",
            help=f"Total: {metrics['solved']['total_output_tokens']:,} across {metrics['solved_count']} breakdowns"
        )

    with col2:
        st.metric(
            "Avg Output Tokens (Unsolved)",
            f"{metrics['unsolved']['avg_output_tokens']:.0f}",
            help=f"Total: {metrics['unsolved']['total_output_tokens']:,} across {metrics['unsolved_count']} breakdowns"
        )

    with col3:
        st.metric(
            "Avg Output Tokens (Overall)",
            f"{metrics['overall']['avg_output_tokens']:.0f}",
            help=f"Total: {metrics['overall']['total_output_tokens']:,} across {metrics['total_count']} breakdowns"
        )

    # Additional info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Total output tokens used:** {metrics['overall']['total_output_tokens']:,}")

    with col2:
        if metrics['solved_count'] > 0 and metrics['unsolved_count'] > 0:
            difference = metrics['solved']['avg_output_tokens'] - metrics['unsolved']['avg_output_tokens']
            pct_diff = (difference / metrics['unsolved']['avg_output_tokens'] * 100) if metrics['unsolved']['avg_output_tokens'] > 0 else 0
            if difference > 0:
                st.info(f"**Solved use {pct_diff:.1f}% more output tokens** than unsolved on average")
            else:
                st.info(f"**Unsolved use {abs(pct_diff):.1f}% more output tokens** than solved on average")


def render_prover_attempts_metrics(metrics: Dict):
    """
    Render prover attempt metrics in a card layout.

    Args:
        metrics: Dictionary from _calculate_metrics()
    """
    st.markdown("### Prover Attempts Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Avg Attempts (Solved)",
            f"{metrics['solved']['avg_prover_attempts']:.1f}",
            help=f"Total: {metrics['solved']['total_prover_attempts']} across {metrics['solved_count']} breakdowns"
        )

    with col2:
        st.metric(
            "Avg Attempts (Unsolved)",
            f"{metrics['unsolved']['avg_prover_attempts']:.1f}",
            help=f"Total: {metrics['unsolved']['total_prover_attempts']} across {metrics['unsolved_count']} breakdowns"
        )

    with col3:
        st.metric(
            "Avg Attempts (Overall)",
            f"{metrics['overall']['avg_prover_attempts']:.1f}",
            help=f"Total: {metrics['overall']['total_prover_attempts']} across {metrics['total_count']} breakdowns"
        )

    # Additional info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Total prover attempts:** {metrics['overall']['total_prover_attempts']}")

    with col2:
        if metrics['solved_count'] > 0 and metrics['unsolved_count'] > 0:
            difference = metrics['solved']['avg_prover_attempts'] - metrics['unsolved']['avg_prover_attempts']
            pct_diff = (difference / metrics['unsolved']['avg_prover_attempts'] * 100) if metrics['unsolved']['avg_prover_attempts'] > 0 else 0
            if difference > 0:
                st.info(f"**Solved use {pct_diff:.1f}% more attempts** than unsolved on average")
            else:
                st.info(f"**Unsolved use {abs(pct_diff):.1f}% more attempts** than solved on average")


def render_comparison_charts(metrics: Dict):
    """
    Render comparison visualizations between solved and unsolved breakdowns.

    Args:
        metrics: Dictionary from _calculate_metrics()
    """
    st.markdown("### Comparison Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        # Output tokens comparison
        fig = go.Figure(data=[
            go.Bar(
                name='Solved',
                x=['Avg Output Tokens'],
                y=[metrics['solved']['avg_output_tokens']],
                marker=dict(color='rgba(0, 200, 0, 0.7)')
            ),
            go.Bar(
                name='Unsolved',
                x=['Avg Output Tokens'],
                y=[metrics['unsolved']['avg_output_tokens']],
                marker=dict(color='rgba(200, 0, 0, 0.7)')
            )
        ])
        fig.update_layout(
            title="Average Output Tokens Comparison",
            barmode='group',
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Prover attempts comparison
        fig = go.Figure(data=[
            go.Bar(
                name='Solved',
                x=['Avg Prover Attempts'],
                y=[metrics['solved']['avg_prover_attempts']],
                marker=dict(color='rgba(0, 200, 0, 0.7)')
            ),
            go.Bar(
                name='Unsolved',
                x=['Avg Prover Attempts'],
                y=[metrics['unsolved']['avg_prover_attempts']],
                marker=dict(color='rgba(200, 0, 0, 0.7)')
            )
        ])
        fig.update_layout(
            title="Average Prover Attempts Comparison",
            barmode='group',
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Breakdown count pie chart
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(data=[
            go.Pie(
                labels=['Solved', 'Unsolved'],
                values=[metrics['solved_count'], metrics['unsolved_count']],
                marker=dict(colors=['rgba(0, 200, 0, 0.7)', 'rgba(200, 0, 0, 0.7)'])
            )
        ])
        fig.update_layout(
            title="Breakdown Distribution (Solved vs Unsolved)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Summary statistics table
        st.markdown("### Summary Statistics")
        summary_data = {
            'Metric': [
                'Total Breakdowns',
                'Solved',
                'Unsolved',
                'Success Rate'
            ],
            'Count/Rate': [
                f"{metrics['total_count']}",
                f"{metrics['solved_count']}",
                f"{metrics['unsolved_count']}",
                f"{(metrics['solved_count'] / metrics['total_count'] * 100):.1f}%" if metrics['total_count'] > 0 else "N/A"
            ]
        }
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_problem_level_analytics(problems: List):
    """
    Render problem-level analytics for reasoning tokens and prover attempts.

    Breaks down metrics by solved vs unsolved problems.

    Args:
        problems: List of Problem or ProblemSummary objects
    """
    st.markdown("### Problem-Level Analytics")

    # Calculate problem-level metrics
    problem_metrics = _calculate_problem_metrics(problems)

    if problem_metrics['total_problems'] == 0:
        st.info("No problem data available.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Avg Attempts (Solved Problems)",
            f"{problem_metrics['solved']['avg_prover_attempts']:.1f}",
            help=f"Total: {problem_metrics['solved']['total_prover_attempts']} across {problem_metrics['solved_count']} problems"
        )

    with col2:
        st.metric(
            "Avg Attempts (Unsolved Problems)",
            f"{problem_metrics['unsolved']['avg_prover_attempts']:.1f}",
            help=f"Total: {problem_metrics['unsolved']['total_prover_attempts']} across {problem_metrics['unsolved_count']} problems"
        )

    with col3:
        st.metric(
            "Avg Attempts (All Problems)",
            f"{problem_metrics['overall']['avg_prover_attempts']:.1f}",
            help=f"Total: {problem_metrics['overall']['total_prover_attempts']} across {problem_metrics['total_problems']} problems"
        )

    # Additional info for prover attempts
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Total prover attempts per problem:** {problem_metrics['overall']['total_prover_attempts']} / {problem_metrics['total_problems']} problems")

    with col2:
        if problem_metrics['solved_count'] > 0 and problem_metrics['unsolved_count'] > 0:
            difference = problem_metrics['solved']['avg_prover_attempts'] - problem_metrics['unsolved']['avg_prover_attempts']
            pct_diff = (difference / problem_metrics['unsolved']['avg_prover_attempts'] * 100) if problem_metrics['unsolved']['avg_prover_attempts'] > 0 else 0
            if difference > 0:
                st.info(f"**Solved problems use {pct_diff:.1f}% more attempts** than unsolved on average")
            else:
                st.info(f"**Unsolved problems use {abs(pct_diff):.1f}% more attempts** than solved on average")

    # Output tokens per problem metrics
    st.markdown("#### Output Tokens per Problem")

    output_token_metrics = _calculate_problem_output_token_metrics(problems)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Avg Output Tokens (Solved Problems)",
            f"{output_token_metrics['solved']['avg_output_tokens']:.0f}",
            help=f"Total: {output_token_metrics['solved']['total_output_tokens']:,} across {output_token_metrics['solved_count']} problems"
        )

    with col2:
        st.metric(
            "Avg Output Tokens (Unsolved Problems)",
            f"{output_token_metrics['unsolved']['avg_output_tokens']:.0f}",
            help=f"Total: {output_token_metrics['unsolved']['total_output_tokens']:,} across {output_token_metrics['unsolved_count']} problems"
        )

    with col3:
        st.metric(
            "Avg Output Tokens (All Problems)",
            f"{output_token_metrics['overall']['avg_output_tokens']:.0f}",
            help=f"Total: {output_token_metrics['overall']['total_output_tokens']:,} across {output_token_metrics['total_problems']} problems"
        )

    # Additional info for output tokens
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Total output tokens per problem:** {output_token_metrics['overall']['total_output_tokens']:,} / {output_token_metrics['total_problems']} problems")

    with col2:
        if output_token_metrics['solved_count'] > 0 and output_token_metrics['unsolved_count'] > 0:
            difference = output_token_metrics['solved']['avg_output_tokens'] - output_token_metrics['unsolved']['avg_output_tokens']
            pct_diff = (difference / output_token_metrics['unsolved']['avg_output_tokens'] * 100) if output_token_metrics['unsolved']['avg_output_tokens'] > 0 else 0
            if difference > 0:
                st.info(f"**Solved problems use {pct_diff:.1f}% more output tokens** than unsolved on average")
            else:
                st.info(f"**Unsolved problems use {abs(pct_diff):.1f}% more output tokens** than solved on average")

    # Render problem-level comparison chart
    st.markdown("#### Problem-Level Comparison")
    col1, col2 = st.columns(2)

    with col1:
        # Prover attempts per problem comparison
        fig = go.Figure(data=[
            go.Bar(
                name='Solved',
                x=['Avg Attempts per Problem'],
                y=[problem_metrics['solved']['avg_prover_attempts']],
                marker=dict(color='rgba(0, 200, 0, 0.7)')
            ),
            go.Bar(
                name='Unsolved',
                x=['Avg Attempts per Problem'],
                y=[problem_metrics['unsolved']['avg_prover_attempts']],
                marker=dict(color='rgba(200, 0, 0, 0.7)')
            )
        ])
        fig.update_layout(
            title="Average Prover Attempts per Problem",
            barmode='group',
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Problem distribution pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=['Solved', 'Unsolved'],
                values=[problem_metrics['solved_count'], problem_metrics['unsolved_count']],
                marker=dict(colors=['rgba(0, 200, 0, 0.7)', 'rgba(200, 0, 0, 0.7)'])
            )
        ])
        fig.update_layout(
            title="Problem Distribution (Solved vs Unsolved)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Render detailed problem table
    st.markdown("#### Per-Problem Breakdown")
    problem_details = _get_problem_details(problems)
    if problem_details:
        df = pd.DataFrame(problem_details)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Problem ID": st.column_config.TextColumn(width="medium"),
                "Status": st.column_config.TextColumn(width="small"),
                "Output Tokens": st.column_config.NumberColumn(width="small"),
                "Prover Attempts": st.column_config.NumberColumn(width="small"),
                "Avg Attempts/Breakdown": st.column_config.NumberColumn(format="%.1f", width="small"),
            }
        )


def _calculate_problem_output_token_metrics(problems: List) -> Dict:
    """
    Calculate output token metrics for solved/unsolved/overall problems.

    Args:
        problems: List of Problem or ProblemSummary objects

    Returns:
        Dictionary with output token metrics for solved, unsolved, and overall problems
    """
    solved_problems = []
    unsolved_problems = []

    for problem in problems:
        if _is_problem_solved(problem):
            solved_problems.append(problem)
        else:
            unsolved_problems.append(problem)

    metrics = {
        'solved': _get_problem_output_token_metrics_list(solved_problems),
        'unsolved': _get_problem_output_token_metrics_list(unsolved_problems),
        'overall': _get_problem_output_token_metrics_list(problems),
        'solved_count': len(solved_problems),
        'unsolved_count': len(unsolved_problems),
        'total_problems': len(problems),
    }

    return metrics


def _get_problem_output_token_metrics_list(problems: List) -> Dict:
    """
    Get aggregated output token metrics for a list of problems.

    Args:
        problems: List of Problem or ProblemSummary objects

    Returns:
        Dictionary with output token metrics
    """
    if not problems:
        return {
            'output_tokens': 0,
            'avg_output_tokens': 0.0,
            'total_output_tokens': 0,
        }

    total_output_tokens = 0
    for problem in problems:
        total_output_tokens += problem.get_total_cost('output_tokens')

    return {
        'output_tokens': total_output_tokens,
        'avg_output_tokens': total_output_tokens / len(problems),
        'total_output_tokens': total_output_tokens,
    }


def _calculate_problem_metrics(problems: List) -> Dict:
    """
    Calculate metrics for solved/unsolved/overall problems.

    Args:
        problems: List of Problem or ProblemSummary objects

    Returns:
        Dictionary with metrics for solved, unsolved, and overall problems
    """
    solved_problems = []
    unsolved_problems = []

    for problem in problems:
        if _is_problem_solved(problem):
            solved_problems.append(problem)
        else:
            unsolved_problems.append(problem)

    metrics = {
        'solved': _get_problem_metrics_list(solved_problems),
        'unsolved': _get_problem_metrics_list(unsolved_problems),
        'overall': _get_problem_metrics_list(problems),
        'solved_count': len(solved_problems),
        'unsolved_count': len(unsolved_problems),
        'total_problems': len(problems),
    }

    return metrics


def _is_problem_solved(problem) -> bool:
    """
    Check if a problem is solved.

    Handles both Problem and ProblemSummary objects.

    Args:
        problem: Problem or ProblemSummary object

    Returns:
        True if problem is solved, False otherwise
    """
    if hasattr(problem, 'is_solved'):
        # Problem object (has method)
        return problem.is_solved()
    elif hasattr(problem, 'solved'):
        # ProblemSummary object (has attribute)
        return problem.solved
    return False


def _get_problem_metrics_list(problems: List) -> Dict:
    """
    Get aggregated metrics for a list of problems.

    Args:
        problems: List of Problem or ProblemSummary objects

    Returns:
        Dictionary with prover attempt metrics
    """
    if not problems:
        return {
            'prover_attempts': 0,
            'avg_prover_attempts': 0.0,
            'total_prover_attempts': 0,
        }

    total_prover_attempts = 0
    for problem in problems:
        if hasattr(problem, 'count_prover_calls'):
            # Problem object (has method)
            total_prover_attempts += problem.count_prover_calls()
        else:
            # ProblemSummary object - need to get from breakdowns
            breakdowns = getattr(problem, 'breakdowns', [])
            for bd in breakdowns:
                if hasattr(bd, 'count_prover_calls'):
                    total_prover_attempts += bd.count_prover_calls()

    return {
        'prover_attempts': total_prover_attempts,
        'avg_prover_attempts': total_prover_attempts / len(problems),
        'total_prover_attempts': total_prover_attempts,
    }


def _get_problem_details(problems: List) -> List[Dict]:
    """
    Get detailed metrics for each problem.

    Args:
        problems: List of Problem or ProblemSummary objects

    Returns:
        List of dictionaries with per-problem details
    """
    details = []

    for problem in problems:
        # Get problem ID
        problem_id = getattr(problem, 'origin_problem_id', 'Unknown')

        # Get status
        status = "Solved" if _is_problem_solved(problem) else "Unsolved"

        total_output_tokens = problem.get_total_cost('output_tokens')
        total_attempts = problem.get_total_cost('prover_calls')

        # Get number of breakdowns
        if hasattr(problem, 'breakdowns'):
            if isinstance(problem.breakdowns, dict):
                num_breakdowns = len(problem.breakdowns)
            else:
                num_breakdowns = len(problem.breakdowns)
        else:
            num_breakdowns = 0

        # Calculate average attempts per breakdown
        avg_per_breakdown = total_attempts / num_breakdowns if num_breakdowns > 0 else 0

        details.append({
            'Problem ID': problem_id,
            'Status': status,
            'Output Tokens': total_output_tokens,
            'Prover Attempts': total_attempts,
            'Avg Attempts/Breakdown': avg_per_breakdown,
        })

    # Sort by problem attempts descending
    details.sort(key=lambda x: x['Prover Attempts'], reverse=True)
    return details
