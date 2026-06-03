"""
Solve types tab for RL Training Analysis.

Shows solve methods and action distribution analysis.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

from .utils import (
    load_rollouts_cached,
    compute_action_distribution,
    compute_action_success_rates,
    get_rollouts_file,
    format_budget_display,
)


def render_solve_types_tab(run_dir: Path, run_metadata: Dict, target_budget: Optional[float], target_lambda: Optional[float] = None):
    """Render solve type and action distribution analysis."""
    lambda_str = f", λ={target_lambda:.0e}" if target_lambda is not None else ""
    st.subheader(f"Solve Types & Actions - {format_budget_display(target_budget)} Budget{lambda_str}")

    solve_type_data = []
    action_data = []
    avg_action_data = []

    for round_id in run_metadata['rounds']:
        round_dir = run_dir / f"round{round_id}"
        rollouts_file = get_rollouts_file(round_dir)

        if rollouts_file is None:
            continue

        rollouts = load_rollouts_cached(str(rollouts_file), target_budget)
        # Filter by lambda if specified
        if target_lambda is not None:
            rollouts = [r for r in rollouts if r.get('lambda') == target_lambda]
        successful_rollouts = [r for r in rollouts if r.get('success', False)]
        num_rollouts = len(rollouts)

        # Count solve methods
        solve_methods = defaultdict(int)
        for r in successful_rollouts:
            method = r.get('solve_method', 'unknown')
            solve_methods[method] += 1

        total_solved = len(successful_rollouts)

        for method, count in solve_methods.items():
            solve_type_data.append({
                'Round': round_id,
                'Solve Method': method,
                'Count': count,
                'Percentage': 100 * count / total_solved if total_solved > 0 else 0
            })

        # Count actions from all rollouts
        action_counts = compute_action_distribution(rollouts)
        total_actions = sum(action_counts.values())

        for action_type, count in action_counts.items():
            action_data.append({
                'Round': round_id,
                'Action': action_type,
                'Count': count,
                'Percentage': 100 * count / total_actions if total_actions > 0 else 0
            })
            # Also compute average per rollout
            avg_action_data.append({
                'Round': round_id,
                'Action': action_type,
                'Avg per Rollout': count / num_rollouts if num_rollouts > 0 else 0
            })

    # Consistent color map for actions/solve methods across all charts
    COLOR_MAP = {
        # Actions
        'FULL_PROOF_8B': '#1f77b4',    # blue
        'FULL_PROOF_32B': '#ff7f0e',   # orange
        'ATTEMPT_8B': '#2ca02c',       # green
        'ATTEMPT_32B': '#d62728',      # red
        'CORRECTION_32B': '#9467bd',   # purple
        'CREATE_BREAKDOWN': '#8c564b', # brown
        'TERMINATE': '#7f7f7f',        # gray
        # Solve methods (matching their action counterparts)
        'full_proof_8b': '#1f77b4',    # blue
        'full_proof_32b': '#ff7f0e',   # orange
        'agentic': '#2ca02c',          # green
        'unknown': '#7f7f7f',          # gray
    }

    # Two columns for solve types and actions
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Solve Methods")
        if solve_type_data:
            solve_df = pd.DataFrame(solve_type_data)

            # Pivot table
            st.markdown("**Distribution by Round**")
            solve_pivot = solve_df.pivot(index='Round', columns='Solve Method', values='Count').fillna(0)
            st.dataframe(solve_pivot.astype(int), use_container_width=True)

            # Stacked bar chart
            fig = px.bar(
                solve_df,
                x='Round',
                y='Count',
                color='Solve Method',
                title='Solve Methods Across Rounds',
                barmode='stack',
                color_discrete_map=COLOR_MAP
            )
            fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1), height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No solve type data available.")

    with col2:
        st.markdown("### Action Distribution")
        if action_data:
            action_df = pd.DataFrame(action_data)

            # Pivot table
            st.markdown("**Distribution by Round**")
            action_pivot = action_df.pivot(index='Round', columns='Action', values='Count').fillna(0)
            st.dataframe(action_pivot.astype(int), use_container_width=True)

            # Stacked bar chart
            fig = px.bar(
                action_df,
                x='Round',
                y='Count',
                color='Action',
                title='Actions Across Rounds',
                barmode='stack',
                color_discrete_map=COLOR_MAP
            )
            fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1), height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No action data available.")

    # Percentage tables below
    st.markdown("---")
    st.markdown("### Percentage Breakdown")

    pct_col1, pct_col2 = st.columns(2)

    with pct_col1:
        if solve_type_data:
            solve_df = pd.DataFrame(solve_type_data)
            st.markdown("**Solve Method %**")
            pct_pivot = solve_df.pivot(index='Round', columns='Solve Method', values='Percentage').fillna(0)
            st.dataframe(pct_pivot.round(1), use_container_width=True)

    with pct_col2:
        if action_data:
            action_df = pd.DataFrame(action_data)
            st.markdown("**Action %**")
            pct_pivot = action_df.pivot(index='Round', columns='Action', values='Percentage').fillna(0)
            st.dataframe(pct_pivot.round(1), use_container_width=True)

    # Action Success Rates table
    st.markdown("---")
    st.markdown("### Action Success Rates")

    success_rate_data = []
    for round_id in run_metadata['rounds']:
        round_dir = run_dir / f"round{round_id}"
        rollouts_file = get_rollouts_file(round_dir)

        if rollouts_file is None:
            continue

        rollouts = load_rollouts_cached(str(rollouts_file), target_budget)
        # Filter by lambda if specified
        if target_lambda is not None:
            rollouts = [r for r in rollouts if r.get('lambda') == target_lambda]
        action_stats = compute_action_success_rates(rollouts)

        for action_type, stats in action_stats.items():
            total = stats['total']
            successful = stats['successful']
            success_rate = successful / total if total > 0 else 0
            success_rate_data.append({
                'Round': round_id,
                'Action': action_type,
                'Total': total,
                'Successful': successful,
                'Success Rate': success_rate
            })

    if success_rate_data:
        success_df = pd.DataFrame(success_rate_data)

        # Pivot table showing success rates
        st.markdown("**Success Rate by Action Type**")
        success_pivot = success_df.pivot(index='Round', columns='Action', values='Success Rate').fillna(0)
        # Format as percentages
        success_pivot_display = success_pivot.applymap(lambda x: f"{x*100:.1f}%")
        st.dataframe(success_pivot_display, use_container_width=True)

        # Also show raw numbers
        st.markdown("**Successful / Total by Action Type**")
        # Create a pivot with formatted strings
        raw_data = []
        for _, row in success_df.iterrows():
            raw_data.append({
                'Round': row['Round'],
                'Action': row['Action'],
                'Ratio': f"{row['Successful']}/{row['Total']}"
            })
        raw_df = pd.DataFrame(raw_data)
        raw_pivot = raw_df.pivot(index='Round', columns='Action', values='Ratio').fillna('-')
        st.dataframe(raw_pivot, use_container_width=True)
    else:
        st.info("No action success rate data available.")

    # Average Actions per Rollout
    st.markdown("---")
    st.markdown("### Average Actions per Rollout")

    if avg_action_data:
        avg_df = pd.DataFrame(avg_action_data)
        avg_pivot = avg_df.pivot(index='Round', columns='Action', values='Avg per Rollout').fillna(0)
        st.dataframe(avg_pivot.round(2), use_container_width=True)
    else:
        st.info("No action data available.")
