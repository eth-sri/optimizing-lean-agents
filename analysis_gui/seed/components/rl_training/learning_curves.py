"""
Learning curves tab for RL Training Analysis.

Shows success rate evolution across training rounds.
"""
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List

from typing import Optional

from .utils import (
    load_baseline_data,
    load_rollouts_cached,
    compute_metrics_from_rollouts,
    get_rollouts_file,
    format_budget_display,
)


def render_learning_curves_tab(
    rollouts_dir: Path,
    run_dir: Path,
    run_metadata: Dict,
    baselines: List[str],
    target_budget: float,  # kept for API compatibility but we show all budgets now
    dataset: Optional[str] = None
):
    """Render learning curves showing success rate evolution across rounds, for all budgets.

    Args:
        rollouts_dir: Root rollouts directory
        run_dir: Path to the RL run directory
        run_metadata: Metadata for the RL run
        baselines: List of baseline strategies to compare
        target_budget: Budget value (kept for API compatibility)
        dataset: Optional dataset name (e.g., 'putnam') for baseline data location
    """
    st.subheader("Learning Curves - All Budgets")

    # Get budgets from metadata
    iter_results = run_metadata.get('iteration_results') or {}
    budgets = iter_results.get('budgets', [])

    # If no budgets in metadata, try to discover them
    if not budgets:
        for round_id in run_metadata['rounds'][:1]:
            round_dir = run_dir / f"round{round_id}"
            rollouts_file = get_rollouts_file(round_dir)
            if rollouts_file is not None:
                # Load without budget filter to discover budgets
                with open(rollouts_file, 'r') as f:
                    all_data = json.load(f)
                if isinstance(all_data, list):
                    budgets = sorted(set(r.get('budget', 0) for r in all_data))
                break

    if not budgets:
        budgets = [target_budget]  # Fallback to provided budget

    # Handle None (unlimited) budgets - don't convert to float
    budgets = [float(b) if b is not None else None for b in budgets]

    # Get lambda values from metadata
    lambda_values = iter_results.get('lambda_values', [])

    # If no lambda values in metadata, try to discover them from rollouts
    if not lambda_values:
        for round_id in run_metadata['rounds'][:1]:
            round_dir = run_dir / f"round{round_id}"
            rollouts_file = get_rollouts_file(round_dir)
            if rollouts_file is not None:
                rollouts = load_rollouts_cached(str(rollouts_file), budgets[0] if budgets else target_budget)
                lambda_values = sorted(set(r.get('lambda', 0) for r in rollouts))
                break

    if not lambda_values:
        lambda_values = [0]  # Fallback

    lambda_values = [float(lv) for lv in lambda_values]

    # Filter baselines to only full_proof strategies
    full_proof_baselines = [b for b in baselines if b.startswith('full_proof')]

    # Create one chart per budget
    for budget in budgets:
        st.markdown(f"### Budget: {format_budget_display(budget, precision=0)}")

        # Load baseline metrics (only full_proof baselines)
        baseline_success_rates = {}
        for strategy in full_proof_baselines:
            baseline_data = load_baseline_data(rollouts_dir, strategy, dataset)
            if baseline_data and 'rollouts_path' in baseline_data:
                rollouts = load_rollouts_cached(str(baseline_data['rollouts_path']), budget)
                metrics = compute_metrics_from_rollouts(rollouts)
                if metrics:
                    baseline_success_rates[strategy] = metrics['success_rate']

        # Load RL round metrics per lambda
        rl_data_by_lambda = {lv: [] for lv in lambda_values}

        for round_id in run_metadata['rounds']:
            round_dir = run_dir / f"round{round_id}"
            rollouts_file = get_rollouts_file(round_dir)

            if rollouts_file is not None:
                all_rollouts = load_rollouts_cached(str(rollouts_file), budget)

                for lambda_val in lambda_values:
                    # Filter by lambda with tolerance for float comparison
                    lambda_rollouts = [r for r in all_rollouts if abs(float(r.get('lambda', 0)) - lambda_val) < 1e-12]
                    metrics = compute_metrics_from_rollouts(lambda_rollouts)
                    if metrics:
                        rl_data_by_lambda[lambda_val].append({
                            'Round': round_id,
                            'Success Rate': metrics['success_rate'],
                            'Solved': metrics['num_successful'],
                            'Total': metrics['total_rollouts'],
                        })

        # Check if we have any data
        has_data = any(len(data) > 0 for data in rl_data_by_lambda.values())
        if not has_data:
            st.warning(f"No RL training data available for {format_budget_display(budget, precision=0)} budget.")
            continue

        # Collect all success rates for y-axis scaling
        all_success_rates = list(baseline_success_rates.values())
        for rl_data in rl_data_by_lambda.values():
            all_success_rates.extend([d['Success Rate'] for d in rl_data])

        if not all_success_rates:
            continue

        min_y = min(all_success_rates)
        max_y = max(all_success_rates)
        y_padding = (max_y - min_y) * 0.1 if max_y > min_y else 0.05

        # Create plotly figure
        fig = go.Figure()

        # Add baseline horizontal lines (only full_proof)
        baseline_colors = {
            'full_proof_8b': '#2ca02c',
            'full_proof_32b': '#d62728',
        }

        for strategy, success_rate in baseline_success_rates.items():
            color = baseline_colors.get(strategy, '#888888')
            fig.add_hline(
                y=success_rate,
                line_dash="dash",
                line_color=color,
                annotation_text=f"{strategy}: {success_rate:.3f}",
                annotation_position="right"
            )

        # Color palette for lambda lines
        lambda_colors = px.colors.qualitative.Set1

        # Add one line per lambda
        for i, lambda_val in enumerate(lambda_values):
            rl_data = rl_data_by_lambda[lambda_val]
            if not rl_data:
                continue

            rl_df = pd.DataFrame(rl_data)

            color = lambda_colors[i % len(lambda_colors)]
            fig.add_trace(go.Scatter(
                x=rl_df['Round'],
                y=rl_df['Success Rate'],
                mode='lines+markers',
                name=f'λ={lambda_val:.2e}',
                line=dict(color=color, width=2),
                marker=dict(size=8),
            ))

        fig.update_layout(
            title=f"Success Rate by Lambda ({format_budget_display(budget, precision=0)} Budget)",
            xaxis_title="Round",
            yaxis_title="Success Rate",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            yaxis=dict(range=[max(0, min_y - y_padding), max_y + y_padding]),
            height=400,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Additional metrics - show table per lambda as a collapsible section
    with st.expander("Detailed Round Metrics"):
        for lambda_val in lambda_values:
            st.markdown(f"**λ = {lambda_val:.2e}**")
            # Show data for all budgets combined
            all_data = []
            for budget in budgets:
                for round_id in run_metadata['rounds']:
                    round_dir = run_dir / f"round{round_id}"
                    rollouts_file = get_rollouts_file(round_dir)
                    if rollouts_file is not None:
                        all_rollouts = load_rollouts_cached(str(rollouts_file), budget)
                        lambda_rollouts = [r for r in all_rollouts if abs(float(r.get('lambda', 0)) - lambda_val) < 1e-12]
                        metrics = compute_metrics_from_rollouts(lambda_rollouts)
                        if metrics:
                            all_data.append({
                                'Budget': format_budget_display(budget, precision=0),
                                'Round': round_id,
                                'Success Rate': f"{metrics['success_rate']:.4f}",
                                'Solved': metrics['num_successful'],
                                'Total': metrics['total_rollouts'],
                            })
            if all_data:
                st.dataframe(pd.DataFrame(all_data), use_container_width=True, hide_index=True)
            else:
                st.info(f"No data for λ={lambda_val:.2e}")
