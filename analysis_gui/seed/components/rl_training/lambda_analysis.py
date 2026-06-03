"""
Lambda analysis tab for RL Training Analysis.

Shows analysis of different lambda values and their performance.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional


def format_budget_display(budget: Optional[float], precision: int = 1) -> str:
    """Format budget for display in UI (e.g., '16.0M' or 'unlimited')."""
    if budget is None:
        return "unlimited"
    return f"{budget/1e6:.{precision}f}M"


def render_lambda_analysis_tab(run_metadata: Dict, target_budget: Optional[float]):
    """Render lambda value analysis."""
    st.subheader(f"Lambda Analysis - {format_budget_display(target_budget)} Budget")

    budget_lambda = run_metadata.get('budget_lambda_comparison')

    if not budget_lambda:
        st.info("No lambda comparison data available.")
        return

    # Handle nested structure: data may be under 'by_budget_lambda' key
    if 'by_budget_lambda' in budget_lambda:
        budget_lambda = budget_lambda['by_budget_lambda']

    # Extract data for target budget
    budget_key = format_budget_display(target_budget)

    if budget_key not in budget_lambda:
        # Try numeric key
        budget_key = str(int(target_budget)) if target_budget is not None else "unlimited"
        if budget_key not in budget_lambda:
            # Try float key
            budget_key = str(target_budget) if target_budget is not None else "unlimited"
            if budget_key not in budget_lambda:
                st.warning(f"No data for budget {format_budget_display(target_budget)}. Available: {list(budget_lambda.keys())}")
                return

    budget_data = budget_lambda[budget_key]

    if isinstance(budget_data, dict):
        # Create dataframe from lambda data
        lambda_data = []
        for lambda_val, lambda_metrics in budget_data.items():
            if isinstance(lambda_metrics, dict):
                # Data is nested under 'rounds' array - use the last round's metrics
                rounds = lambda_metrics.get('rounds', [])
                if rounds:
                    # Use the last round (most trained)
                    metrics = rounds[-1]
                    lambda_data.append({
                        'Lambda': float(lambda_val),
                        'Success Rate': metrics.get('success_rate', 0),
                        'Avg Cost': metrics.get('avg_cost', 0),
                        'Avg Steps': metrics.get('avg_steps', 0),
                        'Num Solved': metrics.get('num_successful', 0),
                        'Total': metrics.get('num_rollouts', 0),
                    })
                else:
                    # Fallback: metrics directly in dict
                    success_rate = (
                        lambda_metrics.get('success_rate') or
                        lambda_metrics.get('solve_rate') or
                        0
                    )
                    avg_cost = lambda_metrics.get('avg_cost', 0)
                    avg_steps = lambda_metrics.get('avg_steps', 0)
                    lambda_data.append({
                        'Lambda': float(lambda_val),
                        'Success Rate': success_rate,
                        'Avg Cost': avg_cost,
                        'Avg Steps': avg_steps,
                        'Num Solved': lambda_metrics.get('num_successful', 0),
                        'Total': lambda_metrics.get('num_rollouts', 0),
                    })

        if lambda_data:
            df = pd.DataFrame(lambda_data)
            df = df.sort_values('Lambda')

            # Display table
            st.markdown("**Performance by Lambda Value (Last Round)**")
            display_df = df.copy()
            display_df['Lambda'] = display_df['Lambda'].apply(lambda x: f"{x:.2e}")
            display_df['Success Rate'] = display_df['Success Rate'].apply(lambda x: f"{x:.4f}")
            display_df['Avg Cost'] = display_df['Avg Cost'].apply(lambda x: f"{x/1e6:.2f}M")
            display_df['Avg Steps'] = display_df['Avg Steps'].apply(lambda x: f"{x:.1f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Plot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Success Rate vs Lambda', 'Avg Cost vs Lambda']
            )

            fig.add_trace(
                go.Scatter(x=df['Lambda'], y=df['Success Rate'], mode='lines+markers', name='Success Rate'),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=df['Lambda'], y=df['Avg Cost'], mode='lines+markers', name='Avg Cost'),
                row=1, col=2
            )

            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Show raw data for debugging
        with st.expander("Raw Data Structure"):
            # Show first lambda entry to understand structure
            first_lambda = list(budget_data.keys())[0] if budget_data else None
            if first_lambda:
                st.write(f"**Sample entry for lambda={first_lambda}:**")
                st.json(budget_data[first_lambda])
    else:
        st.json(budget_data)
