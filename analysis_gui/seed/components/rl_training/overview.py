"""
Overview tab for RL Training Analysis.

Shows overview metrics comparing RL rounds vs baselines for all budgets.
"""
import streamlit as st
import pandas as pd
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


def render_overview_tab(
    rollouts_dir: Path,
    run_dir: Path,
    run_metadata: Dict,
    baselines: List[str],
    budgets: List[Optional[float]],
    dataset: Optional[str] = None
):
    """Render overview metrics comparing RL rounds vs baselines for all budgets, broken down by lambda.

    Args:
        rollouts_dir: Root rollouts directory
        run_dir: Path to the RL run directory
        run_metadata: Metadata for the RL run
        baselines: List of baseline strategies to compare
        budgets: List of budget values to show
        dataset: Optional dataset name (e.g., 'putnam') for baseline data location
    """
    st.subheader("Overview - All Budgets & Lambdas")

    # Get lambda values from metadata
    iter_results = run_metadata.get('iteration_results') or {}
    lambda_values = iter_results.get('lambda_values', [])

    # Render a section for each budget
    for target_budget in budgets:
        st.markdown(f"## Budget: {format_budget_display(target_budget)}")

        # Load baseline metrics (baselines don't have lambda, so show once per budget)
        baseline_metrics = []
        for strategy in baselines:
            baseline_data = load_baseline_data(rollouts_dir, strategy, dataset)
            if baseline_data and 'rollouts_path' in baseline_data:
                rollouts = load_rollouts_cached(str(baseline_data['rollouts_path']), target_budget)
                metrics = compute_metrics_from_rollouts(rollouts)
                if metrics:
                    metrics['strategy'] = strategy
                    baseline_metrics.append(metrics)

        # Show baselines once per budget
        st.markdown("### Baseline Strategies")
        if baseline_metrics:
            baseline_df = pd.DataFrame(baseline_metrics)
            baseline_df = baseline_df[['strategy', 'success_rate', 'num_successful', 'unique_solved', 'total_rollouts', 'avg_cost', 'avg_steps']]
            baseline_df.columns = ['Strategy', 'Success Rate', 'Solved', 'Unique', 'Total', 'Avg Cost', 'Avg Steps']
            baseline_df['Success Rate'] = baseline_df['Success Rate'].apply(lambda x: f"{x:.3f}")
            baseline_df['Avg Cost'] = baseline_df['Avg Cost'].apply(lambda x: f"{x/1e6:.2f}M")
            baseline_df['Avg Steps'] = baseline_df['Avg Steps'].apply(lambda x: f"{x:.1f}")
            st.dataframe(baseline_df, use_container_width=True, hide_index=True)
        else:
            st.info("No baseline data available.")

        # Show RL rounds per lambda
        st.markdown("### RL Training Rounds (by Lambda)")

        # If no lambda values in metadata, try to discover them from rollouts
        if not lambda_values:
            # Load first round's rollouts to discover lambda values
            for round_id in run_metadata['rounds'][:1]:
                round_dir = run_dir / f"round{round_id}"
                rollouts_file = get_rollouts_file(round_dir)
                if rollouts_file is not None:
                    rollouts = load_rollouts_cached(str(rollouts_file), target_budget)
                    lambda_values = sorted(set(r.get('lambda', 0) for r in rollouts))
                    break

        if not lambda_values:
            lambda_values = [0]  # Fallback

        # Ensure lambda values are floats
        lambda_values = [float(lv) for lv in lambda_values]

        # Show each lambda as a separate section (list format)
        for lambda_val in lambda_values:
            st.markdown(f"#### λ = {lambda_val:.2e}")

            # Load RL round metrics filtered by lambda
            rl_metrics = []
            for round_id in run_metadata['rounds']:
                round_dir = run_dir / f"round{round_id}"
                rollouts_file = get_rollouts_file(round_dir)

                if rollouts_file is not None:
                    rollouts = load_rollouts_cached(str(rollouts_file), target_budget)
                    # Filter by lambda with tolerance for float comparison
                    lambda_rollouts = [r for r in rollouts if abs(float(r.get('lambda', 0)) - lambda_val) < 1e-12]
                    metrics = compute_metrics_from_rollouts(lambda_rollouts)
                    if metrics:
                        metrics['round'] = round_id
                        rl_metrics.append(metrics)

            if rl_metrics:
                rl_df = pd.DataFrame(rl_metrics)
                rl_df = rl_df[['round', 'success_rate', 'num_successful', 'unique_solved', 'total_rollouts', 'avg_cost', 'avg_steps']]
                rl_df.columns = ['Round', 'Success Rate', 'Solved', 'Unique', 'Total', 'Avg Cost', 'Avg Steps']
                rl_df['Success Rate'] = rl_df['Success Rate'].apply(lambda x: f"{x:.3f}")
                rl_df['Avg Cost'] = rl_df['Avg Cost'].apply(lambda x: f"{x/1e6:.2f}M")
                rl_df['Avg Steps'] = rl_df['Avg Steps'].apply(lambda x: f"{x:.1f}")
                st.dataframe(rl_df, use_container_width=True, hide_index=True)

                # Summary metrics for this lambda
                if baseline_metrics:
                    best_baseline = max(baseline_metrics, key=lambda x: x['success_rate'])
                    best_rl = max(rl_metrics, key=lambda x: x['success_rate'])
                    initial_rl = rl_metrics[0] if rl_metrics else None

                    summary_cols = st.columns(4)

                    with summary_cols[0]:
                        st.metric(
                            "Best Baseline",
                            f"{best_baseline['success_rate']:.3f}",
                            help=f"Strategy: {best_baseline['strategy']}"
                        )

                    with summary_cols[1]:
                        st.metric(
                            "Initial RL (Round 0)",
                            f"{initial_rl['success_rate']:.3f}" if initial_rl else "N/A",
                        )

                    with summary_cols[2]:
                        st.metric(
                            f"Best RL (Round {best_rl['round']})",
                            f"{best_rl['success_rate']:.3f}",
                        )

                    with summary_cols[3]:
                        if initial_rl:
                            improvement = best_rl['success_rate'] - initial_rl['success_rate']
                            st.metric(
                                "RL Improvement",
                                f"+{improvement:.3f}" if improvement >= 0 else f"{improvement:.3f}",
                                delta=f"{improvement*100:.1f}%"
                            )
            else:
                st.info(f"No RL training data available for λ={lambda_val:.2e}.")

        st.markdown("---")
