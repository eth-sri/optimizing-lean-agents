"""
Per-problem tab for RL Training Analysis.

Shows per-problem performance analysis with per-lambda breakdown
and baseline success rate comparison for sanity checking.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

from .utils import (
    load_round_per_problem,
    load_rollouts_cached,
    get_rollouts_file,
    format_budget_display,
)


# Repo root for resolving paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent

# Dataset-specific baseline paths
BASELINE_PATHS = {
    'minif2f': {
        '8b': REPO_ROOT / "dataset" / "minif2f_gpv2_8b_256pass_solved_num.csv",
        '32b': REPO_ROOT / "dataset" / "minif2f_32b_256pass_summary.csv",
        'format': 'csv',  # tab-separated CSV with sum/count columns
    },
    'putnam': {
        '8b': REPO_ROOT / "dataset" / "putnam_solved_8b_summary.csv",
        '32b': REPO_ROOT / "dataset" / "putnam_solved_32b_summary.csv",
        'format': 'csv',  # tab-separated CSV with sum/count columns
    },
}


def load_baseline_success_rates(dataset: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Load baseline per-attempt success rates from dataset-specific files.

    Args:
        dataset: Dataset name (e.g., 'minif2f', 'putnam'). If None, tries to detect.

    Returns dict: {problem_id: {'p_8b': float, 'p_32b': float}}
    """
    baselines = {}

    # Determine which dataset paths to use
    dataset_key = None
    if dataset:
        # Try to match dataset name to known datasets
        dataset_lower = dataset.lower()
        for key in BASELINE_PATHS:
            if key in dataset_lower:
                dataset_key = key
                break

    # Default to minif2f if no match
    if dataset_key is None:
        dataset_key = 'minif2f'

    paths = BASELINE_PATHS.get(dataset_key, BASELINE_PATHS['minif2f'])
    file_format = paths.get('format', 'csv')

    if file_format == 'csv':
        # Load from tab-separated CSV files (minif2f format)
        baseline_8b_path = paths['8b']
        baseline_32b_path = paths['32b']

        # Load 8B baseline
        if baseline_8b_path.exists():
            df_8b = pd.read_csv(baseline_8b_path, sep='\t')
            for _, row in df_8b.iterrows():
                problem_id = row['origin_problem_id']
                if problem_id not in baselines:
                    baselines[problem_id] = {}
                baselines[problem_id]['p_8b'] = row['sum'] / row['count']

        # Load 32B baseline
        if baseline_32b_path.exists():
            df_32b = pd.read_csv(baseline_32b_path, sep='\t')
            for _, row in df_32b.iterrows():
                problem_id = row['origin_problem_id']
                if problem_id not in baselines:
                    baselines[problem_id] = {}
                baselines[problem_id]['p_32b'] = row['sum'] / row['count']

    elif file_format == 'json':
        # Load from JSON files (putnam format)
        import json

        baseline_8b_path = paths['8b']
        baseline_32b_path = paths['32b']

        # Load 8B baseline
        if baseline_8b_path.exists():
            with open(baseline_8b_path, 'r') as f:
                data = json.load(f)
            for item in data:
                problem_id = item.get('origin_problem_id', item.get('problem_id'))
                if problem_id:
                    if problem_id not in baselines:
                        baselines[problem_id] = {}
                    # Compute success rate from results
                    results = item.get('results', [])
                    if results:
                        successes = sum(1 for r in results if r.get('complete', False))
                        baselines[problem_id]['p_8b'] = successes / len(results)

        # Load 32B baseline
        if baseline_32b_path.exists():
            with open(baseline_32b_path, 'r') as f:
                data = json.load(f)
            for item in data:
                problem_id = item.get('origin_problem_id', item.get('problem_id'))
                if problem_id:
                    if problem_id not in baselines:
                        baselines[problem_id] = {}
                    # Compute success rate from results
                    results = item.get('results', [])
                    if results:
                        successes = sum(1 for r in results if r.get('complete', False))
                        baselines[problem_id]['p_32b'] = successes / len(results)

    return baselines


def render_per_problem_tab(run_dir: Path, run_metadata: Dict, target_budget: Optional[float], dataset: Optional[str] = None):
    """Render per-problem analysis with per-lambda breakdown."""
    st.subheader(f"Per-Problem Analysis - {format_budget_display(target_budget)} Budget")

    # Load baseline success rates for sanity checking (dataset-specific)
    baselines = load_baseline_success_rates(dataset)
    if baselines:
        dataset_info = f" (dataset: {dataset})" if dataset else ""
        st.caption(f"Loaded baseline success rates for {len(baselines)} problems (p_8b, p_32b){dataset_info}")

    # Round selector
    available_rounds = run_metadata.get('rounds', [])
    if not available_rounds:
        st.warning("No rounds available.")
        return

    round_options = ["All Rounds"] + [f"Round {r}" for r in available_rounds]
    selected_round_option = st.selectbox(
        "Select Round",
        options=round_options,
        index=0,
        key="per_problem_round_select"
    )

    # Parse selected round
    if selected_round_option == "All Rounds":
        selected_rounds = available_rounds
    else:
        selected_round_id = int(selected_round_option.replace("Round ", ""))
        selected_rounds = [selected_round_id]

    # Load per-problem summaries - these have per-lambda breakdown
    # Structure: {problem_id: {budget_key: {lambda_str: {...metrics...}}}}
    per_problem_by_lambda = {}

    for round_id in selected_rounds:
        round_dir = run_dir / f"round{round_id}"
        per_problem_file = round_dir / "per_problem_summary.json"

        if per_problem_file.exists():
            import json
            with open(per_problem_file, 'r') as f:
                data = json.load(f)
                # Data is {problem_id: {budget_key: {lambda_str: metrics}}}
                for problem_id, budget_data in data.items():
                    if problem_id not in per_problem_by_lambda:
                        per_problem_by_lambda[problem_id] = {}

                    budget_key = format_budget_display(target_budget)
                    if budget_key in budget_data:
                        # Merge lambda data from this round
                        for lambda_str, metrics in budget_data[budget_key].items():
                            if lambda_str not in per_problem_by_lambda[problem_id]:
                                per_problem_by_lambda[problem_id][lambda_str] = metrics
                            else:
                                # If showing all rounds, we could aggregate, but for now just overwrite
                                per_problem_by_lambda[problem_id][lambda_str] = metrics

    if not per_problem_by_lambda:
        st.warning("No per-problem data available for this budget.")
        _render_fallback_from_rollouts(run_dir, selected_rounds, target_budget, baselines)
        return

    # Get all lambda values
    all_lambdas = set()
    for problem_data in per_problem_by_lambda.values():
        all_lambdas.update(problem_data.keys())
    all_lambdas = sorted(all_lambdas, key=lambda x: float(x))

    # Lambda selector
    selected_lambda = st.selectbox(
        "Select Lambda",
        options=["All"] + all_lambdas,
        index=0,
        key="per_problem_lambda_select"
    )

    # Build dataframe
    rows = []
    for problem_id, lambda_data in per_problem_by_lambda.items():
        if selected_lambda == "All":
            # Show one row per problem with aggregated stats across lambdas
            all_success_rates = []
            all_avg_costs = []
            total_rollouts = 0

            for lam, metrics in lambda_data.items():
                all_success_rates.append(metrics.get('success_rate', 0))
                all_avg_costs.append(metrics.get('avg_cost', 0))
                total_rollouts += metrics.get('num_rollouts', 0)

            row = {
                'Problem': problem_id,
                'Avg Success Rate': np.mean(all_success_rates) if all_success_rates else 0,
                'Max Success Rate': max(all_success_rates) if all_success_rates else 0,
                'Min Avg Cost': min(all_avg_costs) if all_avg_costs else 0,
                'Num Lambdas': len(lambda_data),
                'Total Rollouts': total_rollouts,
            }

            # Add baseline columns
            if problem_id in baselines:
                row['p_8b'] = baselines[problem_id].get('p_8b', None)
                row['p_32b'] = baselines[problem_id].get('p_32b', None)
            else:
                row['p_8b'] = None
                row['p_32b'] = None

            rows.append(row)
        else:
            # Show stats for selected lambda only
            if selected_lambda in lambda_data:
                metrics = lambda_data[selected_lambda]
                row = {
                    'Problem': problem_id,
                    'Success Rate': metrics.get('success_rate', 0),
                    'Num Successful': metrics.get('num_successful', 0),
                    'Num Rollouts': metrics.get('num_rollouts', 0),
                    'Avg Cost': metrics.get('avg_cost', 0),
                    'Avg Steps': metrics.get('avg_steps', 0),
                }

                # Add baseline columns
                if problem_id in baselines:
                    row['p_8b'] = baselines[problem_id].get('p_8b', None)
                    row['p_32b'] = baselines[problem_id].get('p_32b', None)
                else:
                    row['p_8b'] = None
                    row['p_32b'] = None

                rows.append(row)

    if rows:
        df = pd.DataFrame(rows)

        # Sort by success rate descending
        sort_col = 'Avg Success Rate' if selected_lambda == "All" else 'Success Rate'
        df = df.sort_values(sort_col, ascending=False)

        # Format percentage columns
        pct_cols = [c for c in df.columns if 'Rate' in c or c in ['p_8b', 'p_32b']]
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")

        # Format cost columns
        cost_cols = [c for c in df.columns if 'Cost' in c]
        for col in cost_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x/1e6:.2f}M" if pd.notna(x) and x > 0 else "-")

        st.dataframe(df, use_container_width=True, hide_index=True)

        # Summary stats
        st.markdown("---")
        st.markdown("### Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Problems", len(rows))
        with col2:
            if selected_lambda == "All":
                solved = sum(1 for r in rows if r.get('Max Success Rate', 0) > 0)
            else:
                solved = sum(1 for r in rows if r.get('Success Rate', 0) > 0)
            st.metric("Problems with >0% Success", solved)
        with col3:
            if baselines:
                in_baseline = sum(1 for r in rows if r.get('p_8b') is not None)
                st.metric("Problems with Baseline", in_baseline)
    else:
        st.info("No data for selected lambda.")


def _render_fallback_from_rollouts(run_dir: Path, selected_rounds: list, target_budget: Optional[float], baselines: Dict):
    """Fallback: load from rollouts directly if per_problem_summary.json doesn't exist."""
    st.info("Attempting to load from rollouts...")

    for round_id in selected_rounds[:1]:  # Just first selected round for now
        round_dir = run_dir / f"round{round_id}"
        rollouts_file = get_rollouts_file(round_dir)

        if rollouts_file is not None:
            rollouts = load_rollouts_cached(str(rollouts_file), target_budget)

            # Group by problem
            problems = defaultdict(list)
            for r in rollouts:
                problem_id = r.get('problem_id', r.get('origin_problem_id', 'unknown'))
                problems[problem_id].append(r)

            # Create summary
            problem_summary = []
            for problem_id, problem_rollouts in problems.items():
                num_solved = sum(1 for r in problem_rollouts if r.get('success', False))
                avg_cost = np.mean([r.get('final_cost', 0) for r in problem_rollouts])
                row = {
                    'Problem': problem_id,
                    'Solved': num_solved > 0,
                    'Success Rate': num_solved / len(problem_rollouts),
                    'Avg Cost': avg_cost,
                    'Rollouts': len(problem_rollouts)
                }

                # Add baseline columns
                if problem_id in baselines:
                    row['p_8b'] = baselines[problem_id].get('p_8b', None)
                    row['p_32b'] = baselines[problem_id].get('p_32b', None)
                else:
                    row['p_8b'] = None
                    row['p_32b'] = None

                problem_summary.append(row)

            if problem_summary:
                df = pd.DataFrame(problem_summary)
                df = df.sort_values('Success Rate', ascending=False)

                # Format percentage columns
                for col in ['Success Rate', 'p_8b', 'p_32b']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")

                st.dataframe(df, use_container_width=True, hide_index=True)
