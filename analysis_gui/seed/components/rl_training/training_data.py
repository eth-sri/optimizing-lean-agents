"""
Training Data Tab Component.

Transforms rollouts into a clean training data dataframe with features
defined in the YAML config as columns and success/cost as response variables.
"""
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .utils import (
    get_rollouts_file,
    load_rollouts_cached,
    budget_selector,
    lambda_selector,
    format_budget_display,
)
from .config_viewer import find_config_file, load_config
import json


# Feature registry - maps feature names to extraction functions
# Each function takes a step state dict and returns a float value
FEATURE_EXTRACTORS = {
    'num_attempts_made': lambda s: s.get('num_attempts_made', 0),
    'cost_so_far': lambda s: s.get('cost_so_far', 0),
    'avg_proof_length_full_8b': lambda s: s.get('avg_proof_length_full_8b', 0.0),
    'avg_proof_length_full_32b': lambda s: s.get('avg_proof_length_full_32b', 0.0),
    'avg_proof_length_attempt_8b': lambda s: s.get('avg_proof_length_attempt_8b', 0.0),
    'avg_proof_length_attempt_32b': lambda s: s.get('avg_proof_length_attempt_32b', 0.0),
    'num_attempt_8b_on_lemma': lambda s: s.get('num_attempt_8b_on_lemma', 0),
    'num_attempt_32b_on_lemma': lambda s: s.get('num_attempt_32b_on_lemma', 0),
    'num_full_proof_8b_used': lambda s: s.get('num_full_proof_8b_used', 0),
    'num_full_proof_32b_used': lambda s: s.get('num_full_proof_32b_used', 0),
    # Log-transformed attempt counts: log(1 + x) to handle zeros
    'log_num_full_proof_8b_used': lambda s: np.log1p(s.get('num_full_proof_8b_used', 0)),
    'log_num_full_proof_32b_used': lambda s: np.log1p(s.get('num_full_proof_32b_used', 0)),
    'is_lemma': lambda s: 1.0 if s.get('is_lemma', False) else 0.0,
    'proof_length': lambda s: s.get('proof_length', 0),
    'num_errors': lambda s: s.get('num_errors', 0),
}


def extract_training_data_from_rollout(
    rollout: Dict[str, Any],
    features: List[str],
    round_id: int
) -> List[Dict[str, Any]]:
    """
    Extract training data points from a single rollout.

    Each step in the rollout history becomes one data point with:
    - Features from the state before the action
    - Response variables: action_success and action_cost

    Args:
        rollout: Single rollout dictionary with history
        features: List of feature names to extract
        round_id: Training round this rollout is from

    Returns:
        List of data point dictionaries
    """
    data_points = []
    history = rollout.get('history', [])

    if not history:
        return data_points

    # Rollout-level response variables
    rollout_success = rollout.get('success', False)
    rollout_cost = rollout.get('final_cost', 0)

    prev_cost = 0
    for step_idx, step in enumerate(history):
        # Skip steps not marked for training (tree rollouts have this field)
        if 'include_in_training' in step and not step['include_in_training']:
            prev_cost = step.get('cost_so_far', prev_cost)
            continue

        # Get state before the action
        state = step.get('state_before', step.get('state', {}))
        if state is None:
            state = {}

        # Get action info
        action = step.get('action', {})
        action_type = action.get('action_type', 'unknown')
        if hasattr(action_type, 'name'):
            action_type = action_type.name

        # Get result info - handle both formats:
        # Format 1 (regular): result dict with 'success' and 'cost'
        # Format 2 (tree): 'action_success' at step level, cost from cost_so_far delta
        result = step.get('result', {})
        if 'action_success' in step:
            # Tree rollout format
            action_success = step.get('action_success', False)
        else:
            # Regular format
            action_success = result.get('success', False)

        # Get cost_so_far
        cost_so_far = step.get('cost_so_far', 0)

        # Calculate action_cost
        if 'step_cost' in step:
            action_cost = step['step_cost']
        elif result.get('cost'):
            action_cost = result['cost']
        else:
            # Calculate from cost_so_far delta
            action_cost = cost_so_far - prev_cost

        prev_cost = cost_so_far
        remaining_cost = rollout_cost - cost_so_far

        # Build data point - action type, features, and response variables
        data_point = {
            'action_type': action_type,
            'action_success': int(action_success),
            'action_cost': action_cost,
            'rollout_success': int(rollout_success),
            'rollout_cost': rollout_cost,
            'cost_so_far': cost_so_far,
            'remaining_cost': remaining_cost,
        }

        # Extract features
        for feature_name in features:
            if feature_name in FEATURE_EXTRACTORS:
                data_point[feature_name] = FEATURE_EXTRACTORS[feature_name](state)
            else:
                # Try direct extraction from state
                data_point[feature_name] = state.get(feature_name, 0.0)

        data_points.append(data_point)

    return data_points


def extract_rollout_level_data(
    rollout: Dict[str, Any],
    features: List[str],
    round_id: int
) -> Dict[str, Any]:
    """
    Extract rollout-level summary data (one row per rollout).

    This is a simpler representation where each rollout is one data point,
    using the final state features.

    Args:
        rollout: Single rollout dictionary
        features: List of feature names to extract
        round_id: Training round this rollout is from

    Returns:
        Single data point dictionary
    """
    final_cost = rollout.get('final_cost', 0)

    # Response variable only
    data_point = {
        'final_cost': final_cost,
    }

    # For rollout-level, use the final state or aggregate features
    # Try to get features directly from rollout summary fields
    for feature_name in features:
        # First check if it's directly in the rollout (summary format)
        if feature_name in rollout:
            data_point[feature_name] = rollout[feature_name]
        elif feature_name in FEATURE_EXTRACTORS:
            # For rollout level, use the last state if history exists
            history = rollout.get('history', [])
            if history:
                last_state = history[-1].get('state_before', history[-1].get('state', {})) or {}
                data_point[feature_name] = FEATURE_EXTRACTORS[feature_name](last_state)
            else:
                data_point[feature_name] = 0.0
        else:
            data_point[feature_name] = 0.0

    return data_point


def _format_budget_dirname(budget: Optional[float]) -> str:
    """Format budget as directory name (e.g., 8M, 16M, or 'unlimited')."""
    if budget is None:
        return "unlimited"
    return f"{int(budget / 1e6)}M"


def load_full_rollouts_from_directory(
    round_dir: Path,
    target_budget: Optional[float],
    progress_callback: Optional[callable] = None
) -> List[Dict]:
    """
    Load full rollouts (with history) from the nested directory structure.

    Supports two structures:
    - Regular: round_dir/rollouts/<problem_id>/<budget>/rollouts.json
    - Tree: round_dir/rollouts/<problem_id>/<budget>/switch_X_sample_Y.json

    Args:
        round_dir: Path to round directory
        target_budget: Budget to filter by
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        List of full rollout dictionaries with history
    """
    rollouts_base = round_dir / "rollouts"
    if not rollouts_base.exists():
        return []

    all_rollouts = []
    budget_dirname = _format_budget_dirname(target_budget)

    # Get list of problem directories
    problem_dirs = [d for d in rollouts_base.iterdir() if d.is_dir()]
    total = len(problem_dirs)

    # Iterate over problem directories
    for idx, problem_dir in enumerate(problem_dirs):
        if progress_callback:
            progress_callback(idx, total)

        # Look for the budget subdirectory
        budget_dir = problem_dir / budget_dirname
        if not budget_dir.exists():
            continue

        # Try loading rollouts.json first (regular format)
        rollouts_file = budget_dir / "rollouts.json"
        if rollouts_file.exists():
            try:
                with open(rollouts_file, 'r') as f:
                    rollouts = json.load(f)
                    all_rollouts.extend(rollouts)
            except Exception:
                pass
        else:
            # Try loading individual switch_*.json files (tree rollouts format)
            for json_file in budget_dir.glob("switch_*.json"):
                try:
                    with open(json_file, 'r') as f:
                        rollout = json.load(f)
                        all_rollouts.append(rollout)
                except Exception:
                    pass

    if progress_callback:
        progress_callback(total, total)

    return all_rollouts


@st.cache_data(ttl=300, show_spinner=False)
def load_and_transform_training_data(
    run_dir: str,
    rounds: Tuple[int, ...],
    features: Tuple[str, ...],
    target_budget: Optional[float],
    granularity: str = "rollout",
    include_tree_rollouts: bool = False
) -> pd.DataFrame:
    """
    Load rollouts and transform into training data dataframe.

    This function caches the result and clears raw rollout data from memory.

    Args:
        run_dir: Path to run directory (as string for caching)
        rounds: Tuple of round IDs to load
        features: Tuple of feature names to extract
        target_budget: Budget to filter by (None = unlimited)
        granularity: "step" for per-step data, "rollout" for per-rollout
        include_tree_rollouts: Whether to include tree rollouts

    Returns:
        DataFrame with features and response variables
    """
    run_path = Path(run_dir)
    all_data_points = []

    # Load from selected rounds
    for round_id in rounds:
        round_dir = run_path / f"round{round_id}"

        if granularity == "step":
            # For step-level, load full rollouts from nested directory structure
            rollouts = load_full_rollouts_from_directory(round_dir, target_budget)
        else:
            # For rollout-level, use the summary file
            rollouts_file = get_rollouts_file(round_dir)
            if rollouts_file is None:
                continue
            rollouts = load_rollouts_cached(str(rollouts_file), target_budget)

        # Extract data points
        for rollout in rollouts:
            if granularity == "step":
                data_points = extract_training_data_from_rollout(
                    rollout, list(features), round_id
                )
                all_data_points.extend(data_points)
            else:
                data_point = extract_rollout_level_data(
                    rollout, list(features), round_id
                )
                all_data_points.append(data_point)

        # Clear rollouts from memory after processing
        del rollouts

    # Load from tree rollouts if requested
    if include_tree_rollouts:
        tree_rollouts_dir = run_path / "tree_rollouts"

        if granularity == "step":
            rollouts = load_full_rollouts_from_directory(tree_rollouts_dir, target_budget)
        else:
            rollouts_file = get_rollouts_file(tree_rollouts_dir)
            if rollouts_file is not None:
                rollouts = load_rollouts_cached(str(rollouts_file), target_budget)
            else:
                rollouts = []

        for rollout in rollouts:
            if granularity == "step":
                data_points = extract_training_data_from_rollout(
                    rollout, list(features), -1  # Use -1 for tree rollouts
                )
                all_data_points.extend(data_points)
            else:
                data_point = extract_rollout_level_data(
                    rollout, list(features), -1
                )
                all_data_points.append(data_point)

        if rollouts:
            del rollouts

    # Force garbage collection to free memory
    gc.collect()

    if not all_data_points:
        return pd.DataFrame()

    return pd.DataFrame(all_data_points)


def render_training_data_tab(
    run_dir: Path,
    run_metadata: Dict[str, Any],
    budgets: List[Optional[float]]
):
    """
    Render the Training Data tab.

    Loads rollouts and transforms them into a clean training data dataframe.

    Args:
        run_dir: Path to the RL training run directory
        run_metadata: Metadata about the run
        budgets: List of available budgets
    """
    st.subheader("Training Data")

    st.markdown("""
    This tab transforms rollout data into a clean training dataset suitable for analysis.
    Features are extracted based on the config file, and response variables include
    success and cost metrics.
    """)

    # Try to load config for feature list
    config_file = find_config_file(run_dir)
    if config_file:
        config = load_config(config_file)
        config_features = config.get('active_features', []) if config else []
    else:
        config_features = []

    # Settings columns
    col1, col2 = st.columns(2)

    with col1:
        target_budget = budget_selector(budgets, "training_data")

    with col2:
        granularity = st.radio(
            "Data Granularity",
            options=["rollout", "step"],
            index=1,
            help="Rollout: one row per rollout. Step: one row per action taken.",
            horizontal=True
        )

    # Data source selection
    available_rounds = run_metadata.get('rounds', [])
    has_tree_rollouts = (run_dir / "tree_rollouts" / "rollouts").exists()

    col1, col2 = st.columns(2)

    with col1:
        # Build data source options
        source_options = [f"Round {r}" for r in available_rounds]
        if has_tree_rollouts:
            source_options.append("Tree Rollouts")

        selected_sources = st.multiselect(
            "Data Source",
            options=source_options,
            default=source_options[:1] if source_options else [],
            help="Select which rounds or tree rollouts to load data from"
        )

    with col2:
        # Feature selection
        all_available_features = list(FEATURE_EXTRACTORS.keys())
        default_features = config_features if config_features else ['num_attempts_made']

        # Ensure default features are in available features
        default_features = [f for f in default_features if f in all_available_features]
        if not default_features:
            default_features = ['num_attempts_made']

        selected_features = st.multiselect(
            "Features to Extract",
            options=all_available_features,
            default=default_features,
            help="Select which features to include in the training data"
        )

    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    if not selected_sources:
        st.warning("Please select at least one data source.")
        return

    # Parse selected sources
    selected_rounds = []
    include_tree_rollouts = False
    for source in selected_sources:
        if source == "Tree Rollouts":
            include_tree_rollouts = True
        elif source.startswith("Round "):
            selected_rounds.append(int(source.replace("Round ", "")))

    # Load button
    if st.button("Load Training Data", type="primary"):
        progress_bar = st.progress(0, text="Loading rollout data...")

        # Count total sources for progress
        total_sources = len(selected_rounds) + (1 if include_tree_rollouts else 0)
        current_source = 0

        all_data_points = []
        run_path = Path(run_dir)

        # Load from selected rounds
        for round_id in selected_rounds:
            progress_bar.progress(
                current_source / total_sources,
                text=f"Loading Round {round_id}..."
            )
            round_dir = run_path / f"round{round_id}"

            if granularity == "step":
                rollouts = load_full_rollouts_from_directory(round_dir, target_budget)
            else:
                rollouts_file = get_rollouts_file(round_dir)
                if rollouts_file:
                    rollouts = load_rollouts_cached(str(rollouts_file), target_budget)
                else:
                    rollouts = []

            for rollout in rollouts:
                if granularity == "step":
                    data_points = extract_training_data_from_rollout(
                        rollout, list(selected_features), round_id
                    )
                    all_data_points.extend(data_points)
                else:
                    data_point = extract_rollout_level_data(
                        rollout, list(selected_features), round_id
                    )
                    all_data_points.append(data_point)

            current_source += 1

        # Load from tree rollouts if requested
        if include_tree_rollouts:
            progress_bar.progress(
                current_source / total_sources,
                text="Loading Tree Rollouts..."
            )
            tree_rollouts_dir = run_path / "tree_rollouts"

            if granularity == "step":
                rollouts = load_full_rollouts_from_directory(tree_rollouts_dir, target_budget)
            else:
                rollouts_file = get_rollouts_file(tree_rollouts_dir)
                if rollouts_file:
                    rollouts = load_rollouts_cached(str(rollouts_file), target_budget)
                else:
                    rollouts = []

            for rollout in rollouts:
                if granularity == "step":
                    data_points = extract_training_data_from_rollout(
                        rollout, list(selected_features), -1
                    )
                    all_data_points.extend(data_points)
                else:
                    data_point = extract_rollout_level_data(
                        rollout, list(selected_features), -1
                    )
                    all_data_points.append(data_point)

        progress_bar.progress(1.0, text="Done!")

        # Create dataframe
        if all_data_points:
            df = pd.DataFrame(all_data_points)
        else:
            df = pd.DataFrame()

        # Clear memory
        gc.collect()

        # Store in session state
        st.session_state['training_data_df'] = df
        st.session_state['training_data_features'] = selected_features
        st.session_state['training_data_granularity'] = granularity

        progress_bar.empty()

    # Display the data if loaded
    if 'training_data_df' in st.session_state:
        df = st.session_state['training_data_df']

        if df.empty:
            st.warning("No data loaded. Check that rollouts exist for the selected budget.")
            return

        st.success(f"Loaded {len(df):,} data points")

        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            if 'action_success' in df.columns:
                success_rate = df['action_success'].mean() * 100
                st.metric("Action Success Rate", f"{success_rate:.1f}%")
            else:
                avg_cost = df['final_cost'].mean() if 'final_cost' in df.columns else 0
                st.metric("Avg Cost", f"{avg_cost/1e6:.2f}M")
        with col3:
            if 'action_cost' in df.columns:
                avg_cost = df['action_cost'].mean()
                st.metric("Avg Action Cost", f"{avg_cost/1e6:.2f}M")
            elif 'final_cost' in df.columns:
                avg_cost = df['final_cost'].mean()
                st.metric("Avg Final Cost", f"{avg_cost/1e6:.2f}M")

        # Show data shape and columns
        st.markdown("### Data Schema")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Feature Columns:**")
            for feat in st.session_state.get('training_data_features', []):
                if feat in df.columns:
                    st.write(f"- `{feat}`")

        with col2:
            st.markdown("**Response Columns:**")
            if st.session_state.get('training_data_granularity') == 'step':
                st.write("- `action_type`")
                st.write("- `action_success`")
                st.write("- `action_cost`")
                st.write("- `rollout_success`")
                st.write("- `rollout_cost`")
                st.write("- `cost_so_far`")
                st.write("- `remaining_cost`")
            else:
                st.write("- `final_cost`")

        # Data preview
        st.markdown("### Data Preview")
        st.dataframe(df.head(100), use_container_width=True)

        # Basic statistics for numeric columns
        st.markdown("### Feature Statistics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        # Plots for step-level data
        if st.session_state.get('training_data_granularity') == 'step' and 'num_attempts_made' in df.columns:
            st.markdown("### Success & Cost vs Attempts")

            # Filter to only 8B and 32B actions
            df_plot = df[df['action_type'].isin(['FULL_PROOF_8B', 'FULL_PROOF_32B'])].copy()

            if not df_plot.empty:
                # Group by action_type and num_attempts_made
                grouped = df_plot.groupby(['action_type', 'num_attempts_made']).agg({
                    'rollout_success': 'mean',
                    'action_cost': 'mean',
                    'remaining_cost': 'mean',
                }).reset_index()

                # Rename for cleaner display
                grouped['action_type'] = grouped['action_type'].replace({
                    'FULL_PROOF_8B': '8B',
                    'FULL_PROOF_32B': '32B'
                })

                import plotly.express as px

                col1, col2 = st.columns(2)

                with col1:
                    # Success rate plot
                    fig_success = px.line(
                        grouped,
                        x='num_attempts_made',
                        y='rollout_success',
                        color='action_type',
                        markers=True,
                        title='Avg Rollout Success vs Attempts Made',
                        labels={
                            'num_attempts_made': 'Attempts Made',
                            'rollout_success': 'Avg Rollout Success',
                            'action_type': 'Model'
                        }
                    )
                    fig_success.update_layout(
                        yaxis_tickformat='.0%',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    st.plotly_chart(fig_success, use_container_width=True)

                with col2:
                    # Remaining cost plot
                    grouped['remaining_cost_M'] = grouped['remaining_cost'] / 1e6
                    fig_remaining = px.line(
                        grouped,
                        x='num_attempts_made',
                        y='remaining_cost_M',
                        color='action_type',
                        markers=True,
                        title='Avg Remaining Cost vs Attempts Made',
                        labels={
                            'num_attempts_made': 'Attempts Made',
                            'remaining_cost_M': 'Avg Remaining Cost (M SFLOPs)',
                            'action_type': 'Model'
                        }
                    )
                    fig_remaining.update_layout(
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    st.plotly_chart(fig_remaining, use_container_width=True)

                # Action cost plot (full width)
                grouped['action_cost_M'] = grouped['action_cost'] / 1e6
                fig_action = px.line(
                    grouped,
                    x='num_attempts_made',
                    y='action_cost_M',
                    color='action_type',
                    markers=True,
                    title='Avg Action Cost vs Attempts Made',
                    labels={
                        'num_attempts_made': 'Attempts Made',
                        'action_cost_M': 'Avg Action Cost (M SFLOPs)',
                        'action_type': 'Model'
                    }
                )
                fig_action.update_layout(
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                st.plotly_chart(fig_action, use_container_width=True)

                # Logistic regression curves
                st.markdown("### Logistic Regression: P(Success) vs Attempts")

                from sklearn.linear_model import LogisticRegression
                import plotly.graph_objects as go

                # Prepare data
                df_scatter = df_plot.copy()
                df_scatter['action_type'] = df_scatter['action_type'].replace({
                    'FULL_PROOF_8B': '8B',
                    'FULL_PROOF_32B': '32B'
                })

                col1, col2 = st.columns(2)

                with col1:
                    fig_logistic = go.Figure()
                    colors = {'8B': '#636EFA', '32B': '#EF553B'}

                    for action_type in ['8B', '32B']:
                        df_action = df_scatter[df_scatter['action_type'] == action_type]
                        if df_action.empty:
                            continue

                        X = df_action[['num_attempts_made']].values
                        y = df_action['rollout_success'].values

                        # Fit logistic regression
                        if len(np.unique(y)) > 1:  # Need both classes
                            model = LogisticRegression()
                            model.fit(X, y)

                            # Generate smooth curve
                            x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                            y_pred = model.predict_proba(x_range)[:, 1]

                            fig_logistic.add_trace(go.Scatter(
                                x=x_range.flatten(),
                                y=y_pred,
                                mode='lines',
                                name=action_type,
                                line=dict(color=colors[action_type], width=3)
                            ))

                    fig_logistic.update_layout(
                        title='P(Rollout Success) vs Attempts Made',
                        xaxis_title='Attempts Made',
                        yaxis_title='P(Rollout Success)',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        yaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig_logistic, use_container_width=True)

                with col2:
                    # Histogram of num_attempts_made counts
                    fig_hist = px.histogram(
                        df_scatter,
                        x='num_attempts_made',
                        color='action_type',
                        barmode='group',
                        title='Distribution of Attempts Made',
                        labels={
                            'num_attempts_made': 'Attempts Made',
                            'count': 'Count',
                            'action_type': 'Model'
                        }
                    )
                    fig_hist.update_layout(
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Also show the grouped data table
                with st.expander("View Grouped Data"):
                    st.dataframe(grouped, use_container_width=True)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"training_data_{granularity}_{format_budget_display(target_budget)}.csv",
            mime="text/csv"
        )
