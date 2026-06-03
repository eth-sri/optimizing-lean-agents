"""
RL Training Analysis Viewer Component.

Provides interactive analysis of RL training rollouts, comparing
RL training rounds against baseline strategies.

This is the main entry point that imports tab components from the
rl_training subpackage.
"""
import streamlit as st
from pathlib import Path

from .rl_training import (
    get_available_rl_runs,
    get_available_baseline_datasets,
    get_available_baselines,
    load_baseline_data,
    load_rl_run_metadata,
    load_rollouts_cached,
    get_rollouts_file,
    budget_selector,
    lambda_selector,
    render_overview_tab,
    render_learning_curves_tab,
    render_solve_types_tab,
    render_per_problem_tab,
    render_lambda_analysis_tab,
    render_rollout_detail_tab,
    render_model_parameters_tab,
    render_cost_quality_tab,
    render_config_tab,
    render_training_data_tab,
)
from .rl_training.baseline_model_analysis import render_baseline_model_analysis_tab, render_tree_rollouts_calibration_tab
from .rl_training.utils import format_budget_display


def render_rl_training_viewer(base_dir: Path):
    """
    Main render function for RL Training Analysis view.

    Args:
        base_dir: Base directory containing the rollouts folder
    """
    st.header("RL Training Analysis")

    # Find rollouts directory - check various possible locations
    # A valid rollouts dir should have a 'baselines' subfolder or timestamped run folders
    rollouts_dir = None
    possible_paths = [
        base_dir / "rollouts",
        base_dir / "dump" / "rollouts",
        base_dir.parent / "rollouts" if base_dir.name == "minified" else None,
        # For combined_dump structure: traverse up to find rollouts sibling
        base_dir.parent / "rollouts" if base_dir.parent else None,
        base_dir.parent.parent / "rollouts" if base_dir.parent and base_dir.parent.parent else None,
    ]

    # Also check if we're in a timestamped results dir (YYYY/MM/DD/HHMMSS)
    # and need to go up multiple levels to combined_dump
    current = base_dir
    for _ in range(6):  # Go up max 6 levels
        rollouts_candidate = current / "rollouts"
        if rollouts_candidate.exists():
            possible_paths.append(rollouts_candidate)
        if current.parent and current.parent != current:
            current = current.parent
        else:
            break

    # Check paths - prefer ones with 'baselines' subfolder (indicates root rollouts dir)
    for path in possible_paths:
        if path and path.exists() and (path / "baselines").exists():
            rollouts_dir = path
            break

    # Fallback: use first existing path if no baselines found
    if rollouts_dir is None:
        for path in possible_paths:
            if path and path.exists():
                rollouts_dir = path
                break

    # Final fallback: hardcoded default
    default_rollouts = Path("scratch/dump/combined_dump/rollouts")

    if rollouts_dir is None or not (rollouts_dir / "baselines").exists():
        if default_rollouts.exists():
            rollouts_dir = default_rollouts

    if rollouts_dir is None:
        st.warning("No rollouts directory found. Please ensure the selected run contains a 'rollouts' folder.")

        # Allow manual path input
        manual_path = st.text_input(
            "Manual rollouts directory path",
            value=str(default_rollouts) if default_rollouts.exists() else "",
            placeholder="/path/to/rollouts"
        )
        if manual_path and Path(manual_path).exists():
            rollouts_dir = Path(manual_path)
        else:
            return

    col_info, col_clear = st.columns([4, 1])
    with col_info:
        st.info(f"Rollouts directory: `{rollouts_dir}`")
    with col_clear:
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.rerun()

    # Get available RL runs
    rl_runs = get_available_rl_runs(rollouts_dir)

    # Get available baseline datasets
    baseline_datasets = get_available_baseline_datasets(rollouts_dir)

    if not rl_runs:
        st.warning("No RL training runs found in the rollouts directory.")
        if baseline_datasets:
            st.info(f"Found {len(baseline_datasets)} baseline datasets: {', '.join(baseline_datasets)}")
        return

    # Run selection, dataset selection, and baseline checkboxes
    col_run, col_dataset, col_baselines, col_load = st.columns([2, 1.5, 2.5, 1])

    with col_run:
        run_options = [name for name, _ in rl_runs]
        selected_run_name = st.selectbox(
            "Select RL Training Run",
            options=run_options,
            index=0
        )

    # Dataset selection
    with col_dataset:
        if baseline_datasets:
            selected_dataset = st.selectbox(
                "Baseline Dataset",
                options=baseline_datasets,
                index=0,
                key="baseline_dataset"
            )
        else:
            selected_dataset = None
            st.caption("No datasets found")

    # Get baselines for the selected dataset
    baselines = get_available_baselines(rollouts_dir, selected_dataset) if selected_dataset else []

    # Baseline selection checkboxes
    with col_baselines:
        st.write("**Baselines to compare:**")
        if baselines:
            selected_baselines = []
            for baseline in baselines:
                if st.checkbox(baseline, value=False, key=f"baseline_{selected_dataset}_{baseline}"):
                    selected_baselines.append(baseline)
        else:
            st.caption("No baselines found")
            selected_baselines = []

    # Find selected run path
    selected_run_path = None
    for name, path in rl_runs:
        if name == selected_run_name:
            selected_run_path = path
            break

    if selected_run_path is None:
        st.error("Selected run not found.")
        return

    # Load run metadata (lightweight, always load)
    run_metadata = load_rl_run_metadata(selected_run_path)

    # Get all budgets (None means unlimited/no budget constraint)
    if run_metadata['iteration_results']:
        budgets = run_metadata['iteration_results'].get('budgets')
        # Handle None or empty - means unlimited budget
        if not budgets:
            budgets = [None]  # Represents unlimited
    else:
        budgets = [None]  # Default to unlimited if no metadata

    # Load Data button
    with col_load:
        st.write("")  # Spacing
        load_clicked = st.button("Load Data", type="primary")

    # Show run info
    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("Training Rounds", len(run_metadata['rounds']))
    if run_metadata['iteration_results']:
        iter_results = run_metadata['iteration_results']
        with info_cols[1]:
            budget_strs = ["unlimited" if b is None else f"{b/1e6:.0f}M" for b in budgets]
            st.metric("Budgets", ', '.join(budget_strs))
        with info_cols[2]:
            lambdas = iter_results.get('lambda_values', [])
            st.metric("Lambda values", len(lambdas))
        with info_cols[3]:
            num_problems = iter_results.get('num_problems', 'N/A')
            st.metric("Problems", num_problems)

    st.markdown("---")

    # Create cache key based on run + dataset + selected baselines
    baselines_key = "_".join(sorted(selected_baselines)) if selected_baselines else "none"
    dataset_key = selected_dataset if selected_dataset else "none"
    cache_key = f"{selected_run_path}_{dataset_key}_{baselines_key}"

    # Check if data needs to be loaded
    if f"loaded_{cache_key}" not in st.session_state:
        if not load_clicked:
            st.info("Select baselines to compare and click **Load Data** to begin analysis.")
            return

        # Load data with progress bar
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            st.info("Loading rollout data...")
            progress_bar = st.progress(0, text="Initializing...")

            # Total items: (rounds + selected_baselines) * budgets
            total_items = (len(run_metadata['rounds']) + len(selected_baselines)) * len(budgets)
            loaded = 0

            for budget in budgets:
                for round_id in run_metadata['rounds']:
                    progress_bar.progress(
                        loaded / total_items if total_items > 0 else 1,
                        text=f"Loading round {round_id} @ {format_budget_display(budget)}..."
                    )
                    round_dir = selected_run_path / f"round{round_id}"
                    rollouts_file = get_rollouts_file(round_dir)
                    if rollouts_file is not None:
                        _ = load_rollouts_cached(str(rollouts_file), budget)
                    loaded += 1

                # Load only selected baselines
                for strategy in selected_baselines:
                    progress_bar.progress(
                        loaded / total_items if total_items > 0 else 1,
                        text=f"Loading {strategy} @ {format_budget_display(budget)}..."
                    )
                    baseline_data = load_baseline_data(rollouts_dir, strategy, selected_dataset)
                    if baseline_data and 'rollouts_path' in baseline_data:
                        _ = load_rollouts_cached(str(baseline_data['rollouts_path']), budget)
                    loaded += 1

            progress_bar.progress(1.0, text="Done!")

        st.session_state[f"loaded_{cache_key}"] = True
        st.session_state["active_baselines"] = selected_baselines
        st.session_state["active_dataset"] = selected_dataset
        progress_placeholder.empty()
        st.rerun()

    # Use the baselines and dataset that were loaded (stored in session state)
    active_baselines = st.session_state.get("active_baselines", selected_baselines)
    active_dataset = st.session_state.get("active_dataset", selected_dataset)

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "📊 Overview",
        "📈 Learning Curves",
        "🎯 Solve Types",
        "📋 Per-Problem",
        "λ Lambda",
        "🔍 Rollout Detail",
        "🧠 Model Parameters",
        "💰 Cost/Quality",
        "📦 Training Data",
        "🎯 Model Calibration",
        "🌳 Tree Rollouts Calib",
        "⚙️ Config"
    ])

    with tab1:
        # Overview shows ALL budgets
        render_overview_tab(rollouts_dir, selected_run_path, run_metadata, active_baselines, budgets, active_dataset)

    with tab2:
        # Budget selector for this tab
        target_budget = budget_selector(budgets, "learning_curves")
        render_learning_curves_tab(rollouts_dir, selected_run_path, run_metadata, active_baselines, target_budget, active_dataset)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            target_budget = budget_selector(budgets, "solve_types")
        with col2:
            iter_results = run_metadata.get('iteration_results', {})
            lambda_values = iter_results.get('lambda_values', [])
            target_lambda = lambda_selector(lambda_values, "solve_types", include_all=True)
        render_solve_types_tab(selected_run_path, run_metadata, target_budget, target_lambda)

    with tab4:
        target_budget = budget_selector(budgets, "per_problem")
        render_per_problem_tab(selected_run_path, run_metadata, target_budget, active_dataset)

    with tab5:
        target_budget = budget_selector(budgets, "lambda")
        render_lambda_analysis_tab(run_metadata, target_budget)

    with tab6:
        target_budget = budget_selector(budgets, "rollout_detail")
        render_rollout_detail_tab(selected_run_path, run_metadata, target_budget)

    with tab7:
        render_model_parameters_tab(selected_run_path, run_metadata)

    with tab8:
        render_cost_quality_tab(rollouts_dir, selected_run_path, run_metadata, active_dataset)

    with tab9:
        render_training_data_tab(selected_run_path, run_metadata, budgets)

    with tab10:
        render_baseline_model_analysis_tab(selected_run_path, run_metadata)

    with tab11:
        render_tree_rollouts_calibration_tab(selected_run_path)

    with tab12:
        render_config_tab(selected_run_path, run_metadata)
