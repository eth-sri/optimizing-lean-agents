"""
Component for comparing problems across multiple runs.

Provides a user interface to select two runs and compare their results.
"""
import streamlit as st
from pathlib import Path
from typing import Optional
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from folder_browser import get_available_runs, format_runs_for_dropdown
from comparison import load_run, create_comparison_table


def calculate_metrics(problems, run_path=None):
    """Calculate pipeline metrics for a run."""
    from pathlib import Path
    import json

    metrics = {
        "total_problems": len(problems),
        "breakdowns_with_parsed": 0,
        "breakdowns_with_full_formalization": 0,
        "problems_with_full_formalization": 0,
        "problems_with_valid_theorem": 0,
        "problems_solved": 0,
        "total_attempted_formalizations": 0,
    }

    # Load validation results if run_path provided
    # Note: None means no validation file (count all formalized as valid)
    #       {} or dict means validation file exists (check validation data)
    validation_results = None
    if run_path:
        validation_file = Path(run_path) / "round0" / "formalizer" / "validation_results.json"
        if validation_file.exists():
            validation_results = {}
            try:
                with open(validation_file, 'r') as f:
                    validation_data = json.load(f)
                    # Group by problem_id
                    for entry in validation_data:
                        problem_id = entry.get('problem_id')
                        if problem_id not in validation_results:
                            validation_results[problem_id] = []
                        validation_results[problem_id].append(entry)
            except Exception:
                pass

    for problem in problems:
        has_valid_theorem = False
        has_any_fully_validated = False

        for breakdown in problem.breakdowns.values():
            # Count all attempted formalizations
            metrics["total_attempted_formalizations"] += 1

            # Check if parsed
            if breakdown.parsed_breakdown is not None:
                # parsed_breakdown is now OOP ParsedBreakdown - simply check if it exists
                metrics["breakdowns_with_parsed"] += 1

            # Check if fully formalized (has theorem prover attempts)
            if breakdown.theorem_prover_results and breakdown.theorem_prover_results.get('attempts', []):
                metrics["breakdowns_with_full_formalization"] += 1
                has_any_fully_validated = True

            # Check for valid theorem
            if breakdown.theorem_prover_results:
                attempts = breakdown.theorem_prover_results.get('attempts', [])
                if attempts:
                    for attempt in attempts:
                        comp_result = attempt.get('data', {}).get('compilation_result', {})
                        if comp_result.get('pass', False) and comp_result.get('complete', False):
                            has_valid_theorem = True
                            break

        # Count problems with at least one fully validated formalization
        if has_any_fully_validated:
            metrics["problems_with_full_formalization"] += 1

        # Count problems with valid theorem
        if has_valid_theorem:
            metrics["problems_with_valid_theorem"] += 1

        # Count solved problems
        if problem.is_solved():
            metrics["problems_solved"] += 1

    return metrics


def render_comparison_viewer() -> None:
    """
    Render a comparison interface for analyzing differences between two runs.
    """
    st.header("🔄 Compare Runs")

    # Get available runs
    runs = get_available_runs()

    if not runs:
        st.warning("No runs found in scratch/results/combined directory")
        return

    # Format runs for dropdown
    display_names, path_mapping = format_runs_for_dropdown(runs)

    # Create two columns for run selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Run 1")
        selected_run1_display = st.selectbox(
            "Select first run",
            options=display_names,
            index=0,
            key="run1_select",
            help="Choose the first run to compare"
        )
        selected_run1 = path_mapping[selected_run1_display] if selected_run1_display else None

    with col2:
        st.subheader("Run 2")
        selected_run2_display = st.selectbox(
            "Select second run",
            options=display_names,
            index=1 if len(display_names) > 1 else 0,
            key="run2_select",
            help="Choose the second run to compare"
        )
        selected_run2 = path_mapping[selected_run2_display] if selected_run2_display else None

    # Check if both runs are the same
    if selected_run1 == selected_run2:
        st.warning("Please select two different runs to compare")
        return

    # Load button
    if not st.button("Load Results", type="primary", use_container_width=True):
        return

    # Load both runs
    st.info("Loading runs...")

    col1, col2 = st.columns(2)

    with col1:
        result1 = load_run(selected_run1, round_num=0)
        if result1:
            _, problems1 = result1
            st.success(f"✓ Loaded {len(problems1)} problems from Run 1")
        else:
            st.error("Failed to load Run 1")
            return

    with col2:
        result2 = load_run(selected_run2, round_num=0) if selected_run2 else None
        if result2:
            _, problems2 = result2
            st.success(f"✓ Loaded {len(problems2)} problems from Run 2")
        else:
            st.error("Failed to load Run 2")
            return

    # Calculate metrics for both runs
    metrics1 = calculate_metrics(problems1, selected_run1)
    metrics2 = calculate_metrics(problems2, selected_run2)

    # Display metrics side by side
    st.markdown("---")
    st.subheader("📊 Pipeline Metrics (Run 1)")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            "Breakdowns Parsed",
            metrics1["breakdowns_with_parsed"],
            f"{metrics1['breakdowns_with_parsed']}/{metrics1['total_attempted_formalizations']}"
        )

    with metric_col2:
        st.metric(
            "Breakdowns Formalized",
            metrics1["breakdowns_with_full_formalization"],
            f"{metrics1['breakdowns_with_full_formalization']}/{metrics1['breakdowns_with_parsed'] if metrics1['breakdowns_with_parsed'] > 0 else metrics1['total_attempted_formalizations']}"
        )

    with metric_col3:
        st.metric(
            "Problems with Theorem Proven",
            metrics1["problems_with_valid_theorem"],
            f"{metrics1['problems_with_valid_theorem']}/{metrics1['total_problems']}"
        )

    with metric_col4:
        st.metric(
            "Problems Solved",
            metrics1["problems_solved"],
            f"{metrics1['problems_solved']}/{metrics1['total_problems']}"
        )

    # Run 2 metrics
    st.subheader("📊 Pipeline Metrics (Run 2)")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            "Breakdowns Parsed",
            metrics2["breakdowns_with_parsed"],
            f"{metrics2['breakdowns_with_parsed']}/{metrics2['total_attempted_formalizations']}"
        )

    with metric_col2:
        st.metric(
            "Breakdowns Formalized",
            metrics2["breakdowns_with_full_formalization"],
            f"{metrics2['breakdowns_with_full_formalization']}/{metrics2['breakdowns_with_parsed'] if metrics2['breakdowns_with_parsed'] > 0 else metrics2['total_attempted_formalizations']}"
        )

    with metric_col3:
        st.metric(
            "Problems with Theorem Proven",
            metrics2["problems_with_valid_theorem"],
            f"{metrics2['problems_with_valid_theorem']}/{metrics2['total_problems']}"
        )

    with metric_col4:
        st.metric(
            "Problems Solved",
            metrics2["problems_solved"],
            f"{metrics2['problems_solved']}/{metrics2['total_problems']}"
        )

    # Load validation results for both runs
    from pathlib import Path

    validation_results1 = {}
    validation_results2 = {}

    validation_file1 = Path(selected_run1) / "round0" / "formalizer" / "validation_results.json"
    if validation_file1.exists():
        try:
            import json
            with open(validation_file1, 'r') as f:
                validation_data = json.load(f)
                for entry in validation_data:
                    problem_id = entry.get('problem_id')
                    if problem_id not in validation_results1:
                        validation_results1[problem_id] = []
                    validation_results1[problem_id].append(entry)
        except Exception:
            pass

    validation_file2 = Path(selected_run2) / "round0" / "formalizer" / "validation_results.json"
    if validation_file2.exists():
        try:
            import json
            with open(validation_file2, 'r') as f:
                validation_data = json.load(f)
                for entry in validation_data:
                    problem_id = entry.get('problem_id')
                    if problem_id not in validation_results2:
                        validation_results2[problem_id] = []
                    validation_results2[problem_id].append(entry)
        except Exception:
            pass

    # Create comparison table
    comparison_data = create_comparison_table(problems1, problems2, validation_results1 if validation_results1 else None, validation_results2 if validation_results2 else None)

    if not comparison_data:
        st.warning("No problems found to compare")
        return

    # Display comparison table
    st.markdown("---")
    st.subheader("📋 Problem-by-Problem Comparison")

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Display with styling
    st.dataframe(
        df,
        use_container_width=True,
        height=min(600, len(df) * 40 + 50)
    )

    # Comparison summary
    st.markdown("---")
    st.subheader("📈 Comparison Summary")

    comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)

    with comp_col1:
        st.metric(
            "📊 Total Problems",
            len(df),
        )

    with comp_col2:
        # Count Run 1 problems where at least one breakdown was solved (check if not "N/A" and not "0/...")
        run1_solved = sum(1 for row in comparison_data if row.get("Run 1 Solved") != "N/A" and row.get("Run 1 Solved") != "—" and not row.get("Run 1 Solved", "").startswith("0/"))
        st.metric(
            "🏃 Run 1 Solved",
            run1_solved,
            f"{run1_solved}/{len(df)}"
        )

    with comp_col3:
        # Count Run 2 problems where at least one breakdown was solved (check if not "N/A" and not "0/...")
        run2_solved = sum(1 for row in comparison_data if row.get("Run 2 Solved") != "N/A" and row.get("Run 2 Solved") != "—" and not row.get("Run 2 Solved", "").startswith("0/"))
        st.metric(
            "🏃 Run 2 Solved",
            run2_solved,
            f"{run2_solved}/{len(df)}"
        )

    with comp_col4:
        # Count problems where Run 1 had 0 solved but Run 2 had at least 1 solved
        improved = sum(1 for row in comparison_data if (row.get("Run 1 Solved") == "N/A" or row.get("Run 1 Solved") == "—" or row.get("Run 1 Solved", "").startswith("0/")) and row.get("Run 2 Solved") != "N/A" and row.get("Run 2 Solved") != "—" and not row.get("Run 2 Solved", "").startswith("0/"))
        delta = f"+{improved}" if improved > 0 else "0"
        st.metric(
            "📈 Improved",
            improved,
            delta
        )
