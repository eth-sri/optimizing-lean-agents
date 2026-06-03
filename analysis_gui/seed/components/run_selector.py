"""
Component for selecting and browsing available runs from the scratch folder.

Provides a user-friendly dropdown to browse and select runs.
"""
import streamlit as st
from pathlib import Path
from typing import Optional, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from folder_browser import get_available_runs, format_runs_for_dropdown


def get_putnam_seed_prover_runs() -> List[Tuple[str, str, str]]:
    """
    Scan outputs/putnam/seed_prover/runs for available runs.

    Returns:
        List of tuples (display_name, full_path, data_type)
    """
    runs = []
    base = Path(__file__).parent.parent.parent.parent / "outputs" / "putnam" / "seed_prover" / "runs"

    if not base.exists():
        return runs

    for run_dir in sorted(base.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue

        run_name = run_dir.name

        # Check for different structures: minified/, dump/minified/, or direct round0/
        minified_dir = run_dir / "minified"
        dump_minified_dir = run_dir / "dump" / "minified"
        round0_dir = run_dir / "round0"

        if minified_dir.exists():
            runs.append((f"🆕 {run_name}", str(minified_dir), "minified"))
        elif dump_minified_dir.exists():
            runs.append((f"🆕 {run_name} (dump)", str(dump_minified_dir), "minified"))
        elif round0_dir.exists():
            # Direct round0 structure - point to parent
            runs.append((f"🆕 {run_name}", str(run_dir), "minified"))

    return runs


def render_run_selector() -> Optional[str]:
    """
    Render a dropdown selector for available runs.

    Provides options to browse runs from:
    - Repository routing (scratch/results/combined, scratch/dump, etc.)
    - outputs/putnam/seed_prover/runs (new runs)

    Returns:
        Selected run path, or None if no selection made
    """
    st.subheader("📁 Browse Runs")

    # Data source selector
    data_source = st.radio(
        "Data Source",
        options=["Repository (scratch/)", "Putnam Seed Prover (outputs/)"],
        horizontal=True,
        help="Choose where to load runs from"
    )

    # Get runs based on selected source
    if data_source == "Repository (scratch/)":
        runs = get_available_runs()
        empty_msg = "No runs found in scratch/results/combined or scratch/dump directories"
    else:
        runs = get_putnam_seed_prover_runs()
        empty_msg = "No runs found in outputs/putnam/seed_prover/runs"

    if not runs:
        st.warning(empty_msg)
        return None

    # Format runs for dropdown
    display_names, path_mapping, data_type_mapping = format_runs_for_dropdown(runs)

    # Create dropdown with most recent run as default
    selected_display = st.selectbox(
        "Select a run to analyze",
        options=display_names,
        index=0,  # Most recent run (first in reversed-sorted list)
        help="Choose a run to analyze"
    )

    if selected_display:
        selected_path = path_mapping[selected_display]
        data_type = data_type_mapping[selected_display]

        # Store data type in session state for app to use
        st.session_state.selected_data_type = data_type

        return selected_path

    return None
