"""
Main Streamlit application for Seed Prover Analysis GUI.
"""
import streamlit as st
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from seed_data_models import Session
from components.problem_browser import render_problem_browser, render_problem_summary_card
from components.breakdown_viewer import render_breakdown_tabs, comparison_table_with_load, render_problem_component_costs

st.set_page_config(
    page_title="Seed Prover Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    st.title("🔍 Seed Prover Analysis GUI")

    st.sidebar.header("Configuration")

    view_mode = st.sidebar.radio(
        "View Mode",
        options=["Run Analysis", "Analyze Trajectories"],
        horizontal=False
    )

    if view_mode == "Analyze Trajectories":
        st.markdown("---")
        from components.trajectory_analysis_viewer import render_trajectory_analysis_viewer
        render_trajectory_analysis_viewer()
        return

    # --- Run Analysis ---
    st.sidebar.markdown("---")
    minified_dir = st.sidebar.text_input(
        "Minified folder path",
        value="",
        help="Path to the minified folder (e.g., scratch/results/.../dump/minified)"
    )

    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.minified_dir = None
        st.session_state.selected_problem_id = None
        st.session_state.active_tab = 0
        st.session_state.session = None
        st.session_state.loaded_breakdown = None

    load_clicked = st.sidebar.button("Load Data", type="primary")
    if load_clicked and minified_dir:
        with st.spinner("Loading minified data..."):
            try:
                st.session_state.session = Session.load_from_minified(Path(minified_dir))
                st.session_state.data_loaded = True
                st.session_state.minified_dir = minified_dir
                st.sidebar.success(f"✅ Loaded {st.session_state.session.get_problem_count()} problems")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                import traceback
                st.error(traceback.format_exc())
                st.session_state.data_loaded = False
                st.stop()

    if not (st.session_state.data_loaded and st.session_state.session):
        st.info("👈 Enter the path to the minified folder and click 'Load Data' to begin")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.text(f"Problems: {st.session_state.session.get_problem_count()}")
    st.sidebar.text(f"Solved: {st.session_state.session.get_solved_count()}")

    st.markdown("---")

    render_problem_browser(st.session_state.session)

    if st.session_state.get("selected_problem_id"):
        selected_problem = st.session_state.session.get_problem(st.session_state.selected_problem_id)
        if selected_problem:
            st.markdown("---")
            st.header(f"Breakdown Details: {selected_problem.origin_problem_id}")
            render_problem_summary_card(selected_problem)
            st.markdown("---")

            def collect_all_breakdowns(problem):
                all_bds = dict(problem.breakdowns)
                for rp in problem.recursive_attempts:
                    all_bds.update(collect_all_breakdowns(rp))
                return all_bds

            def collect_all_breakdowns_for_cost(problem):
                all_bds = list(problem.breakdowns.values())
                for rp in problem.recursive_attempts:
                    all_bds.extend(collect_all_breakdowns_for_cost(rp))
                return all_bds

            all_breakdowns = collect_all_breakdowns(selected_problem)
            with st.expander("📊 Component Costs", expanded=False):
                render_problem_component_costs(collect_all_breakdowns_for_cost(selected_problem))
            st.markdown("---")
            st.subheader("Breakdown Comparison")
            comparison_table_with_load(list(all_breakdowns.values()))
            st.markdown("---")

            if st.session_state.get("loaded_breakdown"):
                bd = st.session_state.loaded_breakdown
                st.subheader(f"Loaded Breakdown: {bd.origin_problem_id} (R{bd.round_id} B{bd.breakdown_id})")
                render_breakdown_tabs(bd)
        else:
            st.warning("Selected problem not found.")


if __name__ == "__main__":
    main()
