"""
Recursive breakdown viewer component.
Renders a breakdown with its proving path tree, main attempt details, and nested recursive lemmas.
"""
import streamlit as st
from typing import Optional, Dict, List, Any
import sys
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

from seed_data_models import Breakdown, Problem
from components.breakdown_viewer import render_breakdown_tabs


def render_recursive_breakdown_viewer(
    breakdown: Breakdown,
    parent_problem: Problem,
    level: int = 0,
    indent_prefix: str = ""
):
    """
    Recursively render a breakdown with its proving path tree, details, and nested recursive lemmas.

    Args:
        breakdown: The breakdown to display
        parent_problem: The problem containing this breakdown
        level: Current recursion level (0 for round 0, 1 for round 1, etc.)
        indent_prefix: Prefix for nested headers (visual indentation)
    """
    # Show the prover path tree for this level (includes all recursion depths)
    st.markdown("### 🌳 Prover Path")
    st.markdown("*🔵 Blue=Theorem | 🟢 Green=Directly Solved | 🟡 Gold=Solved via Recursion | 🔴 Red=Unsolved*")
    fig = breakdown.plot_prover_path()
    if fig:
        st.pyplot(fig, use_container_width=False)
    st.markdown("---")

    # Get lemmas from this breakdown
    lemma_ids = []
    if breakdown.parsed_breakdown and breakdown.parsed_breakdown.lemmas:
        lemma_ids = sorted(breakdown.parsed_breakdown.lemmas.keys())

    # Find recursive attempts for lemmas in THIS breakdown
    related_recursive = []
    if parent_problem and parent_problem.recursive_attempts:
        for lemma_id in lemma_ids:
            # Build the expected UID pattern for this lemma
            expected_uid_prefix = f"{parent_problem.origin_problem_id}_r{level}_b{breakdown.breakdown_id}_l{lemma_id}"

            # Find matching recursive attempt
            for recursive_problem in parent_problem.recursive_attempts:
                if recursive_problem.origin_problem_id.startswith(expected_uid_prefix):
                    related_recursive.append((lemma_id, recursive_problem))
                    break

    # Prepare sub-tabs: Main Attempt + Recursive Lemmas
    sub_tab_labels = ["Main Attempt"]
    for lemma_id, _ in related_recursive:
        sub_tab_labels.append(f"Lemma {lemma_id}")

    sub_tabs = st.tabs(sub_tab_labels)

    # Tab 0: Main Attempt (this breakdown's details)
    with sub_tabs[0]:
        st.markdown(f"**Main Attempt - Round {level}, Breakdown {breakdown.breakdown_id}**")
        st.markdown("---")
        render_breakdown_tabs(breakdown)

    # Tabs 1+: Recursive Lemmas (recursive calls to this function)
    for tab_idx, (lemma_id, recursive_problem) in enumerate(related_recursive, start=1):
        with sub_tabs[tab_idx]:
            st.markdown(f"**Recursive Attempt for Lemma {lemma_id}**")
            st.markdown(f"**Lemma UID:** `{recursive_problem.origin_problem_id}`")
            st.markdown("---")

            # Group breakdowns by breakdown_id for this recursive attempt
            recursive_by_breakdown = {}
            for (parent_id, round_id, bd_id), bd in recursive_problem.breakdowns.items():
                if bd_id not in recursive_by_breakdown:
                    recursive_by_breakdown[bd_id] = []
                recursive_by_breakdown[bd_id].append(bd)

            if recursive_by_breakdown:
                # Create tabs for each breakdown in the recursive attempt
                recursive_breakdown_labels = []
                for bd_id in sorted(recursive_by_breakdown.keys()):
                    # Get round number from any breakdown with this bd_id
                    round_num = None
                    for (parent_id, round_id, check_bd_id) in recursive_problem.breakdowns.keys():
                        if check_bd_id == bd_id:
                            round_num = round_id
                            break
                    label = f"Round {round_num}, Breakdown {bd_id}" if round_num is not None else f"Breakdown {bd_id}"
                    recursive_breakdown_labels.append(label)

                recursive_breakdown_tabs = st.tabs(recursive_breakdown_labels)

                for rbd_tab, rbd_id in zip(recursive_breakdown_tabs, sorted(recursive_by_breakdown.keys())):
                    with rbd_tab:
                        # Recursively render each breakdown in the recursive problem
                        for rbd in recursive_by_breakdown[rbd_id]:
                            # Get the round number for this recursive breakdown
                            round_num = None
                            for (parent_id, round_id, check_bd_id) in recursive_problem.breakdowns.keys():
                                if check_bd_id == rbd_id:
                                    round_num = round_id
                                    break

                            # RECURSIVE CALL: Render this breakdown and its children
                            render_recursive_breakdown_viewer(
                                rbd,
                                recursive_problem,
                                level=round_num if round_num is not None else level + 1
                            )
            else:
                st.info("No breakdowns found for this recursive attempt")
