"""
Problem browser component - displays and filters list of all problems.
"""
import streamlit as st
import pandas as pd
import re
from typing import List, Optional, Set, Union, Any
from pathlib import Path
import sys

# Add root directory to path to import seed_data_models
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

from seed_data_models import Session, Problem


def get_max_recursion_depth(problem: 'Problem') -> int:
    """
    Calculate the maximum recursion depth of a problem.

    Depth 0: main problem with round 0 breakdowns only
    Depth 1: has recursive attempts (round 1+ lemmas)
    Depth 2: has recursive attempts of recursive attempts (round 2+ nested)
    etc.
    """
    max_depth = 0
    for recursive_prob in problem.recursive_attempts:
        recursive_depth = 1 + get_max_recursion_depth(recursive_prob)
        max_depth = max(max_depth, recursive_depth)
    return max_depth


def render_problem_browser(session_or_problems: Union['Session', List], key_prefix: str = "") -> Optional[str]:
    """
    Render the problem browser interface.

    Args:
        session_or_problems: Session object (new) or List[Any] (old)
        key_prefix: Prefix for widget keys to avoid conflicts when rendering multiple browsers

    Returns:
        Selected origin_problem_id or None
    """
    st.header("Problem Browser")

    # Convert Session to list of problems if needed
    if isinstance(session_or_problems, Session):
        problems = sorted(session_or_problems.problems.values(), key=lambda p: p.origin_problem_id)
        session = session_or_problems
    else:
        problems = sorted(session_or_problems, key=lambda p: p.origin_problem_id if hasattr(p, 'origin_problem_id') else str(p))
        session = None

    if not problems:
        st.warning("No problems found in this run.")
        return None

    # Display total run cost at the top
    if session:
        total_run_cost = session.get_total_cost('cost')
        st.markdown(f"## 💰 Total Run Cost: **${total_run_cost:.4f}**")
        st.markdown("---")

    # Search and filter controls
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input(
            "Search problems",
            placeholder="Enter problem name or ID...",
            key=f"{key_prefix}problem_search"
        )

    with col2:
        status_filter = st.selectbox(
            "Filter by status",
            options=["All", "Solved", "Unsolved"],
            key=f"{key_prefix}status_filter"
        )

    # Apply filters
    filtered_problems = problems

    if search_query:
        filtered_problems = [
            p for p in filtered_problems
            if search_query.lower() in p.origin_problem_id.lower()
        ]

    if status_filter == "Solved":
        filtered_problems = [p for p in filtered_problems if p.is_solved()]
    elif status_filter == "Unsolved":
        filtered_problems = [p for p in filtered_problems if not p.is_solved()]

    # Display statistics
    st.markdown("---")

    # Calculate statistics from model objects
    solved_count = sum(1 for p in problems if p.is_solved())
    total_breakdowns = sum(len(p.breakdowns) for p in problems)
    solved_breakdowns = sum(len(p.get_solved_breakdowns()) for p in problems)

    # Count problems with theorem proofs
    problems_with_theorem_proof = 0
    for problem in problems:
        for breakdown in problem.breakdowns.values():
            if breakdown.parsed_breakdown and breakdown.parsed_breakdown.theorem:
                if breakdown.parsed_breakdown.theorem.get_best_attempt():
                    problems_with_theorem_proof += 1
                    break  # Count problem only once

    # Count problems with lemma proofs
    total_lemmas_proven = 0
    for problem in problems:
        for breakdown in problem.breakdowns.values():
            if breakdown.parsed_breakdown:
                for lemma in breakdown.parsed_breakdown.lemmas.values():
                    if lemma.get_best_attempt():
                        total_lemmas_proven += 1

    # Display proving statistics
    st.subheader("📊 Problem Solving Statistics")

    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)

    with stat_col1:
        st.metric("Total Problems", len(problems))

    with stat_col2:
        st.metric("Solved", solved_count)

    with stat_col3:
        st.metric("Unsolved", len(problems) - solved_count)

    with stat_col4:
        st.metric("With Theorem Proof", problems_with_theorem_proof)

    with stat_col5:
        st.metric("Lemmas Proven", total_lemmas_proven)

    st.markdown("---")

    # Display breakdown statistics
    st.subheader("⚙️ Breakdown Statistics")

    breakdown_col1, breakdown_col2 = st.columns(2)

    with breakdown_col1:
        st.metric("Total Breakdowns", total_breakdowns)

    with breakdown_col2:
        st.metric("Solved Breakdowns", f"{solved_breakdowns}/{total_breakdowns}")

    st.markdown("---")

    st.markdown(f"**Showing {len(filtered_problems)} of {len(problems)} problems**")

    # Check if any problem has difficulty data
    has_difficulty = any(p.difficulty for p in filtered_problems)

    # Create table data
    table_data = []
    for problem in filtered_problems:
        # Count statistics using new models
        total_breakdowns_in_problem = len(problem.breakdowns)

        formalized_breakdowns = 0
        theorem_proven_count = 0
        solved_breakdowns_count = 0

        for breakdown in problem.breakdowns.values():
            # Count formalized - check if the breakdown is fully formalized
            # Use the data model (ParsedBreakdown from parsed_breakdown)
            is_formalized = False
            if breakdown.parsed_breakdown:
                # Use the new is_formalized() method which checks if ALL lemmas and theorem
                # have at least one formalization that compiled
                is_formalized = breakdown.parsed_breakdown.is_formalized()

            if is_formalized:
                formalized_breakdowns += 1

            # Count theorem proofs
            if (breakdown.parsed_breakdown and
                breakdown.parsed_breakdown.theorem and
                breakdown.parsed_breakdown.theorem.get_best_attempt()):
                theorem_proven_count += 1

            # Count solved breakdowns
            if breakdown.is_solved():
                solved_breakdowns_count += 1

        # Format display strings
        # Formalized: show emoji and count
        formalized_emoji = '✅' if formalized_breakdowns > 0 else '❌'
        formalized_display = f"{formalized_breakdowns}/{total_breakdowns_in_problem} ({formalized_emoji})"

        # Theorem: N/A if no formalized breakdowns, otherwise show count with emoji
        if formalized_breakdowns > 0:
            theorem_emoji = '✅' if theorem_proven_count > 0 else '❌'
            theorem_display = f"{theorem_proven_count}/{formalized_breakdowns} ({theorem_emoji})"
        else:
            theorem_display = "N/A"

        # Solved: N/A if no theorems proven, otherwise show count with emoji
        if theorem_proven_count > 0:
            solved_emoji = '✅' if solved_breakdowns_count > 0 else '❌'
            solved_display = f"{solved_breakdowns_count}/{theorem_proven_count} ({solved_emoji})"
        else:
            solved_display = "N/A"

        # Overall solved status
        overall_solved_emoji = '✅' if problem.is_solved() else '❌'

        # Cost metrics
        problem_output_tokens = problem.get_total_cost("output_tokens")
        problem_prover_calls = problem.get_total_cost("prover_calls")

        # Recursion depth
        max_recursion_depth = get_max_recursion_depth(problem)

        row_data = {
            "Problem ID": problem.origin_problem_id,
            "Formalized": formalized_display,
            "Theorem": theorem_display,
            "Solved": solved_display,
            "Overall": overall_solved_emoji,
            "Depth": max_recursion_depth,
            "Output Tokens": int(problem_output_tokens),
            "Pass": int(problem_prover_calls),
        }

        # Add difficulty only if it exists in any problem
        if has_difficulty and problem.difficulty:
            row_data["Difficulty"] = problem.difficulty

        table_data.append(row_data)

    if table_data:
        df = pd.DataFrame(table_data)

        # Display as interactive table
        st.markdown("**Click 'View' button on any row to view its breakdown details:**")

        # Determine column widths based on whether difficulty is included
        if has_difficulty:
            col_widths = [2.5, 1.0, 1.2, 1.5, 2.0, 1.0, 1.2, 1.5, 1.2, 1.0]
        else:
            col_widths = [3.0, 1.2, 1.5, 2.0, 1.0, 1.2, 1.5, 1.2, 1.0]

        # Create columns for the header
        col_headers = st.columns(col_widths)
        col_idx = 0
        with col_headers[col_idx]:
            st.markdown("**Problem ID**")
        col_idx += 1

        if has_difficulty:
            with col_headers[col_idx]:
                st.markdown("**Difficulty**")
            col_idx += 1

        with col_headers[col_idx]:
            st.markdown("**Formalized**")
        col_idx += 1
        with col_headers[col_idx]:
            st.markdown("**Theorem**")
        col_idx += 1
        with col_headers[col_idx]:
            st.markdown("**Solved**")
        col_idx += 1
        with col_headers[col_idx]:
            st.markdown("**Overall**")
        col_idx += 1
        with col_headers[col_idx]:
            st.markdown("**Depth**")
        col_idx += 1
        with col_headers[col_idx]:
            st.markdown("**Output Tokens**")
        col_idx += 1
        with col_headers[col_idx]:
            st.markdown("**Pass**")
        col_idx += 1
        with col_headers[col_idx]:
            st.markdown("**Action**")

        # Create scrollable container
        container = st.container(height=600)

        # Display each problem as a clickable row
        for idx, row in df.iterrows():
            with container:
                cols = st.columns(col_widths)
                col_idx = 0

                with cols[col_idx]:
                    st.text(row["Problem ID"])
                col_idx += 1

                if has_difficulty:
                    with cols[col_idx]:
                        st.text(row.get("Difficulty", "-"))
                    col_idx += 1

                with cols[col_idx]:
                    st.text(row["Formalized"])
                col_idx += 1
                with cols[col_idx]:
                    st.text(row["Theorem"])
                col_idx += 1
                with cols[col_idx]:
                    st.text(row["Solved"])
                col_idx += 1
                with cols[col_idx]:
                    st.text(row["Overall"])
                col_idx += 1
                with cols[col_idx]:
                    st.text(str(row["Depth"]))
                col_idx += 1
                with cols[col_idx]:
                    st.text(str(row["Output Tokens"]))
                col_idx += 1
                with cols[col_idx]:
                    st.text(str(row["Pass"]))
                col_idx += 1
                with cols[col_idx]:
                    if st.button("View", key=f"{key_prefix}view_{idx}_{row['Problem ID']}"):
                        # Set session state to show breakdown details
                        st.session_state.selected_problem_id = row["Problem ID"]

    return st.session_state.get("selected_problem_id", None)


def render_problem_summary_card(problem: Any):
    """
    Render a summary card for a specific problem.

    Args:
        problem: Problem object (new) or ProblemSummary object (old)
    """
    st.subheader(f"Problem: {problem.origin_problem_id}")

    # Count solved breakdowns
    solved_breakdowns = problem.get_solved_breakdowns() if isinstance(problem, Problem) else []
    solved_breakdown_numbers = []

    if isinstance(problem, Problem):
        for breakdown in solved_breakdowns:
            # Extract the breakdown number from problem_id (e.g., "mathd_algebra_209_r0_b2" -> 2)
            match = re.search(r'_b(\d+)', breakdown.problem_id)
            if match:
                solved_breakdown_numbers.append(int(match.group(1)))

    # Basic info
    col1, col2, col3 = st.columns(3)

    with col1:
        total_breakdowns = len(problem.breakdowns)
        st.metric("Breakdowns Generated", total_breakdowns)

    with col2:
        solved_count = len(solved_breakdown_numbers) if isinstance(problem, Problem) else len(solved_breakdowns)
        st.metric("Breakdowns Solved", f"{solved_count}/{total_breakdowns}")

    with col3:
        solved_status = problem.is_solved() if isinstance(problem, Problem) else problem.solved
        solved_emoji = '✅' if solved_status else '❌'
        st.metric("Solved", solved_emoji)

    # Cost metrics (only for Problem objects, not ProblemSummary)
    if isinstance(problem, Problem):
        st.markdown("---")
        st.subheader("💰 Cost & Token Breakdown")

        cost_col1, cost_col2, cost_col3, cost_col4, cost_col5 = st.columns(5)

        with cost_col1:
            problem_cost = problem.get_total_cost("cost")
            st.metric("Total Cost", f"${problem_cost:.6f}")

        with cost_col2:
            problem_calls = problem.get_total_cost("prover_calls")
            st.metric("Prover Calls", problem_calls)

        with cost_col3:
            problem_rounds = problem.count_rounds()
            st.metric("Breakdown Rounds", problem_rounds)

        with cost_col4:
            reasoning_tokens = problem.get_total_cost("output_tokens")
            st.metric("Output Tokens", reasoning_tokens)

        with cost_col5:
            output_sflops = problem.get_total_cost("output_sflops")
            st.metric("Output SFLOPs", f"{output_sflops:,}")

        # Show cost by round breakdown
        cost_by_round = problem.get_cost_by_round()
        if len(cost_by_round) > 1:
            st.markdown("**Cost by Round:**")
            for round_id in sorted(cost_by_round.keys()):
                st.write(f"  Round {round_id}: ${cost_by_round[round_id]:.6f}")

        # Show token summary
        st.markdown("**Token Summary:**")
        col_tokens1, col_tokens2, col_tokens3 = st.columns(3)
        with col_tokens1:
            input_tokens = problem.get_total_cost("input_tokens")
            st.write(f"Input Tokens: {input_tokens:,}")
        with col_tokens2:
            output_tokens = problem.get_total_cost("output_tokens")
            st.write(f"Output Tokens: {output_tokens:,}")
        with col_tokens3:
            total_tokens = input_tokens + output_tokens
            st.write(f"Total Tokens: {total_tokens:,}")

    # Show per-round breakdown statistics (for problems with multiple rounds)
    if isinstance(problem, Problem) and problem.count_rounds() > 1:
        st.markdown("---")
        st.subheader("📋 Breakdowns by Round")

        # Group breakdowns by round
        breakdowns_by_round = {}
        for breakdown in problem.breakdowns.values():
            round_id = breakdown.round_id
            if round_id not in breakdowns_by_round:
                breakdowns_by_round[round_id] = []
            breakdowns_by_round[round_id].append(breakdown)

        # Display per-round stats
        for round_id in sorted(breakdowns_by_round.keys()):
            round_breakdowns = breakdowns_by_round[round_id]
            round_solved = sum(1 for bd in round_breakdowns if bd.is_solved())
            round_cost = sum(bd.get_total_cost("cost") for bd in round_breakdowns)

            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric(f"Round {round_id} - Breakdowns", len(round_breakdowns))
            with col_r2:
                st.metric(f"Round {round_id} - Solved", f"{round_solved}/{len(round_breakdowns)}")
            with col_r3:
                st.metric(f"Round {round_id} - Cost", f"${round_cost:.6f}")

    # Show which breakdowns are solved
    if solved_breakdown_numbers:
        breakdown_names = [f"Breakdown {i}" for i in sorted(solved_breakdown_numbers)]
        solved_names_str = ", ".join(breakdown_names)
        st.markdown(f"**Solved Breakdowns:** {solved_names_str}")

    # Problem statement from first breakdown (simplified)
    if problem.breakdowns:
        first_breakdown = list(problem.breakdowns.values())[0] if isinstance(problem, Problem) else problem.breakdowns[0]
        with st.expander("View Breakdown Info", expanded=False):
            if hasattr(first_breakdown, 'informal_breakdown') and first_breakdown.informal_breakdown:
                st.markdown("**Informal Breakdown:**")
                st.markdown(first_breakdown.informal_breakdown)
            else:
                st.info("No breakdown information available")
