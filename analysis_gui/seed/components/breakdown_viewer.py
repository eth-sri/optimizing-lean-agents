"""
Breakdown viewer component - displays all breakdowns for a selected problem.
Uses OOP data models from models.py exclusively.
"""
import re
import streamlit as st
import pandas as pd
from typing import List, Optional, Union, Dict, Any
import sys
from pathlib import Path

# Add root directory to path to import seed_data_models
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

from seed_data_models import Breakdown

from utils import format_cost, truncate_text
from components.enhanced_breakdown_details import render_enhanced_breakdown_details_inner


def group_breakdowns_by_parent_problem(breakdowns_dict: Union[Dict[int, Breakdown], List[Breakdown]]) -> Dict[str, Dict[int, Breakdown]]:
    """
    Group breakdowns by parent problem (parent_problem_id if round 1+, otherwise origin_problem_id).

    This groups Round 0 breakdowns with their corresponding Round 1 recursive proving attempts.

    Args:
        breakdowns_dict: Dictionary or list of Breakdown objects

    Returns:
        Dict mapping problem_key to dict of breakdowns for that problem
        Structure: {problem_key: {breakdown_id: Breakdown, ...}, ...}
        where problem_key is either parent_problem_id (round 1+) or origin_problem_id (round 0)
    """
    # Handle both dict and list inputs
    if isinstance(breakdowns_dict, dict):
        breakdowns_list = list(breakdowns_dict.values())
    else:
        breakdowns_list = breakdowns_dict

    by_problem = {}
    for breakdown in breakdowns_list:
        # Use parent_problem_id if available (round 1+), otherwise origin_problem_id
        problem_key = breakdown.parent_problem_id if breakdown.parent_problem_id else breakdown.origin_problem_id

        if problem_key not in by_problem:
            by_problem[problem_key] = {}
        by_problem[problem_key][breakdown.breakdown_id] = breakdown

    return by_problem


def render_round_tabs(problem_breakdowns: Union[Dict[int, Breakdown], List[Breakdown]], analysis: Optional[Any] = None):
    """
    Render tabs for different parent problems (groups breakdowns by parent problem and breakdown id).

    Round 0 breakdowns are grouped under their origin_problem_id.
    Round 1+ breakdowns are grouped under their parent_problem_id (the lemma being recursively proven).

    Args:
        problem_breakdowns: Dict or list of all Breakdown objects for a problem
        analysis: Optional analysis data for proof attempt details
    """
    by_problem = group_breakdowns_by_parent_problem(problem_breakdowns)

    if len(by_problem) == 1:
        # Only one problem - show directly
        problem_key = list(by_problem.keys())[0]
        st.subheader(f"Problem: {problem_key}")
        render_breakdown_viewer(by_problem[problem_key], analysis)
    else:
        # Multiple problems - use tabs
        sorted_problems = sorted(by_problem.keys())
        tabs = st.tabs([p.split('_r')[0] if '_r' in p else p for p in sorted_problems])  # Shorten tab names
        for tab, problem_key in zip(tabs, sorted_problems):
            with tab:
                st.subheader(f"Problem: {problem_key}")
                render_breakdown_viewer(by_problem[problem_key], analysis)


def render_breakdown_viewer(
    breakdowns: Union[Dict[int, Breakdown], List[Breakdown]],
    analysis: Optional[Any] = None
):
    """
    Render the breakdown viewer for a list of breakdowns.

    Args:
        breakdowns: Dict of Breakdown OOP objects or List of Breakdown OOP objects
        analysis: Optional analysis data for proof attempt details
    """
    # Convert dict to list if needed
    if isinstance(breakdowns, dict):
        breakdowns_list = list(breakdowns.values())
    else:
        breakdowns_list = breakdowns

    if not breakdowns_list:
        st.warning("No breakdowns found for this problem.")
        return

    st.header(f"Breakdowns")
    st.markdown(f"**Total breakdowns:** {len(breakdowns_list)}")

    st.markdown("---")

    # Comparison table with Load buttons
    st.subheader("Breakdown Comparison")
    comparison_table_with_load(breakdowns_list, analysis)

    st.markdown("---")

    # Component Cost - display directly below comparison table
    render_problem_component_costs(breakdowns_list)

    st.markdown("---")

    # Loaded breakdown details (will be populated when user clicks Load button)
    if 'loaded_breakdown' in st.session_state and st.session_state.loaded_breakdown:
        st.subheader(f"Loaded Breakdown: {st.session_state.loaded_breakdown.origin_problem_id} (R{st.session_state.loaded_breakdown.round_id} B{st.session_state.loaded_breakdown.breakdown_id})")
        render_breakdown_tabs(st.session_state.loaded_breakdown, analysis)


def render_breakdown_pipeline_analysis(breakdown: Breakdown, analysis: Optional['ProblemAnalysis'] = None):
    """
    Render pipeline analysis for a single breakdown.
    Shows formalization, parsing, and proof progression data.

    Args:
        breakdown: Breakdown OOP object
        analysis: Optional analysis object
    """
    st.subheader("Pipeline Overview")

    # Parsing status
    st.markdown("#### 📊 Parsing")
    if breakdown.parse_failure:
        st.error("❌ Parsing Failed")
        if breakdown.parse_failure.get('error'):
            st.text(breakdown.parse_failure['error'])
    else:
        st.success("✅ Parsed Successfully")
        if breakdown.parsed_breakdown:
            parsed_data = breakdown.parsed_breakdown
            # parsed_breakdown is now OOP ParsedBreakdown - directly access lemmas
            lemma_count = len(parsed_data.lemmas)
            st.text(f"Found {lemma_count} lemmas in breakdown")

    st.markdown("---")

    # Formalization status
    st.markdown("#### ⚙️ Formalization")
    if breakdown.parsed_breakdown:
        st.success("✅ Formalized Successfully")
        parsed_data = breakdown.parsed_breakdown
        # Handle both OOP ParsedBreakdown and legacy dict structures
        if hasattr(parsed_data, 'lemmas'):
            # OOP ParsedBreakdown
            parsed_lemmas = len(parsed_data.lemmas)
        elif isinstance(parsed_data, dict):
            # Legacy dict structure
            parsed_lemmas = len(parsed_data.get('lemmas', []))
        else:
            parsed_lemmas = 0
        if parsed_lemmas > 0:
            st.text(f"Formalized {parsed_lemmas} lemmas in Lean 4")
    else:
        st.info("ℹ️ No parsed breakdown available")

    st.markdown("---")

    # Proof progression
    st.markdown("#### 🎯 Proof Progression")

    # Theorem attempts
    if breakdown.theorem_prover_results and breakdown.theorem_prover_results.get('attempts'):
        attempts = breakdown.theorem_prover_results.get('attempts', [])
        col1, col2 = st.columns(2)

        with col1:
            total_attempts = len(attempts)
            st.metric("Theorem Attempts", total_attempts)

        with col2:
            passed_attempts = sum(1 for a in attempts if a.get('data', {}).get('compilation_result', {}).get('pass'))
            st.metric("Passed Attempts", passed_attempts)

        # Show iteration progression
        iterations = {}
        for attempt in attempts:
            it = attempt.get('iteration', 0)
            comp = attempt.get('data', {}).get('compilation_result', {})
            if it not in iterations:
                iterations[it] = {'attempts': 0, 'passed': 0}
            iterations[it]['attempts'] += 1
            if comp.get('pass'):
                iterations[it]['passed'] += 1

        if iterations:
            st.markdown("**Progression by Iteration:**")
            prog_rows = []
            for it in sorted(iterations.keys()):
                data = iterations[it]
                prog_rows.append({
                    'Iteration': it,
                    'Attempts': data['attempts'],
                    'Passed': data['passed'],
                    'Status': '✅ Passed' if data['passed'] > 0 else '❌ Failed'
                })
            prog_df = pd.DataFrame(prog_rows)
            st.dataframe(prog_df, width="stretch", hide_index=True)

    st.markdown("---")

    # Lemma attempts
    st.markdown("#### 📝 Lemma Prover")
    if breakdown.lemma_prover_results and breakdown.lemma_prover_results.get('all_attempts'):
        all_attempts = breakdown.lemma_prover_results.get('all_attempts', [])

        # Group by lemma
        lemmas_data = {}
        for attempt in all_attempts:
            data = attempt.get('data', {})
            metadata = data.get('metadata', {})
            lemma_id = metadata.get('lemma_id')

            # Get lemma_id from metadata or attempt to extract from uid
            if lemma_id is None:
                uid = data.get('uid', '')
                # Try to extract lemma_id from uid format like "xxx_l1_f0_a0"
                import re
                match = re.search(r'_l(\d+)', uid)
                if match:
                    lemma_id = f"l{match.group(1)}"
            else:
                lemma_id = f"l{lemma_id}"

            if lemma_id:
                if lemma_id not in lemmas_data:
                    lemmas_data[lemma_id] = []
                lemmas_data[lemma_id].append(attempt)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Lemmas Attempted", len(lemmas_data))

        with col2:
            proven = sum(1 for lemma_attempts in lemmas_data.values()
                        if any(a.get('data', {}).get('compilation_result', {}).get('pass') and
                              a.get('data', {}).get('compilation_result', {}).get('complete')
                              for a in lemma_attempts))
            st.metric("Lemmas Proven", proven)

        # Lemma details
        if lemmas_data:
            st.markdown("**Lemma Attempt Details:**")
            lemma_rows = []
            for lemma_id in sorted(lemmas_data.keys()):
                attempts_list = lemmas_data[lemma_id]
                total = len(attempts_list)
                passed = sum(1 for a in attempts_list if a.get('data', {}).get('compilation_result', {}).get('pass'))
                proven = any(a.get('data', {}).get('compilation_result', {}).get('pass') and
                            a.get('data', {}).get('compilation_result', {}).get('complete')
                            for a in attempts_list)

                lemma_rows.append({
                    'Lemma': lemma_id,
                    'Attempts': total,
                    'Passed': passed,
                    'Complete': '✅' if proven else '❌',
                    'Status': 'Proven' if proven else 'Not Proven'
                })

            lemma_df = pd.DataFrame(lemma_rows)
            st.dataframe(lemma_df, width="stretch", hide_index=True)
    else:
        st.info("No lemma prover attempts recorded")


def render_breakdown_tabs(breakdown: Breakdown, analysis: Optional[Any] = None):
    """
    Render tabs for a single breakdown with "Breakdown Details" and "Informal Breakdown".

    Args:
        breakdown: Breakdown OOP object
        analysis: Optional analysis data
    """
    # Display basic info
    problem_id = breakdown.problem_id
    origin_id = breakdown.origin_problem_id

    st.markdown(f"**Problem ID:** `{problem_id}`")
    st.markdown(f"**Origin Problem:** `{origin_id}`")

    # Display optional properties
    if breakdown.tags:
        st.markdown(f"**Tags:** {', '.join(breakdown.tags)}")

    if breakdown.detailed_cost:
        st.markdown(f"**Cost:** {format_cost(breakdown.detailed_cost)}")

    st.markdown("---")

    # Create tabs: Details, Informal, Component Costs
    details_tab, informal_tab, costs_tab = st.tabs(["Breakdown Details", "Informal Breakdown", "Component Costs"])

    with details_tab:
        # Render the enhanced breakdown details without the top tabs
        render_enhanced_breakdown_details_inner(breakdown, analysis)

    with informal_tab:
        # Render informal breakdown
        st.subheader("Informal Breakdown")
        if hasattr(breakdown, 'informal_breakdown') and breakdown.informal_breakdown:
            # Always use markdown for LaTeX support
            st.markdown(breakdown.informal_breakdown)
        else:
            st.info("No informal breakdown available.")

    with costs_tab:
        # Render component cost analysis
        render_breakdown_component_costs(breakdown)


def render_breakdown_details(breakdown: Breakdown):
    """
    Render detailed information for a single breakdown.

    Args:
        breakdown: Breakdown OOP object
    """
    # Basic info
    st.markdown(f"**Problem ID:** `{breakdown.problem_id}`")
    st.markdown(f"**Origin Problem:** `{breakdown.origin_problem_id}`")

    if breakdown.tags:
        st.markdown(f"**Tags:** {', '.join(breakdown.tags)}")

    # Cost information
    if breakdown.detailed_cost:
        st.markdown(f"**Cost:** {format_cost(breakdown.detailed_cost)}")

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Informal Breakdown",
        "Problem Statement",
        "Lean Code",
        "Raw Data"
    ])

    with tab1:
        st.subheader("Informal Breakdown")
        if breakdown.informal_breakdown:
            # Always use markdown for LaTeX support
            st.markdown(breakdown.informal_breakdown)
        else:
            st.info("No informal breakdown available.")

    with tab2:
        st.subheader("Problem Statement")

        if breakdown.informal_prefix:
            st.markdown("**Informal Prefix:**")
            st.markdown(breakdown.informal_prefix)

        if breakdown.informal_solution:
            st.markdown("**Informal Solution:**")
            st.markdown(breakdown.informal_solution)

        if breakdown.formal_statement:
            st.markdown("**Formal Statement:**")
            st.code(breakdown.formal_statement, language="lean")

    with tab3:
        st.subheader("Lean Code")
        if breakdown.lean4_code:
            st.code(breakdown.lean4_code, language="lean")

            # Add copy button
            if st.button(f"Copy to clipboard", key=f"copy_{breakdown.problem_id}"):
                st.code(breakdown.lean4_code)
                st.success("Code displayed above - use your browser to copy")
        else:
            st.info("No Lean code available.")

    with tab4:
        st.subheader("Raw Breakdown Data")

        # Display all fields
        st.json({
            "problem_id": breakdown.problem_id,
            "origin_problem_id": breakdown.origin_problem_id,
            "name": breakdown.name,
            "tags": breakdown.tags,
            "detailed_cost": breakdown.detailed_cost,
            "informal_prefix": truncate_text(breakdown.informal_prefix, 500),
            "informal_solution": truncate_text(breakdown.informal_solution, 500),
            "formal_statement": truncate_text(breakdown.formal_statement, 500),
            "lean4_code": truncate_text(breakdown.lean4_code, 500),
            "informal_breakdown_length": len(breakdown.informal_breakdown) if breakdown.informal_breakdown else 0,
        })


def comparison_table_with_load(breakdowns: Union[Dict[int, Breakdown], List[Breakdown]], analysis: Optional[Any] = None):
    """
    Render a comparison table with Load buttons for each breakdown.
    Clicking Load loads the hierarchical data and displays it below.

    Args:
        breakdowns: Dict of Breakdown OOP objects or List of Breakdown OOP objects
        analysis: Optional analysis data
    """
    # Convert dict to list if needed
    if isinstance(breakdowns, dict):
        breakdowns_list = list(breakdowns.values())
    else:
        breakdowns_list = breakdowns

    if len(breakdowns_list) < 1:
        st.info("No breakdowns available.")
        return

    # Build comparison data rows (same as compare_breakdowns)
    comparison_data = []
    for bd in breakdowns_list:
        # Handle both old and new breakdown structures
        cost = 0
        if hasattr(bd, 'detailed_cost') and bd.detailed_cost:
            cost = bd.detailed_cost.get('cost', 0)

        breakdown_length = 0
        if hasattr(bd, 'informal_breakdown') and bd.informal_breakdown:
            breakdown_length = len(bd.informal_breakdown)

        # Check if parsed
        parsed = "✅"
        if isinstance(bd, Breakdown):
            if bd.parse_failure:
                parsed = "❌"
            elif not bd.parsed_breakdown:
                parsed = "❌"
        elif hasattr(bd, 'parsed_breakdown'):
            if hasattr(bd, 'parse_failure') and bd.parse_failure:
                parsed = "❌"
            elif not bd.parsed_breakdown:
                parsed = "❌"
            elif isinstance(bd.parsed_breakdown, dict):
                parsed_data = bd.parsed_breakdown
                if isinstance(parsed_data, dict) and 'parsed_breakdown' in parsed_data:
                    parsed_data = parsed_data['parsed_breakdown']
                if isinstance(parsed_data, dict) and parsed_data.get('error'):
                    parsed = "❌"

        # Check if formalized
        formalized = False
        if isinstance(bd, Breakdown):
            if bd.parsed_breakdown:
                formalized = bd.parsed_breakdown.is_formalized()

        # Check if theorem was proven and count attempts
        theorem_proven = False
        theorem_passing_count = 0
        theorem_total_count = 0
        if isinstance(bd, Breakdown):
            if bd.parsed_breakdown and bd.parsed_breakdown.theorem:
                theorem_proven = bd.parsed_breakdown.theorem.get_best_attempt() is not None
                # Count all attempts across all formalizations
                for form in bd.parsed_breakdown.theorem.formalizations:
                    for attempt in form.proof_attempts:
                        theorem_total_count += 1
                        if attempt.is_passing():
                            theorem_passing_count += 1
        elif hasattr(bd, 'theorem_prover_results') and bd.theorem_prover_results:
            attempts = bd.theorem_prover_results.get('attempts', [])
            theorem_total_count = len(attempts)
            for a in attempts:
                if (a.get('data', {}).get('compilation_result', {}).get('pass', False) and
                    a.get('data', {}).get('compilation_result', {}).get('complete', False)):
                    theorem_passing_count += 1
            theorem_proven = theorem_passing_count > 0

        # Get used lemmas and their proof status
        used_lemmas_count = 0
        proven_used_lemmas_count = 0

        if isinstance(bd, Breakdown) and bd.parsed_breakdown:
            used_lemmas_count, proven_used_lemmas_count = bd.get_used_lemmas_count()

        solved = bd.is_solved()

        # Get actual combined proof status
        from pathlib import Path
        actual_proof_status = "❌"
        run_path = None

        # Try to get run_dir from session state first
        if 'run_dir' in st.session_state and st.session_state.run_dir:
            run_path = Path(st.session_state.run_dir)

        if run_path:
            # Check for combined proof in either location
            lean_file_dirs = [
                run_path / "combined" / "lean_files",
                run_path.parent / "combined" / "lean_files" if run_path.name == "minified" else None
            ]
            for lean_dir in filter(None, lean_file_dirs):
                for subdir in ["complete", "incomplete"]:
                    dir_path = lean_dir / subdir
                    if dir_path.exists():
                        # Look for any file matching this breakdown's pattern
                        matches = list(dir_path.glob(f"{bd.origin_problem_id}_r{bd.round_id}_b{bd.breakdown_id}.lean"))
                        if matches:
                            actual_proof_status = "✅"
                            break
                if actual_proof_status == "✅":
                    break

        problem_key = bd.parent_problem_id if bd.parent_problem_id else bd.origin_problem_id
        breakdown_info = f"Breakdown {bd.breakdown_id}"
        display_text = f"{problem_key} - {breakdown_info}"

        comparison_data.append({
            "Breakdown": display_text,
            "Parsed": parsed,
            "Formalized": "✅" if formalized else "❌",
            "Theorem": f"{theorem_passing_count}/{theorem_total_count} {'✅' if theorem_proven else '❌'}",
            "Used Lemmas": f"{proven_used_lemmas_count}/{used_lemmas_count}" if used_lemmas_count > 0 else "N/A",
            "Solved": "✅" if solved else "❌",
            "Actual": actual_proof_status,
            "_breakdown_obj": bd,  # Store the breakdown object for Load button
            "_sort_key": (problem_key, bd.breakdown_id),
        })

    # Sort by parent problem, then breakdown_id
    comparison_data_sorted = sorted(comparison_data, key=lambda x: x["_sort_key"])

    # Create table with Load buttons as the last column
    # First, display all rows with Load buttons
    cols = st.columns([3, 1, 1, 1, 1, 1, 1, 2])  # 7 data columns + 1 for Load button

    # Header row
    with cols[0]:
        st.markdown("**Breakdown**")
    with cols[1]:
        st.markdown("**Parsed**")
    with cols[2]:
        st.markdown("**Formalized**")
    with cols[3]:
        st.markdown("**Theorem**")
    with cols[4]:
        st.markdown("**Used Lemmas**")
    with cols[5]:
        st.markdown("**Solved**")
    with cols[6]:
        st.markdown("**Actual**")
    with cols[7]:
        st.markdown("**Load**")

    st.divider()

    # Data rows with Load buttons
    for idx, row in enumerate(comparison_data_sorted):
        cols = st.columns([3, 1, 1, 1, 1, 1, 1, 2])

        with cols[0]:
            st.text(row["Breakdown"])
        with cols[1]:
            st.text(row["Parsed"])
        with cols[2]:
            st.text(row["Formalized"])
        with cols[3]:
            st.text(row["Theorem"])
        with cols[4]:
            st.text(row["Used Lemmas"])
        with cols[5]:
            st.text(row["Solved"])
        with cols[6]:
            st.text(row["Actual"])
        with cols[7]:
            breakdown_obj = row["_breakdown_obj"]
            if st.button("📥 Load", key=f"load_from_table_{breakdown_obj.origin_problem_id}_{breakdown_obj.round_id}_{breakdown_obj.breakdown_id}", help="Load breakdown data from hierarchical dump"):
                _load_breakdown_and_display(breakdown_obj)


def _load_breakdown_and_display(breakdown: Breakdown):
    """
    Load hierarchical breakdown data and set it in session state for display.

    Args:
        breakdown: Breakdown object to load
    """
    try:
        # Get hierarchical_dir from session state
        hierarchical_dir = st.session_state.get('hierarchical_dir')

        # Get the session object
        session = st.session_state.get('session')
        if not session:
            st.error("❌ Session not loaded.")
            return

        if not hierarchical_dir:
            # Hierarchical directory not available - show warning but continue
            st.warning("⚠️ Hierarchical directory not found. Displaying data from current session only.")
            # Store in session state for display with current data
            st.session_state.loaded_breakdown = breakdown
            st.info("💡 Showing breakdown with available data. Some details may be limited.")
        else:
            # Try to load the breakdown from hierarchical data
            try:
                with st.spinner(f"Loading breakdown data..."):
                    session.load_breakdown(
                        origin_problem_id=breakdown.origin_problem_id,
                        round_id=breakdown.round_id,
                        breakdown_id=breakdown.breakdown_id,
                        hierarchical_dir=hierarchical_dir
                    )

                # Store in session state for display
                st.session_state.loaded_breakdown = breakdown
                st.success("✅ Loaded! Scroll down to see details.")

            except (FileNotFoundError, ValueError) as e:
                # Breakdown directory doesn't exist - show warning but continue with available data
                st.warning(f"⚠️ Breakdown directory not found in hierarchical dump. Displaying data from current session only.")
                st.info("💡 Showing breakdown with available data. Some details may be limited.")
                # Store in session state for display with current data
                st.session_state.loaded_breakdown = breakdown

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        import traceback
        st.error(traceback.format_exc())


def compare_breakdowns(breakdowns: Union[Dict[int, Breakdown], List[Breakdown]]):
    """
    Render a comparison view of multiple breakdowns.

    Args:
        breakdowns: Dict of Breakdown OOP objects or List of Breakdown OOP objects
    """
    st.subheader("Breakdown Comparison")

    # Convert dict to list if needed
    if isinstance(breakdowns, dict):
        breakdowns_list = list(breakdowns.values())
    else:
        breakdowns_list = breakdowns

    if len(breakdowns_list) < 2:
        st.info("Need at least 2 breakdowns to compare.")
        return

    breakdowns = breakdowns_list

    # Create comparison table
    comparison_data = []
    for bd in breakdowns:
        # Handle both old and new breakdown structures
        cost = 0
        if hasattr(bd, 'detailed_cost') and bd.detailed_cost:
            cost = bd.detailed_cost.get('cost', 0)

        breakdown_length = 0
        if hasattr(bd, 'informal_breakdown') and bd.informal_breakdown:
            breakdown_length = len(bd.informal_breakdown)

        # Check if parsed - use data model
        parsed = "✅"  # Default to success
        if isinstance(bd, Breakdown):
            # New structure - use ParsedBreakdown directly
            if bd.parse_failure:
                parsed = "❌"  # Parse failed
            elif not bd.parsed_breakdown:
                parsed = "❌"  # Parse failed
        elif hasattr(bd, 'parsed_breakdown'):
            # Old structure
            if hasattr(bd, 'parse_failure') and bd.parse_failure:
                parsed = "❌"
            elif not bd.parsed_breakdown:
                parsed = "❌"
            elif isinstance(bd.parsed_breakdown, dict):
                # Check for errors in dict structure
                parsed_data = bd.parsed_breakdown
                if isinstance(parsed_data, dict) and 'parsed_breakdown' in parsed_data:
                    parsed_data = parsed_data['parsed_breakdown']
                if isinstance(parsed_data, dict) and parsed_data.get('error'):
                    parsed = "❌"

        # Check if formalized - all lemmas and theorem must have at least one successful formalization
        # Use the data model (Breakdown object with ParsedBreakdown)
        formalized = False
        if isinstance(bd, Breakdown):
            if bd.parsed_breakdown:
                # Use the new is_formalized() method which checks if ALL lemmas and theorem
                # have at least one formalization that compiled
                formalized = bd.parsed_breakdown.is_formalized()

        # Check if theorem was proven and count attempts
        theorem_proven = False
        theorem_passing_count = 0
        theorem_total_count = 0

        if bd.parsed_breakdown and bd.parsed_breakdown.theorem:
            theorem_proven = bd.parsed_breakdown.theorem.get_best_attempt() is not None
            # Count all attempts across all formalizations
            for form in bd.parsed_breakdown.theorem.formalizations:
                for attempt in form.proof_attempts:
                    theorem_total_count += 1
                    if attempt.is_passing():
                        theorem_passing_count += 1

        used_lemmas_count, proven_used_lemmas_count = bd.get_used_lemmas_count()
       
        # Check if breakdown is solved (according to our metric)
        solved = bd.is_solved()

        # Get actual compilation result from combined proof
        # If our metric says not solved, just show ❌ (no file should exist)
        # If our metric says solved, show actual compilation result
        if not solved:
            # Not solved, so file shouldn't exist - just show ❌
            actual_compilation_status = "❌"
            metric_vs_actual = "✅"  # Expected behavior - matches our metric
        else:
            # We say it's solved, check if it actually compiled
            actual_compilation_status = "⚠️"  # Default to file not found
            try:
                import json
                from pathlib import Path

                if 'run_dir' in st.session_state:
                    run_path = Path(st.session_state.run_dir)

                    # Extract round number from problem_id
                    round_match = None
                    for part in bd.problem_id.split('_'):
                        if part.startswith('r'):
                            round_match = part
                            break

                    if round_match:
                        round_num = round_match[1:]
                        # Try top-level combined folder first (new location)
                        compilation_results_path = run_path / "combined" / "compilation_results.json"

                        # Fall back to round-specific folder if not found
                        if not compilation_results_path.exists():
                            compilation_results_path = run_path / f"round{round_num}" / "combined" / "compilation_results.json"

                        if compilation_results_path.exists():
                            with open(compilation_results_path, 'r') as f:
                                compilation_results = json.load(f)

                            # Use 3-level fallback strategy to find compilation result
                            try:
                                from id_utils import get_breakdown_id
                            except ImportError:
                                def get_breakdown_id(pid: str) -> str:
                                    """Fallback breakdown ID extraction."""
                                    canonical = str(pid)
                                    if "_l" in canonical:
                                        parts = canonical.rsplit("_l", 1)
                                        if len(parts) == 2 and parts[1].isdigit():
                                            canonical = parts[0]
                                    if "_theorem" in canonical:
                                        canonical = canonical.replace("_theorem", "")
                                    canonical = re.sub(r'_s\d+$', '', canonical)
                                    return canonical

                            found = False

                            # Level 1: Try exact match
                            for item in compilation_results:
                                if item.get('name') == bd.problem_id:
                                    found = True
                                    comp_result = item.get('compilation_result', {})
                                    if comp_result.get('pass', False) and comp_result.get('complete', False):
                                        actual_compilation_status = "✅"
                                    else:
                                        actual_compilation_status = "❌"
                                    break

                            # Level 2: Try breakdown_id match
                            if not found:
                                target_bid = get_breakdown_id(bd.problem_id)
                                for item in compilation_results:
                                    item_name = item.get('name', '')
                                    if item_name and get_breakdown_id(item_name) == target_bid:
                                        found = True
                                        comp_result = item.get('compilation_result', {})
                                        if comp_result.get('pass', False) and comp_result.get('complete', False):
                                            actual_compilation_status = "✅"
                                        else:
                                            actual_compilation_status = "❌"
                                        break

                            # Level 3: Try origin problem fallback
                            if not found:
                                target_bid = get_breakdown_id(bd.problem_id)
                                origin = '_'.join(target_bid.split('_')[:3])
                                for item in compilation_results:
                                    item_name = item.get('name', '')
                                    if item_name and origin in item_name and '_r' in item_name:
                                        found = True
                                        comp_result = item.get('compilation_result', {})
                                        if comp_result.get('pass', False) and comp_result.get('complete', False):
                                            actual_compilation_status = "✅"
                                        else:
                                            actual_compilation_status = "❌"
                                        break

                            if not found:
                                actual_compilation_status = "⚠️"
            except Exception:
                pass  # If we can't load compilation results, default to ⚠️

            # Compare our metric with actual compilation
            if actual_compilation_status == "⚠️":
                metric_vs_actual = "⚠️"  # Can't compare if file not found
            else:
                actual_says_solved = (actual_compilation_status == "✅")
                # We said solved (True), check if actual matches
                metric_vs_actual = "✅" if actual_says_solved else "❌"

        # Extract breakdown info - use parent_problem_id if available (round 1+), otherwise origin_problem_id
        problem_key = bd.parent_problem_id if bd.parent_problem_id else bd.origin_problem_id
        breakdown_info = f"Breakdown {bd.breakdown_id}"
        display_text = f"{problem_key} - {breakdown_info}"

        comparison_data.append({
            "Breakdown": display_text,
            "Parsed": parsed,  # Now displays ✅, ❌, or "X" for parse failures
            "Formalized": "✅" if formalized else "❌",
            "Theorem": f"{theorem_passing_count}/{theorem_total_count} {'✅' if theorem_proven else '❌'}",
            "Used Lemmas": f"{proven_used_lemmas_count}/{used_lemmas_count}" if used_lemmas_count > 0 else "N/A",
            "Solved (Ours)": "✅" if solved else "❌",
            "Actual": actual_compilation_status,
            "_sort_key": (problem_key, bd.breakdown_id),  # Hidden sort key for grouping by parent problem, then breakdown
        })

    # Sort by parent problem (or origin problem), then breakdown_id
    comparison_data_sorted = sorted(comparison_data, key=lambda x: x["_sort_key"])
    # Remove the sort key from display
    for row in comparison_data_sorted:
        del row["_sort_key"]

    st.dataframe(comparison_data_sorted, width="stretch", hide_index=True)


def render_problem_component_costs(breakdowns_list: List[Breakdown]):
    """
    Render problem-level component cost analysis with per-round breakdown.
    Shows component costs across all breakdowns with round-by-round comparison.

    Args:
        breakdowns_list: List of Breakdown objects for this problem
    """
    try:
        import plotly.express as px
    except ImportError:
        st.warning("Plotly not available for component cost visualization")
        return

    if not breakdowns_list:
        st.info("No breakdowns to analyze.")
        return

    st.subheader("💰 Problem-Level Component Costs")

    # Aggregate component costs
    components = ["breakdown", "breakdown_parser", "formalization", "prover"]
    nice_names = ["Breakdown", "Parser", "Formalization", "Prover"]

    component_totals = {c: 0 for c in components}
    round_component_totals = {}  # Dict[int, Dict[str, int]]

    for breakdown in breakdowns_list:
        costs = breakdown.get_component_costs()
        round_id = breakdown.round_id

        if round_id not in round_component_totals:
            round_component_totals[round_id] = {c: 0 for c in components}

        for component in components:
            output_tokens = costs.get(component, {}).get("output_tokens", 0)
            component_totals[component] += output_tokens
            round_component_totals[round_id][component] += output_tokens

    # Display metrics by component
    st.markdown("**Total Output Tokens by Component:**")
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    for component, nice_name, col in zip(components, nice_names, cols):
        total_tokens = component_totals[component]
        with col:
            st.metric(nice_name, f"{total_tokens:,}", "output tokens")

    st.markdown("---")

    # Chart 1: Pie chart showing component distribution
    st.subheader("🥧 Component Distribution (Overall)")

    component_dist = []
    for component, nice_name in zip(components, nice_names):
        tokens = component_totals[component]
        if tokens > 0:
            component_dist.append({"Component": nice_name, "Tokens": tokens})

    if component_dist:
        df_dist = pd.DataFrame(component_dist)
        fig_pie = px.pie(
            df_dist,
            values="Tokens",
            names="Component",
            title="Component Token Distribution (All Rounds)",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True, key=f"problem_component_pie_{id(breakdowns_list)}")
    else:
        st.info("No component cost data available.")

    st.markdown("---")

    # Chart 2: Stacked bar chart by round
    if round_component_totals:
        st.subheader("📊 Component Costs by Round (Stacked)")

        rounds = sorted(round_component_totals.keys())
        df_rounds = pd.DataFrame({
            "Round": [str(r) for r in rounds],
            "Breakdown": [round_component_totals[r]["breakdown"] for r in rounds],
            "Parser": [round_component_totals[r]["breakdown_parser"] for r in rounds],
            "Formalization": [round_component_totals[r]["formalization"] for r in rounds],
            "Prover": [round_component_totals[r]["prover"] for r in rounds]
        })

        # Melt for plotly
        df_melted = df_rounds.melt(
            id_vars=["Round"],
            value_vars=["Breakdown", "Parser", "Formalization", "Prover"],
            var_name="Component",
            value_name="Output Tokens"
        )

        fig = px.bar(
            df_melted,
            x="Round",
            y="Output Tokens",
            color="Component",
            title="Total Output Tokens per Component by Round (Stacked)",
            barmode="stack",
            height=400
        )
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True, key=f"problem_component_stacked_{id(breakdowns_list)}")

        st.markdown("---")

        # Show detailed table
        st.subheader("📋 Detailed Per-Round Component Costs")
        st.dataframe(df_rounds.set_index("Round"), use_container_width=True)


def render_breakdown_component_costs(breakdown: Breakdown):
    """
    Render component-level token cost breakdown for a single breakdown.
    Shows a horizontal bar chart comparing token usage across pipeline components.

    Args:
        breakdown: Breakdown OOP object
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        st.warning("Plotly not available for component cost visualization")
        return

    st.subheader("💰 Component Token Costs")

    # Get component costs
    component_costs = breakdown.get_component_costs()

    if not component_costs or all(c["output_tokens"] == 0 for c in component_costs.values()):
        st.info("No component cost data available for this breakdown.")
        return

    # Prepare data for visualization
    components = ["breakdown", "breakdown_parser", "formalization", "prover"]
    nice_names = ["Breakdown", "Parser", "Formalization", "Prover"]
    output_tokens = [component_costs.get(c, {}).get("output_tokens", 0) for c in components]

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    for i, (col, nice_name, tokens) in enumerate(zip([col1, col2, col3, col4], nice_names, output_tokens)):
        with col:
            st.metric(nice_name, f"{tokens:,}", "output tokens")

    st.markdown("---")

    # Horizontal bar chart
    df = pd.DataFrame({
        "Component": nice_names,
        "Output Tokens": output_tokens
    })

    # Create horizontal bar chart
    fig = px.bar(
        df,
        x="Output Tokens",
        y="Component",
        orientation="h",
        title="Output Tokens per Component",
        labels={"Output Tokens": "Output Tokens"},
        height=400,
        color="Component"
    )
    fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True, key=f"component_costs_bar_{breakdown.problem_id}_{id(breakdown)}")

    st.markdown("---")

    # Show detailed breakdown table
    st.subheader("📋 Detailed Breakdown")
    details_rows = []
    total_tokens = sum(output_tokens)

    for component, nice_name, tokens in zip(components, nice_names, output_tokens):
        percentage = (tokens / total_tokens * 100) if total_tokens > 0 else 0
        details_rows.append({
            "Component": nice_name,
            "Output Tokens": f"{tokens:,}",
            "Percentage": f"{percentage:.1f}%"
        })

    df_details = pd.DataFrame(details_rows)
    st.dataframe(df_details, width="stretch", hide_index=True)


def _handle_load_breakdown(breakdown: Breakdown):
    """
    Handle the load breakdown button click.
    Loads full breakdown data from hierarchical dump into the session.

    Args:
        breakdown: Breakdown object to load data for
    """
    try:
        # Get hierarchical_dir from session state
        hierarchical_dir = st.session_state.get('hierarchical_dir')

        # Get the session object
        session = st.session_state.get('session')
        if not session:
            st.error("❌ Session not loaded. Please load data first.")
            return

        if not hierarchical_dir:
            st.warning("⚠️ Hierarchical directory not found. Cannot load full breakdown data from disk.")
            st.info("💡 Displaying data from current session only. Some details may be limited.")
            return

        # Try to load the breakdown
        try:
            with st.spinner(f"Loading breakdown {breakdown.origin_problem_id} r{breakdown.round_id} b{breakdown.breakdown_id}..."):
                session.load_breakdown(
                    origin_problem_id=breakdown.origin_problem_id,
                    round_id=breakdown.round_id,
                    breakdown_id=breakdown.breakdown_id,
                    hierarchical_dir=hierarchical_dir
                )

            st.success(f"✅ Successfully loaded breakdown data!")
            st.info("💡 The detail view has been updated with full hierarchical data. "
                   "You can now see full code, reasoning traces, and complete formalization details.")

        except (FileNotFoundError, ValueError) as e:
            st.warning(f"⚠️ Breakdown directory not found in hierarchical dump.")
            st.info("💡 Displaying data from current session only. Some details may be limited.")

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        import traceback
        st.error(traceback.format_exc())


def _handle_load_attempts(breakdown: Breakdown, lemma_id: int = -1):
    """
    Handle the load attempts button click.
    Loads full proof attempts from hierarchical dump into the session.

    Args:
        breakdown: Breakdown object to load attempts for
        lemma_id: Lemma ID (-1 for theorem, 0+ for specific lemmas)
    """
    try:
        # Determine what we're loading
        component = "theorem" if lemma_id == -1 else f"lemma {lemma_id}"
        print(f"\n{'='*80}")
        print(f"[LOAD ATTEMPTS] Starting to load {component} proof attempts")
        print(f"[LOAD ATTEMPTS] Problem: {breakdown.origin_problem_id}, Round: {breakdown.round_id}, Breakdown: {breakdown.breakdown_id}")

        # Get hierarchical_dir from session state
        hierarchical_dir = st.session_state.get('hierarchical_dir')

        # Get the session object
        session = st.session_state.get('session')
        if not session:
            print(f"[LOAD ATTEMPTS ERROR] Session not loaded.")
            st.error("❌ Session not loaded.")
            return

        if not hierarchical_dir:
            print(f"[LOAD ATTEMPTS WARNING] Hierarchical directory path not set in session state.")
            st.warning(f"⚠️ Hierarchical directory not found. Cannot load {component} attempts from disk.")
            st.info("💡 Displaying data from current session only.")
            return

        print(f"[LOAD ATTEMPTS] Hierarchical dir: {hierarchical_dir}")

        # Check if the hierarchical directory actually exists
        from pathlib import Path
        hierarchical_path = Path(hierarchical_dir)
        if not hierarchical_path.exists():
            print(f"[LOAD ATTEMPTS WARNING] Hierarchical directory not found at: {hierarchical_path}")
            st.warning(f"⚠️ Hierarchical directory not found at: {hierarchical_path}")
            st.info("💡 Displaying data from current session only.")
            return

        print(f"[LOAD ATTEMPTS] Hierarchical path exists: {hierarchical_path}")

        # Load the attempts
        print(f"[LOAD ATTEMPTS] Calling session.load_attempts()...")
        with st.spinner(f"Loading {component} proof attempts..."):
            session.load_attempts(
                origin_problem_id=breakdown.origin_problem_id,
                round_id=breakdown.round_id,
                breakdown_id=breakdown.breakdown_id,
                lemma_id=lemma_id,
                hierarchical_dir=hierarchical_path
            )

        print(f"[LOAD ATTEMPTS SUCCESS] Successfully loaded {component} proof attempts!")
        print(f"{'='*80}\n")

        # Rerun to refresh the UI with the newly loaded data
        st.rerun()

    except FileNotFoundError as e:
        print(f"[LOAD ATTEMPTS ERROR] File not found: {e}")
        import traceback
        traceback.print_exc()
    except ValueError as e:
        print(f"[LOAD ATTEMPTS ERROR] Invalid data: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"[LOAD ATTEMPTS ERROR] Error loading {component} attempts: {e}")
        import traceback
        traceback.print_exc()
