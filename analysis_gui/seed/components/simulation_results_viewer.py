"""
Simulation results viewer component.

Displays statistics and analysis from proof search strategy simulations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

from seed_prover.simulations.strategies import Simple, Probabilistic, ReasoningTrace
from .problem_browser import render_problem_browser


def render_simulation_results(simulation_results: List[Dict[str, Any]] = None, run_dir: str = None) -> None:
    """
    Render simulation results with summary statistics and per-seed analysis.

    Args:
        simulation_results: List of dicts with 'seed' and 'problems' keys
        run_dir: Path to the run directory (for accessing experiments)
    """
    st.header("🎲 Simulations")

    # Add tab selection between interactive simulations and saved experiments
    sim_mode = st.radio(
        "Simulation Mode",
        options=["Interactive Simulations", "View Experiments"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if sim_mode == "View Experiments" and run_dir:
        from .simulation_experiments_viewer import render_simulation_experiments_viewer
        from pathlib import Path

        render_simulation_experiments_viewer(Path(run_dir))
        return

    # Show simulation control panel at the top
    st.subheader("Run Simulations")

    # Create two columns for inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        num_simulations = st.number_input(
            "Number of Simulations",
            min_value=1,
            max_value=100,
            value=st.session_state.get('sim_num_simulations', 5),
            step=1,
            help="Number of simulations to run with different random seeds"
        )

    with col2:
        strategy_choice = st.selectbox(
            "Strategy",
            options=["Simple", "Probabilistic", "Reasoning Trace Correctness"],
            index=st.session_state.get('sim_strategy_index', 0),
            help="Select which strategy to use for simulations"
        )

    # Initialize variables
    stop_prob = None
    confidence_threshold = None
    explore_steps = None

    with col3:
        if strategy_choice == "Probabilistic":
            stop_prob = st.slider(
                "Stop Probability",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('sim_stop_prob', 0.1),
                step=0.05,
                help="Probability of stopping at each step (0.0 = continue indefinitely, 1.0 = always stop)"
            )
        elif strategy_choice == "Reasoning Trace Correctness":
            # Create a container for the two sliders
            st.write("**Reasoning Trace Configuration**")

            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0,
                max_value=10,
                value=st.session_state.get('sim_confidence_threshold', 5),
                step=1,
                help="Stop when average confidence exceeds this threshold"
            )

            explore_steps = st.slider(
                "Explore Steps",
                min_value=1,
                max_value=10,
                value=st.session_state.get('sim_explore_steps', 1),
                step=1,
                help="Number of steps to explore before evaluating average confidence"
            )

    # Run Simulations button
    col1, col2, _ = st.columns([1, 1, 2])

    run_simulations_clicked = False
    with col1:
        run_simulations_clicked = st.button("▶️ Run Simulations", type="primary", use_container_width=True)

    with col2:
        if st.session_state.get('simulations_completed', False):
            st.button("✅ Last run complete", disabled=True, use_container_width=True)

    # Save settings to session state
    st.session_state.sim_num_simulations = num_simulations
    strategy_index_map = {"Simple": 0, "Probabilistic": 1, "Reasoning Trace Correctness": 2}
    st.session_state.sim_strategy_index = strategy_index_map.get(strategy_choice, 0)
    if stop_prob is not None:
        st.session_state.sim_stop_prob = stop_prob
    if confidence_threshold is not None:
        st.session_state.sim_confidence_threshold = confidence_threshold
    if explore_steps is not None:
        st.session_state.sim_explore_steps = explore_steps

    # Run simulations if button clicked
    if run_simulations_clicked and st.session_state.get('data_loaded') and st.session_state.get('session'):
        try:
            # Clear previous results for memory efficiency
            st.session_state.simulation_results = None
            st.session_state.simulations_completed = False

            st.info(f"Running {num_simulations} simulations with {strategy_choice} strategy...")

            # Run simulations with seeds from 0 to num_simulations-1
            simulation_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for seed in range(num_simulations):
                status_text.text(f"Simulation {seed + 1}/{num_simulations}")

                # Simulate each problem in the session
                simulated_problems = {}
                for problem_id, problem in st.session_state.session.problems.items():
                    try:
                        # Select strategy
                        if strategy_choice == "Simple":
                            strategy = Simple()
                        elif strategy_choice == "Probabilistic":
                            strategy = Probabilistic(stop_prob=stop_prob)
                        else:  # Reasoning Trace Correctness
                            strategy = ReasoningTrace(
                                threshold=confidence_threshold,
                                explore=st.session_state.get('sim_explore_steps', 1)
                            )

                        simulated_problems[problem_id] = problem.simulate(seed=seed, max_depth=None, search_policy="sequential", strategy=strategy)
                    except Exception as sim_error:
                        # Store failure info but continue
                        st.warning(f"Simulation failed for {problem_id} seed {seed}: {sim_error}")
                        simulated_problems[problem_id] = None

                simulation_results.append({
                    'seed': seed,
                    'problems': simulated_problems
                })

                progress_bar.progress((seed + 1) / num_simulations)

            # Store results in session state
            st.session_state.simulation_results = simulation_results
            st.session_state.simulations_completed = True

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            st.success(f"✅ Completed {num_simulations} simulations")
            st.rerun()

        except Exception as e:
            st.error(f"Error running simulations: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    # Display results if they exist
    if not st.session_state.get('simulation_results'):
        st.info("No simulation results yet. Configure the settings above and click 'Run Simulations' to get started.")
        return

    simulation_results = st.session_state.get('simulation_results')

    # Calculate aggregate statistics
    num_seeds = len(simulation_results)
    total_problems = None

    # Collect data for each seed
    seed_data = []
    problem_solve_counts = {}

    for result in simulation_results:
        seed = result['seed']
        problems = result['problems']

        if total_problems is None:
            total_problems = len(problems)

        solved_count = 0
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_prover_calls = 0

        for problem_id, simulated_problem in problems.items():
            if simulated_problem is None:
                continue

            # Track solve status - check if problem is actually solved
            if simulated_problem.is_solved():
                solved_count += 1
                if problem_id not in problem_solve_counts:
                    problem_solve_counts[problem_id] = 0
                problem_solve_counts[problem_id] += 1

            # Aggregate costs
            total_cost += simulated_problem.get_total_cost("cost")
            total_input_tokens += simulated_problem.get_total_cost("input_tokens")
            total_output_tokens += simulated_problem.get_total_cost("output_tokens")
            total_prover_calls += simulated_problem.get_total_cost("prover_calls")
            total_input_sflops += simulated_problem.get_total_cost("input_sflops")
            total_output_sflops += simulated_problem.get_total_cost("output_sflops")

        seed_data.append({
            'Seed': seed,
            'Solved': solved_count,
            'Success Rate': f"{(solved_count/total_problems*100):.1f}%",
            'Total Cost': f"${total_cost:.4f}",
            'Input Tokens': total_input_tokens,
            'Output Tokens': total_output_tokens,
            'Input SFLOPs': total_input_sflops,
            'Output SFLOPs': total_output_sflops,
            'Prover Calls': total_prover_calls,
        })

    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_solved = sum(d['Solved'] for d in seed_data) / num_seeds
        st.metric("Avg Problems Solved", f"{avg_solved:.1f}/{total_problems}")

    with col2:
        avg_success_rate = sum(d['Solved'] for d in seed_data) / (num_seeds * total_problems) * 100
        st.metric("Avg Success Rate", f"{avg_success_rate:.1f}%")

    with col3:
        total_costs = sum(float(d['Total Cost'].replace('$', '')) for d in seed_data)
        avg_cost = total_costs / num_seeds if num_seeds > 0 else 0.0
        st.metric("Avg Cost per Seed", f"${avg_cost:.4f}")

    with col4:
        total_calls = sum(d['Prover Calls'] for d in seed_data)
        avg_calls = total_calls / num_seeds if num_seeds > 0 else 0
        st.metric("Avg Prover Calls", f"{avg_calls:.0f}")

    # Display detailed seed-by-seed results
    st.subheader("Per-Seed Statistics")
    df = pd.DataFrame(seed_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Display consistency analysis (which problems were solved in how many seeds)
    # Collect per-problem statistics across all seeds (for ALL problems, not just solved ones)
    problem_stats = {}
    all_problem_ids = set()

    for result in simulation_results:
        for problem_id, simulated_problem in result['problems'].items():
            all_problem_ids.add(problem_id)

            if simulated_problem is None:
                continue

            if problem_id not in problem_stats:
                problem_stats[problem_id] = {
                    'solve_count': 0,
                    'output_tokens_list': [],
                    'output_sflops_list': [],
                    'prover_calls_list': []
                }

            problem_stats[problem_id]['output_tokens_list'].append(simulated_problem.get_total_cost("output_tokens"))
            problem_stats[problem_id]['output_sflops_list'].append(simulated_problem.get_total_cost("output_sflops"))
            problem_stats[problem_id]['prover_calls_list'].append(simulated_problem.get_total_cost("prover_calls"))

            if simulated_problem.is_solved():
                problem_stats[problem_id]['solve_count'] += 1

    if problem_stats:
        st.subheader("Problem Consistency")
        consistency_data = []
        for problem_id in sorted(all_problem_ids):
            stats = problem_stats.get(problem_id, {})
            solve_count = stats.get('solve_count', 0)

            # Calculate averages from the lists (including unsolved attempts)
            prover_calls_list = stats.get('prover_calls_list', [])
            output_tokens_list = stats.get('output_tokens_list', [])

            avg_output_tokens = sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0
            avg_prover_calls = sum(prover_calls_list) / len(prover_calls_list) if prover_calls_list else 0

            consistency_data.append({
                'Problem': problem_id,
                'Solved in Runs': solve_count,
                'Success Rate': f"{(solve_count/num_seeds)*100:.0f}%",
                'Avg Output Tokens': f"{avg_output_tokens:.0f}",
                'Avg Prover Calls': f"{avg_prover_calls:.1f}",
                'Status': '✅ Always' if solve_count == num_seeds else '⚠️ Sometimes' if solve_count > 0 else '❌ Never'
            })

        df_consistency = pd.DataFrame(consistency_data)
        st.dataframe(df_consistency, use_container_width=True, hide_index=True)

        # Categorize problems
        always_solved = [p for p in all_problem_ids if problem_stats.get(p, {}).get('solve_count', 0) == num_seeds]
        sometimes_solved = [p for p in all_problem_ids if 0 < problem_stats.get(p, {}).get('solve_count', 0) < num_seeds]
        never_solved = [p for p in all_problem_ids if problem_stats.get(p, {}).get('solve_count', 0) == 0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Always Solved", len(always_solved))
        with col2:
            st.metric("Sometimes Solved", len(sometimes_solved))
        with col3:
            st.metric("Never Solved", len(never_solved))

    # Cost analysis
    st.subheader("Cost Analysis")

    costs_per_seed = []
    for result in simulation_results:
        total_cost = 0.0
        for simulated_problem in result['problems'].values():
            if simulated_problem:
                total_cost += simulated_problem.get_total_cost("cost")
        costs_per_seed.append({
            'Seed': result['seed'],
            'Total Cost': total_cost
        })

    if costs_per_seed:
        df_costs = pd.DataFrame(costs_per_seed)
        col1, col2 = st.columns(2)

        with col1:
            st.write("Cost Distribution")
            # Create histogram with matplotlib
            fig, ax = plt.subplots(figsize=(8, 5))
            costs = df_costs['Total Cost'].values
            ax.hist(costs, bins=10, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Total Cost ($)')
            ax.set_ylabel('Count')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)

        with col2:
            avg_cost = df_costs['Total Cost'].mean()
            min_cost = df_costs['Total Cost'].min()
            max_cost = df_costs['Total Cost'].max()

            st.metric("Min Cost", f"${min_cost:.4f}")
            st.metric("Max Cost", f"${max_cost:.4f}")
            st.metric("Avg Cost", f"${avg_cost:.4f}")

    # Per-problem analysis
    st.subheader("Per-Problem Analysis")

    if problem_stats:
        selected_problem = st.selectbox(
            "Select a problem to view simulation details",
            options=sorted(all_problem_ids),
            key="problem_analysis_select"
        )

        if selected_problem:
            stats = problem_stats.get(selected_problem, {})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Solved in Runs", f"{stats.get('solve_count', 0)}/{num_seeds}")
            with col2:
                success_rate = (stats.get('solve_count', 0) / num_seeds * 100) if num_seeds > 0 else 0
                st.metric("Success Rate", f"{success_rate:.0f}%")
            with col3:
                avg_output = sum(stats.get('output_tokens_list', [])) / len(stats.get('output_tokens_list', [])) if stats.get('output_tokens_list') else 0
                st.metric("Avg Output Tokens", f"{avg_output:.0f}")
            with col4:
                avg_calls = sum(stats.get('prover_calls_list', [])) / len(stats.get('prover_calls_list', [])) if stats.get('prover_calls_list') else 0
                st.metric("Avg Prover Calls", f"{avg_calls:.1f}")

            # Distribution of output tokens and prover calls across seeds
            output_tokens_list = stats.get('output_tokens_list', [])
            prover_calls_list = stats.get('prover_calls_list', [])

            col1, col2 = st.columns(2)

            with col1:
                if output_tokens_list:
                    st.write("Output Tokens Distribution")
                    # Create histogram with matplotlib
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(output_tokens_list, bins=10, edgecolor='black', alpha=0.7)
                    ax.set_xlabel('Output Tokens')
                    ax.set_ylabel('Count')
                    ax.grid(axis='y', alpha=0.3)
                    st.pyplot(fig)

            with col2:
                if prover_calls_list:
                    st.write("Prover Calls Distribution")
                    # Create histogram with matplotlib
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(prover_calls_list, bins=10, edgecolor='black', alpha=0.7)
                    ax.set_xlabel('Prover Calls')
                    ax.set_ylabel('Count')
                    ax.grid(axis='y', alpha=0.3)
                    st.pyplot(fig)

    # Problem browser for each seed
    st.subheader("Problems by Seed")

    if simulation_results:
        # Create tabs for each seed
        seed_tabs = st.tabs([f"Seed {result['seed']}" for result in simulation_results])

        for tab, result in zip(seed_tabs, simulation_results):
            with tab:
                # Get simulated problems for this seed and convert to list
                simulated_problems = result['problems']
                problems_list = [p for p in simulated_problems.values() if p is not None]

                if problems_list:
                    render_problem_browser(problems_list, key_prefix=f"seed_{result['seed']}_")
                else:
                    st.info("No problems in this simulation.")

    # Download results as CSV
    st.subheader("Export Results")
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download seed results as CSV",
        data=csv_data,
        file_name="simulation_results.csv",
        mime="text/csv"
    )
