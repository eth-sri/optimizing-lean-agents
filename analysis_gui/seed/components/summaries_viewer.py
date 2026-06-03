"""
Summaries viewer component.

Displays statistics about reasoning traces and compilation summaries.
Includes:
1. Histogram of confidence scores from reasoning summaries
2. Proof success rate by confidence score
3. Compilation error frequency plot
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter


def render_summaries_viewer(session) -> None:
    """
    Render summaries analysis with reasoning traces and compilation summaries.

    Args:
        session: The Session object with all loaded data
    """
    st.header("📊 Summaries")

    if not session or not session.problems:
        st.info("No data loaded. Please load a run directory first.")
        return

    # Collect all proof attempts from the session
    all_attempts = _collect_all_attempts(session)

    if not all_attempts:
        st.warning("No proof attempts found in the session.")
        return

    # Create tabs for reasoning traces and compilation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Reasoning Traces",
        "✓ Correctness Scores",
        "❌ Compilation Errors",
        "📄 Proof",
        "💰 Cost Analysis"
    ])

    # Tab 1: Reasoning traces (confidence distribution and success rate)
    with tab1:
        render_reasoning_traces(all_attempts, session)

    # Tab 2: Correctness scores
    with tab2:
        render_correctness_analysis(all_attempts, session)

    # Tab 3: Compilation error frequency
    with tab3:
        render_compilation_errors(all_attempts)
        st.markdown("---")
        render_error_correctability_analysis(session)

    # Tab 4: Proof analysis (proof length, etc.)
    with tab4:
        render_proof_analysis(all_attempts, session)

    # Tab 5: Cost analysis
    with tab5:
        render_cost_analysis(session)


def _collect_all_attempts(session) -> List[Dict[str, Any]]:
    """
    Collect all proof attempts from all problems/breakdowns/formalizations.

    Returns:
        List of dictionaries with attempt info and reasoning_summary, compilation_summary
    """
    attempts = []

    for problem in session.problems.values():
        for breakdown in problem.breakdowns.values():
            if not breakdown.parsed_breakdown:
                continue

            # Collect theorem attempts
            theorem = breakdown.parsed_breakdown.theorem
            for formalization in theorem.formalizations:
                for attempt in formalization.proof_attempts:
                    attempts.append({
                        'problem_id': problem.origin_problem_id,
                        'breakdown_id': breakdown.breakdown_id,
                        'lemma_id': attempt.lemma_id,
                        'is_passing': attempt.is_passing(),
                        'iteration_id': attempt.iteration_id,
                        'correction_round_id': attempt.correction_round_id,
                        'reasoning_summary': attempt.reasoning_summary,
                        'compilation_summary': attempt.compilation_summary,
                        'used_lemma_ids': attempt.used_lemma_ids,
                        'attempt': attempt
                    })

            # Collect lemma attempts
            for lemma in breakdown.parsed_breakdown.lemmas.values():
                for formalization in lemma.formalizations:
                    for attempt in formalization.proof_attempts:
                        attempts.append({
                            'problem_id': problem.origin_problem_id,
                            'breakdown_id': breakdown.breakdown_id,
                            'lemma_id': attempt.lemma_id,
                            'is_passing': attempt.is_passing(),
                            'iteration_id': attempt.iteration_id,
                            'correction_round_id': attempt.correction_round_id,
                            'reasoning_summary': attempt.reasoning_summary,
                            'compilation_summary': attempt.compilation_summary,
                            'used_lemma_ids': attempt.used_lemma_ids,
                            'attempt': attempt
                        })

    return attempts


def render_reasoning_traces(all_attempts: List[Dict[str, Any]], session=None) -> None:
    """
    Render reasoning traces analysis combining confidence distribution and success rate.

    Shows:
    - Histogram of confidence scores
    - Table of success rates by confidence
    - Trend plot of success rate vs confidence (with difficulty breakdown)
    - Confidence impact on correction round success
    - Average confidence by iteration
    - Average confidence by correction round
    - Average confidence per origin_problem_id
    - Average confidence per theorem
    - Average confidence per lemma
    """
    render_confidence_histogram(all_attempts)
    st.markdown("---")
    render_success_by_confidence(all_attempts, session)
    st.markdown("---")
    render_confidence_vs_correction_success(all_attempts)
    st.markdown("---")
    render_confidence_by_iteration(all_attempts)
    st.markdown("---")
    render_confidence_by_correction_round(all_attempts)
    st.markdown("---")
    render_confidence_per_origin_problem(all_attempts)
    st.markdown("---")
    render_confidence_per_theorem(all_attempts)
    st.markdown("---")
    render_confidence_per_lemma(all_attempts)
    st.markdown("---")
    render_confidence_vs_correct_proof_likelihood(all_attempts)
    st.markdown("---")
    render_confidence_threshold_analysis(all_attempts)
    st.markdown("---")
    render_min_max_confidence_histograms(all_attempts)


def render_correctness_analysis(all_attempts: List[Dict[str, Any]], session=None) -> None:
    """
    Render correctness score analysis combining histogram and success rate.

    Shows:
    - Histogram of correctness scores
    - Table of success rates by correctness
    - Trend plot of success rate vs correctness (with difficulty breakdown)
    - Correctness impact on correction round success
    - Average correctness by iteration
    - Average correctness by correction round
    - Average correctness per origin_problem_id
    - Average correctness per theorem
    - Average correctness per lemma
    """
    render_correctness_histogram(all_attempts)
    st.markdown("---")
    render_success_by_correctness(all_attempts, session)
    st.markdown("---")
    render_correctness_vs_correction_success(all_attempts)
    st.markdown("---")
    render_correctness_by_iteration(all_attempts, session)
    st.markdown("---")
    render_correctness_by_correction_round(all_attempts, session)


def render_correctness_histogram(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render a histogram of correctness scores from reasoning summaries.

    Shows:
    - Histogram of correctness values (scale 1-10)
    - Count of None values
    """
    st.subheader("Reasoning Correctness Distribution")

    correctness_scores = []
    none_count = 0

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')

        if reasoning_summary is None:
            none_count += 1
        elif isinstance(reasoning_summary, dict):
            correctness = reasoning_summary.get('correctness')
            if correctness is not None:
                correctness_scores.append(float(correctness))
            else:
                none_count += 1
        else:
            none_count += 1

    # Display statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Attempts", len(all_attempts))

    with col2:
        st.metric("With Correctness", len(correctness_scores))

    with col3:
        st.metric("None Correctness", none_count)

    with col4:
        if correctness_scores:
            avg_correctness = np.mean(correctness_scores)
            st.metric("Avg Correctness", f"{avg_correctness:.2f}")

    # Create histogram
    if correctness_scores:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histogram with bins aligned to integer values (1-2, 2-3, ..., 10-11)
        bins = np.arange(0.5, 11.5, 1)
        ax.hist(correctness_scores, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

        ax.set_xlabel('Correctness Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Reasoning Correctness Scores', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 11))
        ax.set_xticklabels(range(1, 11))
        ax.grid(axis='y', alpha=0.3)

        # Add a vertical line for the mean
        if correctness_scores:
            mean_val = np.mean(correctness_scores)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.legend()

        st.pyplot(fig)

        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min", f"{np.min(correctness_scores):.2f}")
        with col2:
            st.metric("Median", f"{np.median(correctness_scores):.2f}")
        with col3:
            st.metric("Max", f"{np.max(correctness_scores):.2f}")
        with col4:
            std_dev = np.std(correctness_scores)
            st.metric("Std Dev", f"{std_dev:.2f}")
    else:
        st.warning("No correctness scores found in reasoning summaries.")


def render_success_by_correctness(all_attempts: List[Dict[str, Any]], session=None) -> None:
    """
    Render a table showing proof success rate by correctness score.

    Shows:
    - Individual correctness scores (integers 1-10, or binned by integer ranges if correctness > 10)
    - Total proofs at each correctness level
    - Passed proofs (both passing & complete)
    - Success rate percentage
    - Trend plot with difficulty breakdown (if session is available)
    """
    st.subheader("Proof Success Rate by Correctness Score")

    # Collect correctness scores and success status
    correctness_success_map = defaultdict(lambda: {'total': 0, 'passed': 0})

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')

        if reasoning_summary and isinstance(reasoning_summary, dict):
            correctness = reasoning_summary.get('correctness')
            if correctness is not None:
                correctness_val = float(correctness)
                is_passed = 1 if attempt_info['is_passing'] else 0

                # Check if correctness is an integer or near-integer
                if correctness_val == int(correctness_val):
                    # Use the integer as-is
                    key = int(correctness_val)
                else:
                    # For non-integer correctness, bin by integer ranges (1-2, 2-3, etc.)
                    key = f"{int(correctness_val)}-{int(correctness_val) + 1}"

                correctness_success_map[key]['total'] += 1
                correctness_success_map[key]['passed'] += is_passed

    if not correctness_success_map:
        st.warning("No correctness scores found in reasoning summaries.")
        return

    # Sort keys: integers first in ascending order, then range strings
    integer_keys = sorted([k for k in correctness_success_map.keys() if isinstance(k, int)])
    range_keys = sorted([k for k in correctness_success_map.keys() if isinstance(k, str)])
    sorted_keys = integer_keys + range_keys

    # Create summary table
    table_data = []
    for key in sorted_keys:
        stats = correctness_success_map[key]
        total_count = stats['total']
        success_count = stats['passed']
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0

        # Format the correctness label
        if isinstance(key, int):
            correctness_label = str(key)
        else:
            correctness_label = key

        table_data.append({
            'Correctness': correctness_label,
            'Total Proofs': int(total_count),
            'Passed Proofs': int(success_count),
            'Success Rate': f"{success_rate:.1f}%"
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Plot success rate trend
        st.subheader("Success Rate Trend")

        # Prepare data for plotting
        correctnesses = []
        success_rates = []

        for item in table_data:
            corr_label = item['Correctness']
            success_rate_str = item['Success Rate'].rstrip('%')
            success_rate = float(success_rate_str)

            # Extract numeric value from label (handle both "5" and "5-6" formats)
            if '-' in corr_label:
                # For ranges like "1-2", use the start value
                corr_num = float(corr_label.split('-')[0])
            else:
                # For integers like "5"
                corr_num = float(corr_label)

            correctnesses.append(corr_num)
            success_rates.append(success_rate)

        # Create trend plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot line with markers
        ax.plot(correctnesses, success_rates, marker='o', linewidth=2, markersize=8, color='steelblue', label='Success Rate')

        # Add shaded area under the curve
        ax.fill_between(correctnesses, success_rates, alpha=0.3, color='steelblue')

        ax.set_xlabel('Correctness Score', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Proof Success Rate by Correctness Score', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 105])

        # Add value labels on points
        for x, y in zip(correctnesses, success_rates):
            ax.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom', fontsize=9)

        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # Additional statistics
        st.subheader("Overall Statistics")
        col1, col2, col3 = st.columns(3)

        total_proofs = sum(stats['total'] for stats in correctness_success_map.values())
        passed_proofs = sum(stats['passed'] for stats in correctness_success_map.values())
        overall_success_rate = (passed_proofs / total_proofs * 100) if total_proofs > 0 else 0

        with col1:
            st.metric("Total Proofs with Correctness", total_proofs)
        with col2:
            st.metric("Passed Proofs", passed_proofs)
        with col3:
            st.metric("Overall Success Rate", f"{overall_success_rate:.1f}%")
    else:
        st.warning("No data to display.")


def render_correctness_vs_correction_success(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Analyze relationship between correctness score and success rate in correction rounds.
    Shows line plot with one line per correction round.
    """
    st.subheader("Correctness Score Impact on Correction Round Success")

    # Collect data for correction round analysis by binned correctness scores
    correction_round_correctness_map = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'passed': 0}))

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        correction_round_id = attempt_info.get('correction_round_id', 0)

        if reasoning_summary and isinstance(reasoning_summary, dict):
            correctness = reasoning_summary.get('correctness')
            if correctness is not None:
                correctness_val = float(correctness)
                is_passed = 1 if attempt_info['is_passing'] else 0

                # Bin correctness by integer scores only (for cleaner line plot)
                key = int(correctness_val)

                correction_round_correctness_map[correction_round_id][key]['total'] += 1
                correction_round_correctness_map[correction_round_id][key]['passed'] += is_passed

    if not correction_round_correctness_map:
        st.warning("No correction round data available.")
        return

    # Sort correction rounds
    correction_rounds_sorted = sorted(correction_round_correctness_map.keys())

    # Get all integer correctness scores (1-10)
    all_correctness_scores = set()
    for round_data in correction_round_correctness_map.values():
        all_correctness_scores.update(round_data.keys())

    sorted_correctness_scores = sorted([k for k in all_correctness_scores if isinstance(k, int)])

    # Define colors for each correction round
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_map = {round_id: colors[idx % len(colors)] for idx, round_id in enumerate(correction_rounds_sorted)}

    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot one line per correction round
    for corr_round in correction_rounds_sorted:
        correctness_scores = []
        success_rates = []

        for correctness_score in sorted_correctness_scores:
            stats = correction_round_correctness_map[corr_round].get(correctness_score, {'total': 0, 'passed': 0})
            if stats['total'] > 0:
                success_rate = (stats['passed'] / stats['total'] * 100)
                correctness_scores.append(correctness_score)
                success_rates.append(success_rate)

        # Plot line with markers for this correction round
        if correctness_scores:
            color = color_map[corr_round]
            ax.plot(correctness_scores, success_rates, marker='o', linewidth=2.5, markersize=8,
                    label=f"Round {corr_round}", color=color, alpha=0.8)

            # Add value labels on points
            for x, y in zip(correctness_scores, success_rates):
                ax.text(x, y + 2, f'{y:.0f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Correctness Score', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate vs Correctness Score by Correction Round', fontsize=14, fontweight='bold')
    ax.set_xticks(sorted_correctness_scores)
    ax.set_xticklabels(sorted_correctness_scores, fontsize=10)
    ax.set_ylim([-5, 110])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()
    st.pyplot(fig)


def render_correctness_by_iteration(all_attempts: List[Dict[str, Any]], session=None) -> None:
    """
    Render success rate vs correctness score by iteration and difficulty.
    One plot per iteration, with three lines (one per difficulty: easy, medium, hard).
    """
    st.subheader("Success Rate vs Correctness Score by Iteration")

    # Build a map: problem_id -> difficulty
    problem_difficulty_map = {}
    if session:
        for problem in session.problems.values():
            problem_difficulty_map[problem.origin_problem_id] = problem.difficulty

    # Collect data: iteration -> correctness -> difficulty -> {total, passed}
    iteration_correctness_difficulty_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'total': 0, 'passed': 0})))

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        iteration_id = attempt_info.get('iteration_id', 0)
        problem_id = attempt_info.get('problem_id')

        if reasoning_summary and isinstance(reasoning_summary, dict):
            correctness = reasoning_summary.get('correctness')
            if correctness is not None:
                correctness_val = float(correctness)
                correctness_key = int(correctness_val)
                is_passed = 1 if attempt_info['is_passing'] else 0

                # Get difficulty
                difficulty = problem_difficulty_map.get(problem_id, 'Unknown') if session else 'Unknown'

                iteration_correctness_difficulty_map[iteration_id][correctness_key][difficulty]['total'] += 1
                iteration_correctness_difficulty_map[iteration_id][correctness_key][difficulty]['passed'] += is_passed

    if not iteration_correctness_difficulty_map:
        st.warning("No iteration data available for correctness breakdown.")
        return

    # Sort iterations
    iterations_sorted = sorted(iteration_correctness_difficulty_map.keys())
    num_iterations = len(iterations_sorted)

    # Get all difficulties
    all_difficulties = set()
    for iter_data in iteration_correctness_difficulty_map.values():
        for difficulty_data in iter_data.values():
            all_difficulties.update(difficulty_data.keys())

    # Sort difficulties: easy, medium, hard
    difficulties_sorted = sorted(all_difficulties,
                                 key=lambda x: {'easy': 0, 'Easy': 0, 'medium': 1, 'Medium': 1, 'hard': 2, 'Hard': 2}.get(x, 3))

    # Define colors for each difficulty
    difficulty_colors = {
        'easy': 'green',
        'Easy': 'green',
        'medium': 'orange',
        'Medium': 'orange',
        'hard': 'red',
        'Hard': 'red'
    }

    # Create subplots: one per iteration, arranged in a row
    num_cols = min(4, num_iterations)
    num_rows = (num_iterations + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Handle single plot case
    if num_iterations == 1:
        axes_flat = [axes]
    elif num_rows == 1:
        axes_flat = list(axes) if isinstance(axes, np.ndarray) else [axes]
    else:
        axes_flat = axes.flatten().tolist()

    # Plot for each iteration
    for idx, iteration_id in enumerate(iterations_sorted):
        ax = axes_flat[idx]
        correctness_data = iteration_correctness_difficulty_map[iteration_id]

        # Get all correctness scores for this iteration
        sorted_correctness_scores = sorted(correctness_data.keys())

        # Plot a line for each difficulty
        for difficulty in difficulties_sorted:
            correctness_scores = []
            success_rates = []

            for correctness_score in sorted_correctness_scores:
                stats = correctness_data[correctness_score].get(difficulty, {'total': 0, 'passed': 0})
                if stats['total'] > 0:
                    success_rate = (stats['passed'] / stats['total'] * 100)
                    correctness_scores.append(correctness_score)
                    success_rates.append(success_rate)

            # Plot line for this difficulty
            if correctness_scores:
                color = difficulty_colors.get(difficulty, 'blue')
                difficulty_label = difficulty.capitalize() if difficulty else 'Unknown'
                ax.plot(correctness_scores, success_rates, marker='o', linewidth=2, markersize=6,
                        label=difficulty_label, color=color, alpha=0.8)

        ax.set_xlabel('Correctness Score', fontsize=10)
        ax.set_ylabel('Success Rate (%)', fontsize=10)
        ax.set_title(f'Iteration {iteration_id}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 105])
        ax.legend(loc='best', fontsize=9)

    # Hide extra subplots if any
    for idx in range(num_iterations, len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle('Success Rate vs Correctness Score by Iteration and Difficulty', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)


def render_correctness_by_correction_round(all_attempts: List[Dict[str, Any]], session=None) -> None:
    """
    Render success rate vs correctness score by correction round and difficulty.
    One row per correction round, with multiple plots per row (one per difficulty comparison).
    """
    st.subheader("Success Rate vs Correctness Score by Correction Round")

    # Build a map: problem_id -> difficulty
    problem_difficulty_map = {}
    if session:
        for problem in session.problems.values():
            problem_difficulty_map[problem.origin_problem_id] = problem.difficulty

    # Collect data: correction_round -> correctness -> difficulty -> {total, passed}
    correction_round_correctness_difficulty_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'total': 0, 'passed': 0})))

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        correction_round_id = attempt_info.get('correction_round_id', 0)
        problem_id = attempt_info.get('problem_id')

        if reasoning_summary and isinstance(reasoning_summary, dict):
            correctness = reasoning_summary.get('correctness')
            if correctness is not None:
                correctness_val = float(correctness)
                correctness_key = int(correctness_val)
                is_passed = 1 if attempt_info['is_passing'] else 0

                # Get difficulty
                difficulty = problem_difficulty_map.get(problem_id, 'Unknown') if session else 'Unknown'

                correction_round_correctness_difficulty_map[correction_round_id][correctness_key][difficulty]['total'] += 1
                correction_round_correctness_difficulty_map[correction_round_id][correctness_key][difficulty]['passed'] += is_passed

    if not correction_round_correctness_difficulty_map:
        st.warning("No correction round data available.")
        return

    # Sort correction rounds
    correction_rounds_sorted = sorted(correction_round_correctness_difficulty_map.keys())
    num_correction_rounds = len(correction_rounds_sorted)

    # Get all difficulties
    all_difficulties = set()
    for round_data in correction_round_correctness_difficulty_map.values():
        for difficulty_data in round_data.values():
            all_difficulties.update(difficulty_data.keys())

    # Sort difficulties: easy, medium, hard
    difficulties_sorted = sorted(all_difficulties,
                                 key=lambda x: {'easy': 0, 'Easy': 0, 'medium': 1, 'Medium': 1, 'hard': 2, 'Hard': 2}.get(x, 3))

    # Define colors for each difficulty
    difficulty_colors = {
        'easy': 'green',
        'Easy': 'green',
        'medium': 'orange',
        'Medium': 'orange',
        'hard': 'red',
        'Hard': 'red'
    }

    # Create subplots: one row per correction round, max 4 plots per row
    num_cols = min(4, num_correction_rounds)
    num_rows = num_correction_rounds
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Handle single plot case
    if num_correction_rounds == 1:
        axes_flat = [axes] if num_cols == 1 else list(axes)
    else:
        axes_flat = axes.flatten().tolist() if num_rows > 1 else (list(axes) if isinstance(axes, np.ndarray) else [axes])

    # Plot for each correction round
    for idx, correction_round_id in enumerate(correction_rounds_sorted):
        ax = axes_flat[idx]
        correctness_data = correction_round_correctness_difficulty_map[correction_round_id]

        # Get all correctness scores for this correction round
        sorted_correctness_scores = sorted(correctness_data.keys())

        # Plot a line for each difficulty
        for difficulty in difficulties_sorted:
            correctness_scores = []
            success_rates = []

            for correctness_score in sorted_correctness_scores:
                stats = correctness_data[correctness_score].get(difficulty, {'total': 0, 'passed': 0})
                if stats['total'] > 0:
                    success_rate = (stats['passed'] / stats['total'] * 100)
                    correctness_scores.append(correctness_score)
                    success_rates.append(success_rate)

            # Plot line for this difficulty
            if correctness_scores:
                color = difficulty_colors.get(difficulty, 'blue')
                difficulty_label = difficulty.capitalize() if difficulty else 'Unknown'
                ax.plot(correctness_scores, success_rates, marker='o', linewidth=2, markersize=6,
                        label=difficulty_label, color=color, alpha=0.8)

        ax.set_xlabel('Correctness Score', fontsize=10)
        ax.set_ylabel('Success Rate (%)', fontsize=10)
        ax.set_title(f'Correction Round {correction_round_id}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 105])
        ax.legend(loc='best', fontsize=9)

    # Hide extra subplots if any
    for idx in range(num_correction_rounds, len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle('Success Rate vs Correctness Score by Correction Round and Difficulty', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)


def render_confidence_histogram(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render a histogram of confidence scores from reasoning summaries.

    Shows:
    - Histogram of confidence values
    - Count of None values
    """
    st.subheader("Reasoning Confidence Distribution")

    confidence_scores = []
    none_count = 0

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')

        if reasoning_summary is None:
            none_count += 1
        elif isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                confidence_scores.append(float(confidence))
            else:
                none_count += 1
        else:
            none_count += 1

    # Display statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Attempts", len(all_attempts))

    with col2:
        st.metric("With Confidence", len(confidence_scores))

    with col3:
        st.metric("None Confidence", none_count)

    with col4:
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")

    # Create histogram
    if confidence_scores:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histogram with bins aligned to integer values (0-1, 1-2, ..., 10-11)
        bins = np.arange(-0.5, 11.5, 1)
        counts, bin_edges, patches = ax.hist(confidence_scores, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Reasoning Confidence Scores', fontsize=14, fontweight='bold')
        ax.set_xticks(range(0, 11))
        ax.set_xticklabels(range(0, 11))
        ax.grid(axis='y', alpha=0.3)

        # Add a vertical line for the mean
        if confidence_scores:
            mean_val = np.mean(confidence_scores)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.legend()

        st.pyplot(fig)

        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min", f"{np.min(confidence_scores):.2f}")
        with col2:
            st.metric("Median", f"{np.median(confidence_scores):.2f}")
        with col3:
            st.metric("Max", f"{np.max(confidence_scores):.2f}")
        with col4:
            std_dev = np.std(confidence_scores)
            st.metric("Std Dev", f"{std_dev:.2f}")
    else:
        st.warning("No confidence scores found in reasoning summaries.")


def render_success_by_confidence(all_attempts: List[Dict[str, Any]], session=None) -> None:
    """
    Render a table showing proof success rate by confidence score.

    Shows:
    - Individual confidence scores (integers 0-10, or binned by integer ranges if confidence > 10)
    - Total proofs at each confidence level
    - Passed proofs (both passing & complete)
    - Success rate percentage
    - Trend plot with difficulty breakdown (if session is available)
    """
    st.subheader("Proof Success Rate by Confidence Score")

    # Collect confidence scores and success status
    confidence_success_map = defaultdict(lambda: {'total': 0, 'passed': 0})

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')

        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                confidence_val = float(confidence)
                is_passed = 1 if attempt_info['is_passing'] else 0

                # Check if confidence is an integer or near-integer
                if confidence_val == int(confidence_val):
                    # Use the integer as-is
                    key = int(confidence_val)
                else:
                    # For non-integer confidences, bin by integer ranges (0-1, 1-2, etc.)
                    key = f"{int(confidence_val)}-{int(confidence_val) + 1}"

                confidence_success_map[key]['total'] += 1
                confidence_success_map[key]['passed'] += is_passed

    if not confidence_success_map:
        st.warning("No confidence scores found in reasoning summaries.")
        return

    # Sort keys: integers first in ascending order, then range strings
    integer_keys = sorted([k for k in confidence_success_map.keys() if isinstance(k, int)])
    range_keys = sorted([k for k in confidence_success_map.keys() if isinstance(k, str)])
    sorted_keys = integer_keys + range_keys

    # Create summary table
    table_data = []
    for key in sorted_keys:
        stats = confidence_success_map[key]
        total_count = stats['total']
        success_count = stats['passed']
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0

        # Format the confidence label
        if isinstance(key, int):
            confidence_label = str(key)
        else:
            confidence_label = key

        table_data.append({
            'Confidence': confidence_label,
            'Total Proofs': int(total_count),
            'Passed Proofs': int(success_count),
            'Success Rate': f"{success_rate:.1f}%"
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Plot success rate trend
        st.subheader("Success Rate Trend")

        # Prepare data for plotting
        confidences = []
        success_rates = []

        for item in table_data:
            conf_label = item['Confidence']
            success_rate_str = item['Success Rate'].rstrip('%')
            success_rate = float(success_rate_str)

            # Extract numeric value from label (handle both "5" and "5-6" formats)
            if '-' in conf_label:
                # For ranges like "0-1", use the start value
                conf_num = float(conf_label.split('-')[0])
            else:
                # For integers like "5"
                conf_num = float(conf_label)

            confidences.append(conf_num)
            success_rates.append(success_rate)

        # Create trend plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot line with markers
        ax.plot(confidences, success_rates, marker='o', linewidth=2, markersize=8, color='steelblue', label='Success Rate')

        # Add shaded area under the curve
        ax.fill_between(confidences, success_rates, alpha=0.3, color='steelblue')

        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Proof Success Rate by Confidence Score', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 105])

        # Add value labels on points
        for x, y in zip(confidences, success_rates):
            ax.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom', fontsize=9)

        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # Additional statistics
        st.subheader("Overall Statistics")
        col1, col2, col3 = st.columns(3)

        total_proofs = sum(stats['total'] for stats in confidence_success_map.values())
        passed_proofs = sum(stats['passed'] for stats in confidence_success_map.values())
        overall_success_rate = (passed_proofs / total_proofs * 100) if total_proofs > 0 else 0

        with col1:
            st.metric("Total Proofs with Confidence", total_proofs)
        with col2:
            st.metric("Passed Proofs", passed_proofs)
        with col3:
            st.metric("Overall Success Rate", f"{overall_success_rate:.1f}%")

        # Difficulty breakdown if session is available
        if session:
            render_success_rate_by_difficulty(all_attempts, session)
            st.markdown("---")
            render_success_rate_by_iteration(all_attempts, session)
    else:
        st.warning("No data to display.")


def render_success_rate_by_iteration(all_attempts: List[Dict[str, Any]], session=None) -> None:
    """
    Render success rate by confidence score for each iteration of the prover.

    First shows plots of confidence vs success rate for each iteration (one plot per iteration).
    Then shows plots of confidence vs success rate for each iteration, with multiple lines
    showing breakdown by difficulty (Easy, Medium, Hard).
    """
    st.subheader("Success Rate by Confidence Score - By Iteration")

    # Build a map: problem_id -> difficulty
    problem_difficulty_map = {}
    if session:
        for problem in session.problems.values():
            problem_difficulty_map[problem.origin_problem_id] = problem.difficulty

    # Collect confidence scores, success status, and iteration (overall)
    iteration_confidence_map = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'passed': 0}))

    # Collect confidence scores, success status, iteration, and difficulty
    iteration_difficulty_confidence_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'total': 0, 'passed': 0})))

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        iteration_id = attempt_info.get('iteration_id', 0)
        problem_id = attempt_info.get('problem_id')

        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                confidence_val = float(confidence)
                is_passed = 1 if attempt_info['is_passing'] else 0

                # Bin confidence
                if confidence_val == int(confidence_val):
                    key = int(confidence_val)
                else:
                    key = f"{int(confidence_val)}-{int(confidence_val) + 1}"

                # Overall iteration data
                iteration_confidence_map[iteration_id][key]['total'] += 1
                iteration_confidence_map[iteration_id][key]['passed'] += is_passed

                # Iteration + difficulty data
                if session:
                    difficulty = problem_difficulty_map.get(problem_id, 'Unknown')
                    iteration_difficulty_confidence_map[iteration_id][difficulty][key]['total'] += 1
                    iteration_difficulty_confidence_map[iteration_id][difficulty][key]['passed'] += is_passed

    if not iteration_confidence_map:
        st.warning("No iteration data available for confidence breakdown.")
        return

    # Sort iterations
    iterations_sorted = sorted(iteration_confidence_map.keys())

    # ===== SECTION 1: Overall plots (one per iteration) =====
    st.markdown("**Overall Success Rate by Iteration**")

    num_iterations = len(iterations_sorted)
    num_cols = min(3, num_iterations)
    num_rows = (num_iterations + num_cols - 1) // num_cols

    # Create subplots for overall iteration plots
    fig1, axes1 = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Handle single plot case
    if num_iterations == 1:
        axes1_flat = [axes1]
    elif num_rows == 1:
        axes1_flat = list(axes1) if isinstance(axes1, np.ndarray) else [axes1]
    else:
        axes1_flat = axes1.flatten().tolist()

    # Plot for each iteration (overall, no difficulty breakdown)
    for idx, iteration_id in enumerate(iterations_sorted):
        ax = axes1_flat[idx]
        confidence_data = iteration_confidence_map[iteration_id]

        # Sort keys
        integer_keys = sorted([k for k in confidence_data.keys() if isinstance(k, int)])
        range_keys = sorted([k for k in confidence_data.keys() if isinstance(k, str)])
        sorted_keys = integer_keys + range_keys

        # Prepare data
        confidences = []
        success_rates = []
        counts = []

        for key in sorted_keys:
            stats = confidence_data[key]
            total_count = stats['total']
            success_count = stats['passed']
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0

            # Extract numeric value from label
            if isinstance(key, int):
                conf_num = float(key)
            else:
                conf_num = float(key.split('-')[0])

            confidences.append(conf_num)
            success_rates.append(success_rate)
            counts.append(total_count)

        # Plot line with markers
        ax.plot(confidences, success_rates, marker='o', linewidth=2, markersize=6,
                color='steelblue', alpha=0.8)

        # Add shaded area under curve
        ax.fill_between(confidences, success_rates, alpha=0.2, color='steelblue')

        ax.set_xlabel('Confidence Score', fontsize=10)
        ax.set_ylabel('Success Rate (%)', fontsize=10)
        ax.set_title(f'Iteration {iteration_id}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 105])

        # Add value labels on points with count
        for x, y, count in zip(confidences, success_rates, counts):
            label_text = f'{y:.0f}%\n(n={count})'
            ax.text(x, y + 3, label_text, ha='center', va='bottom', fontsize=8)

    # Hide extra subplots if any
    for idx in range(num_iterations, len(axes1_flat)):
        axes1_flat[idx].axis('off')

    plt.tight_layout()
    st.pyplot(fig1)

    # ===== SECTION 2: Difficulty breakdown plots (one per iteration with multiple lines) =====
    if session and iteration_difficulty_confidence_map:
        st.markdown("---")
        st.markdown("**Success Rate by Confidence Score - Difficulty Breakdown per Iteration**")

        # Get all difficulties
        all_difficulties = set()
        for iter_data in iteration_difficulty_confidence_map.values():
            all_difficulties.update(iter_data.keys())

        # Sort difficulties: easy, medium, hard
        difficulties_sorted = sorted(all_difficulties,
                                     key=lambda x: {'easy': 0, 'Easy': 0, 'medium': 1, 'Medium': 1, 'hard': 2, 'Hard': 2}.get(x, 3))

        # Define colors for each difficulty
        difficulty_colors = {
            'easy': 'green',
            'Easy': 'green',
            'medium': 'orange',
            'Medium': 'orange',
            'hard': 'red',
            'Hard': 'red'
        }

        # Create subplots for difficulty breakdown (one plot per iteration, with multiple lines)
        fig2, axes2 = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

        # Handle single plot case
        if num_iterations == 1:
            axes2_flat = [axes2]
        elif num_rows == 1:
            axes2_flat = list(axes2) if isinstance(axes2, np.ndarray) else [axes2]
        else:
            axes2_flat = axes2.flatten().tolist()

        # Plot for each iteration (with difficulty breakdown)
        for idx, iteration_id in enumerate(iterations_sorted):
            ax = axes2_flat[idx]

            # Plot a line for each difficulty
            for difficulty in difficulties_sorted:
                confidence_data = iteration_difficulty_confidence_map[iteration_id][difficulty]

                if not confidence_data:
                    continue

                # Sort keys
                integer_keys = sorted([k for k in confidence_data.keys() if isinstance(k, int)])
                range_keys = sorted([k for k in confidence_data.keys() if isinstance(k, str)])
                sorted_keys = integer_keys + range_keys

                # Prepare data
                confidences = []
                success_rates = []

                for key in sorted_keys:
                    stats = confidence_data[key]
                    total_count = stats['total']
                    success_count = stats['passed']
                    success_rate = (success_count / total_count * 100) if total_count > 0 else 0

                    # Extract numeric value from label
                    if isinstance(key, int):
                        conf_num = float(key)
                    else:
                        conf_num = float(key.split('-')[0])

                    confidences.append(conf_num)
                    success_rates.append(success_rate)

                # Plot line with markers for this difficulty
                color = difficulty_colors.get(difficulty, 'blue')
                difficulty_label = difficulty.capitalize() if difficulty else 'Unknown' if difficulty else 'Unknown'
                ax.plot(confidences, success_rates, marker='o', linewidth=2, markersize=5,
                        color=color, label=difficulty_label, alpha=0.8)

            ax.set_xlabel('Confidence Score', fontsize=10)
            ax.set_ylabel('Success Rate (%)', fontsize=10)
            ax.set_title(f'Iteration {iteration_id} - By Difficulty', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-5, 105])
            ax.legend(loc='lower right', fontsize=9)

        # Hide extra subplots if any
        for idx in range(num_iterations, len(axes2_flat)):
            axes2_flat[idx].axis('off')

        plt.tight_layout()
        st.pyplot(fig2)

    # ===== SECTION 3: Correction Round Breakdown =====
    st.markdown("---")
    st.markdown("**Success Rate by Confidence Score - By Correction Round**")

    # Collect data separated by correction round
    correction_round_confidence_map = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'passed': 0}))
    correction_round_difficulty_confidence_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'total': 0, 'passed': 0})))

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        correction_round_id = attempt_info.get('correction_round_id', 0)
        problem_id = attempt_info.get('problem_id')

        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                confidence_val = float(confidence)
                is_passed = 1 if attempt_info['is_passing'] else 0

                # Bin confidence
                if confidence_val == int(confidence_val):
                    key = int(confidence_val)
                else:
                    key = f"{int(confidence_val)}-{int(confidence_val) + 1}"

                # Overall correction round data
                correction_round_confidence_map[correction_round_id][key]['total'] += 1
                correction_round_confidence_map[correction_round_id][key]['passed'] += is_passed

                # Correction round + difficulty data
                if session:
                    difficulty = problem_difficulty_map.get(problem_id, 'Unknown')
                    correction_round_difficulty_confidence_map[correction_round_id][difficulty][key]['total'] += 1
                    correction_round_difficulty_confidence_map[correction_round_id][difficulty][key]['passed'] += is_passed

    # Get correction rounds
    correction_rounds_sorted = sorted(correction_round_confidence_map.keys())

    if correction_rounds_sorted:
        # Create two subplots for correction rounds: overall and with difficulty breakdown
        st.markdown("**Overall Success Rate by Correction Round**")

        num_correction_rounds = len(correction_rounds_sorted)
        num_cols = min(3, num_correction_rounds)
        num_rows = (num_correction_rounds + num_cols - 1) // num_cols

        # Create subplots for overall correction round plots
        fig1, axes1 = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

        # Handle single plot case
        if num_correction_rounds == 1:
            axes1_flat = [axes1]
        elif num_rows == 1:
            axes1_flat = list(axes1) if isinstance(axes1, np.ndarray) else [axes1]
        else:
            axes1_flat = axes1.flatten().tolist()

        # Plot for each correction round (overall, no difficulty breakdown)
        for idx, correction_round_id in enumerate(correction_rounds_sorted):
            ax = axes1_flat[idx]
            confidence_data = correction_round_confidence_map[correction_round_id]

            # Sort keys
            integer_keys = sorted([k for k in confidence_data.keys() if isinstance(k, int)])
            range_keys = sorted([k for k in confidence_data.keys() if isinstance(k, str)])
            sorted_keys = integer_keys + range_keys

            # Prepare data
            confidences = []
            success_rates = []
            counts = []

            for key in sorted_keys:
                stats = confidence_data[key]
                total_count = stats['total']
                success_count = stats['passed']
                success_rate = (success_count / total_count * 100) if total_count > 0 else 0

                # Extract numeric value from label
                if isinstance(key, int):
                    conf_num = float(key)
                else:
                    conf_num = float(key.split('-')[0])

                confidences.append(conf_num)
                success_rates.append(success_rate)
                counts.append(total_count)

            # Plot line with markers
            ax.plot(confidences, success_rates, marker='o', linewidth=2, markersize=6,
                    color='steelblue', alpha=0.8)

            # Add shaded area under curve
            ax.fill_between(confidences, success_rates, alpha=0.2, color='steelblue')

            ax.set_xlabel('Confidence Score', fontsize=10)
            ax.set_ylabel('Success Rate (%)', fontsize=10)
            ax.set_title(f'Correction Round {correction_round_id}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-5, 105])

            # Add value labels on points with count
            for x, y, count in zip(confidences, success_rates, counts):
                label_text = f'{y:.0f}%\n(n={count})'
                ax.text(x, y + 3, label_text, ha='center', va='bottom', fontsize=8)

        # Hide extra subplots if any
        for idx in range(num_correction_rounds, len(axes1_flat)):
            axes1_flat[idx].axis('off')

        plt.tight_layout()
        st.pyplot(fig1)

        # Difficulty breakdown for correction rounds
        if session and correction_round_difficulty_confidence_map:
            st.markdown("---")
            st.markdown("**Success Rate by Confidence Score - Difficulty Breakdown per Correction Round**")

            # Get all difficulties
            all_difficulties = set()
            for corr_data in correction_round_difficulty_confidence_map.values():
                all_difficulties.update(corr_data.keys())

            # Sort difficulties: easy, medium, hard
            difficulties_sorted = sorted(all_difficulties,
                                         key=lambda x: {'easy': 0, 'Easy': 0, 'medium': 1, 'Medium': 1, 'hard': 2, 'Hard': 2}.get(x, 3))

            # Define colors for each difficulty
            difficulty_colors = {
                'easy': 'green',
                'Easy': 'green',
                'medium': 'orange',
                'Medium': 'orange',
                'hard': 'red',
                'Hard': 'red'
            }

            # Create subplots for difficulty breakdown (one plot per correction round, with multiple lines)
            fig2, axes2 = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

            # Handle single plot case
            if num_correction_rounds == 1:
                axes2_flat = [axes2]
            elif num_rows == 1:
                axes2_flat = list(axes2) if isinstance(axes2, np.ndarray) else [axes2]
            else:
                axes2_flat = axes2.flatten().tolist()

            # Plot for each correction round (with difficulty breakdown)
            for idx, correction_round_id in enumerate(correction_rounds_sorted):
                ax = axes2_flat[idx]

                # Plot a line for each difficulty
                for difficulty in difficulties_sorted:
                    confidence_data = correction_round_difficulty_confidence_map[correction_round_id][difficulty]

                    if not confidence_data:
                        continue

                    # Sort keys
                    integer_keys = sorted([k for k in confidence_data.keys() if isinstance(k, int)])
                    range_keys = sorted([k for k in confidence_data.keys() if isinstance(k, str)])
                    sorted_keys = integer_keys + range_keys

                    # Prepare data
                    confidences = []
                    success_rates = []

                    for key in sorted_keys:
                        stats = confidence_data[key]
                        total_count = stats['total']
                        success_count = stats['passed']
                        success_rate = (success_count / total_count * 100) if total_count > 0 else 0

                        # Extract numeric value from label
                        if isinstance(key, int):
                            conf_num = float(key)
                        else:
                            conf_num = float(key.split('-')[0])

                        confidences.append(conf_num)
                        success_rates.append(success_rate)

                    # Plot line with markers for this difficulty
                    color = difficulty_colors.get(difficulty, 'blue')
                    ax.plot(confidences, success_rates, marker='o', linewidth=2, markersize=5,
                            color=color, label=(difficulty.capitalize() if difficulty else 'Unknown'), alpha=0.8)

                ax.set_xlabel('Confidence Score', fontsize=10)
                ax.set_ylabel('Success Rate (%)', fontsize=10)
                ax.set_title(f'Correction Round {correction_round_id} - By Difficulty', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-5, 105])
                ax.legend(loc='lower right', fontsize=9)

            # Hide extra subplots if any
            for idx in range(num_correction_rounds, len(axes2_flat)):
                axes2_flat[idx].axis('off')

            plt.tight_layout()
            st.pyplot(fig2)

    # ===== SECTION 4: Combined Iteration and Correction Round Breakdown =====
    st.markdown("---")
    st.markdown("**Success Rate by Confidence Score - By Iteration and Correction Round (Difficulty Breakdown)**")

    # Collect data separated by iteration and correction round
    iter_corr_difficulty_confidence_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'total': 0, 'passed': 0}))))

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        iteration_id = attempt_info.get('iteration_id', 0)
        correction_round_id = attempt_info.get('correction_round_id', 0)
        problem_id = attempt_info.get('problem_id')

        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                confidence_val = float(confidence)
                is_passed = 1 if attempt_info['is_passing'] else 0

                # Bin confidence
                if confidence_val == int(confidence_val):
                    key = int(confidence_val)
                else:
                    key = f"{int(confidence_val)}-{int(confidence_val) + 1}"

                # Iteration + correction round + difficulty data
                if session:
                    difficulty = problem_difficulty_map.get(problem_id, 'Unknown')
                    iter_corr_difficulty_confidence_map[iteration_id][correction_round_id][difficulty][key]['total'] += 1
                    iter_corr_difficulty_confidence_map[iteration_id][correction_round_id][difficulty][key]['passed'] += is_passed

    # Get unique iterations and correction rounds
    iter_corr_iterations = sorted(iter_corr_difficulty_confidence_map.keys())
    iter_corr_correction_rounds = set()
    for iter_data in iter_corr_difficulty_confidence_map.values():
        iter_corr_correction_rounds.update(iter_data.keys())
    iter_corr_correction_rounds = sorted(iter_corr_correction_rounds)

    if iter_corr_iterations and iter_corr_correction_rounds and session:
        # Get all difficulties
        all_iter_corr_difficulties = set()
        for iter_data in iter_corr_difficulty_confidence_map.values():
            for corr_data in iter_data.values():
                all_iter_corr_difficulties.update(corr_data.keys())

        difficulties_sorted = sorted(all_iter_corr_difficulties,
                                     key=lambda x: {'easy': 0, 'Easy': 0, 'medium': 1, 'Medium': 1, 'hard': 2, 'Hard': 2}.get(x, 3))

        difficulty_colors = {
            'easy': 'green',
            'Easy': 'green',
            'medium': 'orange',
            'Medium': 'orange',
            'hard': 'red',
            'Hard': 'red'
        }

        # Create plots organized by correction round (rows) and iteration (columns)
        # Limit to up to 3 iterations per correction round (6 plots max)
        num_iterations = min(3, len(iter_corr_iterations))
        num_correction_rounds = len(iter_corr_correction_rounds)

        fig, axes = plt.subplots(num_correction_rounds, num_iterations, figsize=(5 * num_iterations, 5 * num_correction_rounds))

        # Handle single plot case
        if num_correction_rounds == 1 and num_iterations == 1:
            axes_list = [[axes]]
        elif num_correction_rounds == 1:
            axes_list = [list(axes) if isinstance(axes, np.ndarray) else [axes]]
        elif num_iterations == 1:
            axes_list = [[ax] for ax in (axes if isinstance(axes, np.ndarray) else [axes])]
        else:
            axes_list = axes.tolist() if hasattr(axes, 'tolist') else [[axes[i, j] for j in range(num_iterations)] for i in range(num_correction_rounds)]

        # Plot for each correction round (rows) and iteration (columns)
        for corr_idx, correction_round_id in enumerate(iter_corr_correction_rounds):
            for iter_idx, iteration_id in enumerate(iter_corr_iterations[:num_iterations]):
                ax = axes_list[corr_idx][iter_idx]
                confidence_data_dict = iter_corr_difficulty_confidence_map[iteration_id][correction_round_id]

                if not confidence_data_dict or not any(confidence_data_dict.values()):
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Iter {iteration_id} - Corr Round {correction_round_id}', fontsize=10, fontweight='bold')
                    continue

                # Plot a line for each difficulty
                for difficulty in difficulties_sorted:
                    confidence_data = confidence_data_dict[difficulty]

                    if not confidence_data:
                        continue

                    # Sort keys
                    integer_keys = sorted([k for k in confidence_data.keys() if isinstance(k, int)])
                    range_keys = sorted([k for k in confidence_data.keys() if isinstance(k, str)])
                    sorted_keys = integer_keys + range_keys

                    # Prepare data
                    confidences = []
                    success_rates = []

                    for key in sorted_keys:
                        stats = confidence_data[key]
                        total_count = stats['total']
                        success_count = stats['passed']
                        success_rate = (success_count / total_count * 100) if total_count > 0 else 0

                        # Extract numeric value from label
                        if isinstance(key, int):
                            conf_num = float(key)
                        else:
                            conf_num = float(key.split('-')[0])

                        confidences.append(conf_num)
                        success_rates.append(success_rate)

                    # Plot line with markers for this difficulty
                    color = difficulty_colors.get(difficulty, 'blue')
                    ax.plot(confidences, success_rates, marker='o', linewidth=2, markersize=5,
                            color=color, label=(difficulty.capitalize() if difficulty else 'Unknown'), alpha=0.8)

                ax.set_xlabel('Confidence Score', fontsize=10)
                ax.set_ylabel('Success Rate (%)', fontsize=10)
                ax.set_title(f'Iter {iteration_id} - Corr Round {correction_round_id}', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-5, 105])
                ax.legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)

    # Show summary table for iterations
    st.markdown("---")
    st.subheader("Success Rate Statistics by Iteration")

    summary_data = []
    for iteration_id in iterations_sorted:
        confidence_data = iteration_confidence_map[iteration_id]
        total = sum(stats['total'] for stats in confidence_data.values())
        passed = sum(stats['passed'] for stats in confidence_data.values())
        rate = (passed / total * 100) if total > 0 else 0

        summary_data.append({
            'Iteration': int(iteration_id),
            'Total Attempts': int(total),
            'Passed': int(passed),
            'Success Rate': f"{rate:.1f}%"
        })

    if summary_data:
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_compilation_errors(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render a plot showing the proportion of proofs with each compilation error type.

    Shows:
    - Bar chart of error types and proportion of proofs with that error
    - Count of attempts without errors
    - Table of unknown errors with full error messages
    """
    st.subheader("Compilation Error Frequency")

    # Track which proofs have which error types (binary: yes/no)
    error_proofs = defaultdict(int)  # error_type -> count of proofs with this error
    unknown_errors = []  # List of unknown error details
    no_error_count = 0
    total_failed_count = 0
    total_attempts = len(all_attempts)

    for attempt_info in all_attempts:
        attempt = attempt_info['attempt']

        # Check if compilation failed
        if not attempt.compilation_result.passed:
            total_failed_count += 1
            compilation_summary = attempt_info.get('compilation_summary')

            if compilation_summary and isinstance(compilation_summary, dict):
                error_counts = compilation_summary.get('error_counts', {})
                if error_counts:
                    # error_counts is a dict - mark which error types appear in this proof
                    for error_type in error_counts.keys():
                        error_proofs[error_type] += 1
                else:
                    # No error_counts - capture the full compilation summary
                    error_proofs['Unknown Error'] += 1
                    unknown_errors.append({
                        'problem_id': attempt_info.get('problem_id', 'N/A'),
                        'full_summary': str(compilation_summary)
                    })
            else:
                error_proofs['Unknown Error'] += 1
                unknown_errors.append({
                    'problem_id': attempt_info.get('problem_id', 'N/A'),
                    'full_summary': 'No compilation summary available'
                })
        else:
            no_error_count += 1

    # Display statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Attempts", total_attempts)

    with col2:
        st.metric("Failed Compilations", total_failed_count)

    with col3:
        st.metric("Successful Compilations", no_error_count)

    # Create bar chart (excluding Unknown Error from histogram)
    if error_proofs:
        # Separate known and unknown errors
        known_errors = {k: v for k, v in error_proofs.items() if k != 'Unknown Error'}
        unknown_count = error_proofs.get('Unknown Error', 0)

        # Show summary table with known/unknown/successful
        st.markdown("---")
        st.subheader("Compilation Status Summary")

        summary_data = [
            {'Status': 'Successful Compilations', 'Count': no_error_count, 'Proportion': f"{(no_error_count/total_attempts*100):.1f}%"},
            {'Status': 'Known Error Types', 'Count': sum(known_errors.values()), 'Proportion': f"{(sum(known_errors.values())/total_attempts*100):.1f}%"},
            {'Status': 'Unknown Errors', 'Count': unknown_count, 'Proportion': f"{(unknown_count/total_attempts*100):.1f}%"}
        ]
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        if known_errors:
            # Sort known errors by proportion (descending)
            sorted_errors = sorted(known_errors.items(), key=lambda x: x[1], reverse=True)
            error_types = [item[0] for item in sorted_errors]
            error_counts = [item[1] for item in sorted_errors]
            # Calculate proportions as percentage of total attempts
            error_proportions = [count / total_attempts * 100 for count in error_counts]

            fig, ax = plt.subplots(figsize=(12, 6))

            # Create bar chart
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(error_types)))
            bars = ax.bar(range(len(error_types)), error_proportions, color=colors, edgecolor='black', alpha=0.8)

            ax.set_xlabel('Error Type', fontsize=12)
            ax.set_ylabel('Proportion of Proofs (%)', fontsize=12)
            ax.set_title('Proportion of Proofs with Each Error Type', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(error_types)))
            ax.set_xticklabels(error_types, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 100])

            # Add value labels on bars
            for bar, prop in zip(bars, error_proportions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prop:.1f}%',
                       ha='center', va='bottom', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)

            # Show error frequency table
            st.subheader("Error Type Proportions")
            table_data = []
            for error_type, count in sorted_errors:
                proportion = (count / total_attempts * 100) if total_attempts > 0 else 0
                table_data.append({
                    'Error Type': error_type,
                    'Proofs with Error': int(count),
                    'Proportion': f"{proportion:.1f}%"
                })

            df_errors = pd.DataFrame(table_data)
            st.dataframe(df_errors, use_container_width=True, hide_index=True)
        else:
            st.info("No known error types found in the data (only successful compilations and/or unknown errors).")

        # Plot errors by correction round
        st.markdown("---")
        render_errors_by_correction_round(all_attempts)

        # Plot error count vs correction round success
        st.markdown("---")
        render_error_count_vs_correction_success(all_attempts)
    else:
        st.info("No compilation errors found in the data.")


def render_errors_by_correction_round(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render errors by correction round (initial round 0 vs correction rounds 1+).

    Shows:
    - Grouped bar chart with error types on x-axis
    - Bars for each correction round (0, 1, 2, etc.)
    - Proportion of failed proofs with each error type in each correction round
    - Unknown errors are excluded from the chart
    - Proportions are relative to failed compilations only (not all attempts)
    """
    st.subheader("Error Types by Correction Round")

    # Collect errors by (error_type, correction_round)
    error_corr_data = defaultdict(lambda: defaultdict(int))
    total_failed_by_corr = defaultdict(int)

    for attempt_info in all_attempts:
        attempt = attempt_info['attempt']

        if not attempt.compilation_result.passed:
            correction_round = attempt_info.get('correction_round_id', 0)
            total_failed_by_corr[correction_round] += 1
            compilation_summary = attempt_info.get('compilation_summary')

            if compilation_summary and isinstance(compilation_summary, dict):
                error_counts = compilation_summary.get('error_counts', {})
                if error_counts:
                    # Only track known error types
                    for error_type in error_counts.keys():
                        error_corr_data[error_type][correction_round] += 1

    if not error_corr_data:
        st.info("No known error types found across correction rounds.")
        return

    # Get all error types and correction rounds
    error_types = sorted(error_corr_data.keys())
    correction_rounds = sorted(set().union(*[set(d.keys()) for d in error_corr_data.values()]))

    if not correction_rounds:
        st.info("No correction round data available.")
        return

    # Prepare data for grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(error_types))
    width = 0.8 / len(correction_rounds)  # Width of bars per correction round
    colors = plt.cm.Set3(np.linspace(0, 1, len(correction_rounds)))

    for i, corr_round in enumerate(correction_rounds):
        proportions = []
        for error_type in error_types:
            count = error_corr_data[error_type].get(corr_round, 0)
            total_failed = total_failed_by_corr[corr_round] if total_failed_by_corr[corr_round] > 0 else 1
            proportion = (count / total_failed * 100)
            proportions.append(proportion)

        # Plot bars for this correction round
        offset = (i - len(correction_rounds) / 2 + 0.5) * width
        bars = ax.bar(x + offset, proportions, width, label=f'Correction Round {corr_round}',
                     color=colors[i], edgecolor='black', alpha=0.8)

        # Add value labels on bars
        for bar, prop in zip(bars, proportions):
            if prop > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{prop:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Error Type', fontsize=12)
    ax.set_ylabel('Proportion of Failed Proofs (%)', fontsize=12)
    ax.set_title('Error Distribution Among Failed Proofs by Correction Round', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(error_types, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


def render_error_correctability_analysis(session) -> None:
    """
    Analyze the correctability of each error type.

    For each error type found in round 0, calculate:
    - How many proofs had that error in round 0
    - How many of those were corrected (became PASS) in round 1
    - Correctability rate = (corrected / total) * 100%

    Shows a bar chart and table with error types and their correction rates.
    """
    st.subheader("🔧 Error Correctability Analysis")
    st.markdown("For each error type, what % of proofs with that error get corrected in round 1?")

    # Collect data: for each error type, track which attempts had it and if they were corrected
    error_type_stats = defaultdict(lambda: {'total': 0, 'corrected': 0, 'attempts': []})

    # Iterate through all problems and their formalizations
    for problem in session.problems.values():
        for breakdown in problem.breakdowns.values():
            if not breakdown.parsed_breakdown:
                continue

            # Process theorem
            theorem = breakdown.parsed_breakdown.theorem
            for formalization in theorem.formalizations:
                _process_formalization_for_correctability(formalization, error_type_stats)

            # Process lemmas
            for lemma in breakdown.parsed_breakdown.lemmas.values():
                for formalization in lemma.formalizations:
                    _process_formalization_for_correctability(formalization, error_type_stats)

    if not error_type_stats:
        st.info("No error data available for correctability analysis.")
        return

    # Calculate correctability rates
    table_data = []
    error_types_list = []
    correctability_rates = []

    for error_type in sorted(error_type_stats.keys()):
        stats = error_type_stats[error_type]
        total = stats['total']
        corrected = stats['corrected']
        rate = (corrected / total * 100) if total > 0 else 0

        table_data.append({
            'Error Type': error_type,
            'Round 0 Failures': total,
            'Corrected in Round 1': corrected,
            'Correctability Rate': f"{rate:.1f}%"
        })

        error_types_list.append(error_type)
        correctability_rates.append(rate)

    # Show statistics
    col1, col2, col3 = st.columns(3)

    total_attempts_with_errors = sum(stats['total'] for stats in error_type_stats.values())
    total_corrected = sum(stats['corrected'] for stats in error_type_stats.values())
    overall_rate = (total_corrected / total_attempts_with_errors * 100) if total_attempts_with_errors > 0 else 0

    with col1:
        st.metric("Total Errors in Round 0", total_attempts_with_errors)

    with col2:
        st.metric("Corrected in Round 1", total_corrected)

    with col3:
        st.metric("Overall Correctability", f"{overall_rate:.1f}%")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(error_types_list)))
    bars = ax.bar(error_types_list, correctability_rates, color=colors, edgecolor='black', alpha=0.8)

    ax.set_xlabel('Error Type', fontsize=12)
    ax.set_ylabel('Correctability Rate (%)', fontsize=12)
    ax.set_title('Error Correctability Rate by Error Type', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, rate in zip(bars, correctability_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # Show table
    st.subheader("📋 Detailed Correctability Table")
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _process_formalization_for_correctability(formalization, error_type_stats: Dict) -> None:
    """
    Process a single formalization to extract error correctability data.

    For each error type in round 0 failed attempts, check if the corresponding
    attempt in round 1 (same initial_attempt_index) becomes successful.
    """
    if not formalization.proof_attempts_by_round:
        return

    # Get round 0 attempts
    round0_attempts = formalization.proof_attempts_by_round.get(0, [])
    if not round0_attempts:
        return

    # Get round 1 attempts
    round1_attempts = formalization.proof_attempts_by_round.get(1, [])
    if not round1_attempts:
        return

    # Create map of initial_attempt_index -> round 1 attempt
    round1_map = {attempt.initial_attempt_index: attempt for attempt in round1_attempts}

    # Process round 0 failed attempts
    for r0_attempt in round0_attempts:
        if r0_attempt.is_passing():
            continue  # Skip successful attempts

        # Check if this attempt was corrected in round 1
        initial_idx = r0_attempt.initial_attempt_index
        r1_attempt = round1_map.get(initial_idx)
        is_corrected = r1_attempt is not None and r1_attempt.is_passing()

        # Extract error types from compilation summary
        compilation_summary = r0_attempt.compilation_summary
        if compilation_summary and isinstance(compilation_summary, dict):
            error_counts = compilation_summary.get('error_counts', {})
            for error_type in error_counts.keys():
                error_type_stats[error_type]['total'] += 1
                if is_corrected:
                    error_type_stats[error_type]['corrected'] += 1
                error_type_stats[error_type]['attempts'].append({
                    'error_type': error_type,
                    'corrected': is_corrected,
                    'initial_index': initial_idx
                })


def render_confidence_vs_correction_success(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Analyze confidence scores from round 0 and their impact on correction round success.

    For each confidence score level, determine:
    - How many failed attempts in round 0 had that confidence
    - How many of those became successful in round 1 (correction round)
    - Success rate = (corrected / total) * 100%

    Shows a bar chart and table comparing confidence to correction success rates.
    """
    st.subheader("Confidence Score Impact on Correction Round Success")
    st.markdown("For failed round 0 attempts, how does confidence score predict correction round success?")

    # Collect data: for each confidence score, track failed round 0 attempts and their correction round results
    confidence_correction_stats = defaultdict(lambda: {'total': 0, 'corrected': 0})

    for attempt_info in all_attempts:
        attempt = attempt_info['attempt']

        # Only process failed round 0 attempts
        if attempt.correction_round_id != 0 or attempt.is_passing():
            continue

        # Get confidence from reasoning summary
        reasoning_summary = attempt_info.get('reasoning_summary')
        confidence = None

        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')

        if confidence is None:
            confidence = 'None'
        else:
            confidence = float(confidence)
            # Round to nearest integer for binning
            confidence = int(round(confidence))

        # Track this failed attempt
        confidence_correction_stats[confidence]['total'] += 1

        # Check if this attempt was corrected in round 1
        # We need to find the corresponding round 1 attempt using initial_attempt_index
        initial_idx = attempt.initial_attempt_index

        # Look for matching round 1 attempt in the same formalization
        is_corrected = False
        for other_attempt_info in all_attempts:
            other_attempt = other_attempt_info['attempt']
            if (other_attempt.correction_round_id == 1 and
                other_attempt.initial_attempt_index == initial_idx and
                other_attempt.lemma_id == attempt.lemma_id and
                other_attempt_info['problem_id'] == attempt_info['problem_id'] and
                other_attempt_info['breakdown_id'] == attempt_info['breakdown_id']):
                if other_attempt.is_passing():
                    is_corrected = True
                break

        if is_corrected:
            confidence_correction_stats[confidence]['corrected'] += 1

    if not confidence_correction_stats:
        st.info("No failed round 0 attempts found for correction analysis.")
        return

    # Calculate success rates
    table_data = []
    confidence_labels = []
    success_rates = []

    # Sort by confidence (None first, then numbers)
    sorted_keys = sorted(confidence_correction_stats.keys(),
                         key=lambda x: (x != 'None', x))

    for confidence in sorted_keys:
        stats = confidence_correction_stats[confidence]
        total = stats['total']
        corrected = stats['corrected']
        rate = (corrected / total * 100) if total > 0 else 0

        # Format confidence label
        confidence_label = str(confidence)

        table_data.append({
            'Confidence Score': confidence_label,
            'Failed in Round 0': total,
            'Corrected in Round 1': corrected,
            'Correction Success Rate': f"{rate:.1f}%"
        })

        confidence_labels.append(confidence_label)
        success_rates.append(rate)

    # Show statistics
    col1, col2, col3 = st.columns(3)

    total_failed_attempts = sum(stats['total'] for stats in confidence_correction_stats.values())
    total_corrected = sum(stats['corrected'] for stats in confidence_correction_stats.values())
    overall_rate = (total_corrected / total_failed_attempts * 100) if total_failed_attempts > 0 else 0

    with col1:
        st.metric("Failed Round 0 Attempts", total_failed_attempts)

    with col2:
        st.metric("Corrected in Round 1", total_corrected)

    with col3:
        st.metric("Overall Correction Rate", f"{overall_rate:.1f}%")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(confidence_labels)))
    bars = ax.bar(confidence_labels, success_rates, color=colors, edgecolor='black', alpha=0.8)

    ax.set_xlabel('Confidence Score (Round 0)', fontsize=12)
    ax.set_ylabel('Correction Success Rate (%)', fontsize=12)
    ax.set_title('Confidence Score Impact on Correction Round Success', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # Show table
    st.subheader("📋 Detailed Correction Success by Confidence")
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_confidence_by_iteration(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render average confidence scores grouped by iteration.

    Shows a table with:
    - Iteration ID
    - Total attempts in that iteration
    - Average confidence score
    """
    st.subheader("Average Confidence by Iteration")

    # Collect confidence scores by iteration
    iteration_data = defaultdict(lambda: {'total': 0, 'confidence_sum': 0, 'count': 0})

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        iteration_id = attempt_info.get('iteration_id', 0)

        iteration_data[iteration_id]['total'] += 1

        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                confidence_val = float(confidence)
                iteration_data[iteration_id]['confidence_sum'] += confidence_val
                iteration_data[iteration_id]['count'] += 1

    if not iteration_data:
        st.warning("No iteration data available.")
        return

    # Build table
    table_data = []
    for iteration_id in sorted(iteration_data.keys()):
        stats = iteration_data[iteration_id]
        avg_confidence = (stats['confidence_sum'] / stats['count']) if stats['count'] > 0 else None

        table_data.append({
            'Iteration': int(iteration_id),
            'Total Attempts': int(stats['total']),
            'Attempts with Confidence': int(stats['count']),
            'Average Confidence': f"{avg_confidence:.2f}" if avg_confidence is not None else "N/A"
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("No confidence data available for iterations.")


def render_confidence_by_correction_round(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render average confidence scores grouped by correction round.

    Shows a table with:
    - Correction Round ID
    - Total attempts in that correction round
    - Average confidence score
    """
    st.subheader("Average Confidence by Correction Round")

    # Collect confidence scores by correction round
    correction_round_data = defaultdict(lambda: {'total': 0, 'confidence_sum': 0, 'count': 0})

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        correction_round_id = attempt_info.get('correction_round_id', 0)

        correction_round_data[correction_round_id]['total'] += 1

        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                confidence_val = float(confidence)
                correction_round_data[correction_round_id]['confidence_sum'] += confidence_val
                correction_round_data[correction_round_id]['count'] += 1

    if not correction_round_data:
        st.warning("No correction round data available.")
        return

    # Build table
    table_data = []
    for correction_round_id in sorted(correction_round_data.keys()):
        stats = correction_round_data[correction_round_id]
        avg_confidence = (stats['confidence_sum'] / stats['count']) if stats['count'] > 0 else None

        table_data.append({
            'Correction Round': int(correction_round_id),
            'Total Attempts': int(stats['total']),
            'Attempts with Confidence': int(stats['count']),
            'Average Confidence': f"{avg_confidence:.2f}" if avg_confidence is not None else "N/A"
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("No confidence data available for correction rounds.")


def render_confidence_per_origin_problem(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render histogram of average confidence per origin_problem_id.

    For each problem, calculates the average confidence of all its proofs,
    then shows a histogram of these averages.
    """
    st.subheader("Average Confidence per Origin Problem ID")

    # Collect confidence scores per problem
    problem_data = defaultdict(lambda: {'confidence_sum': 0, 'count': 0})

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        problem_id = attempt_info.get('problem_id')

        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None and problem_id is not None:
                confidence_val = float(confidence)
                problem_data[problem_id]['confidence_sum'] += confidence_val
                problem_data[problem_id]['count'] += 1

    if not problem_data:
        st.warning("No problem-level confidence data available.")
        return

    # Calculate average confidence per problem
    problem_averages = []
    for problem_id, stats in problem_data.items():
        if stats['count'] > 0:
            avg_confidence = stats['confidence_sum'] / stats['count']
            problem_averages.append(avg_confidence)

    if not problem_averages:
        st.warning("No average confidence data to display.")
        return

    # Display statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Problems", len(problem_averages))

    with col2:
        st.metric("Min Avg Confidence", f"{np.min(problem_averages):.2f}")

    with col3:
        st.metric("Median Avg Confidence", f"{np.median(problem_averages):.2f}")

    with col4:
        st.metric("Max Avg Confidence", f"{np.max(problem_averages):.2f}")

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.arange(0, 10.5, 0.5)  # Bins of size 0.5
    ax.hist(problem_averages, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

    ax.set_xlabel('Average Confidence Score', fontsize=12)
    ax.set_ylabel('Number of Problems', fontsize=12)
    ax.set_title('Distribution of Average Confidence Scores per Problem', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add mean line
    if problem_averages:
        mean_val = np.mean(problem_averages)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.legend()

    st.pyplot(fig)


def render_confidence_per_theorem(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render histogram of average confidence per theorem (iter0 only).

    For each unique theorem in iteration 0, calculates the average confidence
    of all its attempts (across correction rounds), then shows a histogram.
    """
    st.subheader("Average Confidence per Theorem (Iteration 0)")

    # Collect confidence scores per theorem in iter0
    theorem_data = defaultdict(lambda: {'confidence_sum': 0, 'count': 0})

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        iteration_id = attempt_info.get('iteration_id', 0)
        problem_id = attempt_info.get('problem_id')
        lemma_id = attempt_info.get('lemma_id')

        # Only process iteration 0 and theorems (lemma_id == -1)
        if iteration_id != 0 or lemma_id != -1:
            continue

        theorem_key = f"{problem_id}_theorem"

        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                confidence_val = float(confidence)
                theorem_data[theorem_key]['confidence_sum'] += confidence_val
                theorem_data[theorem_key]['count'] += 1

    if not theorem_data:
        st.warning("No theorem-level confidence data available in iteration 0.")
        return

    # Calculate average confidence per theorem
    theorem_averages = []
    for theorem_key, stats in theorem_data.items():
        if stats['count'] > 0:
            avg_confidence = stats['confidence_sum'] / stats['count']
            theorem_averages.append(avg_confidence)

    if not theorem_averages:
        st.warning("No average confidence data to display.")
        return

    # Display statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Theorems", len(theorem_averages))

    with col2:
        st.metric("Min Avg Confidence", f"{np.min(theorem_averages):.2f}")

    with col3:
        st.metric("Median Avg Confidence", f"{np.median(theorem_averages):.2f}")

    with col4:
        st.metric("Max Avg Confidence", f"{np.max(theorem_averages):.2f}")

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.arange(0, 10.5, 0.5)  # Bins of size 0.5
    ax.hist(theorem_averages, bins=bins, edgecolor='black', alpha=0.7, color='seagreen')

    ax.set_xlabel('Average Confidence Score', fontsize=12)
    ax.set_ylabel('Number of Theorems', fontsize=12)
    ax.set_title('Distribution of Average Confidence Scores per Theorem (Iteration 0)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add mean line
    if theorem_averages:
        mean_val = np.mean(theorem_averages)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.legend()

    st.pyplot(fig)


def render_confidence_per_lemma(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render histogram of average confidence per unique lemma.

    For each unique lemma (identified by problem_id and lemma_id),
    calculates the average confidence across all attempts,
    then shows a histogram.
    """
    st.subheader("Average Confidence per Lemma")

    # Collect confidence scores per lemma (across all attempts)
    lemma_data = defaultdict(lambda: {'confidence_sum': 0, 'count': 0})

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        problem_id = attempt_info.get('problem_id')
        lemma_id = attempt_info.get('lemma_id')

        # Only process lemmas (lemma_id >= 0, exclude theorems which are lemma_id == -1)
        if lemma_id is None or lemma_id < 0:
            continue

        lemma_key = f"{problem_id}_l{lemma_id}"

        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                confidence_val = float(confidence)
                lemma_data[lemma_key]['confidence_sum'] += confidence_val
                lemma_data[lemma_key]['count'] += 1

    if not lemma_data:
        st.warning("No lemma-level confidence data available.")
        return

    # Calculate average confidence per lemma
    lemma_averages = []
    for lemma_key, stats in lemma_data.items():
        if stats['count'] > 0:
            avg_confidence = stats['confidence_sum'] / stats['count']
            lemma_averages.append(avg_confidence)

    if not lemma_averages:
        st.warning("No average confidence data to display.")
        return

    # Display statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Unique Lemmas", len(lemma_averages))

    with col2:
        st.metric("Min Avg Confidence", f"{np.min(lemma_averages):.2f}")

    with col3:
        st.metric("Median Avg Confidence", f"{np.median(lemma_averages):.2f}")

    with col4:
        st.metric("Max Avg Confidence", f"{np.max(lemma_averages):.2f}")

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.arange(0, 10.5, 0.5)  # Bins of size 0.5
    ax.hist(lemma_averages, bins=bins, edgecolor='black', alpha=0.7, color='coral')

    ax.set_xlabel('Average Confidence Score', fontsize=12)
    ax.set_ylabel('Number of Lemmas', fontsize=12)
    ax.set_title('Distribution of Average Confidence Scores per Lemma', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add mean line
    if lemma_averages:
        mean_val = np.mean(lemma_averages)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.legend()

    st.pyplot(fig)


def render_confidence_vs_correct_proof_likelihood(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render plots showing how average confidence predicts likelihood of at least one correct proof.

    For theorems and lemmas separately:
    - Groups by average confidence (rounded to nearest integer)
    - Calculates % with at least one passing proof
    - Shows scatter plot with trend line
    """
    st.subheader("Average Confidence vs Likelihood of Correct Proof")
    st.markdown("For theorems and lemmas separately: what % have at least one correct proof at each confidence level?")

    # Create two columns for theorem and lemma plots
    col1, col2 = st.columns(2)

    # Process theorems
    with col1:
        render_confidence_correctness_plot(all_attempts, is_theorem=True)

    # Process lemmas
    with col2:
        render_confidence_correctness_plot(all_attempts, is_theorem=False)


def render_confidence_correctness_plot(all_attempts: List[Dict[str, Any]], is_theorem: bool) -> None:
    """
    Helper function to render confidence vs correctness plot for either theorems or lemmas.

    Args:
        all_attempts: All proof attempts
        is_theorem: If True, filter for theorems (lemma_id == -1). If False, filter for lemmas (lemma_id >= 0)
    """
    # Determine title and filter
    if is_theorem:
        title_prefix = "Theorem"
        filter_name = "theorems"
    else:
        title_prefix = "Lemma"
        filter_name = "lemmas"

    # Collect data: for each theorem/lemma, track average confidence and whether it has at least one correct proof
    item_data = defaultdict(lambda: {
        'confidence_sum': 0,
        'confidence_count': 0,
        'has_correct': False,
        'problem_id': None,
        'lemma_id': None
    })

    for attempt_info in all_attempts:
        problem_id = attempt_info.get('problem_id')
        lemma_id = attempt_info.get('lemma_id')
        is_passing = attempt_info.get('is_passing', False)
        reasoning_summary = attempt_info.get('reasoning_summary')

        # Filter by theorem/lemma
        if is_theorem:
            if lemma_id != -1:
                continue
            item_key = f"{problem_id}_theorem"
        else:
            if lemma_id is None or lemma_id < 0:
                continue
            item_key = f"{problem_id}_l{lemma_id}"

        # Track confidence
        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                item_data[item_key]['confidence_sum'] += float(confidence)
                item_data[item_key]['confidence_count'] += 1

        # Track if has at least one correct proof
        if is_passing:
            item_data[item_key]['has_correct'] = True

        # Store metadata
        item_data[item_key]['problem_id'] = problem_id
        item_data[item_key]['lemma_id'] = lemma_id

    if not item_data:
        st.warning(f"No {filter_name} data available.")
        return

    # Collect data per theorem/lemma: average confidence and proportion of successful attempts
    item_stats = defaultdict(lambda: {
        'confidence_sum': 0,
        'confidence_count': 0,
        'attempts_by_round': defaultdict(lambda: {'total': 0, 'passing': 0})
    })

    for attempt_info in all_attempts:
        problem_id = attempt_info.get('problem_id')
        lemma_id = attempt_info.get('lemma_id')
        is_passing = attempt_info.get('is_passing', False)
        correction_round_id = attempt_info.get('correction_round_id', 0)
        reasoning_summary = attempt_info.get('reasoning_summary')

        # Filter by theorem/lemma
        if is_theorem:
            if lemma_id != -1:
                continue
            item_key = f"{problem_id}_theorem"
        else:
            if lemma_id is None or lemma_id < 0:
                continue
            item_key = f"{problem_id}_l{lemma_id}"

        # Track confidence
        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                confidence_val = float(confidence)
                item_stats[item_key]['confidence_sum'] += confidence_val
                item_stats[item_key]['confidence_count'] += 1

        # Track success by round
        round_type = 'initial' if correction_round_id == 0 else 'correction'
        item_stats[item_key]['attempts_by_round'][round_type]['total'] += 1
        if is_passing:
            item_stats[item_key]['attempts_by_round'][round_type]['passing'] += 1

    if not item_stats:
        st.warning(f"No confidence data available for {filter_name}.")
        return

    # Prepare data for plotting: one point per theorem/lemma per round
    initial_confidences = []
    initial_props = []
    correction_confidences = []
    correction_props = []

    for item_key, stats in item_stats.items():
        if stats['confidence_count'] > 0:
            avg_confidence = stats['confidence_sum'] / stats['confidence_count']

            # Initial round
            if 'initial' in stats['attempts_by_round']:
                initial_stats = stats['attempts_by_round']['initial']
                if initial_stats['total'] > 0:
                    prop = initial_stats['passing'] / initial_stats['total']
                    initial_confidences.append(avg_confidence)
                    initial_props.append(prop * 100)

            # Correction round
            if 'correction' in stats['attempts_by_round']:
                correction_stats = stats['attempts_by_round']['correction']
                if correction_stats['total'] > 0:
                    prop = correction_stats['passing'] / correction_stats['total']
                    correction_confidences.append(avg_confidence)
                    correction_props.append(prop * 100)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine colors based on theorem/lemma
    if is_theorem:
        color_initial = 'steelblue'
        color_correction = 'orange'
    else:
        color_initial = 'darkslateblue'
        color_correction = 'darkred'

    # Plot initial round
    if initial_confidences:
        ax.scatter(initial_confidences, initial_props, s=120, color=color_initial, alpha=0.7, marker='o', edgecolors='black', linewidth=1.5, label='Initial Round (Round 0)', zorder=3)

    # Plot correction round
    if correction_confidences:
        ax.scatter(correction_confidences, correction_props, s=120, color=color_correction, alpha=0.7, marker='^', edgecolors='black', linewidth=1.5, label='Correction Round (Round 1+)', zorder=3)

    # Fit linear regression to combined data
    all_confidences = initial_confidences + correction_confidences
    all_props = initial_props + correction_props

    if len(all_confidences) > 1:
        z = np.polyfit(all_confidences, all_props, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(0, 10, 100)
        y_trend = p(x_trend)
        y_trend = np.clip(y_trend, 0, 100)  # Clamp to [0, 100]
        correlation = np.corrcoef(all_confidences, all_props)[0, 1]
        ax.plot(x_trend, y_trend, color='red', linestyle='--', linewidth=2.5,
                label=f'Linear Fit: y={z[0]:.2f}x+{z[1]:.2f} (r={correlation:.3f})', zorder=2)

    ax.set_xlabel('Average Confidence Score', fontsize=12)
    ax.set_ylabel('Proportion of Attempts Solved (%)', fontsize=12)
    ax.set_title(f'{title_prefix}: Confidence vs Success Rate', fontsize=13, fontweight='bold')
    ax.set_xlim([-0.5, 10.5])
    ax.set_ylim([-5, 105])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    st.pyplot(fig)

    # Show summary statistics
    st.markdown(f"**{title_prefix} Summary Statistics**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Initial Round Points", len(initial_confidences))
    with col2:
        st.metric("Correction Round Points", len(correction_confidences))
    with col3:
        initial_avg = np.mean(initial_props) if initial_props else 0
        st.metric("Initial Avg Success Rate", f"{initial_avg:.1f}%")
    with col4:
        correction_avg = np.mean(correction_props) if correction_props else 0
        st.metric("Correction Avg Success Rate", f"{correction_avg:.1f}%")


def render_error_count_vs_correction_success(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render a scatter plot showing the relationship between error count (from compilation summary)
    and success rate in correction rounds.

    For each attempt in round 0 that failed, extract:
    - Number of errors in compilation_summary
    - Whether it was corrected in round 1 (became passing)
    Then plot: x = error count, y = success rate in correction round
    """
    st.subheader("Error Count vs Correction Round Success Rate")

    # Collect data: for each error count level, track correction success
    error_count_stats = defaultdict(lambda: {'total': 0, 'corrected': 0})

    for attempt_info in all_attempts:
        attempt = attempt_info['attempt']
        correction_round_id = attempt_info.get('correction_round_id', 0)

        # Only process round 0 failed attempts
        if correction_round_id != 0 or attempt.is_passing():
            continue

        # Extract error count from compilation summary
        compilation_summary = attempt_info.get('compilation_summary')
        error_count = 0

        if compilation_summary and isinstance(compilation_summary, dict):
            error_counts = compilation_summary.get('error_counts', {})
            if error_counts:
                # Sum all error counts
                error_count = sum(error_counts.values())

        # Track this failed attempt
        error_count_stats[error_count]['total'] += 1

        # Check if this attempt was corrected in round 1
        initial_idx = attempt.initial_attempt_index
        problem_id = attempt_info.get('problem_id')
        breakdown_id = attempt_info.get('breakdown_id')
        lemma_id = attempt_info.get('lemma_id')

        # Look for matching round 1 attempt
        is_corrected = False
        for other_attempt_info in all_attempts:
            other_attempt = other_attempt_info['attempt']
            if (other_attempt.correction_round_id == 1 and
                other_attempt.initial_attempt_index == initial_idx and
                other_attempt.lemma_id == lemma_id and
                other_attempt_info['problem_id'] == problem_id and
                other_attempt_info['breakdown_id'] == breakdown_id):
                if other_attempt.is_passing():
                    is_corrected = True
                break

        if is_corrected:
            error_count_stats[error_count]['corrected'] += 1

    if not error_count_stats:
        st.info("No error count data available for correction analysis.")
        return

    # Prepare data for plotting
    error_counts = []
    success_rates = []

    for err_count in sorted(error_count_stats.keys()):
        stats = error_count_stats[err_count]
        total = stats['total']
        corrected = stats['corrected']
        rate = (corrected / total * 100) if total > 0 else 0

        error_counts.append(err_count)
        success_rates.append(rate)

    # Create interactive Plotly plot
    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=error_counts,
        y=success_rates,
        mode='markers',
        marker=dict(size=12, color='steelblue', opacity=0.7, line=dict(color='black', width=1.5)),
        text=[f'Error Count: {ec}<br>Success Rate: {sr:.1f}%' for ec, sr in zip(error_counts, success_rates)],
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Correction Success Rate',
        showlegend=True
    ))

    # Fit linear regression if we have enough points
    if len(error_counts) > 1:
        z = np.polyfit(error_counts, success_rates, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(error_counts), max(error_counts), 100)
        y_trend = p(x_trend)
        y_trend = np.clip(y_trend, 0, 100)
        correlation = np.corrcoef(error_counts, success_rates)[0, 1]

        fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            line=dict(color='red', width=2.5, dash='dash'),
            name=f'Linear Fit: y={z[0]:.2f}x+{z[1]:.2f} (r={correlation:.3f})',
            hovertemplate='Error Count: %{x:.1f}<br>Predicted Success Rate: %{y:.1f}%<extra></extra>',
            showlegend=True
        ))

    fig.update_layout(
        title='Error Count vs Correction Round Success',
        xaxis_title='Number of Compilation Errors (Round 0)',
        yaxis_title='Correction Round Success Rate (%)',
        hovermode='closest',
        template='plotly_white',
        width=1000,
        height=600,
        yaxis=dict(range=[-5, 105]),
        font=dict(size=12),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='black', borderwidth=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show summary statistics
    st.markdown("**Summary Statistics**")
    col1, col2, col3 = st.columns(3)

    total_failed = sum(stats['total'] for stats in error_count_stats.values())
    total_corrected = sum(stats['corrected'] for stats in error_count_stats.values())
    overall_rate = (total_corrected / total_failed * 100) if total_failed > 0 else 0

    with col1:
        st.metric("Failed Round 0 Attempts", total_failed)
    with col2:
        st.metric("Corrected in Round 1", total_corrected)
    with col3:
        st.metric("Overall Correction Rate", f"{overall_rate:.1f}%")

    # Show table
    st.markdown("**Error Count vs Correction Success Table**")
    table_data = []
    for err_count, rate in zip(error_counts, success_rates):
        total = error_count_stats[err_count]['total']
        corrected = error_count_stats[err_count]['corrected']
        table_data.append({
            'Error Count': int(err_count),
            'Failed Attempts': int(total),
            'Corrected': int(corrected),
            'Success Rate': f"{rate:.1f}%"
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_confidence_threshold_analysis(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render tables showing how confidence thresholds relate to success for each lemma/theorem.

    For each confidence score (0-10):
    1. Lower bound: Of lemmas/theorems with at least one attempt at or below this confidence level,
       what proportion eventually get solved?
    2. Upper bound: Of lemmas/theorems with at least one attempt at or above this confidence level,
       what proportion eventually get solved?

    This helps understand whether low/high confidence correlates with eventual success.
    """
    st.subheader("Confidence Threshold Analysis (Per Lemma/Theorem)")
    st.markdown("For each confidence level, what proportion of lemmas/theorems with attempts at that confidence get solved?")

    # Collect data per lemma/theorem: all attempt confidences and whether it was eventually solved
    item_data = defaultdict(lambda: {
        'confidences': [],
        'is_solved': False,
        'problem_id': None,
        'lemma_id': None
    })

    for attempt_info in all_attempts:
        problem_id = attempt_info.get('problem_id')
        lemma_id = attempt_info.get('lemma_id')
        is_passing = attempt_info.get('is_passing', False)
        reasoning_summary = attempt_info.get('reasoning_summary')

        if lemma_id is None:
            continue

        # Create unique key for each lemma/theorem
        if lemma_id == -1:
            item_key = f"{problem_id}_theorem"
        else:
            item_key = f"{problem_id}_l{lemma_id}"

        # Track confidence
        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                item_data[item_key]['confidences'].append(float(confidence))

        # Track if solved
        if is_passing:
            item_data[item_key]['is_solved'] = True

        item_data[item_key]['problem_id'] = problem_id
        item_data[item_key]['lemma_id'] = lemma_id

    if not item_data:
        st.warning("No data available for threshold analysis.")
        return

    # Calculate statistics for each confidence threshold
    lower_bound_stats = {}  # confidence_threshold -> {total, solved}
    upper_bound_stats = {}  # confidence_threshold -> {total, solved}

    for item_key, data in item_data.items():
        if not data['confidences']:
            continue

        is_solved = data['is_solved']
        min_conf = min(data['confidences'])
        max_conf = max(data['confidences'])

        # Lower bound: item has at least one attempt <= threshold
        for threshold in range(11):
            if min_conf <= threshold:
                if threshold not in lower_bound_stats:
                    lower_bound_stats[threshold] = {'total': 0, 'solved': 0}
                lower_bound_stats[threshold]['total'] += 1
                if is_solved:
                    lower_bound_stats[threshold]['solved'] += 1

        # Upper bound: item has at least one attempt >= threshold
        for threshold in range(11):
            if max_conf >= threshold:
                if threshold not in upper_bound_stats:
                    upper_bound_stats[threshold] = {'total': 0, 'solved': 0}
                upper_bound_stats[threshold]['total'] += 1
                if is_solved:
                    upper_bound_stats[threshold]['solved'] += 1

    # Create lower bound table
    st.subheader("Lower Bound: ≤ Confidence Threshold")
    st.markdown("Lemmas/Theorems with at least one attempt at or below this confidence level")

    lower_table_data = []
    for threshold in sorted(lower_bound_stats.keys()):
        stats = lower_bound_stats[threshold]
        total = stats['total']
        solved = stats['solved']
        rate = (solved / total * 100) if total > 0 else 0

        lower_table_data.append({
            'Confidence ≤': threshold,
            'Total Items': int(total),
            'Solved': int(solved),
            'Success Rate': f"{rate:.1f}%"
        })

    df_lower = pd.DataFrame(lower_table_data)
    st.dataframe(df_lower, use_container_width=True, hide_index=True)

    # Create upper bound table
    st.subheader("Upper Bound: ≥ Confidence Threshold")
    st.markdown("Lemmas/Theorems with at least one attempt at or above this confidence level")

    upper_table_data = []
    for threshold in sorted(upper_bound_stats.keys()):
        stats = upper_bound_stats[threshold]
        total = stats['total']
        solved = stats['solved']
        rate = (solved / total * 100) if total > 0 else 0

        upper_table_data.append({
            'Confidence ≥': threshold,
            'Total Items': int(total),
            'Solved': int(solved),
            'Success Rate': f"{rate:.1f}%"
        })

    df_upper = pd.DataFrame(upper_table_data)
    st.dataframe(df_upper, use_container_width=True, hide_index=True)

    # Summary interpretation
    st.markdown("---")
    st.markdown("**Interpretation:**")
    st.markdown("""
    - **Lower Bound**: Shows what happens when we include lemmas/theorems with very low confidence attempts.
      If this rate increases as the threshold increases, it suggests that having low confidence doesn't prevent eventual success.
    - **Upper Bound**: Shows what happens when we only look at lemmas/theorems with high confidence attempts.
      If this rate decreases as the threshold increases, it suggests high confidence doesn't guarantee success.
    """)


def render_min_max_confidence_histograms(all_attempts: List[Dict[str, Any]]) -> None:
    """
    Render stacked histograms of minimum and maximum confidence scores per lemma/theorem,
    showing the proportion of solved vs unsolved items for each confidence level.

    For each unique lemma/theorem:
    - Calculate the minimum confidence score across all its attempts
    - Calculate the maximum confidence score across all its attempts
    - Track whether it was eventually solved
    - Show stacked histograms with solved/unsolved breakdown
    """
    st.subheader("Min/Max Confidence Score Distributions (Stacked by Solve Status)")
    st.markdown("Proportion of solved vs unsolved lemmas/theorems for each min/max confidence score")

    # Collect data per lemma/theorem: min/max confidence scores and solve status
    item_data = defaultdict(lambda: {
        'confidences': [],
        'is_solved': False
    })

    for attempt_info in all_attempts:
        problem_id = attempt_info.get('problem_id')
        lemma_id = attempt_info.get('lemma_id')
        is_passing = attempt_info.get('is_passing', False)
        reasoning_summary = attempt_info.get('reasoning_summary')

        if lemma_id is None:
            continue

        # Create unique key for each lemma/theorem
        if lemma_id == -1:
            item_key = f"{problem_id}_theorem"
        else:
            item_key = f"{problem_id}_l{lemma_id}"

        # Track confidence
        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                item_data[item_key]['confidences'].append(float(confidence))

        # Track if solved
        if is_passing:
            item_data[item_key]['is_solved'] = True

    if not item_data:
        st.warning("No data available for min/max confidence analysis.")
        return

    # Calculate min and max for each item and track solved status
    min_conf_solved = []
    min_conf_unsolved = []
    max_conf_solved = []
    max_conf_unsolved = []

    for item_key, data in item_data.items():
        if data['confidences']:
            min_conf = min(data['confidences'])
            max_conf = max(data['confidences'])
            is_solved = data['is_solved']

            if is_solved:
                min_conf_solved.append(min_conf)
                max_conf_solved.append(max_conf)
            else:
                min_conf_unsolved.append(min_conf)
                max_conf_unsolved.append(max_conf)

    if not (min_conf_solved or min_conf_unsolved) or not (max_conf_solved or max_conf_unsolved):
        st.warning("No confidence data available.")
        return

    # Create two-column layout for histograms
    col1, col2 = st.columns(2)

    # Min confidence histogram (stacked)
    with col1:
        st.subheader("Minimum Confidence Scores")

        fig_min = go.Figure()

        # Add solved items
        fig_min.add_trace(go.Histogram(
            x=min_conf_solved,
            nbinsx=11,
            name='Solved',
            marker=dict(color='seagreen', line=dict(color='black', width=1)),
            opacity=0.8
        ))

        # Add unsolved items
        fig_min.add_trace(go.Histogram(
            x=min_conf_unsolved,
            nbinsx=11,
            name='Unsolved',
            marker=dict(color='lightcoral', line=dict(color='black', width=1)),
            opacity=0.8
        ))

        fig_min.update_layout(
            title='Distribution of Minimum Confidence (Stacked by Solve Status)',
            xaxis_title='Minimum Confidence Score',
            yaxis_title='Count',
            template='plotly_white',
            hovermode='x unified',
            barmode='stack',
            font=dict(size=11),
            xaxis=dict(dtick=1),
            showlegend=True
        )

        st.plotly_chart(fig_min, use_container_width=True)

        # Stats for min
        col_a, col_b, col_c = st.columns(3)
        total_min = len(min_conf_solved) + len(min_conf_unsolved)
        solved_min = len(min_conf_solved)
        unsolved_min = len(min_conf_unsolved)
        with col_a:
            st.metric("Total Items", total_min)
        with col_b:
            st.metric("Solved", solved_min)
        with col_c:
            st.metric("Unsolved", unsolved_min)

    # Max confidence histogram (stacked)
    with col2:
        st.subheader("Maximum Confidence Scores")

        fig_max = go.Figure()

        # Add solved items
        fig_max.add_trace(go.Histogram(
            x=max_conf_solved,
            nbinsx=11,
            name='Solved',
            marker=dict(color='seagreen', line=dict(color='black', width=1)),
            opacity=0.8
        ))

        # Add unsolved items
        fig_max.add_trace(go.Histogram(
            x=max_conf_unsolved,
            nbinsx=11,
            name='Unsolved',
            marker=dict(color='lightcoral', line=dict(color='black', width=1)),
            opacity=0.8
        ))

        fig_max.update_layout(
            title='Distribution of Maximum Confidence (Stacked by Solve Status)',
            xaxis_title='Maximum Confidence Score',
            yaxis_title='Count',
            template='plotly_white',
            hovermode='x unified',
            barmode='stack',
            font=dict(size=11),
            xaxis=dict(dtick=1),
            showlegend=True
        )

        st.plotly_chart(fig_max, use_container_width=True)

        # Stats for max
        col_a, col_b, col_c = st.columns(3)
        total_max = len(max_conf_solved) + len(max_conf_unsolved)
        solved_max = len(max_conf_solved)
        unsolved_max = len(max_conf_unsolved)
        with col_a:
            st.metric("Total Items", total_max)
        with col_b:
            st.metric("Solved", solved_max)
        with col_c:
            st.metric("Unsolved", unsolved_max)


def render_success_rate_by_difficulty(all_attempts: List[Dict[str, Any]], session) -> None:
    """
    Render success rate by confidence score, broken down by problem difficulty.

    Shows three separate line plots (Easy, Medium, Hard) on the same figure.
    """
    st.subheader("Success Rate by Confidence Score - Breakdown by Problem Difficulty")

    # Build a map: problem_id -> difficulty
    problem_difficulty_map = {}
    for problem in session.problems.values():
        problem_difficulty_map[problem.origin_problem_id] = problem.difficulty

    # Collect confidence scores, success status, and difficulty
    difficulty_confidence_map = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'passed': 0}))

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        problem_id = attempt_info.get('problem_id')

        if reasoning_summary and isinstance(reasoning_summary, dict):
            confidence = reasoning_summary.get('confidence')
            if confidence is not None:
                confidence_val = float(confidence)
                is_passed = 1 if attempt_info['is_passing'] else 0

                # Get difficulty for this problem
                difficulty = problem_difficulty_map.get(problem_id, 'Unknown')

                # Bin confidence
                if confidence_val == int(confidence_val):
                    key = int(confidence_val)
                else:
                    key = f"{int(confidence_val)}-{int(confidence_val) + 1}"

                difficulty_confidence_map[difficulty][key]['total'] += 1
                difficulty_confidence_map[difficulty][key]['passed'] += is_passed

    if not difficulty_confidence_map:
        st.warning("No difficulty data available for confidence breakdown.")
        return

    # Create separate trend lines for each difficulty
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors for each difficulty
    difficulty_colors = {
        'easy': 'green',
        'medium': 'orange',
        'hard': 'red',
        'Easy': 'green',
        'Medium': 'orange',
        'Hard': 'red'
    }

    # Sort difficulties: easy, medium, hard
    difficulties_sorted = sorted(difficulty_confidence_map.keys(),
                                  key=lambda x: {'easy': 0, 'Easy': 0, 'medium': 1, 'Medium': 1, 'hard': 2, 'Hard': 2}.get(x, 3))

    for difficulty in difficulties_sorted:
        confidence_data = difficulty_confidence_map[difficulty]

        # Sort keys
        integer_keys = sorted([k for k in confidence_data.keys() if isinstance(k, int)])
        range_keys = sorted([k for k in confidence_data.keys() if isinstance(k, str)])
        sorted_keys = integer_keys + range_keys

        # Prepare data for this difficulty
        confidences = []
        success_rates = []

        for key in sorted_keys:
            stats = confidence_data[key]
            total_count = stats['total']
            success_count = stats['passed']
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0

            # Extract numeric value from label
            if isinstance(key, int):
                conf_num = float(key)
            else:
                conf_num = float(key.split('-')[0])

            confidences.append(conf_num)
            success_rates.append(success_rate)

        # Plot line with markers for this difficulty
        color = difficulty_colors.get(difficulty, 'blue')
        difficulty_label = difficulty.capitalize() if difficulty else 'Unknown' if difficulty else 'Unknown'
        ax.plot(confidences, success_rates, marker='o', linewidth=2.5, markersize=7,
                color=color, label=difficulty_label, alpha=0.8)

        # Add value labels on points
        for x, y in zip(confidences, success_rates):
            ax.text(x, y + 1.5, f'{y:.0f}%', ha='center', va='bottom', fontsize=8, alpha=0.7)

    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Proof Success Rate by Confidence Score - Breakdown by Problem Difficulty', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-5, 105])
    ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig)

    # Show summary table by difficulty
    st.markdown("---")
    st.subheader("Summary Statistics by Difficulty")

    summary_data = []
    for difficulty in difficulties_sorted:
        confidence_data = difficulty_confidence_map[difficulty]
        total = sum(stats['total'] for stats in confidence_data.values())
        passed = sum(stats['passed'] for stats in confidence_data.values())
        rate = (passed / total * 100) if total > 0 else 0

        summary_data.append({
            'Difficulty': difficulty.capitalize() if difficulty else 'Unknown' if difficulty else 'Unknown',
            'Total Attempts': int(total),
            'Passed': int(passed),
            'Success Rate': f"{rate:.1f}%"
        })

    if summary_data:
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_proof_analysis(all_attempts: List[Dict[str, Any]], session) -> None:
    """
    Render proof analysis including:
    - Scatter plot: Average proof length vs solve rate (one point per theorem/lemma)
    - Combined table: Proof length statistics by iteration and difficulty
    """
    st.subheader("Proof Length Analysis")

    # Build a map: problem_id -> difficulty
    problem_difficulty_map = {}
    for problem in session.problems.values():
        problem_difficulty_map[problem.origin_problem_id] = problem.difficulty

    # Collect data for scatter plot: one entry per unique theorem/lemma
    theorem_lemma_data = defaultdict(lambda: {
        'proof_lengths': [],
        'solved_count': 0,
        'total_attempts': 0,
        'iterations': set(),
        'difficulties': set()
    })

    # Collect proof length data by iteration and difficulty
    proof_length_by_iter_diff = defaultdict(lambda: defaultdict(lambda: {'lengths': []}))

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        iteration_id = attempt_info.get('iteration_id', 0)
        problem_id = attempt_info.get('problem_id')
        lemma_id = attempt_info.get('lemma_id', -1)
        is_passing = attempt_info.get('is_passing', False)

        if reasoning_summary and isinstance(reasoning_summary, dict):
            proof_length = reasoning_summary.get('proof_length')

            if proof_length is not None:
                try:
                    proof_length_val = float(proof_length)

                    # Create unique key for theorem/lemma
                    if lemma_id == -1:
                        item_key = f"{problem_id}_theorem"
                    else:
                        item_key = f"{problem_id}_l{lemma_id}"

                    # Track for scatter plot
                    theorem_lemma_data[item_key]['proof_lengths'].append(proof_length_val)
                    theorem_lemma_data[item_key]['total_attempts'] += 1
                    if is_passing:
                        theorem_lemma_data[item_key]['solved_count'] += 1
                    theorem_lemma_data[item_key]['iterations'].add(iteration_id)

                    # Track difficulty
                    difficulty = problem_difficulty_map.get(problem_id, 'Unknown')
                    theorem_lemma_data[item_key]['difficulties'].add(difficulty)

                    # Track by iteration and difficulty for table (include all attempts)
                    proof_length_by_iter_diff[iteration_id][difficulty]['lengths'].append(proof_length_val)

                except (ValueError, TypeError):
                    pass

    if not theorem_lemma_data and not proof_length_by_iter_diff:
        st.warning("No proof length data available in reasoning summaries.")
        return

    # ===== SCATTER PLOT: Proof Length vs Solve Rate =====
    st.markdown("**Proof Length vs Solve Rate (per Theorem/Lemma)**")

    if theorem_lemma_data:
        # Prepare data for scatter plot
        # For each theorem/lemma, average all its proof lengths into a single point
        scatter_data = defaultdict(lambda: {
            'avg_proof_length': [],
            'solve_rate': None,
            'difficulty': None
        })

        for item_key, data in theorem_lemma_data.items():
            if data['proof_lengths']:
                # Average all proof lengths for this theorem/lemma
                avg_proof_length = np.mean(data['proof_lengths'])

                # Solve rate: percentage of attempts that were successful
                if data['total_attempts'] > 0:
                    solve_rate = (data['solved_count'] / data['total_attempts']) * 100.0
                else:
                    solve_rate = 0.0

                # Get difficulty
                difficulty = list(data['difficulties'])[0] if data['difficulties'] else 'Unknown'

                scatter_data[item_key]['avg_proof_length'] = avg_proof_length
                scatter_data[item_key]['solve_rate'] = solve_rate
                scatter_data[item_key]['difficulty'] = difficulty

        if scatter_data:
            fig, ax = plt.subplots(figsize=(12, 7))

            # Create scatter plot with colors by difficulty
            difficulties_unique = set(data['difficulty'] for data in scatter_data.values())
            difficulty_colors = {
                'easy': 'green',
                'Easy': 'green',
                'medium': 'orange',
                'Medium': 'orange',
                'hard': 'red',
                'Hard': 'red',
                'Unknown': 'gray'
            }

            # Plot points for each difficulty
            for difficulty in sorted(difficulties_unique, key=lambda x: {'easy': 0, 'Easy': 0, 'medium': 1, 'Medium': 1, 'hard': 2, 'Hard': 2, 'Unknown': 3}.get(x, 3)):
                lengths_for_diff = []
                rates_for_diff = []
                for item_key, data in scatter_data.items():
                    if data['difficulty'] == difficulty:
                        lengths_for_diff.append(data['avg_proof_length'])
                        rates_for_diff.append(data['solve_rate'])  # Already in percentage

                if lengths_for_diff:
                    diff_label = difficulty.capitalize() if difficulty else "Unknown"
                    ax.scatter(lengths_for_diff, rates_for_diff, s=100,
                              color=difficulty_colors.get(difficulty, 'gray'),
                              alpha=0.6, edgecolors='black', linewidth=1.5,
                              label=f'{diff_label} (n={len(lengths_for_diff)})')

            ax.set_xlabel('Average Proof Length (per Theorem/Lemma)', fontsize=12)
            ax.set_ylabel('Solve Rate (\%)', fontsize=12)
            ax.set_title('Proof Length vs Solve Rate (per Theorem/Lemma)', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-5, 105])
            ax.legend(loc='upper right', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)

            # Show summary statistics
            st.markdown("**Summary Statistics**")
            col1, col2, col3, col4, col5 = st.columns(5)

            total_items = len(scatter_data)
            solved_items = sum(1 for data in scatter_data.values() if data['solve_rate'] > 0)
            avg_lengths = [data['avg_proof_length'] for data in scatter_data.values()]
            avg_solve_rates = [data['solve_rate'] for data in scatter_data.values()]

            with col1:
                st.metric("Total Theorems/Lemmas", total_items)
            with col2:
                st.metric("Solved (≥1 success)", solved_items)
            with col3:
                st.metric("Unsolved (0% success)", total_items - solved_items)
            with col4:
                avg_proof_len = np.mean(avg_lengths) if avg_lengths else 0
                st.metric("Avg Proof Length", f"{avg_proof_len:.1f}")
            with col5:
                avg_solve_rate = np.mean(avg_solve_rates) if avg_solve_rates else 0
                st.metric("Avg Solve Rate", f"{avg_solve_rate:.1f}%")

        # ===== BINNED ANALYSIS: Solve Rate by Proof Length Ranges =====
        st.markdown("---")
        st.markdown("**Binned Analysis: Solve Rate by Proof Length Range**")
        st.markdown("All proof attempts grouped in bins of 50 lines, showing average solve rate per bin")

        # Create bins of 50 lines each - use ALL proof attempts, not just averaged values
        bin_size = 50
        binned_data = defaultdict(lambda: {'solve_rates': []})

        for attempt_info in all_attempts:
            reasoning_summary = attempt_info.get('reasoning_summary')
            is_passing = attempt_info.get('is_passing', False)

            if reasoning_summary and isinstance(reasoning_summary, dict):
                proof_length = reasoning_summary.get('proof_length')

                if proof_length is not None:
                    try:
                        proof_length_val = float(proof_length)
                        # Only include proofs up to 1000 lines
                        if proof_length_val <= 1000:
                            # Determine which bin this proof belongs to
                            bin_start = int(proof_length_val // bin_size) * bin_size
                            bin_key = f"{int(bin_start)}-{int(bin_start + bin_size)}"
                            solve_rate = 100.0 if is_passing else 0.0
                            binned_data[bin_key]['solve_rates'].append(solve_rate)
                    except (ValueError, TypeError):
                        pass

        if binned_data:
            # Sort bins by their numeric value
            sorted_bins = sorted(binned_data.keys(), key=lambda x: int(x.split('-')[0]))

            # Calculate statistics for each bin
            bin_labels = []
            bin_avg_rates = []
            bin_counts = []

            for bin_key in sorted_bins:
                rates = binned_data[bin_key]['solve_rates']
                avg_rate = np.mean(rates)
                count = len(rates)

                bin_labels.append(bin_key)
                bin_avg_rates.append(avg_rate)
                bin_counts.append(count)

            # Create binned plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # Color bars based on solve rate (green for high, red for low)
            colors = plt.cm.RdYlGn(np.array(bin_avg_rates) / 100.0)
            bars = ax.bar(range(len(bin_labels)), bin_avg_rates, color=colors, edgecolor='black', alpha=0.8)

            ax.set_xlabel('Proof Length Range (lines)', fontsize=12)
            ax.set_ylabel('Average Solve Rate (\%)', fontsize=12)
            ax.set_title('Average Solve Rate by Proof Length Range', fontsize=13, fontweight='bold')
            ax.set_xticks(range(len(bin_labels)))
            ax.set_xticklabels(bin_labels, rotation=45, ha='right')
            ax.set_ylim([0, 105])
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, rate, count in zip(bars, bin_avg_rates, bin_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{rate:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)

            # Show binned statistics table
            st.markdown("**Binned Statistics Table**")
            binned_table_data = []
            for bin_key, avg_rate, count in zip(bin_labels, bin_avg_rates, bin_counts):
                binned_table_data.append({
                    'Proof Length Range': bin_key,
                    'Count': int(count),
                    'Avg Solve Rate': f"{avg_rate:.1f}%"
                })

            df_binned = pd.DataFrame(binned_table_data)
            st.dataframe(df_binned, use_container_width=True, hide_index=True)

        # ===== STACKED BAR PLOT: Solved vs Unsolved by Max Proof Length =====
        st.markdown("---")
        st.markdown("**Solved vs Unsolved Theorems/Lemmas by Maximum Proof Length**")
        st.markdown("Theorems/Lemmas grouped by their maximum proof length (bins of 100 lines), showing stacked count of solved/unsolved")

        # Create bins for max proof length per theorem/lemma (using 100-line bins)
        max_bin_size = 100
        max_length_bins = defaultdict(lambda: {'solved': 0, 'unsolved': 0})

        for item_key, data in scatter_data.items():
            max_length = max(theorem_lemma_data[item_key]['proof_lengths'])
            bin_start = int(max_length // max_bin_size) * max_bin_size
            bin_key = f"{int(bin_start)}-{int(bin_start + max_bin_size)}"

            if data['solve_rate'] > 0:
                max_length_bins[bin_key]['solved'] += 1
            else:
                max_length_bins[bin_key]['unsolved'] += 1

        if max_length_bins:
            # Sort bins
            sorted_max_bins = sorted(max_length_bins.keys(), key=lambda x: int(x.split('-')[0]))

            solved_counts = []
            unsolved_counts = []
            max_bin_labels = []

            for bin_key in sorted_max_bins:
                solved_counts.append(max_length_bins[bin_key]['solved'])
                unsolved_counts.append(max_length_bins[bin_key]['unsolved'])
                max_bin_labels.append(bin_key)

            # Create stacked bar plot
            fig, ax = plt.subplots(figsize=(12, 6))

            x_pos = np.arange(len(max_bin_labels))
            width = 0.6

            # Stack the bars
            bars1 = ax.bar(x_pos, solved_counts, width, label='Solved', color='seagreen', edgecolor='black', alpha=0.8)
            bars2 = ax.bar(x_pos, unsolved_counts, width, bottom=solved_counts, label='Unsolved', color='lightcoral', edgecolor='black', alpha=0.8)

            ax.set_xlabel('Maximum Proof Length Range (lines)', fontsize=12)
            ax.set_ylabel('Number of Theorems/Lemmas', fontsize=12)
            ax.set_title('Solved vs Unsolved Theorems/Lemmas by Maximum Proof Length', fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(max_bin_labels, rotation=45, ha='right')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for i, (solved, unsolved) in enumerate(zip(solved_counts, unsolved_counts)):
                total = solved + unsolved
                # Label on solved bar
                if solved > 0:
                    ax.text(i, solved / 2, str(solved), ha='center', va='center', fontsize=9, fontweight='bold')
                # Label on unsolved bar
                if unsolved > 0:
                    ax.text(i, solved + unsolved / 2, str(unsolved), ha='center', va='center', fontsize=9, fontweight='bold')
                # Total count above bar
                ax.text(i, total, f'n={total}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)

            # Show stacked bar statistics table
            st.markdown("**Stacked Bar Statistics Table**")
            stacked_table_data = []
            for bin_key, solved, unsolved in zip(max_bin_labels, solved_counts, unsolved_counts):
                total = solved + unsolved
                solve_rate = (solved / total * 100) if total > 0 else 0
                stacked_table_data.append({
                    'Max Proof Length Range': bin_key,
                    'Solved': int(solved),
                    'Unsolved': int(unsolved),
                    'Total': int(total),
                    'Solve Rate': f"{solve_rate:.1f}%"
                })

            df_stacked = pd.DataFrame(stacked_table_data)
            st.dataframe(df_stacked, use_container_width=True, hide_index=True)

    # ===== HISTOGRAM: Proof Length of Solved Theorems =====
    st.markdown("---")
    st.markdown("**Proof Lengths of Solved Theorems/Lemmas**")

    # Collect proof lengths for solved items only
    solved_proof_lengths = []
    for item_key, data in theorem_lemma_data.items():
        if data['solved_count'] > 0:  # Only include items that were solved at least once
            # Use the proof lengths from solved attempts only
            for proof_length in data['proof_lengths']:
                # We would need to track which attempts were solved to only include those
                # For now, we'll include all proof lengths from items that have at least one solve
                solved_proof_lengths.append(proof_length)

    if solved_proof_lengths:
        solved_proof_lengths = sorted(solved_proof_lengths)

        # Create histogram
        fig, ax = plt.subplots(figsize=(12, 6))

        # Determine bins: 0-10, 10-20, ..., etc.
        max_length = max(solved_proof_lengths)
        bins = np.arange(0, max_length + 20, 10)

        counts, bin_edges, patches = ax.hist(solved_proof_lengths, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

        ax.set_xlabel('Proof Length (lines)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Proof Lengths for Solved Theorems/Lemmas', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Display statistics with percentiles
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("Count", len(solved_proof_lengths))
        with col2:
            st.metric("Min", f"{np.min(solved_proof_lengths):.0f}")
        with col3:
            st.metric("P5", f"{np.percentile(solved_proof_lengths, 5):.0f}")
        with col4:
            st.metric("Median", f"{np.median(solved_proof_lengths):.0f}")
        with col5:
            st.metric("P95", f"{np.percentile(solved_proof_lengths, 95):.0f}")
        with col6:
            st.metric("Max", f"{np.max(solved_proof_lengths):.0f}")
    else:
        st.info("No solved proof length data available.")

    # ===== COMBINED TABLE: Proof Length by Iteration and Difficulty =====
    st.markdown("---")
    st.markdown("**Proof Length Statistics by Iteration and Difficulty**")

    if proof_length_by_iter_diff:
        # Get all iterations and difficulties
        iterations_sorted = sorted(proof_length_by_iter_diff.keys())
        all_difficulties = set()
        for iter_data in proof_length_by_iter_diff.values():
            all_difficulties.update(iter_data.keys())

        difficulties_sorted = sorted(all_difficulties,
                                     key=lambda x: {'easy': 0, 'Easy': 0, 'medium': 1, 'Medium': 1, 'hard': 2, 'Hard': 2}.get(x, 3))

        # Build combined table
        table_data = []
        for iteration_id in iterations_sorted:
            for difficulty in difficulties_sorted:
                lengths = proof_length_by_iter_diff[iteration_id][difficulty]['lengths']
                if lengths:
                    table_data.append({
                        'Iteration': int(iteration_id),
                        'Difficulty': difficulty.capitalize() if difficulty else 'Unknown',
                        'Count': len(lengths),
                        'Min': f"{np.min(lengths):.0f}",
                        'P5': f"{np.percentile(lengths, 5):.0f}",
                        'Avg': f"{np.mean(lengths):.1f}",
                        'Median': f"{np.median(lengths):.0f}",
                        'P95': f"{np.percentile(lengths, 95):.0f}",
                        'Max': f"{np.max(lengths):.0f}",
                        'Std Dev': f"{np.std(lengths):.1f}"
                    })

        if table_data:
            df_combined = pd.DataFrame(table_data)
            st.dataframe(df_combined, use_container_width=True, hide_index=True)
        else:
            st.info("No proof length data available.")
    else:
        st.info("No proof length data available.")

    # ===== HISTOGRAM: Shortest Proof Lengths for Solved Theorems/Lemmas =====
    st.markdown("---")
    st.markdown("**Shortest Complete Proof Lengths for Solved Theorems/Lemmas**")

    # Collect shortest proof length for each theorem/lemma that was solved (only from passing attempts)
    shortest_proofs_all = []
    shortest_proofs_by_difficulty = defaultdict(list)

    # Need to re-collect data to track which proof lengths are from passing attempts
    theorem_lemma_passing_lengths = defaultdict(list)

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        is_passing = attempt_info.get('is_passing', False)
        problem_id = attempt_info.get('problem_id')
        lemma_id = attempt_info.get('lemma_id', -1)

        if is_passing and reasoning_summary and isinstance(reasoning_summary, dict):
            proof_length = reasoning_summary.get('proof_length')
            if proof_length is not None:
                try:
                    proof_length_val = float(proof_length)
                    # Create unique key for theorem/lemma
                    if lemma_id == -1:
                        item_key = f"{problem_id}_theorem"
                    else:
                        item_key = f"{problem_id}_l{lemma_id}"
                    theorem_lemma_passing_lengths[item_key].append(proof_length_val)
                except (ValueError, TypeError):
                    pass

    # Now get the shortest proof length from passing attempts only
    for item_key, passing_lengths in theorem_lemma_passing_lengths.items():
        if passing_lengths:
            shortest = min(passing_lengths)
            shortest_proofs_all.append(shortest)

            # Get difficulty from the original data
            if item_key in theorem_lemma_data:
                difficulty = list(theorem_lemma_data[item_key]['difficulties'])[0] if theorem_lemma_data[item_key]['difficulties'] else 'Unknown'
                shortest_proofs_by_difficulty[difficulty].append(shortest)

    # ===== SCATTER PLOT: Average Proof Lengths with Intervals for All Solved Problems =====
    st.markdown("---")
    st.markdown("**Average Proof Lengths of Solved Problems (with Min/Max Range)**")

    # Collect average proof lengths for each problem (only from passing attempts)
    problem_proof_stats = {}

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        is_passing = attempt_info.get('is_passing', False)
        problem_id = attempt_info.get('problem_id')

        if is_passing and reasoning_summary and isinstance(reasoning_summary, dict):
            proof_length = reasoning_summary.get('proof_length')
            if proof_length is not None:
                try:
                    proof_length_val = float(proof_length)
                    if problem_id not in problem_proof_stats:
                        problem_proof_stats[problem_id] = {
                            'lengths': [],
                            'difficulty': problem_difficulty_map.get(problem_id, 'Unknown')
                        }
                    problem_proof_stats[problem_id]['lengths'].append(proof_length_val)
                except (ValueError, TypeError):
                    pass

    if problem_proof_stats:
        # Sort by difficulty, then alphabetically by problem ID
        # Normalize difficulty to lowercase for consistent ordering
        def get_sort_key(item):
            problem_id, stats = item
            difficulty = stats['difficulty'].lower() if stats['difficulty'] else 'unknown'
            difficulty_order = {'easy': 0, 'medium': 1, 'hard': 2}
            return (difficulty_order.get(difficulty, 3), problem_id)

        sorted_problems = sorted(problem_proof_stats.items(), key=get_sort_key)

        # Prepare data for scatter plot
        problem_names = []
        avg_lengths = []
        min_lengths = []
        max_lengths = []
        difficulties = []
        colors = []

        difficulty_colors = {
            'easy': 'green',
            'medium': 'orange',
            'hard': 'red'
        }

        for problem_id, stats in sorted_problems:
            lengths = stats['lengths']
            difficulty = stats['difficulty'].lower() if stats['difficulty'] else 'unknown'

            problem_names.append(problem_id)
            avg_lengths.append(np.mean(lengths))
            min_lengths.append(np.min(lengths))
            max_lengths.append(np.max(lengths))
            difficulties.append(difficulty)
            colors.append(difficulty_colors.get(difficulty, 'blue'))

        # Create scatter plot with many rows
        num_problems = len(problem_names)
        fig_height = max(15, num_problems * 0.25)  # Scale height by number of problems

        fig, ax = plt.subplots(figsize=(14, fig_height))

        y_positions = np.arange(num_problems)

        # Plot interval lines (min to max)
        for i, (min_len, max_len) in enumerate(zip(min_lengths, max_lengths)):
            ax.plot([min_len, max_len], [i, i], color=colors[i], alpha=0.3, linewidth=2)

        # Plot average points
        scatter = ax.scatter(avg_lengths, y_positions, c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=1.5, zorder=3)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(problem_names, fontsize=8)
        ax.set_xlabel('Proof Length (lines)', fontsize=12)
        ax.set_ylabel('Problem', fontsize=12)
        ax.set_title('Average Proof Lengths of Solved Problems\n(point = avg, line = min-max range)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', label='Easy'),
            Patch(facecolor='orange', edgecolor='black', label='Medium'),
            Patch(facecolor='red', edgecolor='black', label='Hard')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

        plt.tight_layout()
        st.pyplot(fig)

        # Summary statistics
        st.markdown("**Summary Statistics**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Solved Problems", num_problems)
        with col2:
            st.metric("Overall Avg Proof Length", f"{np.mean(avg_lengths):.1f}")
        with col3:
            st.metric("Median Avg Proof Length", f"{np.median(avg_lengths):.1f}")
        with col4:
            st.metric("Max Avg Proof Length", f"{np.max(avg_lengths):.1f}")

    if shortest_proofs_all:
        # Overall histogram
        st.markdown("---")
        st.markdown("**All Solved Theorems/Lemmas - Shortest Proof Length**")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Determine bins
        max_length = max(shortest_proofs_all)
        bins = np.arange(0, max_length + 20, 10)

        ax.hist(shortest_proofs_all, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

        ax.set_xlabel('Shortest Proof Length (lines)', fontsize=12)
        ax.set_ylabel('Number of Theorems/Lemmas', fontsize=12)
        ax.set_title('Distribution of Shortest Proof Lengths for Solved Theorems/Lemmas', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Statistics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Count", len(shortest_proofs_all))
        with col2:
            st.metric("Min", f"{np.min(shortest_proofs_all):.0f}")
        with col3:
            st.metric("P5", f"{np.percentile(shortest_proofs_all, 5):.0f}")
        with col4:
            st.metric("Median", f"{np.median(shortest_proofs_all):.0f}")
        with col5:
            st.metric("P95", f"{np.percentile(shortest_proofs_all, 95):.0f}")
        with col6:
            st.metric("Max", f"{np.max(shortest_proofs_all):.0f}")

        # Per-difficulty histograms
        st.markdown("---")
        st.markdown("**Shortest Proof Lengths by Difficulty**")

        # Get all difficulties and sort them
        difficulties_for_shortest = sorted(shortest_proofs_by_difficulty.keys(),
                                          key=lambda x: {'easy': 0, 'Easy': 0, 'medium': 1, 'Medium': 1, 'hard': 2, 'Hard': 2}.get(x, 3))

        if difficulties_for_shortest:
            # Create one histogram per difficulty
            num_difficulties = len(difficulties_for_shortest)
            fig, axes = plt.subplots(1, num_difficulties, figsize=(6 * num_difficulties, 5))

            # Handle single difficulty case
            if num_difficulties == 1:
                axes = [axes]

            difficulty_colors_map = {
                'easy': 'green',
                'Easy': 'green',
                'medium': 'orange',
                'Medium': 'orange',
                'hard': 'red',
                'Hard': 'red'
            }

            for idx, difficulty in enumerate(difficulties_for_shortest):
                ax = axes[idx]
                proof_lengths = shortest_proofs_by_difficulty[difficulty]

                if proof_lengths:
                    max_length = max(proof_lengths)
                    bins = np.arange(0, max_length + 20, 10)
                    color = difficulty_colors_map.get(difficulty, 'blue')

                    ax.hist(proof_lengths, bins=bins, edgecolor='black', alpha=0.7, color=color)

                    ax.set_xlabel('Shortest Proof Length (lines)', fontsize=11)
                    ax.set_ylabel('Count', fontsize=11)
                    ax.set_title(f'{difficulty.capitalize()} - n={len(proof_lengths)}', fontsize=12, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # Statistics per difficulty
            st.markdown("**Statistics by Difficulty**")
            difficulty_stats = []
            for difficulty in difficulties_for_shortest:
                proof_lengths = shortest_proofs_by_difficulty[difficulty]
                if proof_lengths:
                    difficulty_stats.append({
                        'Difficulty': difficulty.capitalize(),
                        'Count': len(proof_lengths),
                        'Min': f"{np.min(proof_lengths):.0f}",
                        'P5': f"{np.percentile(proof_lengths, 5):.0f}",
                        'Median': f"{np.median(proof_lengths):.0f}",
                        'P95': f"{np.percentile(proof_lengths, 95):.0f}",
                        'Max': f"{np.max(proof_lengths):.0f}"
                    })

            if difficulty_stats:
                df_difficulty = pd.DataFrame(difficulty_stats)
                st.dataframe(df_difficulty, use_container_width=True, hide_index=True)
    else:
        st.info("No solved proof length data available.")

    # ===== ANALYSIS: Used Lemmas vs Proof Length =====
    st.markdown("---")
    st.markdown("**Used Lemmas vs Proof Length Analysis**")

    # Collect used lemmas and proof lengths for solved attempts
    used_lemmas_data = []
    used_lemmas_by_difficulty = defaultdict(list)

    for attempt_info in all_attempts:
        reasoning_summary = attempt_info.get('reasoning_summary')
        is_passing = attempt_info.get('is_passing', False)
        problem_id = attempt_info.get('problem_id')
        difficulty = problem_difficulty_map.get(problem_id, 'Unknown')

        if is_passing and reasoning_summary and isinstance(reasoning_summary, dict):
            proof_length = reasoning_summary.get('proof_length')
            # Count used lemmas from used_lemma_ids if available
            used_lemma_ids = attempt_info.get('used_lemma_ids')
            num_used_lemmas = len(used_lemma_ids) if used_lemma_ids else 0

            if proof_length is not None:
                try:
                    proof_length_val = float(proof_length)
                    used_lemmas_data.append({
                        'proof_length': proof_length_val,
                        'num_used_lemmas': num_used_lemmas,
                        'difficulty': difficulty,
                        'problem_id': problem_id
                    })
                    used_lemmas_by_difficulty[difficulty].append(num_used_lemmas)
                except (ValueError, TypeError):
                    pass

    if used_lemmas_data:
        # Create scatter plot: proof_length (x-axis) vs num_used_lemmas (y-axis)
        df_lemmas = pd.DataFrame(used_lemmas_data)
        fig, ax = plt.subplots(figsize=(12, 6))

        difficulty_colors = {
            'easy': 'green',
            'medium': 'orange',
            'hard': 'red'
        }

        for difficulty in df_lemmas['difficulty'].unique():
            mask = df_lemmas['difficulty'] == difficulty
            subset = df_lemmas[mask]
            color = difficulty_colors.get(difficulty.lower() if difficulty else 'unknown', 'blue')
            ax.scatter(subset['proof_length'], subset['num_used_lemmas'],
                      label=difficulty, alpha=0.6, s=80, color=color, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Proof Length (lines)', fontsize=12)
        ax.set_ylabel('Number of Used Lemmas', fontsize=12)
        ax.set_title('Used Lemmas vs Proof Length (for Passing Attempts)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()
        st.pyplot(fig)

        # Summary statistics per difficulty
        st.markdown("**Summary Statistics: Used Lemmas by Difficulty**")
        summary_data = []
        for difficulty in sorted(used_lemmas_by_difficulty.keys(),
                                key=lambda x: {'easy': 0, 'medium': 1, 'hard': 2}.get(x.lower(), 3)):
            lemmas_list = used_lemmas_by_difficulty[difficulty]
            if lemmas_list:
                summary_data.append({
                    'Difficulty': difficulty.capitalize() if difficulty else 'Unknown',
                    'Count': len(lemmas_list),
                    'Avg': f"{np.mean(lemmas_list):.2f}",
                    'Std': f"{np.std(lemmas_list):.2f}",
                    'Min': f"{np.min(lemmas_list):.0f}",
                    'Median': f"{np.median(lemmas_list):.0f}",
                    'Max': f"{np.max(lemmas_list):.0f}"
                })

        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

        # Solve rate by number of used lemmas
        st.markdown("---")
        st.markdown("**Solve Rate by Number of Used Lemmas**")

        # Group attempts by number of used lemmas
        solve_rate_data = defaultdict(lambda: {'total': 0, 'solved': 0})
        for attempt_info in all_attempts:
            used_lemma_ids = attempt_info.get('used_lemma_ids')
            num_used_lemmas = len(used_lemma_ids) if used_lemma_ids else 0
            is_passing = attempt_info.get('is_passing', False)

            solve_rate_data[num_used_lemmas]['total'] += 1
            if is_passing:
                solve_rate_data[num_used_lemmas]['solved'] += 1

        if solve_rate_data:
            solve_rate_plot_data = []
            for num_lemmas in sorted(solve_rate_data.keys()):
                stats = solve_rate_data[num_lemmas]
                rate = (stats['solved'] / stats['total'] * 100) if stats['total'] > 0 else 0
                solve_rate_plot_data.append({
                    'num_lemmas': num_lemmas,
                    'solve_rate': rate,
                    'total': stats['total'],
                    'solved': stats['solved']
                })

            if solve_rate_plot_data:
                df_solve_rate = pd.DataFrame(solve_rate_plot_data)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(df_solve_rate['num_lemmas'], df_solve_rate['solve_rate'],
                      color='steelblue', edgecolor='black', alpha=0.7)
                ax.set_xlabel('Number of Used Lemmas', fontsize=12)
                ax.set_ylabel('Success Rate (%)', fontsize=12)
                ax.set_title('Success Rate by Number of Used Lemmas', fontsize=14, fontweight='bold')
                ax.set_ylim([0, 100])
                ax.grid(axis='y', alpha=0.3)

                # Add labels on bars
                for i, row in df_solve_rate.iterrows():
                    ax.text(row['num_lemmas'], row['solve_rate'] + 2,
                           f"{row['solve_rate']:.0f}%\n(n={row['total']})",
                           ha='center', fontsize=9)

                plt.tight_layout()
                st.pyplot(fig)

                # Table with details
                table_data = []
                for _, row in df_solve_rate.iterrows():
                    table_data.append({
                        'Num Lemmas': int(row['num_lemmas']),
                        'Solved': int(row['solved']),
                        'Total Attempts': int(row['total']),
                        'Success Rate': f"{row['solve_rate']:.1f}%"
                    })

                df_table = pd.DataFrame(table_data)
                st.dataframe(df_table, use_container_width=True, hide_index=True)
    else:
        st.info("No used lemmas data available for analysis.")


def render_cost_analysis(session) -> None:
    """
    Render cost analysis including:
    - Average cost of proofs in non-correction rounds (output tokens)
    - Average cost of proofs in correction rounds (output tokens)
    - Average cost of breakdown operations (excluding prover)
    - Average cost of recursive breakdowns (excluding prover)
    """
    st.subheader("Cost Analysis - Output Tokens")

    if not session or not session.problems:
        st.warning("No session data available for cost analysis.")
        return

    # Collect costs by operation type and difficulty
    proof_costs_non_corr = defaultdict(list)  # Non-correction round proof attempts by difficulty
    proof_costs_corr = defaultdict(list)       # Correction round proof attempts by difficulty
    breakdown_costs = defaultdict(list)        # Breakdown operation costs by difficulty
    recursive_breakdown_costs = defaultdict(list) # Recursive breakdown costs by difficulty

    # Debug: Track what we're finding
    debug_corr_count = 0
    debug_non_corr_count = 0

    for problem in session.problems.values():
        # Check if this is a recursive breakdown (problem ID is a UID)
        is_recursive = '_l' in problem.origin_problem_id or '_theorem' in problem.origin_problem_id
        difficulty = problem.difficulty if problem.difficulty else 'Unknown'

        for breakdown in problem.breakdowns.values():
            # Get component costs
            component_costs = breakdown.get_component_costs()

            # Track breakdown costs (exclude prover component)
            if 'breakdown' in component_costs or 'breakdown_parser' in component_costs or 'formalization' in component_costs:
                breakdown_cost = 0
                if 'breakdown' in component_costs:
                    breakdown_cost += component_costs['breakdown'].get('output_tokens', 0)
                if 'breakdown_parser' in component_costs:
                    breakdown_cost += component_costs['breakdown_parser'].get('output_tokens', 0)
                if 'formalization' in component_costs:
                    breakdown_cost += component_costs['formalization'].get('output_tokens', 0)

                if breakdown_cost > 0:
                    if is_recursive:
                        recursive_breakdown_costs[difficulty].append(breakdown_cost)
                    else:
                        breakdown_costs[difficulty].append(breakdown_cost)

            # Track proof costs directly from proof attempts (not from component costs)
            # This ensures we capture ALL proof attempts including correction rounds
            if breakdown.parsed_breakdown:
                # Collect lemma proof costs
                for lemma in breakdown.parsed_breakdown.lemmas.values():
                    for formalization in lemma.formalizations:
                        for attempt in formalization.proof_attempts:
                            output_tokens = attempt.get_cost("output_tokens")
                            # Include attempts even if they have no cost data (correction rounds might not have costs recorded)
                            # Only skip attempts with 0 tokens if they're passing (they must have been measured)
                            if output_tokens >= 0:  # Changed from > 0 to >= 0 to include all attempts
                                if attempt.correction_round_id > 0:
                                    proof_costs_corr[difficulty].append(output_tokens)
                                    debug_corr_count += 1
                                else:
                                    proof_costs_non_corr[difficulty].append(output_tokens)
                                    debug_non_corr_count += 1

                # Collect theorem proof costs
                if breakdown.parsed_breakdown.theorem:
                    for formalization in breakdown.parsed_breakdown.theorem.formalizations:
                        for attempt in formalization.proof_attempts:
                            output_tokens = attempt.get_cost("output_tokens")
                            # Include attempts even if they have no cost data (correction rounds might not have costs recorded)
                            # Only skip attempts with 0 tokens if they're passing (they must have been measured)
                            if output_tokens >= 0:  # Changed from > 0 to >= 0 to include all attempts
                                if attempt.correction_round_id > 0:
                                    proof_costs_corr[difficulty].append(output_tokens)
                                    debug_corr_count += 1
                                else:
                                    proof_costs_non_corr[difficulty].append(output_tokens)
                                    debug_non_corr_count += 1

    # DEBUG: Display counts
    st.markdown("**DEBUG: Data Collection Summary**")
    debug_col1, debug_col2 = st.columns(2)
    with debug_col1:
        st.metric("Correction Round Attempts Found", debug_corr_count)
    with debug_col2:
        st.metric("Non-Correction Round Attempts Found", debug_non_corr_count)

    # Display overall cost metrics
    st.markdown("**Overall Average Costs (Output Tokens)**")

    col1, col2, col3, col4 = st.columns(4)

    # Flatten all difficulties for overall stats
    all_proof_costs_non_corr = [cost for costs in proof_costs_non_corr.values() for cost in costs]
    all_proof_costs_corr = [cost for costs in proof_costs_corr.values() for cost in costs]
    all_breakdown_costs = [cost for costs in breakdown_costs.values() for cost in costs]
    all_recursive_breakdown_costs = [cost for costs in recursive_breakdown_costs.values() for cost in costs]

    with col1:
        if all_proof_costs_non_corr:
            avg_cost = np.mean(all_proof_costs_non_corr)
            st.metric(
                "Avg Cost - Proofs (Non-Correction)",
                f"{avg_cost:.1f}",
                f"n={len(all_proof_costs_non_corr)}"
            )
        else:
            st.metric("Avg Cost - Proofs (Non-Correction)", "N/A", "n=0")

    with col2:
        if all_proof_costs_corr:
            avg_cost = np.mean(all_proof_costs_corr)
            st.metric(
                "Avg Cost - Proofs (Correction)",
                f"{avg_cost:.1f}",
                f"n={len(all_proof_costs_corr)}"
            )
        else:
            st.metric("Avg Cost - Proofs (Correction)", "N/A", "n=0")

    with col3:
        if all_breakdown_costs:
            avg_cost = np.mean(all_breakdown_costs)
            st.metric(
                "Avg Cost - Breakdown Ops",
                f"{avg_cost:.1f}",
                f"n={len(all_breakdown_costs)}"
            )
        else:
            st.metric("Avg Cost - Breakdown Ops", "N/A", "n=0")

    with col4:
        if all_recursive_breakdown_costs:
            avg_cost = np.mean(all_recursive_breakdown_costs)
            st.metric(
                "Avg Cost - Recursive Breakdowns",
                f"{avg_cost:.1f}",
                f"n={len(all_recursive_breakdown_costs)}"
            )
        else:
            st.metric("Avg Cost - Recursive Breakdowns", "N/A (no recursive breakdowns yet)")

    # Show detailed cost tables with difficulty breakdown
    st.markdown("---")
    st.markdown("**Detailed Cost Statistics by Difficulty**")

    # Get all difficulties
    all_difficulties = set()
    all_difficulties.update(proof_costs_non_corr.keys())
    all_difficulties.update(proof_costs_corr.keys())
    all_difficulties.update(breakdown_costs.keys())
    all_difficulties.update(recursive_breakdown_costs.keys())

    difficulties_sorted = sorted(all_difficulties,
                                 key=lambda x: {'easy': 0, 'Easy': 0, 'medium': 1, 'Medium': 1, 'hard': 2, 'Hard': 2, 'Unknown': 3}.get(x, 3))

    # ===== TABLE 1: Proof Costs - Non-Correction Rounds =====
    st.markdown("**Proof Operation Costs - Non-Correction Rounds**")

    proof_cost_data_non_corr = []

    for difficulty in difficulties_sorted:
        costs = proof_costs_non_corr[difficulty]
        if costs:
            proof_cost_data_non_corr.append({
                'Difficulty': difficulty.capitalize() if difficulty else 'Unknown',
                'Count': len(costs),
                'Min': f"{np.min(costs):.0f}",
                'P5': f"{np.percentile(costs, 5):.0f}",
                'Median': f"{np.median(costs):.0f}",
                'P95': f"{np.percentile(costs, 95):.0f}",
                'Max': f"{np.max(costs):.0f}",
                'Avg': f"{np.mean(costs):.1f}",
                'Total': f"{np.sum(costs):.0f}"
            })

    if proof_cost_data_non_corr:
        # Calculate totals across all difficulties
        all_costs_non_corr = [cost for costs in proof_costs_non_corr.values() for cost in costs]
        if all_costs_non_corr:
            total_row = {
                'Difficulty': 'TOTAL',
                'Count': len(all_costs_non_corr),
                'Min': f"{np.min(all_costs_non_corr):.0f}",
                'P5': f"{np.percentile(all_costs_non_corr, 5):.0f}",
                'Median': f"{np.median(all_costs_non_corr):.0f}",
                'P95': f"{np.percentile(all_costs_non_corr, 95):.0f}",
                'Max': f"{np.max(all_costs_non_corr):.0f}",
                'Avg': f"{np.mean(all_costs_non_corr):.1f}",
                'Total': f"{np.sum(all_costs_non_corr):.0f}"
            }
            proof_cost_data_non_corr.append(total_row)

        df_proof_non_corr = pd.DataFrame(proof_cost_data_non_corr)
        st.dataframe(df_proof_non_corr, use_container_width=True, hide_index=True)
    else:
        st.info("No non-correction round proof cost data available.")

    # ===== TABLE 2: Proof Costs - Correction Rounds =====
    st.markdown("---")
    st.markdown("**Proof Operation Costs - Correction Rounds**")

    proof_cost_data_corr = []

    for difficulty in difficulties_sorted:
        costs = proof_costs_corr[difficulty]
        if costs:
            proof_cost_data_corr.append({
                'Difficulty': difficulty.capitalize() if difficulty else 'Unknown',
                'Count': len(costs),
                'Min': f"{np.min(costs):.0f}",
                'P5': f"{np.percentile(costs, 5):.0f}",
                'Median': f"{np.median(costs):.0f}",
                'P95': f"{np.percentile(costs, 95):.0f}",
                'Max': f"{np.max(costs):.0f}",
                'Avg': f"{np.mean(costs):.1f}",
                'Total': f"{np.sum(costs):.0f}"
            })

    if proof_cost_data_corr:
        # Calculate totals across all difficulties
        all_costs_corr = [cost for costs in proof_costs_corr.values() for cost in costs]
        if all_costs_corr:
            total_row = {
                'Difficulty': 'TOTAL',
                'Count': len(all_costs_corr),
                'Min': f"{np.min(all_costs_corr):.0f}",
                'P5': f"{np.percentile(all_costs_corr, 5):.0f}",
                'Median': f"{np.median(all_costs_corr):.0f}",
                'P95': f"{np.percentile(all_costs_corr, 95):.0f}",
                'Max': f"{np.max(all_costs_corr):.0f}",
                'Avg': f"{np.mean(all_costs_corr):.1f}",
                'Total': f"{np.sum(all_costs_corr):.0f}"
            }
            proof_cost_data_corr.append(total_row)

        df_proof_corr = pd.DataFrame(proof_cost_data_corr)
        st.dataframe(df_proof_corr, use_container_width=True, hide_index=True)
    else:
        st.info("No correction round proof cost data available.")

    # ===== TABLE 3: Breakdown Costs (Non-Recursive) - Separated by Difficulty =====
    st.markdown("---")
    st.markdown("**Breakdown Operation Costs (Non-Recursive) - By Difficulty**")

    breakdown_cost_data = []

    for difficulty in difficulties_sorted:
        costs = breakdown_costs[difficulty]
        if costs:
            breakdown_cost_data.append({
                'Difficulty': difficulty.capitalize() if difficulty else 'Unknown',
                'Count': len(costs),
                'Min': f"{np.min(costs):.0f}",
                'P5': f"{np.percentile(costs, 5):.0f}",
                'Median': f"{np.median(costs):.0f}",
                'P95': f"{np.percentile(costs, 95):.0f}",
                'Max': f"{np.max(costs):.0f}",
                'Avg': f"{np.mean(costs):.1f}",
                'Total': f"{np.sum(costs):.0f}"
            })

    if breakdown_cost_data:
        # Calculate totals across all difficulties
        all_breakdown_costs = [cost for costs in breakdown_costs.values() for cost in costs]
        if all_breakdown_costs:
            total_row = {
                'Difficulty': 'TOTAL',
                'Count': len(all_breakdown_costs),
                'Min': f"{np.min(all_breakdown_costs):.0f}",
                'P5': f"{np.percentile(all_breakdown_costs, 5):.0f}",
                'Median': f"{np.median(all_breakdown_costs):.0f}",
                'P95': f"{np.percentile(all_breakdown_costs, 95):.0f}",
                'Max': f"{np.max(all_breakdown_costs):.0f}",
                'Avg': f"{np.mean(all_breakdown_costs):.1f}",
                'Total': f"{np.sum(all_breakdown_costs):.0f}"
            }
            breakdown_cost_data.append(total_row)

        df_breakdown = pd.DataFrame(breakdown_cost_data)
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
    else:
        st.info("No breakdown cost data available.")

    # ===== TABLE 4: Recursive Breakdown Costs - Separated by Difficulty =====
    st.markdown("---")
    st.markdown("**Breakdown Operation Costs (Recursive) - By Difficulty**")

    recursive_cost_data = []

    for difficulty in difficulties_sorted:
        costs = recursive_breakdown_costs[difficulty]
        if costs:
            recursive_cost_data.append({
                'Difficulty': difficulty.capitalize() if difficulty else 'Unknown',
                'Count': len(costs),
                'Min': f"{np.min(costs):.0f}",
                'P5': f"{np.percentile(costs, 5):.0f}",
                'Median': f"{np.median(costs):.0f}",
                'P95': f"{np.percentile(costs, 95):.0f}",
                'Max': f"{np.max(costs):.0f}",
                'Avg': f"{np.mean(costs):.1f}",
                'Total': f"{np.sum(costs):.0f}"
            })

    if recursive_cost_data:
        # Calculate totals across all difficulties
        all_recursive_costs = [cost for costs in recursive_breakdown_costs.values() for cost in costs]
        if all_recursive_costs:
            total_row = {
                'Difficulty': 'TOTAL',
                'Count': len(all_recursive_costs),
                'Min': f"{np.min(all_recursive_costs):.0f}",
                'P5': f"{np.percentile(all_recursive_costs, 5):.0f}",
                'Median': f"{np.median(all_recursive_costs):.0f}",
                'P95': f"{np.percentile(all_recursive_costs, 95):.0f}",
                'Max': f"{np.max(all_recursive_costs):.0f}",
                'Avg': f"{np.mean(all_recursive_costs):.1f}",
                'Total': f"{np.sum(all_recursive_costs):.0f}"
            }
            recursive_cost_data.append(total_row)

        df_recursive = pd.DataFrame(recursive_cost_data)
        st.dataframe(df_recursive, use_container_width=True, hide_index=True)
    else:
        st.info("No recursive breakdown cost data available (none run yet).")

    # ===== COMPONENT-LEVEL TABLES =====
    st.markdown("---")
    st.markdown("**Component-Level Output Token Costs by Difficulty**")

    # Collect component costs per component
    component_costs = defaultdict(lambda: defaultdict(lambda: []))

    for problem in session.problems.values():
        difficulty = problem.difficulty if problem.difficulty else 'Unknown'

        for breakdown in problem.breakdowns.values():
            component_costs_dict = breakdown.get_component_costs()

            # Collect costs per component
            for component_name in ['breakdown', 'breakdown_parser', 'formalization', 'prover']:
                if component_name in component_costs_dict:
                    output_tokens = component_costs_dict[component_name].get('output_tokens', 0)
                    if output_tokens > 0:
                        component_costs[difficulty][component_name].append(output_tokens)

    # Display each component in its own table
    component_names = ['breakdown', 'breakdown_parser', 'formalization', 'prover']

    for component_name in component_names:
        component_data = []

        for difficulty in difficulties_sorted:
            costs = component_costs[difficulty][component_name]
            if costs:
                component_data.append({
                    'Difficulty': difficulty.capitalize() if difficulty else 'Unknown',
                    'Count': len(costs),
                    'Min': f"{np.min(costs):.0f}",
                    'P5': f"{np.percentile(costs, 5):.0f}",
                    'Median': f"{np.median(costs):.0f}",
                    'P95': f"{np.percentile(costs, 95):.0f}",
                    'Max': f"{np.max(costs):.0f}",
                    'Avg': f"{np.mean(costs):.1f}",
                    'Total': f"{np.sum(costs):.0f}"
                })

        if component_data:
            st.markdown(f"**{component_name.capitalize()} Component**")
            df_component = pd.DataFrame(component_data)
            st.dataframe(df_component, use_container_width=True, hide_index=True)
            st.markdown("---")

    st.info("Component tables show output token costs for each pipeline stage.")
