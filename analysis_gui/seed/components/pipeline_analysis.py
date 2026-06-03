"""
Pipeline Analysis component - analyzes proof generation pipeline progression.

Provides comprehensive analysis of:
- Proof completion by iteration
- Per-theorem analysis
- Per-lemma analysis
- Lemma usage in final proofs
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
import sys
from pathlib import Path

# Add root directory to path to import seed_data_models
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

try:
    from seed_data_models import Session, Problem
    HAS_NEW_MODELS = True
except ImportError:
    HAS_NEW_MODELS = False
    Session = None
    Problem = None


# Try to import get_lemma_id
try:
    from id_utils import get_lemma_id
except ImportError:
    def get_lemma_id(problem_id: str) -> str:
        """Fallback implementation of get_lemma_id."""
        canonical = str(problem_id)
        # Remove correction suffix (_corr<N>)
        if "_corr" in canonical:
            canonical = canonical.split("_corr")[0]
        # Remove proof attempt suffix (_p<N>)
        if "_p" in canonical:
            parts = canonical.rsplit("_p", 1)
            if len(parts) == 2 and parts[1].isdigit():
                canonical = parts[0]
        # Remove round suffix (_r<N>)
        if "_r" in canonical:
            parts = canonical.rsplit("_r", 1)
            if len(parts) == 2 and parts[1].isdigit():
                canonical = parts[0]
        return canonical


def render_pipeline_analysis(data: Union['Session', List[Any]]):
    """
    Render the pipeline analysis view showing proof progression through iterations.

    Args:
        data: Session object (new) or List of ProblemSummary objects (old)
    """
    st.header("🔧 Pipeline Analysis")
    st.markdown("Analyze how proofs progress through the pipeline iterations")

    # Convert Session to list of problems if needed
    if hasattr(data, 'problems') and isinstance(data.problems, dict):
        # Session object (has problems dict)
        problems = list(data.problems.values())
    else:
        # List of ProblemSummary objects
        problems = data

    # Formalization and proof statistics
    render_formalization_statistics(problems)

    st.markdown("---")

    # Pipeline flow statistics (from problem_browser.py, adapted for pipeline tab)
    render_pipeline_flow_statistics(problems)

    st.markdown("---")

    # Extract pipeline data
    pipeline_data = extract_pipeline_data(problems)

    # Overall stats
    render_pipeline_overview(pipeline_data, problems)

    st.markdown("---")

    # Component cost analysis
    st.subheader("💰 Component Token Cost Analysis")
    render_component_cost_analysis(problems)

    st.markdown("---")

    # Proof completion by iteration
    st.subheader("📊 Proof Completion by Iteration")
    render_iteration_analysis(pipeline_data)

    st.markdown("---")

    # Average lemmas used per iteration
    st.subheader("📈 Average Lemmas Used per Iteration")
    render_lemma_usage_by_iteration(pipeline_data)

    st.markdown("---")

    # Per-theorem analysis
    st.subheader("🎯 Per-Theorem Analysis")
    render_theorem_analysis(pipeline_data, problems)

    st.markdown("---")

    # Per-lemma analysis
    st.subheader("📝 Per-Lemma Analysis")
    render_lemma_analysis(pipeline_data, problems)

    st.markdown("---")

    # Lemma validation distribution
    st.subheader("✅ Lemma Validation Distribution")
    render_lemma_validation_distribution(pipeline_data)

    st.markdown("---")

    # Missing lemmas distribution
    st.subheader("❌ Missing Lemmas per Breakdown")
    render_missing_lemmas_distribution(pipeline_data)

    st.markdown("---")

    # Lemma usage analysis
    st.subheader("🧮 Lemma Usage in Final Proofs")
    render_lemma_usage_analysis(pipeline_data, problems)

    st.markdown("---")

    # Error correctability analysis
    if hasattr(data, 'problems') and isinstance(data.problems, dict):
        # Only show if we have a Session object (new models)
        render_error_correctability_analysis(data)


def render_formalization_statistics(problems: List['Problem']):
    """
    Render formalization and proof completion statistics.

    Shows:
    - Total breakdowns that passed formalization
    - Total breakdowns with theorem proofs
    - Total breakdowns that were fully solved

    Reads from full_records to include all iterations and correction rounds.

    Args:
        problems: List of Problem objects
    """
    st.subheader("📊 Formalization & Proof Statistics")

    # Load full_records to get all proof attempts across iterations
    from pathlib import Path
    import json

    all_theorem_records = []
    try:
        run_dir = st.session_state.get('run_dir')
        if run_dir:
            full_records_dir = Path(run_dir) / "full_records"
            if full_records_dir.exists():
                for records_file in full_records_dir.glob("*.json"):
                    try:
                        with open(records_file) as f:
                            records = json.load(f)
                            if isinstance(records, list):
                                for record in records:
                                    meta = record.get('metadata', {})
                                    # Only get theorems (lemma_id == -1)
                                    if meta.get('lemma_id') == -1:
                                        all_theorem_records.append(record)
                    except:
                        pass
    except:
        pass

    # Count statistics from the data model (should match number of breakdowns)
    total_breakdowns = 0
    formalizations_passed = 0
    theorem_proofs = 0
    fully_solved = 0

    for problem in problems:
        if not problem.breakdowns:
            continue

        for breakdown in problem.breakdowns.values():
            total_breakdowns += 1

            # Check if breakdown was fully formalized (theorem AND all lemmas compiled)
            # This matches what problem_browser does
            is_formalized = False
            if breakdown.parsed_breakdown:
                is_formalized = breakdown.parsed_breakdown.is_formalized()

            if is_formalized:
                formalizations_passed += 1

            # Check if theorem has a passing proof
            if breakdown.parsed_breakdown and breakdown.parsed_breakdown.theorem:
                best_attempt = breakdown.parsed_breakdown.theorem.get_best_attempt()
                if best_attempt and best_attempt.is_passing():
                    theorem_proofs += 1

            # Check if fully solved
            if breakdown.is_solved():
                fully_solved += 1

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Breakdowns", total_breakdowns)

    with col2:
        passed_pct = (formalizations_passed / total_breakdowns * 100) if total_breakdowns > 0 else 0
        st.metric(
            "Formalizations Passed",
            f"{formalizations_passed}/{total_breakdowns}",
            f"{passed_pct:.1f}%"
        )

    with col3:
        proof_pct = (theorem_proofs / total_breakdowns * 100) if total_breakdowns > 0 else 0
        st.metric(
            "Theorem Proofs",
            f"{theorem_proofs}/{total_breakdowns}",
            f"{proof_pct:.1f}%"
        )

    with col4:
        solved_pct = (fully_solved / total_breakdowns * 100) if total_breakdowns > 0 else 0
        st.metric(
            "Fully Solved",
            f"{fully_solved}/{total_breakdowns}",
            f"{solved_pct:.1f}%"
        )


def extract_pipeline_data(problems: List['Problem']) -> Dict[str, Any]:
    """
    Extract comprehensive pipeline data from all problems using the data model.

    Traverses the actual data model structure:
    Breakdown -> ParsedBreakdown -> Theorem -> Formalizations -> ProofAttempts

    This automatically includes all iterations and correction rounds.

    Args:
        problems: List of Problem objects (from Session)

    Returns:
        Dictionary containing organized pipeline data
    """
    pipeline_data = {
        'by_iter_corr': defaultdict(lambda: {'attempts': [], 'passed': 0, 'total': 0, 'lemma_usage_list': []}),
        'attempt_distribution': defaultdict(int),  # How many proofs took N attempts? (both theorems and lemmas)
        'theorem_attempt_distribution': defaultdict(int),  # How many theorems took N attempts?
        'unique_theorem_attempt_distribution': defaultdict(int),  # How many unique origin_problem_id theorems took N attempts (min across breakdowns)?
        'by_theorem': defaultdict(dict),
        'by_lemma': defaultdict(dict),
        'lemma_usage': defaultdict(list),
        'all_problems': [],
        'total_theorems': 0,
        'total_lemmas_attempted': 0,
        'total_problems': len(problems),
        'origin_problem_attempts': {},  # Min attempts for each origin_problem_id (only if solved)
        'origin_problem_solved': set(),  # Set of origin_problem_ids that were solved
        'origin_problem_all': set(),  # All origin_problem_ids encountered
        'lemma_usage_by_iteration': defaultdict(list)  # Track lemma usage counts per iteration
    }

    for problem in problems:
        pipeline_data['all_problems'].append(problem.origin_problem_id)
        pipeline_data['origin_problem_all'].add(problem.origin_problem_id)

        if not problem.breakdowns:
            continue

        for breakdown in problem.breakdowns.values():
            # Extract theorem attempts from parsed_breakdown data model
            if not breakdown.parsed_breakdown or not breakdown.parsed_breakdown.theorem:
                continue

            theorem = breakdown.parsed_breakdown.theorem

            # Collect all proof attempts from all formalizations
            all_theorem_attempts = []
            for formalization in theorem.formalizations:
                for attempt in formalization.proof_attempts:
                    all_theorem_attempts.append(attempt)

            if all_theorem_attempts:
                theorem_data = {
                    'problem_id': breakdown.problem_id,
                    'origin_problem': problem.origin_problem_id,
                    'attempts': []
                }

                for attempt in all_theorem_attempts:
                    # Get compilation result from the attempt itself
                    comp_result = attempt.compilation_result
                    iteration = attempt.iteration_id if hasattr(attempt, 'iteration_id') else attempt.iteration
                    correction_round = attempt.correction_round_id if hasattr(attempt, 'correction_round_id') else attempt.correction_round

                    attempt_info = {
                        'problem_id': breakdown.problem_id,
                        'iteration': iteration,
                        'correction_round': correction_round,
                        'pass': comp_result.passed if hasattr(comp_result, 'passed') else comp_result.get('pass', False),
                        'complete': comp_result.complete if hasattr(comp_result, 'complete') else comp_result.get('complete', False),
                        'type': 'theorem'
                    }

                    theorem_data['attempts'].append(attempt_info)

                    # Track by (iteration, correction_round)
                    key = (iteration, correction_round)
                    pipeline_data['by_iter_corr'][key]['total'] += 1
                    pipeline_data['by_iter_corr'][key]['attempts'].append(attempt_info)

                    if attempt_info['pass'] and attempt_info['complete']:
                        pipeline_data['by_iter_corr'][key]['passed'] += 1

                pipeline_data['by_theorem'][breakdown.problem_id] = theorem_data
                pipeline_data['total_theorems'] += 1

                # Track attempt distribution for this theorem
                total_attempts_for_theorem = len(all_theorem_attempts)
                theorem_succeeded = False
                for attempt in all_theorem_attempts:
                    comp_result = attempt.compilation_result
                    is_passing = (comp_result.passed if hasattr(comp_result, 'passed') else comp_result.get('pass', False))
                    is_complete = (comp_result.complete if hasattr(comp_result, 'complete') else comp_result.get('complete', False))

                    if is_passing and is_complete:
                        # This theorem succeeded after N attempts
                        pipeline_data['attempt_distribution'][total_attempts_for_theorem] += 1
                        pipeline_data['theorem_attempt_distribution'][total_attempts_for_theorem] += 1

                        # Track for unique origin_problem_id
                        origin_id = problem.origin_problem_id
                        if origin_id not in pipeline_data['origin_problem_attempts']:
                            pipeline_data['origin_problem_attempts'][origin_id] = total_attempts_for_theorem
                        else:
                            pipeline_data['origin_problem_attempts'][origin_id] = min(
                                pipeline_data['origin_problem_attempts'][origin_id],
                                total_attempts_for_theorem
                            )
                        pipeline_data['origin_problem_solved'].add(origin_id)
                        theorem_succeeded = True
                        break

                # Extract lemma usage from successful theorem
                if theorem_succeeded:
                    if breakdown.parsed_breakdown:
                        best_attempt = theorem.get_best_attempt()
                        iteration = best_attempt.iteration if hasattr(best_attempt, 'iteration') else 0

                        # Get used lemmas from the best attempt, passing the lemmas dict for accurate extraction
                        lemmas_dict = breakdown.parsed_breakdown.lemmas if breakdown.parsed_breakdown else {}
                        used_lemmas = best_attempt.get_used_lemmas(lemmas_dict=lemmas_dict)
                        num_used_lemmas = len(used_lemmas) if used_lemmas else 0

                        pipeline_data['lemma_usage'][breakdown.problem_id] = {
                            'used_lemmas': list(used_lemmas) if used_lemmas else [],
                            'count': num_used_lemmas,
                            'num_proven': 0,  # Legacy field for compatibility
                            'iteration_completed': iteration
                        }
                        # Track lemma usage by iteration
                        pipeline_data['lemma_usage_by_iteration'][iteration].append(num_used_lemmas)

            # Extract lemma attempts from parsed_breakdown data model
            lemmas_dict = breakdown.parsed_breakdown.lemmas if breakdown.parsed_breakdown else {}
            lemma_map = defaultdict(list)

            for lemma_id, lemma in lemmas_dict.items():
                # Collect all proof attempts from all formalizations for this lemma
                for formalization in lemma.formalizations:
                    for attempt in formalization.proof_attempts:
                        iteration = attempt.iteration_id if hasattr(attempt, 'iteration_id') else attempt.iteration
                        correction_round = attempt.correction_round_id if hasattr(attempt, 'correction_round_id') else attempt.correction_round
                        comp_result = attempt.compilation_result

                        is_passing = (comp_result.passed if hasattr(comp_result, 'passed') else comp_result.get('pass', False))
                        is_complete = (comp_result.complete if hasattr(comp_result, 'complete') else comp_result.get('complete', False))

                        attempt_info = {
                            'problem_id': breakdown.problem_id,
                            'lemma_id': lemma_id,
                            'iteration': iteration,
                            'correction_round': correction_round,
                            'pass': is_passing,
                            'complete': is_complete,
                            'type': 'lemma'
                        }

                        lemma_map[lemma_id].append(attempt_info)

                        # Track by (iteration, correction_round)
                        key = (iteration, correction_round)
                        pipeline_data['by_iter_corr'][key]['total'] += 1
                        pipeline_data['by_iter_corr'][key]['attempts'].append(attempt_info)

                        if is_passing and is_complete:
                            pipeline_data['by_iter_corr'][key]['passed'] += 1

            # Store lemma data by breakdown and lemma ID
            for lemma_id, attempts_list in lemma_map.items():
                key = f"{breakdown.problem_id}_{lemma_id}"
                successful = any(a['pass'] and a['complete'] for a in attempts_list)

                pipeline_data['by_lemma'][key] = {
                    'theorem': problem.origin_problem_id,
                    'breakdown': breakdown.problem_id,
                    'lemma_id': lemma_id,
                    'attempts': attempts_list,
                    'total_attempts': len(attempts_list),
                    'successful': successful
                }

                # Track attempt distribution for this lemma
                if successful:
                    pipeline_data['attempt_distribution'][len(attempts_list)] += 1

                pipeline_data['total_lemmas_attempted'] += 1

    # Compute unique theorem attempt distribution (aggregated by origin_problem_id)
    for origin_id, min_attempts in pipeline_data['origin_problem_attempts'].items():
        pipeline_data['unique_theorem_attempt_distribution'][min_attempts] += 1

    return pipeline_data


def render_pipeline_flow_statistics(problems: List[Any]):
    """
    Render pipeline flow statistics - now minimal, just showing proved counts.

    Args:
        problems: List of ProblemSummary objects
    """
    # This section is now removed - the main overview metrics are in render_pipeline_overview
    pass


def render_pipeline_overview(pipeline_data: Dict[str, Any], problems: List[Any]):
    """
    Render overview statistics of the pipeline.

    Args:
        pipeline_data: Pipeline data dictionary
        problems: List of ProblemSummary objects
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Problems", pipeline_data['total_problems'])

    with col2:
        st.metric("Total Theorems", pipeline_data['total_theorems'])

    with col3:
        st.metric("Total Lemmas Attempted", pipeline_data['total_lemmas_attempted'])

    st.markdown("---")

    # Count theorems and lemmas proven by iteration
    col5a, col5b, col5c = st.columns(3)

    with col5a:
        # Theorems proven (iteration 0)
        theorems_proven = sum(1 for it_data in pipeline_data['by_iter_corr'].values()
                             if it_data['passed'] > 0
                             for attempt in it_data['attempts']
                             if attempt.get('iteration') == 0)
        # Better: count unique theorems in iteration 0 that passed
        unique_theorems_proven_iter0 = set()
        for (it, cr), data in pipeline_data['by_iter_corr'].items():
            if it == 0:
                for attempt in data['attempts']:
                    if attempt['pass'] and attempt['complete']:
                        canonical_id = get_lemma_id(attempt.get('problem_id', ''))
                        unique_theorems_proven_iter0.add(canonical_id)
        st.metric("Theorems Proven (Iter 0)", len(unique_theorems_proven_iter0))

    with col5b:
        # Lemmas proven (iterations 1+)
        unique_lemmas_proven_iter1plus = set()
        for (it, cr), data in pipeline_data['by_iter_corr'].items():
            if it and it >= 1:
                for attempt in data['attempts']:
                    if attempt['pass'] and attempt['complete']:
                        canonical_id = get_lemma_id(attempt.get('problem_id', ''))
                        unique_lemmas_proven_iter1plus.add(canonical_id)
        st.metric("Lemmas Proven (Iter 1+)", len(unique_lemmas_proven_iter1plus))

    with col5c:
        # Total proven
        total_proven = len(unique_theorems_proven_iter0) + len(unique_lemmas_proven_iter1plus)
        st.metric("Total Proven", total_proven)

    st.markdown("---")

    # Additional statistics
    col6, col7, col8 = st.columns(3)

    with col6:
        # Success rate for lemmas (iterations 1+ only)
        if pipeline_data['total_lemmas_attempted'] > 0:
            success_rate = (len(unique_lemmas_proven_iter1plus) / pipeline_data['total_lemmas_attempted']) * 100
            st.metric("Lemma Success Rate", f"{success_rate:.1f}%")

    with col7:
        # Count single-attempt failures
        single_failures = sum(1 for ld in pipeline_data['by_lemma'].values() if ld['total_attempts'] == 1 and not ld['successful'])
        st.metric("Single-Attempt Failures", single_failures)

    with col8:
        # Average lemmas per proof
        if pipeline_data['lemma_usage']:
            avg_lemmas = sum(d['count'] for d in pipeline_data['lemma_usage'].values()) / len(pipeline_data['lemma_usage'])
            st.metric("Avg Lemmas/Proof", f"{avg_lemmas:.1f}")


def render_iteration_analysis(pipeline_data: Dict[str, Any]):
    """
    Render analysis of attempts and successful proofs per (iteration, correction_round) combination.
    Also shows distribution of attempts needed for successful proofs.

    Args:
        pipeline_data: Pipeline data dictionary
    """
    st.markdown("### 📊 Attempts and Successes per (Iteration, Correction Round)")

    # Build table from by_iter_corr data
    iter_corr_rows = []
    for (iteration, correction_round), data in sorted(pipeline_data['by_iter_corr'].items()):
        total_attempts = data['total']
        passed = data['passed']
        pass_rate = (passed / total_attempts * 100) if total_attempts > 0 else 0

        iter_corr_rows.append({
            'Iteration': iteration,
            'Correction Round': correction_round,
            'Total Attempts': total_attempts,
            'Passed': passed,
            'Failed': total_attempts - passed,
            'Pass Rate': f"{pass_rate:.1f}%"
        })

    df_iter_corr = pd.DataFrame(iter_corr_rows)
    st.dataframe(df_iter_corr, width="stretch", hide_index=True)

    st.markdown("---")

    # Add unique proofs per iteration summary
    st.markdown("### 🎯 Unique Lemmas/Theorems Attempted and Proven per Iteration")
    st.markdown("How many unique lemmas and theorems were attempted and successfully proven in each iteration?")

    unique_proven_rows = []
    for iteration in sorted(set(it for it, _ in pipeline_data['by_iter_corr'].keys())):
        # Count unique lemmas/theorems by canonical ID (removing corrections and samples)
        successful_ids = set()
        total_attempted_ids = set()

        for (it, cr), data in pipeline_data['by_iter_corr'].items():
            if it == iteration:
                for attempt in data['attempts']:
                    problem_id = attempt.get('problem_id', '')
                    if problem_id:
                        # Use get_lemma_id to get canonical ID (removes _pX, _corrX, etc.)
                        canonical_id = get_lemma_id(problem_id)
                        total_attempted_ids.add(canonical_id)

                        if attempt['pass'] and attempt['complete']:
                            successful_ids.add(canonical_id)

        unique_proven_rows.append({
            'Iteration': iteration,
            'Unique Attempted': len(total_attempted_ids),
            'Unique Proven': len(successful_ids),
            'Success Rate': f"{len(successful_ids)}/{len(total_attempted_ids)}" if total_attempted_ids else "0/0"
        })

    df_unique = pd.DataFrame(unique_proven_rows)
    st.dataframe(df_unique, width="stretch", hide_index=True)

    # Summary statistics
    st.markdown("---")
    st.markdown("### 📈 Summary Statistics")

    col1, col2, col3 = st.columns(3)

    total_all_attempts = sum(data['total'] for data in pipeline_data['by_iter_corr'].values())
    total_all_passed = sum(data['passed'] for data in pipeline_data['by_iter_corr'].values())

    with col1:
        st.metric("Total Attempts (All)", total_all_attempts)

    with col2:
        st.metric("Total Passed (All)", total_all_passed)

    with col3:
        overall_pass_rate = (total_all_passed / total_all_attempts * 100) if total_all_attempts > 0 else 0
        st.metric("Overall Pass Rate", f"{overall_pass_rate:.1f}%")

    st.markdown("---")

    # Distribution of attempts needed for success
    st.markdown("### 🎯 Distribution of Attempts to Success")
    st.markdown("How many successful proofs required 1, 2, 3, ... N attempts?")

    if pipeline_data['attempt_distribution']:
        dist_rows = []
        for num_attempts in sorted(pipeline_data['attempt_distribution'].keys()):
            count = pipeline_data['attempt_distribution'][num_attempts]
            dist_rows.append({
                'Number of Attempts': num_attempts,
                'Successful Proofs': count
            })

        df_dist = pd.DataFrame(dist_rows)
        st.dataframe(df_dist, width="stretch", hide_index=True)

        # Statistics on distribution
        st.markdown("---")
        st.markdown("#### 💡 Insights on Attempt Distribution")

        col1, col2, col3, col4 = st.columns(4)

        total_successful = sum(pipeline_data['attempt_distribution'].values())
        with col1:
            st.metric("Total Successful Proofs", total_successful)

        # Most common number of attempts
        if pipeline_data['attempt_distribution']:
            most_common_attempts = max(pipeline_data['attempt_distribution'].items(), key=lambda x: x[1])
            with col2:
                st.metric("Most Common Attempts", f"{most_common_attempts[0]} ({most_common_attempts[1]} proofs)")

            # Proofs that succeeded on first try
            one_shot = pipeline_data['attempt_distribution'].get(1, 0)
            with col3:
                pct_one_shot = (one_shot / total_successful * 100) if total_successful > 0 else 0
                st.metric("First-Try Success", f"{one_shot} ({pct_one_shot:.1f}%)")

            # Average attempts
            total_attempts_weighted = sum(attempts * count for attempts, count in pipeline_data['attempt_distribution'].items())
            avg_attempts = total_attempts_weighted / total_successful if total_successful > 0 else 0
            with col4:
                st.metric("Avg Attempts/Proof", f"{avg_attempts:.2f}")

        # Bar chart of distribution with unsolved lemmas
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 5))

            attempts = sorted(pipeline_data['attempt_distribution'].keys())
            counts = [pipeline_data['attempt_distribution'][a] for a in attempts]

            # Add unsolved lemmas count as final bar
            unsolved_lemmas = sum(1 for ld in pipeline_data['by_lemma'].values() if not ld['successful'])
            attempts_with_unsolved = attempts + ['Unsolved']
            counts_with_unsolved = counts + [unsolved_lemmas]

            # Color bars differently
            colors = ['steelblue'] * len(attempts) + ['crimson']

            bars = ax.bar(range(len(attempts_with_unsolved)), counts_with_unsolved, color=colors, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Number of Attempts')
            ax.set_ylabel('Count')
            ax.set_title('Distribution: How Many Attempts Needed for Success? (Plus Unsolved)')
            ax.set_xticks(range(len(attempts_with_unsolved)))
            ax.set_xticklabels(attempts_with_unsolved)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts_with_unsolved)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Show note about unsolved lemmas
            st.info(f"🔴 **Unsolved Lemmas**: {unsolved_lemmas} lemmas could not be proven despite attempts")

        except Exception as e:
            st.warning(f"Could not render distribution chart: {e}")

    else:
        st.info("No successful proof data available")

    # Theorem attempt distribution
    if pipeline_data['theorem_attempt_distribution']:
        st.markdown("---")
        st.markdown("### 🎯 Distribution of Attempts to Success (Theorems Only)")
        st.markdown("How many successful theorem proofs required 1, 2, 3, ... N attempts?")

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 5))

            theorem_attempts = sorted(pipeline_data['theorem_attempt_distribution'].keys())
            theorem_counts = [pipeline_data['theorem_attempt_distribution'][a] for a in theorem_attempts]

            # Count unsolved theorems (theorems with no successful attempt)
            unsolved_theorems = pipeline_data['total_theorems'] - sum(theorem_counts)

            # Add unsolved theorems as final bar
            attempts_with_unsolved = theorem_attempts + ['Unsolved']
            counts_with_unsolved = theorem_counts + [unsolved_theorems]

            # Color bars differently
            colors = ['dodgerblue'] * len(theorem_attempts) + ['crimson']

            bars = ax.bar(range(len(attempts_with_unsolved)), counts_with_unsolved, color=colors, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Number of Attempts')
            ax.set_ylabel('Count')
            ax.set_title('Theorem Attempt Distribution: How Many Attempts Needed for Success? (Plus Unsolved)')
            ax.set_xticks(range(len(attempts_with_unsolved)))
            ax.set_xticklabels(attempts_with_unsolved)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, count in zip(bars, counts_with_unsolved):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Statistics on theorem distribution
            total_successful_theorems = sum(pipeline_data['theorem_attempt_distribution'].values())
            if pipeline_data['theorem_attempt_distribution']:
                most_common_attempts = max(pipeline_data['theorem_attempt_distribution'].items(), key=lambda x: x[1])
                one_shot_theorems = pipeline_data['theorem_attempt_distribution'].get(1, 0)
                total_attempts_weighted = sum(attempts * count for attempts, count in pipeline_data['theorem_attempt_distribution'].items())
                avg_attempts = total_attempts_weighted / total_successful_theorems if total_successful_theorems > 0 else 0

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Successful Theorems", total_successful_theorems)

                with col2:
                    st.metric("Most Common Attempts", f"{most_common_attempts[0]} ({most_common_attempts[1]} theorems)")

                with col3:
                    pct_one_shot = (one_shot_theorems / total_successful_theorems * 100) if total_successful_theorems > 0 else 0
                    st.metric("First-Try Success", f"{one_shot_theorems} ({pct_one_shot:.1f}%)")

                with col4:
                    st.metric("Avg Attempts/Theorem", f"{avg_attempts:.2f}")

            # Show note about unsolved theorems
            if unsolved_theorems > 0:
                st.info(f"🔴 **Unsolved Theorems**: {unsolved_theorems} theorems could not be proven despite attempts")

        except Exception as e:
            st.warning(f"Could not render theorem attempt distribution chart: {e}")

    # Unique theorem attempt distribution (aggregated by origin_problem_id)
    if pipeline_data['unique_theorem_attempt_distribution']:
        st.markdown("---")
        st.markdown("### 🎯 Distribution of Minimum Attempts for Unique Theorems")
        st.markdown("For each unique theorem (origin_problem_id), what was the minimum number of attempts needed across all its breakdowns?")

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 5))

            unique_attempts = sorted(pipeline_data['unique_theorem_attempt_distribution'].keys())
            unique_counts = [pipeline_data['unique_theorem_attempt_distribution'][a] for a in unique_attempts]

            # Count unsolved unique theorems
            total_unique_attempted = len(pipeline_data['origin_problem_all'])
            unsolved_unique_theorems = len(pipeline_data['origin_problem_all'] - pipeline_data['origin_problem_solved'])

            # Add unsolved theorems as final bar
            attempts_with_unsolved = unique_attempts + ['Unsolved']
            counts_with_unsolved = unique_counts + [unsolved_unique_theorems]

            # Color bars differently
            colors = ['steelblue'] * len(unique_attempts) + ['crimson']

            bars = ax.bar(range(len(attempts_with_unsolved)), counts_with_unsolved, color=colors, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Minimum Number of Attempts')
            ax.set_ylabel('Count')
            ax.set_title('Unique Theorem Attempt Distribution: Minimum Attempts Needed (Plus Unsolved)')
            ax.set_xticks(range(len(attempts_with_unsolved)))
            ax.set_xticklabels(attempts_with_unsolved)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, count in zip(bars, counts_with_unsolved):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Statistics on unique theorem distribution
            total_successful_unique = len(pipeline_data['origin_problem_solved'])
            if pipeline_data['unique_theorem_attempt_distribution']:
                most_common_unique_attempts = max(pipeline_data['unique_theorem_attempt_distribution'].items(), key=lambda x: x[1])
                one_shot_unique = pipeline_data['unique_theorem_attempt_distribution'].get(1, 0)
                total_attempts_weighted = sum(attempts * count for attempts, count in pipeline_data['unique_theorem_attempt_distribution'].items())
                avg_attempts_unique = total_attempts_weighted / total_successful_unique if total_successful_unique > 0 else 0

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Unique Successful Theorems", total_successful_unique)

                with col2:
                    st.metric("Most Common Min Attempts", f"{most_common_unique_attempts[0]} ({most_common_unique_attempts[1]} theorems)")

                with col3:
                    pct_one_shot_unique = (one_shot_unique / total_successful_unique * 100) if total_successful_unique > 0 else 0
                    st.metric("First-Try Success", f"{one_shot_unique} ({pct_one_shot_unique:.1f}%)")

                with col4:
                    st.metric("Avg Min Attempts/Theorem", f"{avg_attempts_unique:.2f}")

            # Show note about unsolved unique theorems
            if unsolved_unique_theorems > 0:
                st.info(f"🔴 **Unsolved Unique Theorems**: {unsolved_unique_theorems} unique theorems could not be proven in any breakdown")

        except Exception as e:
            st.warning(f"Could not render unique theorem attempt distribution chart: {e}")


def render_lemma_usage_by_iteration(pipeline_data: Dict[str, Any]):
    """
    Render average number of lemmas used per proving iteration.

    Shows a table with:
    - Iteration number
    - Count of proofs completed in that iteration
    - Average number of lemmas used
    - Min/Max lemmas used
    - Standard deviation

    Args:
        pipeline_data: Pipeline data dictionary
    """
    if not pipeline_data['lemma_usage_by_iteration']:
        st.info("No lemma usage data available for iterations")
        return

    # Build table from lemma_usage_by_iteration data
    usage_rows = []
    for iteration in sorted(pipeline_data['lemma_usage_by_iteration'].keys()):
        usage_counts = pipeline_data['lemma_usage_by_iteration'][iteration]

        if not usage_counts:
            continue

        import statistics

        avg_lemmas = sum(usage_counts) / len(usage_counts) if usage_counts else 0
        min_lemmas = min(usage_counts) if usage_counts else 0
        max_lemmas = max(usage_counts) if usage_counts else 0

        # Calculate standard deviation if we have multiple data points
        if len(usage_counts) > 1:
            std_dev = statistics.stdev(usage_counts)
        else:
            std_dev = 0

        usage_rows.append({
            'Iteration': iteration,
            'Proofs Completed': len(usage_counts),
            'Avg Lemmas Used': f"{avg_lemmas:.2f}",
            'Min': min_lemmas,
            'Max': max_lemmas,
            'Std Dev': f"{std_dev:.2f}"
        })

    if usage_rows:
        df_usage = pd.DataFrame(usage_rows)
        st.dataframe(df_usage, width="stretch", hide_index=True)
    else:
        st.info("No lemma usage statistics available")


def render_theorem_analysis(pipeline_data: Dict[str, Any], problems: List[Any]):
    """
    Render per-theorem analysis.

    Args:
        pipeline_data: Pipeline data dictionary
        problems: List of ProblemSummary objects
    """
    theorem_rows = []

    for theorem_id, theorem_data in sorted(pipeline_data['by_theorem'].items()):
        attempts = theorem_data['attempts']
        max_iteration = max([a['iteration'] for a in attempts], default=0)
        completed = any(a['pass'] and a['complete'] for a in attempts)
        total_attempts = len(attempts)
        pass_rate = sum(1 for a in attempts if a['pass']) / len(attempts) * 100 if attempts else 0

        theorem_rows.append({
            'Theorem': theorem_data['origin_problem'],
            'Breakdown': theorem_id,
            'Attempts': total_attempts,
            'Max Iteration': max_iteration,
            'Pass Rate': f"{pass_rate:.0f}%",
            'Completed': '✅' if completed else '❌',
            'Final Status': 'Proven' if completed else 'Not Proven'
        })

    df = pd.DataFrame(theorem_rows)

    st.dataframe(df, width="stretch", hide_index=True)

    # Expandable per-theorem detail
    with st.expander("View iteration breakdown per theorem"):
        for theorem_id, theorem_data in sorted(pipeline_data['by_theorem'].items()):
            with st.expander(f"{theorem_data['origin_problem']} ({theorem_id})"):
                iteration_breakdown = defaultdict(list)
                for attempt in theorem_data['attempts']:
                    iteration_breakdown[attempt['iteration']].append(attempt)

                detail_rows = []
                for iteration in sorted(iteration_breakdown.keys()):
                    iter_attempts = iteration_breakdown[iteration]
                    detail_rows.append({
                        'Iteration': iteration,
                        'Attempts': len(iter_attempts),
                        'Passed': sum(1 for a in iter_attempts if a['pass']),
                        'Complete': sum(1 for a in iter_attempts if a['complete']),
                        'Pass Rate': f"{sum(1 for a in iter_attempts if a['pass']) / len(iter_attempts) * 100:.0f}%"
                    })

                detail_df = pd.DataFrame(detail_rows)
                st.dataframe(detail_df, width="stretch", hide_index=True)
                

def render_lemma_analysis(pipeline_data: Dict[str, Any], problems: List[Any]):
    """
    Render per-lemma analysis.

    Args:
        pipeline_data: Pipeline data dictionary
        problems: List of ProblemSummary objects
    """
    lemma_rows = []
    single_attempt_failures = []

    for lemma_key, lemma_data in sorted(pipeline_data['by_lemma'].items()):
        total_attempts = lemma_data['total_attempts']
        successful = lemma_data['successful']
        max_iteration = max([a['iteration'] for a in lemma_data['attempts']], default=0)

        lemma_rows.append({
            'Theorem': lemma_data['theorem'],
            'Lemma': lemma_data['lemma_id'],
            'Attempts': total_attempts,
            'Max Iteration': max_iteration,
            'Proven': '✅' if successful else '❌',
            'Status': 'Proven' if successful else 'Not Proven'
        })

        # Track single-attempt failures
        if total_attempts == 1 and not successful:
            single_attempt_failures.append({
                'theorem': lemma_data['theorem'],
                'lemma': lemma_key,
                'breakdown': lemma_data['breakdown']
            })

    df = pd.DataFrame(lemma_rows)
    st.dataframe(df, width="stretch", hide_index=True)

    # Highlight problematic cases
    if single_attempt_failures:
        st.warning(f"⚠️ Found {len(single_attempt_failures)} lemmas with single failed attempt")
        with st.expander("View single-attempt failures (likely problematic cases)"):
            for case in single_attempt_failures[:20]:  # Show first 20
                st.markdown(f"- **{case['theorem']}** → {case['lemma']}")
            if len(single_attempt_failures) > 20:
                st.markdown(f"... and {len(single_attempt_failures) - 20} more")


def render_lemma_validation_distribution(pipeline_data: Dict[str, Any]):
    """
    Render a bar chart showing distribution of lemmas by number of validated (yes) verdicts.

    Uses validation_results.json to count the number of "yes" verdicts per (problem_id, lemma_id).
    Shows distribution: 0, 1, 2, 3, 4+ validated samples.

    Args:
        pipeline_data: Pipeline data dictionary
    """
    try:
        import json
        from pathlib import Path
        from collections import defaultdict

        # Get run directory from streamlit session state
        run_dir = st.session_state.get('run_dir')
        round_num = st.session_state.get('round_num')

        if not run_dir or round_num is None:
            st.info("Run directory not available")
            return

        # Load validation results
        validation_file = Path(run_dir) / f"round{round_num}" / "formalizer" / "validation_results.json"
        if not validation_file.exists():
            st.info("No validation_results.json found in this run")
            return

        with open(validation_file) as f:
            validation_data = json.load(f)

        # Count yes verdicts per (problem_id, lemma_id)
        validation_counts = defaultdict(int)  # (problem_id, lemma_id) -> count of 'yes' verdicts

        for entry in validation_data:
            if entry.get('verdict') == 'yes':
                problem_id = entry.get('problem_id')
                lemma_id = entry.get('lemma_id')
                key = (problem_id, lemma_id)
                validation_counts[key] += 1

        # Count distribution: how many (problem_id, lemma_id) pairs have 0, 1, 2, 3, 4+ yes verdicts
        distribution = defaultdict(int)
        for count in validation_counts.values():
            bucket = min(count, 4)  # Group 4+ into a single bucket
            distribution[bucket] += 1

        # Add 0-count lemmas that appear in validation data but have no yes verdicts
        all_lemma_pairs = set()
        for entry in validation_data:
            problem_id = entry.get('problem_id')
            lemma_id = entry.get('lemma_id')
            all_lemma_pairs.add((problem_id, lemma_id))

        for pair in all_lemma_pairs:
            if pair not in validation_counts:
                distribution[0] += 1

        if distribution:
            import matplotlib.pyplot as plt

            # Prepare data for bar chart
            labels = ['0', '1', '2', '3', '4+']
            values = [distribution.get(i, 0) for i in range(5)]

            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("0 Yes Verdicts", values[0])
            with col2:
                st.metric("1 Yes Verdict", values[1])
            with col3:
                st.metric("2 Yes Verdicts", values[2])
            with col4:
                st.metric("3 Yes Verdicts", values[3])
            with col5:
                st.metric("4+ Yes Verdicts", values[4])

            # Create matplotlib bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(labels, values, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'])

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

            ax.set_xlabel('Number of "Yes" Verdicts per (Problem, Lemma)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Lemmas', fontsize=12, fontweight='bold')
            ax.set_title('Distribution of Lemmas by Validation Verdicts', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()
            st.pyplot(fig)

            # Summary statistics
            st.markdown("**Summary:**")
            total_lemmas = sum(values)
            zero_yes = values[0]
            at_least_one_yes = sum(values[1:])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Lemmas", total_lemmas)
            with col2:
                pct_zero = zero_yes / total_lemmas * 100 if total_lemmas > 0 else 0
                st.metric("0 Yes Verdicts", f"{zero_yes} ({pct_zero:.1f}%)")
            with col3:
                pct_at_least_one = at_least_one_yes / total_lemmas * 100 if total_lemmas > 0 else 0
                st.metric("≥1 Yes Verdict", f"{at_least_one_yes} ({pct_at_least_one:.1f}%)")
        else:
            st.info("No validation data found")

    except Exception as e:
        st.error(f"Error loading validation data: {str(e)}")


def render_breakdown_validation_distribution(pipeline_data: Dict[str, Any]):
    """
    Render a bar chart showing distribution of breakdowns by number of lemmas with "no" verdicts.

    Groups breakdowns by: how many of their lemmas had at least one "no" verdict.
    Shows distribution: 0, 1, 2, 3, 4+ lemmas with "no" verdicts.

    Args:
        pipeline_data: Pipeline data dictionary
    """
    try:
        import json
        from pathlib import Path
        from collections import defaultdict

        # Get run directory from streamlit session state
        run_dir = st.session_state.get('run_dir')
        round_num = st.session_state.get('round_num')

        if not run_dir or round_num is None:
            st.info("Run directory not available")
            return

        # Load validation results
        validation_file = Path(run_dir) / f"round{round_num}" / "formalizer" / "validation_results.json"
        if not validation_file.exists():
            st.info("No validation_results.json found in this run")
            return

        with open(validation_file) as f:
            validation_data = json.load(f)

        # Group by problem_id (breakdown) and count lemmas with at least one "no" verdict
        lemmas_with_no_per_breakdown = defaultdict(set)  # problem_id -> set of lemma_ids with "no" verdict

        for entry in validation_data:
            if entry.get('verdict') == 'no':
                problem_id = entry.get('problem_id')
                lemma_id = entry.get('lemma_id')
                lemmas_with_no_per_breakdown[problem_id].add(lemma_id)

        # Count how many lemmas with "no" verdict per breakdown
        breakdown_no_counts = {}
        for problem_id, lemma_ids in lemmas_with_no_per_breakdown.items():
            breakdown_no_counts[problem_id] = len(lemma_ids)

        # Also add breakdowns that have no lemmas with "no" verdict
        all_breakdowns = set()
        for entry in validation_data:
            all_breakdowns.add(entry.get('problem_id'))

        for breakdown in all_breakdowns:
            if breakdown not in breakdown_no_counts:
                breakdown_no_counts[breakdown] = 0

        # Count distribution: how many breakdowns have 0, 1, 2, 3, 4+ lemmas with "no" verdict
        distribution = defaultdict(int)
        for count in breakdown_no_counts.values():
            bucket = min(count, 4)  # Group 4+ into single bucket
            distribution[bucket] += 1

        if distribution:
            import matplotlib.pyplot as plt

            # Prepare data for bar chart
            labels = ['0', '1', '2', '3', '4+']
            values = [distribution.get(i, 0) for i in range(5)]

            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("0 Lemmas with No", values[0])
            with col2:
                st.metric("1 Lemma with No", values[1])
            with col3:
                st.metric("2 Lemmas with No", values[2])
            with col4:
                st.metric("3 Lemmas with No", values[3])
            with col5:
                st.metric("4+ Lemmas with No", values[4])

            # Create matplotlib bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(labels, values, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd'])

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

            ax.set_xlabel('Number of Lemmas with "No" Verdicts per Breakdown', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Breakdowns', fontsize=12, fontweight='bold')
            ax.set_title('Distribution of Breakdowns by Validation Issues', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()
            st.pyplot(fig)

            # Summary statistics
            st.markdown("**Summary:**")
            total_breakdowns = sum(values)
            zero_no = values[0]
            at_least_one_no = sum(values[1:])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Breakdowns", total_breakdowns)
            with col2:
                pct_zero = zero_no / total_breakdowns * 100 if total_breakdowns > 0 else 0
                st.metric("No Issues (0 no)", f"{zero_no} ({pct_zero:.1f}%)")
            with col3:
                pct_issues = at_least_one_no / total_breakdowns * 100 if total_breakdowns > 0 else 0
                st.metric("With Issues (≥1 no)", f"{at_least_one_no} ({pct_issues:.1f}%)")
        else:
            st.info("No validation data found")

    except Exception as e:
        st.error(f"Error loading validation data: {str(e)}")


def render_missing_lemmas_distribution(pipeline_data: Dict[str, Any]):
    """
    Render a bar chart showing distribution of missing lemmas per breakdown.

    For each breakdown:
    - Count total lemmas
    - Count lemmas with at least one "yes" verdict (ok=1)
    - Calculate missing = total - ok

    Groups breakdown by number of missing lemmas: 0, 1, 2, 3, 4+

    Args:
        pipeline_data: Pipeline data dictionary
    """
    try:
        import json
        from pathlib import Path
        from collections import defaultdict

        # Get run directory from streamlit session state
        run_dir = st.session_state.get('run_dir')
        round_num = st.session_state.get('round_num')

        if not run_dir or round_num is None:
            st.info("Run directory not available")
            return

        # Load validation results
        validation_file = Path(run_dir) / f"round{round_num}" / "formalizer" / "validation_results.json"
        if not validation_file.exists():
            st.info("No validation_results.json found in this run")
            return

        with open(validation_file) as f:
            validation_data = json.load(f)

        # Step 1: Group by (problem_id, lemma_id) and count yes verdicts
        lemma_yes_counts = defaultdict(int)  # (problem_id, lemma_id) -> count of yes verdicts

        for entry in validation_data:
            if entry.get('verdict') == 'yes':
                problem_id = entry.get('problem_id')
                lemma_id = entry.get('lemma_id')
                key = (problem_id, lemma_id)
                lemma_yes_counts[key] += 1

        # Step 2: Mark as ok=1 if count > 0, else ok=0
        lemma_status = {}  # (problem_id, lemma_id) -> ok (0 or 1)
        all_lemmas = set()  # all unique (problem_id, lemma_id) pairs

        for entry in validation_data:
            problem_id = entry.get('problem_id')
            lemma_id = entry.get('lemma_id')
            key = (problem_id, lemma_id)
            all_lemmas.add(key)
            if key not in lemma_status:
                lemma_status[key] = 1 if lemma_yes_counts[key] > 0 else 0

        # Step 3: Group by breakdown (problem_id) and calculate missing lemmas
        breakdown_stats = defaultdict(lambda: {'total': 0, 'ok': 0})  # problem_id -> {total, ok}

        for (problem_id, lemma_id), ok_status in lemma_status.items():
            breakdown_stats[problem_id]['total'] += 1
            breakdown_stats[problem_id]['ok'] += ok_status

        # Step 4: Calculate missing lemmas per breakdown
        missing_per_breakdown = {}
        for problem_id, stats in breakdown_stats.items():
            missing = stats['total'] - stats['ok']
            missing_per_breakdown[problem_id] = missing

        # Step 5: Create distribution (0, 1, 2, 3, 4+ missing lemmas)
        distribution = defaultdict(int)
        for missing_count in missing_per_breakdown.values():
            bucket = min(missing_count, 4)  # Group 4+ into single bucket
            distribution[bucket] += 1

        if distribution:
            import matplotlib.pyplot as plt

            # Prepare data for bar chart
            labels = ['0', '1', '2', '3', '4+']
            values = [distribution.get(i, 0) for i in range(5)]

            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("0 Missing", values[0])
            with col2:
                st.metric("1 Missing", values[1])
            with col3:
                st.metric("2 Missing", values[2])
            with col4:
                st.metric("3 Missing", values[3])
            with col5:
                st.metric("4+ Missing", values[4])

            # Create matplotlib bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(labels, values, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd'])

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

            ax.set_xlabel('Number of Missing Lemmas per Breakdown', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Breakdowns', fontsize=12, fontweight='bold')
            ax.set_title('Distribution of Missing Lemmas by Breakdown', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()
            st.pyplot(fig)

            # Summary statistics
            st.markdown("**Summary:**")
            total_breakdowns = sum(values)
            zero_missing = values[0]
            at_least_one_missing = sum(values[1:])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Breakdowns", total_breakdowns)
            with col2:
                pct_complete = zero_missing / total_breakdowns * 100 if total_breakdowns > 0 else 0
                st.metric("Complete (0 missing)", f"{zero_missing} ({pct_complete:.1f}%)")
            with col3:
                pct_incomplete = at_least_one_missing / total_breakdowns * 100 if total_breakdowns > 0 else 0
                st.metric("Incomplete (≥1 missing)", f"{at_least_one_missing} ({pct_incomplete:.1f}%)")
        else:
            st.info("No validation data found")

    except Exception as e:
        st.error(f"Error loading validation data: {str(e)}")


def render_lemma_usage_analysis(pipeline_data: Dict[str, Any], problems: List[Any]):
    """
    Render analysis of lemma usage in final proofs.

    Args:
        pipeline_data: Pipeline data dictionary
        problems: List of ProblemSummary objects
    """
    if not pipeline_data['lemma_usage']:
        st.info("No completed proofs with lemma usage data available")
        return

    # Build usage statistics
    usage_counts = defaultdict(int)
    proof_rows = []

    for proof_id, usage_info in pipeline_data['lemma_usage'].items():
        lemma_count = usage_info['count']
        num_proven = usage_info.get('num_proven', 0)
        usage_counts[lemma_count] += 1

        # Extract theorem name from proof_id
        parts = proof_id.split('_')
        theorem_name = '_'.join(parts[:3]) if len(parts) >= 3 else proof_id

        proof_rows.append({
            'Theorem': theorem_name,
            'Lemmas Used': lemma_count,
            'Lemmas Proven': num_proven,
            'Completion %': f"{(num_proven / lemma_count * 100):.0f}%" if lemma_count > 0 else "N/A",
            'Completed at Iteration': usage_info['iteration_completed']
        })

    # Statistics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        avg_lemmas = sum(d['count'] for d in pipeline_data['lemma_usage'].values()) / len(pipeline_data['lemma_usage'])
        st.metric("Avg Lemmas/Proof", f"{avg_lemmas:.2f}")

    with col2:
        max_lemmas = max(d['count'] for d in pipeline_data['lemma_usage'].values())
        st.metric("Max Lemmas in Proof", max_lemmas)

    with col3:
        min_lemmas = min(d['count'] for d in pipeline_data['lemma_usage'].values())
        st.metric("Min Lemmas in Proof", min_lemmas)

    with col4:
        direct_proofs = usage_counts.get(0, 0)
        if pipeline_data['lemma_usage']:
            pct = (direct_proofs / len(pipeline_data['lemma_usage'])) * 100
            st.metric("Direct Proofs (0 lemmas)", f"{direct_proofs} ({pct:.1f}%)")

    with col5:
        avg_proven = sum(d.get('num_proven', 0) for d in pipeline_data['lemma_usage'].values()) / len(pipeline_data['lemma_usage'])
        st.metric("Avg Lemmas Proven/Proof", f"{avg_proven:.2f}")

    st.markdown("---")

    # Chart: Distribution of lemma usage
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))

        lemma_counts = sorted(usage_counts.keys())
        counts = [usage_counts[lc] for lc in lemma_counts]

        ax.bar(lemma_counts, counts, color='steelblue', edgecolor='black')
        ax.set_xlabel('Number of Lemmas Used')
        ax.set_ylabel('Number of Proofs')
        ax.set_title('Distribution of Lemma Usage in Completed Proofs')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not render chart: {e}")

    st.markdown("---")

    # Table: Proofs with lemma details
    df_proofs = pd.DataFrame(proof_rows)
    st.subheader("Completed Proofs")
    st.dataframe(df_proofs, width="stretch", hide_index=True)


def render_component_cost_analysis(problems: List[Any]):
    """
    Render component-level token cost analysis showing which pipeline components consume the most tokens.

    Shows:
    - Global average token usage per component across all problems
    - Per-round breakdown of component costs
    - Comparison chart of component costs
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        st.warning("Plotly not available for component cost visualization")
        return

    # Aggregate component costs across all problems
    component_totals = {
        "breakdown": {"output_tokens": 0, "count": 0},
        "breakdown_parser": {"output_tokens": 0, "count": 0},
        "formalization": {"output_tokens": 0, "count": 0},
        "prover": {"output_tokens": 0, "count": 0}
    }

    round_component_costs = {}  # Dict[int, Dict[str, int]]

    for problem in problems:
        if not hasattr(problem, 'breakdowns'):
            continue

        for breakdown in problem.breakdowns.values():
            costs = breakdown.get_component_costs()

            # Track by round
            round_id = breakdown.round_id
            if round_id not in round_component_costs:
                round_component_costs[round_id] = {
                    "breakdown": 0,
                    "breakdown_parser": 0,
                    "formalization": 0,
                    "prover": 0
                }

            # Aggregate breakdown and parser from component_costs (these are correct)
            for component in ["breakdown", "breakdown_parser", "prover"]:
                if component in costs:
                    output_tokens = costs[component].get("output_tokens", 0)
                    component_totals[component]["output_tokens"] += output_tokens
                    component_totals[component]["count"] += 1
                    round_component_costs[round_id][component] += output_tokens

            # Calculate formalization costs correctly from Formalization.detailed_cost
            # (component_costs['formalization'] is incorrect - it uses breakdown's detailed_cost)
            formalization_total = 0
            if breakdown.parsed_breakdown:
                # Theorem formalizations
                theorem = breakdown.parsed_breakdown.theorem
                for formalization in theorem.formalizations:
                    if formalization.detailed_cost:
                        formalization_total += formalization.detailed_cost.get('output_tokens', 0)

                # Lemma formalizations
                for lemma in breakdown.parsed_breakdown.lemmas.values():
                    for formalization in lemma.formalizations:
                        if formalization.detailed_cost:
                            formalization_total += formalization.detailed_cost.get('output_tokens', 0)

            component_totals["formalization"]["output_tokens"] += formalization_total
            component_totals["formalization"]["count"] += 1
            round_component_costs[round_id]["formalization"] += formalization_total

    # Display metrics: Average output tokens per component
    col1, col2, col3, col4 = st.columns(4)

    components = ["breakdown", "breakdown_parser", "formalization", "prover"]
    nice_names = ["Breakdown", "Parser", "Formalization", "Prover"]
    cols = [col1, col2, col3, col4]

    for component, nice_name, col in zip(components, nice_names, cols):
        count = component_totals[component]["count"]
        total_tokens = component_totals[component]["output_tokens"]
        avg_tokens = total_tokens / count if count > 0 else 0

        with col:
            st.metric(f"⌛ {nice_name}", f"{avg_tokens:,.0f} tokens", f"avg ({count} breakdowns)")

    st.markdown("---")

    # Chart: Average output tokens per component (global)
    st.subheader("📊 Average Output Tokens by Component")

    avg_tokens_by_component = {}
    for component in components:
        count = component_totals[component]["count"]
        total = component_totals[component]["output_tokens"]
        avg_tokens_by_component[component] = total / count if count > 0 else 0

    # Create bar chart
    df_avg = pd.DataFrame({
        "Component": [nice_names[i] for i in range(len(components))],
        "Output Tokens": [avg_tokens_by_component[c] for c in components]
    })

    fig = px.bar(
        df_avg,
        x="Component",
        y="Output Tokens",
        title="Average Output Tokens per Component (All Breakdowns)",
        labels={"Output Tokens": "Average Output Tokens"},
        color="Component",
        height=400
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="component_avg_bar")

    st.markdown("---")

    # Chart: Component costs by round
    if round_component_costs:
        st.subheader("📈 Component Token Costs by Round")

        # Prepare data for multi-line chart
        rounds = sorted(round_component_costs.keys())
        df_rounds = pd.DataFrame({
            "Round": rounds,
            "Breakdown": [round_component_costs[r]["breakdown"] for r in rounds],
            "Parser": [round_component_costs[r]["breakdown_parser"] for r in rounds],
            "Formalization": [round_component_costs[r]["formalization"] for r in rounds],
            "Prover": [round_component_costs[r]["prover"] for r in rounds]
        })

        # Melt for plotly
        df_melted = df_rounds.melt(
            id_vars=["Round"],
            value_vars=["Breakdown", "Parser", "Formalization", "Prover"],
            var_name="Component",
            value_name="Output Tokens"
        )

        fig2 = px.line(
            df_melted,
            x="Round",
            y="Output Tokens",
            color="Component",
            title="Total Output Tokens per Component across Rounds",
            markers=True,
            height=400
        )
        fig2.update_xaxes(type="category")
        st.plotly_chart(fig2, use_container_width=True, key="component_by_round_line")

        # Show table of per-round costs
        st.subheader("📋 Per-Round Component Costs")
        st.dataframe(df_rounds.set_index("Round"), use_container_width=True)

    # Cost per action analysis
    st.markdown("---")
    render_cost_per_action_analysis(problems)


def render_cost_per_action_analysis(problems: List[Any]):
    """
    Render cost per action analysis showing average tokens per action type.

    Shows:
    - Average cost per breakdown action
    - Average cost per parser action
    - Average cost per formalization (per breakdown)
    - Average cost per prover action (single proof attempt)
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        st.warning("Plotly not available for cost per action visualization")
        return

    st.subheader("🎯 Cost Per Action Analysis")
    st.markdown("Average output token cost per individual action")

    # Collect costs per action type
    breakdown_costs = []  # From component_costs['breakdown']
    parser_costs = []  # From component_costs['breakdown_parser']
    formalization_costs_per_breakdown = []  # Sum of all formalization.detailed_cost per breakdown
    prover_attempt_costs = []  # One per proof attempt

    for problem in problems:
        if not hasattr(problem, 'breakdowns'):
            continue

        for breakdown in problem.breakdowns.values():
            # Get component costs (only use for breakdown and parser - formalization is wrong in component_costs)
            component_costs = breakdown.get_component_costs() if hasattr(breakdown, 'get_component_costs') else {}

            # Breakdown action cost (from component_costs['breakdown'])
            if component_costs and 'breakdown' in component_costs:
                output_tokens = component_costs['breakdown'].get('output_tokens', 0)
                if output_tokens > 0:
                    breakdown_costs.append(output_tokens)

            # Parser action cost (from component_costs['breakdown_parser'])
            if component_costs and 'breakdown_parser' in component_costs:
                output_tokens = component_costs['breakdown_parser'].get('output_tokens', 0)
                if output_tokens > 0:
                    parser_costs.append(output_tokens)

            if not breakdown.parsed_breakdown:
                continue

            # Sum all formalization costs for this breakdown (from actual Formalization.detailed_cost)
            formalization_total = 0

            # Theorem formalizations
            theorem = breakdown.parsed_breakdown.theorem
            for formalization in theorem.formalizations:
                if formalization.detailed_cost:
                    formalization_total += formalization.detailed_cost.get('output_tokens', 0)

                # Prover attempts
                for attempt in formalization.proof_attempts:
                    if attempt.detailed_cost:
                        output_tokens = attempt.detailed_cost.get('output_tokens', 0)
                        if output_tokens > 0:
                            prover_attempt_costs.append(output_tokens)

            # Lemma formalizations
            for lemma in breakdown.parsed_breakdown.lemmas.values():
                for formalization in lemma.formalizations:
                    if formalization.detailed_cost:
                        formalization_total += formalization.detailed_cost.get('output_tokens', 0)

                    # Prover attempts
                    for attempt in formalization.proof_attempts:
                        if attempt.detailed_cost:
                            output_tokens = attempt.detailed_cost.get('output_tokens', 0)
                            if output_tokens > 0:
                                prover_attempt_costs.append(output_tokens)

            if formalization_total > 0:
                formalization_costs_per_breakdown.append(formalization_total)

    # Calculate averages
    avg_breakdown = sum(breakdown_costs) / len(breakdown_costs) if breakdown_costs else 0
    avg_parser = sum(parser_costs) / len(parser_costs) if parser_costs else 0
    avg_formalization = sum(formalization_costs_per_breakdown) / len(formalization_costs_per_breakdown) if formalization_costs_per_breakdown else 0
    avg_prover = sum(prover_attempt_costs) / len(prover_attempt_costs) if prover_attempt_costs else 0

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📋 Breakdown", f"{avg_breakdown:,.0f} tokens", f"avg ({len(breakdown_costs):,} actions)")
    with col2:
        st.metric("🔍 Parser", f"{avg_parser:,.0f} tokens", f"avg ({len(parser_costs):,} actions)")
    with col3:
        st.metric("📝 Formalization", f"{avg_formalization:,.0f} tokens", f"avg ({len(formalization_costs_per_breakdown):,} breakdowns)")
    with col4:
        st.metric("🎯 Prover", f"{avg_prover:,.0f} tokens", f"avg ({len(prover_attempt_costs):,} attempts)")

    # Bar chart comparing action costs
    st.markdown("---")
    st.subheader("📊 Average Cost per Action Type")

    action_data = pd.DataFrame({
        "Action": ["Breakdown", "Parser", "Formalization", "Prover"],
        "Avg Output Tokens": [avg_breakdown, avg_parser, avg_formalization, avg_prover],
        "Count": [len(breakdown_costs), len(parser_costs), len(formalization_costs_per_breakdown), len(prover_attempt_costs)]
    })

    fig = px.bar(
        action_data,
        x="Action",
        y="Avg Output Tokens",
        text="Count",
        title="Average Output Tokens per Action Type",
        color="Action",
        height=400
    )
    fig.update_traces(texttemplate='n=%{text}', textposition='outside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="action_cost_comparison")

    # Distribution of prover attempt costs
    if prover_attempt_costs:
        st.markdown("---")
        st.subheader("📈 Distribution of Prover Attempt Costs")

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=prover_attempt_costs,
            name="Prover Attempts",
            nbinsx=50
        ))
        fig2.update_layout(
            title="Distribution of Output Tokens per Prover Attempt",
            xaxis_title="Output Tokens",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True, key="prover_cost_histogram")


def render_error_correctability_analysis(session: 'Session') -> None:
    """
    Analyze the correctability of each error type.

    For each error type found in round 0, calculate:
    - How many proofs had that error in round 0
    - How many of those were corrected (became PASS) in round 1
    - Correctability rate = (corrected / total) * 100%

    Shows a bar chart and table with error types and their correction rates.
    """
    st.header("🔧 Error Correctability Analysis")
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
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[go.Bar(
            x=error_types_list,
            y=correctability_rates,
            marker=dict(
                color=correctability_rates,
                colorscale='RdYlGn',
                cmin=0,
                cmax=100,
                showscale=True,
                colorbar=dict(title="Correctability %")
            ),
            text=[f"{rate:.1f}%" for rate in correctability_rates],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Correctability: %{y:.1f}%<extra></extra>'
        )]
    )

    fig.update_layout(
        title="Error Correctability Rate by Error Type",
        xaxis_title="Error Type",
        yaxis_title="Correctability Rate (%)",
        yaxis=dict(range=[0, 105]),
        height=500,
        showlegend=False
    )
    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, use_container_width=True)

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
