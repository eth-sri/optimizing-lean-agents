"""Theorem proof code and attempts
"""
import streamlit as st
from typing import List, Dict, Any, Optional
from utils import get_status_emoji

# Import utilities from other components
from .proof_status_component import extract_think_content, count_axioms_in_code, render_proof_attempt_expandables
from .formalization_component import render_compilation_result
from seed_data_models import Breakdown

# Try to import id_utils if available
try:
    from id_utils import get_lemma_id, get_lemma_component
except ImportError:
    import re

    def get_lemma_id(problem_id: str) -> str:
        """Fallback implementation of get_lemma_id - matches id_utils.py."""
        canonical = str(problem_id)

        # Remove correction suffix (_corr<N>)
        if "_corr" in canonical:
            canonical = canonical.split("_corr")[0]

        # Remove proof attempt suffix (_p<N>)
        if "_p" in canonical:
            parts = canonical.rsplit("_p", 1)
            if len(parts) == 2 and parts[1].isdigit():
                canonical = parts[0]

        # Remove proof retry suffix (_r<N>) that comes before lemma/theorem
        if "_r" in canonical:
            if "_l" in canonical or "_theorem" in canonical:
                lemma_match = None
                if "_l" in canonical:
                    lemma_match = canonical.rfind("_l")
                if "_theorem" in canonical:
                    theorem_match = canonical.rfind("_theorem")
                    if lemma_match is None or theorem_match > lemma_match:
                        lemma_match = theorem_match

                if lemma_match is not None:
                    after_lemma = canonical[lemma_match:]
                    r_match = re.search(r'_r(\d+)', after_lemma)
                    if r_match:
                        end_pos = lemma_match + r_match.start()
                        canonical = canonical[:end_pos] + canonical[lemma_match + r_match.end():]

        # Remove formalization sample suffixes
        if "_sample_" in canonical:
            canonical = re.sub(r'_sample_\d+', '', canonical)

        # Remove breakdown sampling _s<N> suffix that appears BEFORE lemma/theorem markers
        canonical = re.sub(r'_s\d+(?=_[lt])', '', canonical)

        return canonical

    def get_lemma_component(problem_id: str):
        """Fallback implementation of get_lemma_component."""
        # Use get_lemma_id to normalize first
        normalized = get_lemma_id(problem_id)

        if "_theorem" in normalized:
            return "theorem"
        elif "_l" in normalized:
            parts = normalized.rsplit("_l", 1)
            if len(parts) == 2 and parts[1].split("_")[0].isdigit():
                return f"l{parts[1].split('_')[0]}"
        return None



def render_lemma_attempts_from_parsed_breakdown(breakdown: Breakdown, lemma_id: int):
    """
    Render lemma proof attempts from the new parsed_breakdown data model.

    This displays proof attempts for a specific lemma from the OOP data structure.
    Used for minified data which doesn't have the legacy lemma_prover_results structure.

    Args:
        breakdown: Breakdown object with parsed_breakdown
        lemma_id: The lemma ID to display attempts for
    """
    if not hasattr(breakdown, 'parsed_breakdown') or not breakdown.parsed_breakdown:
        return

    pb = breakdown.parsed_breakdown
    if lemma_id not in pb.lemmas:
        return

    lemma = pb.lemmas[lemma_id]

    if not hasattr(lemma, 'formalizations') or not lemma.formalizations:
        return

    st.markdown(f"## Lemma {lemma_id} Proof Attempts")

    # Iterate through each formalization and its attempts
    for form_idx, formalization in enumerate(lemma.formalizations):
        if not formalization.proof_attempts:
            continue

        # Create a section for this formalization
        form_label = f"Formalization {form_idx}"
        if formalization.is_selected:
            form_label += " ✅"

        with st.expander(f"{form_label} ({len(formalization.proof_attempts)} attempts)", expanded=False):
            # Show formalization status
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Pass:** {'✅' if formalization.compilation_pass else '❌'}")
            with col2:
                st.markdown(f"**Complete:** {'✅' if formalization.validation_pass else '❌'}")
            with col3:
                st.markdown(f"**Proven:** {'✅' if formalization.is_proven() else '⏳'}")

            st.divider()

            # Group attempts by correction round
            attempts_by_round = {}
            for attempt in formalization.proof_attempts:
                round_id = attempt.correction_round_id
                if round_id not in attempts_by_round:
                    attempts_by_round[round_id] = []
                attempts_by_round[round_id].append(attempt)

            # Display grouped by correction round
            # Get the best attempt for this formalization to mark it visually
            best_attempt = formalization.get_best_attempt(lemmas_dict=pb.lemmas)

            for round_id in sorted(attempts_by_round.keys()):
                round_attempts = attempts_by_round[round_id]
                passing_count = sum(1 for a in round_attempts if a.is_passing())
                round_label = f"Round {round_id}" if round_id > 0 else "Initial"
                round_header = f"{round_label}: {len(round_attempts)} attempts ({passing_count} passing)"

                with st.expander(round_header, expanded=passing_count > 0):
                    # Display each attempt in this round
                    for local_idx, attempt in enumerate(round_attempts):
                        status_emoji = "✅" if attempt.is_passing() else "❌"

                        # Get used lemmas from attempt and count declared lemmas
                        used_lemma_ids = attempt.get_used_lemmas(lemmas_dict=pb.lemmas, recursive=False) if hasattr(attempt, 'get_used_lemmas') else set()
                        total_lemmas = len(pb.lemmas) if pb.lemmas else 0
                        used_count = len(used_lemma_ids)

                        # Mark the best attempt with a star
                        is_best = attempt is best_attempt
                        best_marker = " ⭐" if is_best else ""

                        lemma_info = f" | Lemmas: {used_count}/{total_lemmas}" if total_lemmas > 0 else ""
                        initial_idx_info = f" [orig: {attempt.initial_attempt_index}]" if hasattr(attempt, 'initial_attempt_index') and attempt.initial_attempt_index is not None else ""
                        attempt_label = f"{status_emoji} Attempt {local_idx}{initial_idx_info}{lemma_info}{best_marker} ({', '.join([str(x) for x in used_lemma_ids])})"

                        with st.expander(attempt_label, expanded=False):
                            # Show basic status
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Compiled:** {'✅' if attempt.compilation_result.passed else '❌'}")
                            with col2:
                                st.markdown(f"**Complete:** {'✅' if attempt.compilation_result.complete else '❌'}")

                            # Show reasoning and compilation summaries using shared function
                            # Convert ProofAttempt to dict for the shared function
                            attempt_dict = {
                                'code': attempt.code,
                                'model_reasoning': attempt.model_reasoning,
                                'model_input': attempt.model_input,
                                'reasoning_summary': attempt.reasoning_summary,
                                'compilation_summary': attempt.compilation_summary,
                                'detailed_cost': attempt.detailed_cost,
                                'model_config_path': attempt.model_config_path,
                            }
                            comp_result_dict = {
                                'pass': attempt.compilation_result.passed,
                                'complete': attempt.compilation_result.complete,
                                'errors': attempt.compilation_result.errors,
                                'warnings': attempt.compilation_result.warnings,
                            }

                            render_proof_attempt_expandables(attempt_dict, comp_result_dict)


def render_theorem_attempts_from_parsed_breakdown(breakdown: Breakdown):
    """
    Render theorem proof attempts from the new parsed_breakdown data model.

    This displays proof attempts from the OOP data structure (formalization objects
    with their proof_attempts). Used for minified data which doesn't have the
    legacy theorem_prover_results structure.

    Args:
        breakdown: Breakdown object with parsed_breakdown
    """
    if not hasattr(breakdown, 'parsed_breakdown') or not breakdown.parsed_breakdown:
        return

    pb = breakdown.parsed_breakdown
    theorem = pb.theorem

    if not hasattr(theorem, 'formalizations') or not theorem.formalizations:
        return

    st.markdown("## Proof Attempts")

    # Debug: Log formalizations and their proof attempts
    print(f"[DEBUG] Theorem has {len(theorem.formalizations)} formalizations")
    for form_idx, formalization in enumerate(theorem.formalizations):
        print(f"[DEBUG] Formalization {form_idx}: id={formalization.id}, proof_attempts={len(formalization.proof_attempts) if formalization.proof_attempts else 0}")

    # Iterate through each formalization and its attempts
    for form_idx, formalization in enumerate(theorem.formalizations):
        if not formalization.proof_attempts:
            print(f"[DEBUG] Skipping formalization {form_idx} - no proof attempts")
            continue

        # Create a section for this formalization
        form_label = f"Formalization {form_idx}"
        if formalization.is_selected:
            form_label += " ✅"

        with st.expander(f"{form_label} ({len(formalization.proof_attempts)} attempts)", expanded=False):
            # Show formalization status
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Pass:** {'✅' if formalization.compilation_pass else '❌'}")
            with col2:
                st.markdown(f"**Complete:** {'✅' if formalization.validation_pass else '❌'}")
            with col3:
                st.markdown(f"**Proven:** {'✅' if formalization.is_proven() else '⏳'}")

            st.divider()

            # Group attempts by correction round
            attempts_by_round = {}
            for attempt in formalization.proof_attempts:
                round_id = attempt.correction_round_id
                if round_id not in attempts_by_round:
                    attempts_by_round[round_id] = []
                attempts_by_round[round_id].append(attempt)

            # Display grouped by correction round
            # Get the best attempt for this formalization to mark it visually
            best_attempt = formalization.get_best_attempt(lemmas_dict=pb.lemmas)

            for round_id in sorted(attempts_by_round.keys()):
                round_attempts = attempts_by_round[round_id]
                passing_count = sum(1 for a in round_attempts if a.is_passing())
                round_label = f"Round {round_id}" if round_id > 0 else "Initial"
                round_header = f"{round_label}: {len(round_attempts)} attempts ({passing_count} passing)"

                with st.expander(round_header, expanded=passing_count > 0):
                    # Display each attempt in this round
                    for local_idx, attempt in enumerate(round_attempts):
                        status_emoji = "✅" if attempt.is_passing() else "❌"

                        # Get used lemmas from attempt and count declared lemmas
                        used_lemma_ids = attempt.get_used_lemmas(lemmas_dict=pb.lemmas, recursive=False) if hasattr(attempt, 'get_used_lemmas') else set()
                        total_lemmas = len(pb.lemmas) if pb.lemmas else 0
                        used_count = len(used_lemma_ids)

                        # Mark the best attempt with a star
                        is_best = attempt is best_attempt
                        best_marker = " ⭐" if is_best else ""

                        lemma_info = f" | Lemmas: {used_count}/{total_lemmas}" if total_lemmas > 0 else ""
                        initial_idx_info = f" [orig: {attempt.initial_attempt_index}]" if hasattr(attempt, 'initial_attempt_index') and attempt.initial_attempt_index is not None else ""
                        attempt_label = f"{status_emoji} Attempt {local_idx}{initial_idx_info}{lemma_info}{best_marker} ({', '.join([str(x) for x in used_lemma_ids])})"

                        with st.expander(attempt_label, expanded=False):
                            # Show basic status
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Compiled:** {'✅' if attempt.compilation_result.passed else '❌'}")
                            with col2:
                                st.markdown(f"**Complete:** {'✅' if attempt.compilation_result.complete else '❌'}")

                            # Show reasoning and compilation summaries using shared function
                            # Convert ProofAttempt to dict for the shared function
                            attempt_dict = {
                                'code': attempt.code,
                                'model_reasoning': attempt.model_reasoning,
                                'model_input': attempt.model_input,
                                'reasoning_summary': attempt.reasoning_summary,
                                'compilation_summary': attempt.compilation_summary,
                                'detailed_cost': attempt.detailed_cost,
                                'model_config_path': attempt.model_config_path,
                            }
                            comp_result_dict = {
                                'pass': attempt.compilation_result.passed,
                                'complete': attempt.compilation_result.complete,
                                'errors': attempt.compilation_result.errors,
                                'warnings': attempt.compilation_result.warnings,
                            }

                            render_proof_attempt_expandables(attempt_dict, comp_result_dict)


def render_theorem_prover_code_results(breakdown: Breakdown):
    """
    Render theorem prover results from code compilation attempts.

    Shows all proof attempts for this specific breakdown across correction rounds.

    Args:
        breakdown: Breakdown object with theorem_prover_results
    """
    results = breakdown.theorem_prover_results

    if not results or 'attempts' not in results:
        return

    attempts = results['attempts']
    if not attempts:
        return

    # All attempts in results are already for this specific breakdown
    # (they were filtered during loading by matching origin_problem_id + breakdown_id)
    attempts_sorted = sorted(
        attempts,
        key=lambda x: (x.get('correction_round', 0), x.get('data', {}).get('name', ''))
    )

    # Check if any attempt passed AND is complete (no sorries)
    any_passed = any(
        attempt.get('data', {}).get('compilation_result', {}).get('pass', False) and
        attempt.get('data', {}).get('compilation_result', {}).get('complete', False)
        for attempt in attempts_sorted
    )

    # Display overall status
    st.markdown("### 🤖 Theorem Prover Results")

    col1, col2 = st.columns(2)
    with col1:
        status_emoji = "✅" if any_passed else "❌"
        status_text = "Proven" if any_passed else "Not Proven"
        st.metric("Status", f"{status_emoji} {status_text}")
    with col2:
        st.metric("Total Attempts", len(attempts_sorted))

    # Display attempts in dropdown
    with st.expander("📋 View All Proof Attempts"):
        for round_num in sorted(set(a.get('correction_round', 0) for a in attempts_sorted)):
            round_attempts = [a for a in attempts_sorted if a.get('correction_round') == round_num]

            # Count passed/failed
            passed_count = sum(
                1 for a in round_attempts
                if a.get('data', {}).get('compilation_result', {}).get('pass', False)
            )
            failed_count = len(round_attempts) - passed_count

            round_title = f"Round {round_num}" if round_num == 0 else f"Correction Round {round_num}"
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**{round_title}**")
            with col2:
                st.markdown(f"✅ {passed_count}/{len(round_attempts)} passed")
            with col3:
                st.markdown(f"❌ {failed_count}/{len(round_attempts)} failed")

            st.markdown("---")

            # Display individual samples
            for idx, attempt in enumerate(round_attempts):
                data = attempt.get('data', {})
                comp_result = data.get('compilation_result', {})
                passed = comp_result.get('pass', False)
                complete = comp_result.get('complete', False)

                # Status is ✅ only if BOTH pass AND complete (no sorry)
                status_emoji = "✅" if (passed and complete) else "❌"
                sample_name = data.get('name', f'Sample {idx}')

                # Count axioms used in this proof (try multiple code field names)
                code = data.get('full_code', '') or data.get('code', '') or data.get('lean4_code', '')
                defined_axioms, used_axioms, used_axiom_names = count_axioms_in_code(code)

                expander_title = f"{status_emoji} {sample_name} ({used_axioms}/{defined_axioms} axioms used)"

                with st.expander(expander_title):
                    # Show compilation status
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Compiled:** {'✅ Yes' if passed else '❌ No'}")
                    with col2:
                        complete = comp_result.get('complete', False)
                        st.markdown(f"**Proof Complete:** {'✅ Yes' if complete else '❌ No (has sorry)'}")

                    # Show used axioms
                    if used_axiom_names:
                        axioms_list = ", ".join(sorted(used_axiom_names))
                        st.markdown(f"**Used Axioms:** {axioms_list}")

                    # Show prompt, reasoning summary, and compilation summary using shared function
                    render_proof_attempt_expandables(data, comp_result)

                    # Show model reasoning if available
                    model_output = data.get('model_output')
                    if model_output:
                        with st.expander("💭 Model Reasoning"):
                            st.markdown(model_output)

                    # Show code (expandable)
                    if code:
                        with st.expander("📄 Code"):
                            st.code(code, language="lean")

                    # Show errors in expandable
                    errors = comp_result.get('errors', [])
                    if errors:
                        with st.expander(f"❌ Compilation Errors ({len(errors)})"):
                            for error in errors:
                                if isinstance(error, dict):
                                    st.text(error.get('data', str(error)))
                                else:
                                    st.text(str(error))


def render_lemma_prover_code_results(breakdown: Breakdown):
    """
    Render lemma prover results from code compilation attempts.

    Shows all proof attempts for each lemma across correction rounds.

    Args:
        breakdown: Breakdown object with lemma_prover_results
    """
    results = breakdown.lemma_prover_results

    if not results or 'lemmas' not in results:
        return

    lemmas_dict = results['lemmas']
    if not lemmas_dict:
        return

    st.markdown("### 🤖 Lemma Prover Results")

    # Flatten all attempts from all lemmas
    all_attempts = []
    for lemma_id, lemma_results in lemmas_dict.items():
        for attempt in lemma_results.get('attempts', []):
            attempt['lemma_id'] = lemma_id
            all_attempts.append(attempt)

    if not all_attempts:
        return

    attempts_sorted = sorted(
        all_attempts,
        key=lambda x: (x.get('correction_round', 0), x.get('data', {}).get('name', ''))
    )

    # Check if any attempt passed AND is complete (no sorries)
    any_passed = any(
        attempt.get('data', {}).get('compilation_result', {}).get('pass', False) and
        attempt.get('data', {}).get('compilation_result', {}).get('complete', False)
        for attempt in attempts_sorted
    )

    # Display overall status
    col1, col2 = st.columns(2)
    with col1:
        status_emoji = "✅" if any_passed else "❌"
        status_text = "Proven" if any_passed else "Not Proven"
        st.metric("Status", f"{status_emoji} {status_text}")
    with col2:
        st.metric("Total Attempts", len(attempts_sorted))

    # Display attempts in dropdown
    with st.expander("📋 View All Proof Attempts"):
        for round_num in sorted(set(a.get('correction_round', 0) for a in attempts_sorted)):
            round_attempts = [a for a in attempts_sorted if a.get('correction_round') == round_num]

            # Count passed/failed
            passed_count = sum(
                1 for a in round_attempts
                if a.get('data', {}).get('compilation_result', {}).get('pass', False)
            )
            failed_count = len(round_attempts) - passed_count

            round_title = f"Round {round_num}" if round_num == 0 else f"Correction Round {round_num}"
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**{round_title}**")
            with col2:
                st.markdown(f"✅ {passed_count}/{len(round_attempts)} passed")
            with col3:
                st.markdown(f"❌ {failed_count}/{len(round_attempts)} failed")

            st.markdown("---")

            # Display individual samples
            for idx, attempt in enumerate(round_attempts):
                data = attempt.get('data', {})
                comp_result = data.get('compilation_result', {})
                passed = comp_result.get('pass', False)
                complete = comp_result.get('complete', False)

                # Status is ✅ only if BOTH pass AND complete (no sorry)
                status_emoji = "✅" if (passed and complete) else "❌"
                sample_name = data.get('name', f'Sample {idx}')
                lemma_id = attempt.get('lemma_id', '?')

                # Count axioms used in this proof (try multiple code field names)
                code = data.get('full_code', '') or data.get('code', '') or data.get('lean4_code', '')
                defined_axioms, used_axioms, used_axiom_names = count_axioms_in_code(code)

                expander_title = f"{status_emoji} Lemma {lemma_id} - {sample_name} ({used_axioms}/{defined_axioms} axioms used)"

                with st.expander(expander_title):
                    # Show compilation status
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Compiled:** {'✅ Yes' if passed else '❌ No'}")
                    with col2:
                        complete = comp_result.get('complete', False)
                        st.markdown(f"**Proof Complete:** {'✅ Yes' if complete else '❌ No (has sorry)'}")

                    # Show used axioms
                    if used_axiom_names:
                        axioms_list = ", ".join(sorted(used_axiom_names))
                        st.markdown(f"**Used Axioms:** {axioms_list}")

                    # Show prompt, reasoning summary, and compilation summary using shared function
                    render_proof_attempt_expandables(data, comp_result)

                    # Show model reasoning if available
                    model_output = data.get('model_output')
                    if model_output:
                        with st.expander("💭 Model Reasoning"):
                            st.markdown(model_output)

                    # Show code (expandable)
                    if code:
                        with st.expander("📄 Code"):
                            st.code(code, language="lean")

                    # Show errors
                    errors = comp_result.get('errors', [])
                    if errors:
                        st.error("**Compilation Errors:**")
                        for error in errors:
                            if isinstance(error, dict):
                                st.text(error.get('data', str(error)))
                            else:
                                st.text(str(error))


def render_theorem_prover_attempts(
    breakdown: Breakdown,
    analysis: Optional[Dict[str, Any]]
):
    """
    Display theorem prover attempts and results.

    Args:
        breakdown: Breakdown object
        analysis: ProblemAnalysis object
    """
    st.markdown("### 🤖 Theorem Prover Attempts")

    # Find theorem results for this breakdown
    theorem_results = [
        bd for bd in analysis.theorem_prover_stats.get('breakdown_results', [])
        if bd.get('breakdown_id') == breakdown.problem_id
    ]

    if not theorem_results:
        st.info("No theorem prover attempts recorded for this breakdown.")
        return

    result = theorem_results[0]

    # Display overall result
    col1, col2 = st.columns(2)
    with col1:
        proven_status = result.get('proven', False)
        st.metric(
            "Proven",
            get_status_emoji(proven_status),
            help="Whether the theorem was successfully proven"
        )
    with col2:
        attempts = result.get('attempts', 0)
        st.metric("Attempts", attempts, help="Number of proof attempts made")

    # Display error if present
    if 'error' in result and result['error']:
        st.error(f"**Error:** {result['error']}")

    # Display additional details
    if 'details' in result:
        st.markdown("**Proof Details:**")
        st.json(result['details'])



def render_theorem_formalization_detail(formalization, theorem_statement: str):
    """
    Render a single theorem formalization with its details.

    Args:
        formalization: A Formalization object
        theorem_statement: The theorem statement for display
    """
    # Show formalization reasoning if available
    formalization_reasoning = formalization.formalization_reasoning
    if formalization_reasoning:
        with st.expander("💭 Formalization Reasoning"):
            st.markdown(formalization_reasoning)

    # Show thinking process from formal_statement_raw if available
    formal_statement_raw = formalization.formal_statement_raw if hasattr(formalization, 'formal_statement_raw') else None
    if not formal_statement_raw:
        # Try to extract from properties
        formal_statement_raw = getattr(formalization, 'formal_statement_raw', None)

    if formal_statement_raw:
        thinking = extract_think_content(formal_statement_raw)
        if thinking:
            with st.expander("💭 Reasoning Process"):
                st.markdown(thinking)

    # Show formal statement
    formal_statement = formalization.formal_statement
    if formal_statement:
        st.markdown("**Formal Statement (Lean 4):**")
        cleaned_code = formal_statement.strip()
        st.code(cleaned_code, language="lean")

    # Show compilation and validation status
    st.markdown("**Formalization Status:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        status = "✅ Pass" if formalization.compilation_pass else "❌ Failed"
        st.metric("Compilation", status)

    with col2:
        status = "✅ Pass" if formalization.validation_pass else "❌ Failed"
        st.metric("Validation", status)

    with col3:
        status = "✅ Proven" if formalization.is_proven() else "⏳ Not Proven"
        st.metric("Proven", status)

    # Show detailed compilation results if available
    compilation_result = formalization.compilation_result
    if compilation_result:
        with st.expander("📋 Compilation Output", expanded=False):
            if isinstance(compilation_result, dict):
                st.json(compilation_result)
            else:
                st.code(str(compilation_result), language="text")

    # Show compilation errors if available
    compilation_errors = formalization.compilation_errors
    if compilation_errors:
        with st.expander("❌ Compilation Errors", expanded=True):
            if isinstance(compilation_errors, list):
                for error in compilation_errors:
                    st.code(str(error), language="text")
            else:
                st.code(str(compilation_errors), language="text")


def render_theorem_formalization(
    breakdown: Optional[Breakdown],
    formalized_data: Optional[Dict[str, Any]]
):
    """
    Render formalization details for the theorem including Lean code and compilation results.
    Supports multiple formalizations with tabs to switch between them.

    Args:
        breakdown: Breakdown object
        formalized_data: The formalized breakdown data containing theorem (OOP ParsedBreakdown or dict)
    """
    if not formalized_data:
        return

    # Handle both OOP ParsedBreakdown and legacy dict structures
    theorem_obj = None

    if hasattr(formalized_data, 'theorem'):
        # OOP ParsedBreakdown object
        theorem_obj = formalized_data.theorem

    if not theorem_obj:
        return

    # Check if we have multiple formalizations (new structure)
    if hasattr(theorem_obj, 'formalizations') and len(theorem_obj.formalizations) > 1:
        # New structure with multiple formalizations - use tabs
        # Create tabs for each formalization with labels like "0", "1", "2", "3"
        # Add ✅ emoji if formalization is selected (in selected_formalizations)
        tab_labels = [
            f"{'✅ ' if form.is_selected else ''}{form.id}" if form.id is not None else f"{'✅ ' if form.is_selected else ''}{i}"
            for i, form in enumerate(theorem_obj.formalizations)
        ]
        tabs = st.tabs(tab_labels)

        for tab, form, idx in zip(tabs, theorem_obj.formalizations, range(len(theorem_obj.formalizations))):
            with tab:
                # Add status badge at top of tab
                status_cols = st.columns(3)
                with status_cols[0]:
                    comp_emoji = "✅" if form.compilation_pass else "❌"
                    st.markdown(f"**Compilation:** {comp_emoji}")
                with status_cols[1]:
                    val_emoji = "✅" if form.validation_pass else "❌"
                    st.markdown(f"**Validation:** {val_emoji}")
                with status_cols[2]:
                    proven_emoji = "✅" if form.is_proven() else "⏳"
                    st.markdown(f"**Proven:** {proven_emoji}")

                st.markdown("---")

                # Render the detailed formalization
                render_theorem_formalization_detail(form, theorem_obj.statement)

    elif hasattr(theorem_obj, 'formalizations') and len(theorem_obj.formalizations) == 1:
        # Single formalization - show directly
        formalization = theorem_obj.formalizations[0]
        render_theorem_formalization_detail(formalization, theorem_obj.statement)

    elif hasattr(theorem_obj, 'get_best_formalization'):
        # Fallback: old-style single best formalization
        best_formalization = theorem_obj.get_best_formalization()
        if best_formalization:
            render_theorem_formalization_detail(best_formalization, theorem_obj.statement)



