"""Lemma prover results
"""
import streamlit as st

# Import utilities from proof_status_component
from .proof_status_component import count_axioms_in_code, render_proof_attempt_expandables
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



def render_lemma_prover_results(breakdown: Breakdown, lemma_id):
    """
    Render lemma prover results from code compilation attempts.

    Shows all proof attempts for this specific lemma in the current breakdown across correction rounds.

    Args:
        breakdown: Breakdown object with lemma_prover_results
        lemma_id: The lemma ID (can be full ID like "imo_2019_p1_r0_b0_l1" or just a number)
    """
    if not breakdown.lemma_prover_results:
        st.info("No lemma prover results available for this breakdown.")
        return

    all_attempts = breakdown.lemma_prover_results.get('all_attempts', [])
    if not all_attempts:
        st.info("No lemma prover attempts found.")
        return

    # Get the breakdown_id that was stored during data loading
    # This ensures we only show results for the current breakdown being viewed
    breakdown_id_from_attempt = breakdown.lemma_prover_results.get('breakdown_id')

    if not breakdown_id_from_attempt:
        st.warning("Could not extract breakdown ID from lemma prover results.")
        return

    # Extract the lemma component using id_utils (returns "l1", "l2", "theorem", etc.)
    lemma_component = get_lemma_component(str(lemma_id))

    # If we can't extract a component, it might be just a number, so construct it
    if not lemma_component:
        # Try to construct the full ID if lemma_id is just a number
        try:
            # If it's just a number, create the full lemma ID
            if isinstance(lemma_id, int) or (isinstance(lemma_id, str) and lemma_id.isdigit()):
                full_id = f"{breakdown_id_from_attempt}_l{lemma_id}"
                lemma_component = get_lemma_component(full_id)
        except:
            pass

    if not lemma_component:
        st.warning(f"Could not extract lemma component from lemma ID: {lemma_id}")
        return

    # Build the target lemma ID to match against
    target_lemma_id = f"{breakdown_id_from_attempt}_{lemma_component}"

    # Filter to only include attempts for this specific lemma in this specific breakdown
    attempts_for_lemma = []
    for a in all_attempts:
        data = a.get('data', {})
        metadata = data.get('metadata', {})
        attempt_lemma_id = metadata.get('lemma_id')

        # Try metadata first, then fallback to problem_id parsing
        if attempt_lemma_id is None:
            problem_id = data.get('problem_id', '') or data.get('uid', '')
            if problem_id:
                extracted_id = get_lemma_id(problem_id)
                if extracted_id == target_lemma_id:
                    attempts_for_lemma.append(a)
        else:
            # Extract lemma number from metadata
            try:
                lemma_num = int(attempt_lemma_id)
                # Check if this matches our target (target is like "amc12_2001_p5_r0_b0_l4")
                if f"_l{lemma_num}" in target_lemma_id:
                    attempts_for_lemma.append(a)
            except (ValueError, TypeError):
                pass

    if not attempts_for_lemma:
        st.info(f"No lemma prover attempts found for {target_lemma_id}. (Looking in {len(all_attempts)} total attempts)")
        return

    # Sort attempts by correction_round, then alphabetically by name
    attempts_sorted = sorted(
        attempts_for_lemma,
        key=lambda x: (x.get('correction_round', 0), x.get('data', {}).get('name', ''))
    )

    # Check if any attempt passed
    any_passed = any(
        attempt.get('data', {}).get('compilation_result', {}).get('pass', False)
        for attempt in attempts_sorted
    )

    # Display overall status
    st.markdown("### 🤖 Lemma Prover Results")

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

                # Extract axiom usage from code
                code = data.get('full_code', '') or data.get('code', '') or data.get('lean4_code', '')
                axiom_info_str = ""
                if code and passed and complete:
                    try:
                        defined_axioms, used_axioms, used_axiom_names = count_axioms_in_code(code)
                        if used_axioms > 0:
                            axiom_info_str = f" | {used_axioms}/{defined_axioms} axioms"
                    except:
                        pass

                expander_title = f"{status_emoji} {sample_name}{axiom_info_str}"

                with st.expander(expander_title):
                    # Show compilation status
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Compiled:** {'✅ Yes' if passed else '❌ No'}")
                    with col2:
                        st.markdown(f"**Proof Complete:** {'✅ Yes' if complete else '❌ No (has sorry)'}")

                    # Show axiom usage if available
                    if code and passed and complete:
                        try:
                            defined_axioms, used_axioms, used_axiom_names = count_axioms_in_code(code)

                            if used_axioms > 0:
                                st.markdown(f"**Axioms Used:** {used_axioms}/{defined_axioms}")
                                if used_axiom_names:
                                    axioms_list = ", ".join(sorted(used_axiom_names))
                                    st.markdown(f"_Axioms: {axioms_list}_")
                        except:
                            pass

                    # Show prompt, reasoning summary, and compilation summary using shared function
                    render_proof_attempt_expandables(data, comp_result)

                    # Show model reasoning if available
                    model_output = data.get('model_output')
                    if model_output:
                        with st.expander("💭 Model Reasoning"):
                            st.markdown(model_output)

                    # Show code (expandable) - try multiple code field names
                    code = data.get('full_code', '') or data.get('code', '') or data.get('lean4_code', '')
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



