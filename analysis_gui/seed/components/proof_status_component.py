"""Proof status summary and axiom analysis
"""
"""
Enhanced breakdown details component with nested views for parsing, formalization, and proofs.
"""
import streamlit as st
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add root directory to path to import seed_data_models
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

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



def render_proof_status_summary(breakdown: Breakdown):
    """
    Render a summary of the proof status for this breakdown.

    Shows whether the theorem was proven, lemma prover results, and whether the breakdown is solved.

    Args:
        breakdown: Breakdown object
    """
    # Get theorem prover status
    theorem_proven = False

    # First, try to get theorem proven status from parsed_breakdown (minified data)
    if hasattr(breakdown, 'parsed_breakdown') and breakdown.parsed_breakdown:
        try:
            pb = breakdown.parsed_breakdown
            # Check if theorem has a best attempt (meaning it has a passing proof)
            best_attempt = pb.theorem.get_best_attempt()
            if best_attempt:
                theorem_proven = best_attempt.is_passing()
        except (AttributeError, Exception):
            pass

    # Fallback to theorem_prover_results for legacy data
    if not theorem_proven and breakdown.theorem_prover_results:
        attempts = breakdown.theorem_prover_results.get('attempts', [])
        theorem_proven = any(
            a.get('data', {}).get('compilation_result', {}).get('pass', False)
            for a in attempts
        )


    # Extract used lemmas from the data model - centralized approach
    used_lemmas_count = 0
    proven_used_lemmas = 0

    # If breakdown is a new Breakdown object with parsed_breakdown, use the model method
    if hasattr(breakdown, 'parsed_breakdown') and breakdown.parsed_breakdown:
        used_lemmas_count, proven_used_lemmas = breakdown.get_used_lemmas_count()

    # Determine if breakdown is solved using the data model
    # If it's a Breakdown object with parsed_breakdown, use the model method
    if hasattr(breakdown, 'parsed_breakdown') and breakdown.parsed_breakdown:
        is_solved = breakdown.is_solved()

    # Display status
    st.markdown("### Proof Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        status = "✅ Proven" if theorem_proven else "❌ Not Proven"
        st.metric("Theorem", status)

    with col2:
        if used_lemmas_count > 0:
            st.metric("Used Lemmas Proven", f"{proven_used_lemmas}/{used_lemmas_count}")
        else:
            st.metric("Used Lemmas Proven", "N/A")

    with col3:
        if is_solved:
            st.metric("Breakdown Status", "✅ SOLVED", delta="")
        else:
            st.metric("Breakdown Status", "❌ Not Solved")



def count_axioms_in_code(code: str) -> tuple:
    """
    Count the number of axioms defined and actually used in the proof.

    Args:
        code: The Lean code containing axioms and theorem

    Returns:
        Tuple of (defined_axioms, used_axioms, used_axiom_names) where:
        - defined_axioms: number of axiom definitions
        - used_axioms: number of axioms referenced in the proof
        - used_axiom_names: set of axiom names that were used
    """
    if not code:
        return (0, 0, set())

    # Extract axiom names
    axiom_names = set()
    lines = code.split('\n')
    theorem_start_idx = -1

    for idx, line in enumerate(lines):
        # Find where theorem starts
        if 'theorem' in line.lower():
            theorem_start_idx = idx
            break

        # Extract axiom names
        stripped = line.strip()
        if stripped.startswith('axiom '):
            # Extract the axiom name (e.g., "axiom lemma1" -> "lemma1")
            parts = stripped.split()
            if len(parts) >= 2:
                axiom_names.add(parts[1])

    # Count used axioms in the proof (after theorem definition)
    used_axioms = set()
    if theorem_start_idx >= 0:
        proof_section = '\n'.join(lines[theorem_start_idx:])
        for axiom_name in axiom_names:
            # Look for axiom usage in the proof
            if axiom_name in proof_section:
                used_axioms.add(axiom_name)

    return (len(axiom_names), len(used_axioms), used_axioms)



def extract_think_content(text: str) -> str:
    """
    Extract content from <think> tags.

    Args:
        text: Text potentially containing <think>...</think> tags

    Returns:
        The content inside think tags, or empty string if not found
    """
    if not text:
        return ""

    # Look for <think> tags
    start_tag = "<think>"
    end_tag = "</think>"

    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        # Extract content between tags
        content = text[start_idx + len(start_tag):end_idx].strip()
        return content

    return ""


def parse_reasoning_summary(raw_response: str) -> Optional[Dict[str, Any]]:
    """
    Parse reasoning summary from raw response with lenient JSON parsing.
    Handles malformed JSON with unescaped backslashes (common in LaTeX math).

    Args:
        raw_response: The raw response string potentially containing JSON

    Returns:
        Parsed dictionary or None if parsing fails
    """
    import json
    import re

    if not raw_response:
        return None

    try:
        # Try standard JSON parsing first
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from the response (may be after </think> tag)
    try:
        # Look for JSON object start and end
        start_idx = raw_response.find('{')
        end_idx = raw_response.rfind('}')

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return None

        json_str = raw_response[start_idx:end_idx + 1]

        # Try standard JSON parsing again
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try with lenient decoder for escape sequence issues
    try:
        # Extract JSON portion
        start_idx = raw_response.find('{')
        end_idx = raw_response.rfind('}')

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return None

        json_str = raw_response[start_idx:end_idx + 1]

        # Try with non-strict JSON decoder
        try:
            decoder = json.JSONDecoder(strict=False)
            return decoder.decode(json_str)
        except json.JSONDecodeError:
            pass

        # Last resort: try to work around escape sequence issues
        # The problem is invalid escape sequences like \( and \) in JSON strings
        # Replace them with safe placeholders, parse, then restore
        try:
            # Find all problematic escape sequences and replace with safe ones
            # Common LaTeX sequences: \(, \), \[, \], \{, \}, \neq, \uparrow, etc.
            fixed_str = json_str
            # Replace invalid escape sequences with temporary placeholders
            replacements = {
                '\\(': '__ESCAPED_LPAREN__',
                '\\)': '__ESCAPED_RPAREN__',
                '\\[': '__ESCAPED_LBRACKET__',
                '\\]': '__ESCAPED_RBRACKET__',
                '\\{': '__ESCAPED_LBRACE__',
                '\\}': '__ESCAPED_RBRACE__',
                '\\neq': '__ESCAPED_NEQ__',
                '\\uparrow': '__ESCAPED_UPARROW__',
                '\\cdot': '__ESCAPED_CDOT__',
                '\\times': '__ESCAPED_TIMES__',
                '\\geq': '__ESCAPED_GEQ__',
                '\\leq': '__ESCAPED_LEQ__',
            }

            for original, placeholder in replacements.items():
                fixed_str = fixed_str.replace(original, placeholder)

            result = json.loads(fixed_str)

            # Now fix the strings back
            def fix_backslashes(obj):
                if isinstance(obj, str):
                    for original, placeholder in replacements.items():
                        obj = obj.replace(placeholder, original)
                    return obj
                elif isinstance(obj, dict):
                    return {k: fix_backslashes(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [fix_backslashes(item) for item in obj]
                return obj

            return fix_backslashes(result)
        except json.JSONDecodeError:
            # If placeholder replacement didn't work, try a regex-based approach
            # to extract fields manually
            try:
                result = {}

                # Try to extract "summary" field
                summary_match = re.search(r'"summary"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', json_str)
                if summary_match:
                    # Unescape the string - handle both unicode escapes and simple string content
                    try:
                        result['summary'] = summary_match.group(1).encode().decode('unicode_escape')
                    except:
                        # If unicode_escape fails, just use the raw matched content
                        result['summary'] = summary_match.group(1)

                # Try to extract "confidence" field
                conf_match = re.search(r'"confidence"\s*:\s*([0-9]+|null)', json_str)
                if conf_match:
                    val = conf_match.group(1)
                    result['confidence'] = int(val) if val != 'null' else None

                # Try to extract "correctness" field
                correct_match = re.search(r'"correctness"\s*:\s*(true|false|null)', json_str)
                if correct_match:
                    val = correct_match.group(1)
                    result['correctness'] = val == 'true' if val != 'null' else None

                # Try to extract lemmas array
                lemmas_match = re.search(r'"lemmas"\s*:\s*(\[[\s\S]*?\])', json_str)
                if lemmas_match:
                    # Try to parse the lemmas array
                    try:
                        lemmas_str = lemmas_match.group(1)
                        # Apply the same placeholder replacement to lemmas
                        for original, placeholder in replacements.items():
                            lemmas_str = lemmas_str.replace(original, placeholder)
                        result['lemmas'] = json.loads(lemmas_str)
                    except:
                        result['lemmas'] = []

                if result:
                    return result
                return None
            except:
                pass

        return None

    except Exception:
        return None


def render_proof_attempt_expandables(data: Dict[str, Any], comp_result: Dict[str, Any]):
    """
    Render the expandable sections for a proof attempt:
    - Proof Code
    - Prompt (model_input)
    - Reasoning Summary
    - Compilation Summary

    Args:
        data: The attempt data dictionary
        comp_result: The compilation_result from data
    """

    # Show model config if available
    model_config_path = data.get('model_config_path')
    # Also check in metadata for legacy formats
    if not model_config_path and 'metadata' in data:
        model_config_path = data['metadata'].get('model_config_path')

    if model_config_path:
        st.markdown(f"**Model Config:** `{model_config_path}`")

    # Show prompt if available
    model_input = data.get('model_input')
    if model_input:
        with st.expander("📝 Prompt"):
            if isinstance(model_input, list):
                # model_input is a list of messages
                for msg in model_input:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    st.markdown(f"**{role.upper()}:**")
                    st.markdown(content)
                    st.divider()
            else:
                st.markdown(model_input)

    # Show reasoning summary if available
    reasoning_summary = data.get('reasoning_summary')
    if reasoning_summary:
        # Handle case where reasoning_summary might be a dict or needs parsing
        if isinstance(reasoning_summary, str):
            reasoning_summary = parse_reasoning_summary(reasoning_summary)

        if reasoning_summary:
            # Check if we need to parse the raw_response (when there's a parse_error)
            summary = reasoning_summary.get('summary', '')
            confidence = reasoning_summary.get('confidence')
            correctness = reasoning_summary.get('correctness')
            parse_error = reasoning_summary.get('parse_error')
            raw_response = reasoning_summary.get('raw_response')
            lemmas = reasoning_summary.get('lemmas', [])

            # If there's a parse error and raw_response exists, try to parse it
            if parse_error and raw_response:
                parsed_raw = parse_reasoning_summary(raw_response)
                if parsed_raw:
                    reasoning_summary = parsed_raw
                    summary = reasoning_summary.get('summary', '')
                    confidence = reasoning_summary.get('confidence')
                    correctness = reasoning_summary.get('correctness')
                    lemmas = reasoning_summary.get('lemmas', [])
                    parse_error = None  # Successfully parsed the raw response
            # Also try if main fields are empty
            elif (not summary and confidence is None and correctness is None and not lemmas) and raw_response:
                parsed_raw = parse_reasoning_summary(raw_response)
                if parsed_raw:
                    reasoning_summary = parsed_raw
                    summary = reasoning_summary.get('summary', '')
                    confidence = reasoning_summary.get('confidence')
                    correctness = reasoning_summary.get('correctness')
                    lemmas = reasoning_summary.get('lemmas', [])
                    parse_error = None  # Successfully parsed the raw response

            with st.expander("📊 Reasoning Summary"):
                # Show if we successfully extracted from raw response
                if parse_error and not summary and not confidence and not correctness and not lemmas:
                    st.warning(f"⚠️ Parse error in summary: {parse_error}")
                    st.markdown("*Could not extract data from raw response*")
                elif parse_error:
                    st.info(f"ℹ️ Parse error occurred, but extracted data shown below:")

                if summary:
                    st.markdown(summary)

                if confidence is not None:
                    # Convert to int if needed
                    try:
                        conf_val = int(confidence) if isinstance(confidence, str) else confidence
                        st.markdown(f"**Confidence:** {conf_val}/10")
                    except (ValueError, TypeError):
                        pass

                if correctness is not None:
                    # Convert to bool if needed
                    if isinstance(correctness, str):
                        correctness = correctness.lower() in ('true', '1', 'yes')
                    correctness_text = "✅ Correct" if correctness else "❌ Incorrect"
                    st.markdown(f"**Correctness:** {correctness_text}")

                if lemmas:
                    st.markdown("**Lemmas Analysis:**")
                    for lemma in lemmas:
                        if isinstance(lemma, dict):
                            name = lemma.get('name', 'unknown')
                            used = lemma.get('used', False)
                            mentioned = lemma.get('mentioned', False)
                            correct = lemma.get('correct', False)

                            # Convert string booleans if needed
                            if isinstance(used, str):
                                used = used.lower() in ('true', '1', 'yes')
                            if isinstance(mentioned, str):
                                mentioned = mentioned.lower() in ('true', '1', 'yes')
                            if isinstance(correct, str):
                                correct = correct.lower() in ('true', '1', 'yes')

                            used_emoji = "✅" if used else "❌"
                            mentioned_emoji = "✅" if mentioned else "❌"
                            correct_emoji = "✅" if correct else "❌"

                            st.markdown(f"  **{name}**: used={used_emoji} mentioned={mentioned_emoji} correct={correct_emoji}")

    # Show compilation summary (from consolidated compilation_summary field or derived from compilation_result)
    compilation_summary = data.get('compilation_summary')
    if compilation_summary or comp_result:
        with st.expander("📋 Compilation Summary"):
            # Show compilation_summary from consolidated records if available
            if compilation_summary and isinstance(compilation_summary, dict):
                error_counts = compilation_summary.get('error_counts', {})
                total_errors = compilation_summary.get('total_errors', 0)
                status = compilation_summary.get('status', 'unknown')

                st.markdown(f"**Status:** {status}")
                st.markdown(f"**Total Errors:** {total_errors}")

                if error_counts:
                    st.markdown("**Error Breakdown:**")
                    for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                        st.markdown(f"  - {error_type}: {count}")
            else:
                # Fallback to compilation_result if compilation_summary not available
                st.markdown(f"**Pass:** {'✅ Yes' if comp_result.get('pass') else '❌ No'}")
                st.markdown(f"**Complete:** {'✅ Yes' if comp_result.get('complete') else '❌ No'}")

                sorries = comp_result.get('sorries', [])
                if sorries:
                    st.markdown(f"**Sorries:** {len(sorries)}")
                    with st.expander(f"View {len(sorries)} sorries", expanded=False):
                        for sorry in sorries:
                            st.text(sorry)

                tactics = comp_result.get('tactics', [])
                if tactics:
                    st.markdown(f"**Tactics Used:** {len(tactics)}")
                    with st.expander(f"View {len(tactics)} tactics", expanded=False):
                        for tactic in tactics:
                            st.text(tactic)

                warnings = comp_result.get('warnings', [])
                if warnings:
                    st.markdown(f"**Warnings:** {len(warnings)}")
                    with st.expander(f"View {len(warnings)} warnings", expanded=False):
                        for warning in warnings:
                            if isinstance(warning, dict):
                                st.text(warning.get('data', str(warning)))
                            else:
                                st.text(str(warning))

    # Add expandable for Model Reasoning
    with st.expander("📝 Model Reasoning"):
        model_reasoning = data.get('model_reasoning')
        if model_reasoning:
            st.markdown(model_reasoning)
        else:
            st.info("No model reasoning available")

    # Add expandable for Compilation Results
    with st.expander("⚙️ Compilation Results"):
        errors = comp_result.get('errors', [])
        warnings = comp_result.get('warnings', [])

        if errors:
            with st.expander(f"❌ Errors ({len(errors)})", expanded=False):
                for i, error in enumerate(errors):
                    if isinstance(error, dict):
                        st.text(error.get('data', str(error)))
                    else:
                        st.text(str(error))
                    if i < len(errors) - 1:
                        st.divider()
        else:
            st.success("✅ No errors")

        if warnings:
            with st.expander(f"⚠️ Warnings ({len(warnings)})", expanded=False):
                for i, warning in enumerate(warnings):
                    if isinstance(warning, dict):
                        st.text(warning.get('data', str(warning)))
                    else:
                        st.text(str(warning))
                    if i < len(warnings) - 1:
                        st.divider()

    # Show proof code if available
    code = data.get('code')
    if code:
        with st.expander("💻 Proof Code", expanded=False):
            st.code(code, language="lean")

