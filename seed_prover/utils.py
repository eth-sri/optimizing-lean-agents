# Utility functions for seed prover
# Functions used by seed_prover/, seed_data_models/, and analysis_gui/

import re


def extract_code(inputs):
    """
    Extract Lean 4 code from LLM output with markdown code blocks.

    Args:
        inputs: String potentially containing ```lean4 or ```lean code blocks

    Returns:
        Extracted code with import header, or "None" if no code block found
    """
    import_head = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

    # Remove anything inside <think></think> tags
    inputs = re.sub(r'<think>.*?</think>', '', inputs, flags=re.DOTALL)

    pattern = r'```lean4\n(.*?)\n```'
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    pattern = r'```lean4\n(.*?)```'
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    pattern = r'```lean\n(.*?)```'
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    return "None"


def problem_check(statement, full_code):
    """
    Check and replace statement in proof.

    Args:
        statement: The theorem statement
        full_code: The full proof code

    Returns:
        Modified proof code
    """
    full_code = replace_statement_in_proof(statement, full_code)
    return full_code


def extract_lemma_dependencies(lemma_text: str):
    """
    Extract lemma dependencies from lemma text (statement, assumptions, proof).

    Looks for references to "lemma {i}", "Lemma {i}", and "lemma{i}" patterns.

    Args:
        lemma_text: The combined text to search (statement + assumptions + proof)

    Returns:
        List of lemma numbers that are referenced (e.g., [1, 3, 5])
    """
    if not lemma_text:
        return []

    dependencies = set()

    # Pattern 1: "lemma N" or "Lemma N" (with word boundaries)
    pattern1 = r'[Ll]emma\s+(\d+)'
    matches1 = re.findall(pattern1, lemma_text)
    dependencies.update(matches1)

    # Pattern 2: "lemmaX" (no space)
    pattern2 = r'lemma(\d+)'
    matches2 = re.findall(pattern2, lemma_text)
    dependencies.update(matches2)

    # Convert to integers and sort
    try:
        dep_list = sorted([int(d) for d in dependencies])
        return dep_list
    except ValueError:
        return []


def extract_axiom_names(code):
    """
    Extract axiom names from Lean code.

    This function identifies axiom declarations in Lean code and extracts their names.
    Axioms are typically used in theorem proofs as placeholders for lemmas.

    Args:
        code: Lean code string containing potential axiom declarations

    Returns:
        List of axiom names found in the code (e.g., ["lemma1", "lemma2"])
    """
    axiom_names = []
    lines = code.split('\n')

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('axiom'):
            # Extract the axiom name (word after 'axiom')
            # e.g., "axiom lemma1 (x : ℕ) : ..." -> "lemma1"
            parts = stripped.split()
            if len(parts) >= 2:
                axiom_name = parts[1]
                axiom_names.append(axiom_name)

    return axiom_names


def check_if_axiom_used(theorem_code, axiom_name):
    """
    Check if an axiom is actually referenced in the theorem proof body.

    This function determines whether a declared axiom is actually used in the proof,
    as opposed to just being declared. It searches for the axiom name in the proof
    body (the part after ":= by").

    Args:
        theorem_code: Complete Lean theorem code including declaration and proof
        axiom_name: Name of the axiom to check for usage

    Returns:
        True if the axiom is used in the proof body, False otherwise
    """
    # Find the proof body - handle both ":= by" and ":=\nby" patterns
    proof_body = theorem_code

    # Try to find the start of the proof
    # Pattern: ":=" followed by optional whitespace and "by"
    match = re.search(r':=\s+by\b', theorem_code, re.MULTILINE)
    if match:
        # Get everything after the ":= by"
        proof_body = theorem_code[match.end():]

    # Check if axiom name appears as a word in the proof body
    # Use word boundaries to avoid partial matches
    pattern = r'\b' + re.escape(axiom_name) + r'\b'
    return bool(re.search(pattern, proof_body))


def strip_preamble(code):
    """
    Remove preamble (imports, set_option, open, axioms) from Lean code.

    Keeps everything from the first theorem/lemma/def declaration onward.
    This is used for both theorem and lemma proofs to ensure clean, compilable code.

    Args:
        code: Lean code potentially containing preamble statements

    Returns:
        Code with everything removed except from the first 'theorem'/'lemma'/'def' statement onward
    """
    lines = code.split('\n')

    # Find the first line that starts a proof declaration
    for i, line in enumerate(lines):
        stripped = line.strip()
        if any(stripped.startswith(kw) for kw in ['theorem', 'lemma', 'def']):
            # Return everything from this line onward
            return '\n'.join(lines[i:])

    # If no proof declaration found, return the original code
    return code


def remove_axiom_declarations(code):
    """
    Remove axiom declarations from Lean code while keeping everything else.

    Removes lines that start with 'axiom' keyword, which are placeholders for lemmas.
    This is used to clean proof code before inserting actual lemma implementations.

    Args:
        code: Lean code potentially containing axiom declarations

    Returns:
        Code with all axiom declarations removed
    """
    lines = code.split('\n')
    filtered_lines = []

    for line in lines:
        stripped = line.strip()
        # Skip lines that are axiom declarations
        if not stripped.startswith('axiom '):
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def simplify_theorem_name_in_code(code: str) -> str:
    """
    Simplify theorem names in Lean code by removing breakdown-specific parts.

    For round 1+ proofs, theorem names include full breakdown info like:
      theorem algebra_ineq_nto1onlt2m1on_r0_b3_l3_f0 : ... := by ...

    This removes the _r<num>_b<num>_l<num>_f<num> suffix and replaces with _lemma<id>_f<id>:
      theorem algebra_ineq_nto1onlt2m1on_lemma3_f0 : ... := by ...

    The function extracts lemma_id and formalization_id from the theorem name automatically.

    Replaces all occurrences of such theorem names in the code.

    Args:
        code: The Lean code containing theorem declarations

    Returns:
        Code with simplified theorem names
    """
    # Pattern: match theorem declarations with the full breakdown name format
    # theorem <prefix>_r<num>_b<num>_l<num>_f<num>
    pattern = r'\btheorem\s+(\w+)_r(\d+)_b(\d+)_l(\d+)_f(\d+)'

    def replace_func(match):
        prefix = match.group(1)      # e.g., algebra_ineq_nto1onlt2m1on
        # r_id = match.group(2)      # round id (not used)
        # b_id = match.group(3)      # breakdown id (not used)
        lemma_id = match.group(4)    # e.g., 3
        form_id = match.group(5)     # e.g., 0

        simple_name = f'{prefix}_lemma{lemma_id}_f{form_id}'
        return f'theorem {simple_name}'

    # Replace ALL occurrences, not just the first one
    return re.sub(pattern, replace_func, code)


def remove_comments(text):
    """
    Remove comments from Lean code.

    Args:
        text: Lean code with potential comments

    Returns:
        Code with comments removed
    """
    # First remove all /- ... -/ blocks
    text = re.sub(r'/-.*?-/', '', text, flags=re.DOTALL)
    # Then remove -- comments from each line
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Split on -- and keep only the first part
        cleaned_line = line.split('--', 1)[0]
        cleaned_lines.append(cleaned_line)
    # Join back together
    cleaned_text = '\n'.join(cleaned_lines)
    return cleaned_text.strip()


def return_theorem_to_prove(text):
    """
    Find the span of theorem/lemma declaration ending with ':= by sorry'.

    Args:
        text: Lean code text

    Returns:
        Span tuple (start, end) or None if not found
    """
    # Pattern that matches from 'theorem' or 'lemma' to ':= by sorry' with any content in between
    pattern = r'((?:theorem|lemma).*?:=\s*by\s*sorry)'
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def return_theorem_to_replace(text):
    """
    Find the span of theorem/lemma declaration ending with ':= by'.

    Args:
        text: Lean code text

    Returns:
        Span tuple (start, end) or None if not found
    """
    # Pattern that matches from 'theorem' or 'lemma' to ':= by' with any content in between
    pattern = r'((?:^|\s)(?:theorem|lemma)\s+.*?:=\s*by)'
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def replace_statement_in_proof(statement, proof):
    """
    Replace theorem statement with proof implementation.

    Args:
        statement: Theorem statement (with sorry)
        proof: Proof implementation

    Returns:
        Combined code or error message
    """
    if ("apply?" in proof) or ("exact?" in proof):
        return "**Error**, 'apply?' or 'exact?' is used, which is not allowed."
    stats_re = remove_comments(statement)
    stats_span_= return_theorem_to_prove(stats_re)
    if stats_span_ is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in statement.split('\n')])
        return f"**Error**, can not find 'theorem' seed and ':= sorry' in {error_app}"
    proof_str = remove_comments(proof)
    span = return_theorem_to_replace(proof_str)
    if span is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in proof.split('\n')])
        return f"**Error**, can not find 'theorem' seed and ':=' in {error_app}"
    return stats_re[:stats_span_[1]].replace("sorry", "") + proof_str[span[1]:]
