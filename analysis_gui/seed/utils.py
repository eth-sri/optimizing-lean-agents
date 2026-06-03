"""
Utility functions for the analysis GUI.
"""
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path to import metadata_utils
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from metadata_utils import (
        get_run_costs,
        get_problem_costs,
        get_breakdown_costs,
        extract_detailed_cost
    )
    HAS_METADATA_UTILS = True
except ImportError:
    HAS_METADATA_UTILS = False

# Try to import id_utils if available
try:
    from id_utils import get_lemma_component
except ImportError:
    def get_lemma_component(problem_id: str):
        """Fallback implementation of get_lemma_component."""
        if "_theorem" in problem_id:
            return "theorem"
        elif "_l" in problem_id:
            parts = problem_id.rsplit("_l", 1)
            if len(parts) == 2 and parts[1].split("_")[0].isdigit():
                return f"l{parts[1].split('_')[0]}"
        return None


def format_cost(cost: Optional[Dict[str, Any]]) -> str:
    """
    Format cost information for display.

    Args:
        cost: Dictionary containing cost information

    Returns:
        Formatted string representation
    """
    if not cost:
        return "N/A"

    total_cost = cost.get('cost', 0)
    input_tokens = cost.get('input_tokens', 0)
    output_tokens = cost.get('output_tokens', 0)

    return f"${total_cost:.4f} ({input_tokens} in / {output_tokens} out)"


def get_status_emoji(status: bool) -> str:
    """
    Get emoji for status indicator.

    Args:
        status: Boolean status

    Returns:
        Emoji string
    """
    return "✅" if status else "❌"


def get_breakdown_stage_status(analysis: Any, breakdown_id: str) -> Dict[str, bool]:
    """
    Get status of all pipeline stages for a breakdown.

    Args:
        analysis: Optional[Dict[str, Any]] object
        breakdown_id: The breakdown ID to check

    Returns:
        Dictionary with stage names and their status
    """
    if not analysis:
        return {
            "breakdown": False,
            "parsed": False,
            "formalized": False,
            "theorem_proven": False,
            "lemmas_proven": False,
        }

    # Check if breakdown exists
    breakdown_exists = any(
        bd.get('breakdown_id') == breakdown_id
        for bd in analysis.breakdown_stats.get('breakdowns', [])
    )

    # Check if parsed
    parsed = any(
        bd.get('breakdown_id') == breakdown_id
        for bd in analysis.breakdown_parser_stats.get('parsed_breakdowns', [])
    )

    # Check if formalized
    formalized = any(
        bd.get('breakdown_id') == breakdown_id
        for bd in analysis.formalizer_stats.get('parsed_breakdowns', [])
    )

    # Check if theorem proven
    theorem_proven = any(
        bd.get('breakdown_id') == breakdown_id and bd.get('proven', False)
        for bd in analysis.theorem_prover_stats.get('breakdown_results', [])
    )

    # Check if all lemmas proven
    lemmas_proven = any(
        bd.get('breakdown_id') == breakdown_id and bd.get('all_lemmas_proven', False)
        for bd in analysis.lemma_prover_stats.get('breakdown_results', [])
    )

    return {
        "breakdown": breakdown_exists,
        "parsed": parsed,
        "formalized": formalized,
        "theorem_proven": theorem_proven,
        "lemmas_proven": lemmas_proven,
    }


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to a maximum length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def extract_problem_name(problem_id: str) -> str:
    """
    Extract a clean problem name from problem ID.

    Args:
        problem_id: Problem ID (e.g., "putnam_1962_a5")

    Returns:
        Formatted name (e.g., "Putnam 1962 A5")
    """
    parts = problem_id.split('_')
    if len(parts) >= 3:
        competition = parts[0].capitalize()
        year = parts[1]
        problem = parts[2].upper()
        return f"{competition} {year} {problem}"
    return problem_id

def extract_axiom_names(code: str) -> list:
    """
    Extract axiom names from Lean code.

    Matches the implementation in seed_prover/utils.py.

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


def check_if_axiom_used(theorem_code: str, axiom_name: str) -> bool:
    """
    Check if an axiom is actually referenced in the theorem proof body.

    Matches the implementation in seed_prover/utils.py.

    Args:
        theorem_code: Complete Lean theorem code including declaration and proof
        axiom_name: Name of the axiom to check for usage

    Returns:
        True if the axiom is used in the proof body, False otherwise
    """
    import re

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

def extract_used_lemmas_from_code(code: str, breakdown_id: str) -> tuple:
    """
    Extract used lemmas from Lean code by detecting axiom usage.

    This matches the approach used in LemmaUsageAnalyzer:
    - Scans for axiom declarations
    - Detects axiom usage in the proof (more than just the declaration)
    - Maps axiom names like 'lemma1' to lemma IDs like 'breakdown_id_l1'

    Args:
        code: Lean code to analyze
        breakdown_id: The breakdown ID (for constructing lemma IDs)

    Returns:
        Tuple of (used_lemma_ids, used_lemma_count)
        - used_lemma_ids: list of lemma IDs that are actually used
        - used_lemma_count: count of used lemmas
    """
    import re

    used_lemmas = set()

    # Step 1: Extract all axiom declarations
    axiom_pattern = r'axiom\s+(\w+)'
    axioms = re.findall(axiom_pattern, code)

    if not axioms:
        return [], 0

    # Step 2: For each axiom, check if it's actually used in the proof
    for axiom_name in axioms:
        # Count occurrences of the axiom name
        pattern = r'\b' + re.escape(axiom_name) + r'\b'
        matches = list(re.finditer(pattern, code))

        # If there's more than one match, it means the axiom is used (not just declared)
        # First match = axiom declaration, second+ match = actual usage in proof
        if len(matches) > 1:
            # Extract lemma number from axiom name (e.g., 'imo_1968_p5_1_lemma1' -> 1)
            lemma_match = re.search(r'lemma(\d+)', axiom_name)
            if lemma_match:
                lemma_num = lemma_match.group(1)
                lemma_id = f"{breakdown_id}_l{lemma_num}"
                used_lemmas.add(lemma_id)

    return list(used_lemmas), len(used_lemmas)

def extract_lemma_dependencies(lemma_text: str) -> list:
    """
    Extract lemma dependencies from lemma text (statement, assumptions, proof).

    Looks for references to "lemma {i}", "Lemma {i}", and "lemma{i}" patterns.

    Args:
        lemma_text: The combined text to search (statement + assumptions + proof)

    Returns:
        List of lemma numbers that are referenced (e.g., [1, 3, 5])
    """
    import re

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


def build_dependency_tree_data(lemmas: list, theorem_text: str = "") -> tuple:
    """
    Build hierarchical tree data for visualization.

    Args:
        lemmas: List of lemma dictionaries with statement, assumption, proof fields
        theorem_text: Optional text from theorem proof to detect which lemmas it uses

    Returns:
        Tuple of (root_node, all_nodes, edges) for visualization
    """
    import re

    if not lemmas:
        return None, [], []

    # Build dependency map for all lemmas using regex to find lemma references
    lemma_deps = {}
    for idx, lemma in enumerate(lemmas, 1):
        combined_text = ""

        # OOP Lemma object - direct attribute access
        if lemma.statement:
            combined_text += lemma.statement + "\n"
        if lemma.assumptions:
            combined_text += lemma.assumptions + "\n"
        if hasattr(lemma, 'proof_idea') and lemma.proof_idea:
            combined_text += lemma.proof_idea + "\n"

        # Also check formalized code (formalizations) for actual lemma usage
        if hasattr(lemma, 'formalizations') and lemma.formalizations:
            for form in lemma.formalizations:
                if hasattr(form, 'formal_statement') and form.formal_statement:
                    combined_text += form.formal_statement + "\n"
                # Also check proof attempts
                if hasattr(form, 'proof_attempts') and form.proof_attempts:
                    for attempt in form.proof_attempts:
                        if hasattr(attempt, 'code') and attempt.code:
                            combined_text += attempt.code + "\n"

        # Use regex to find references to other lemmas (e.g., "lemma1", "lemma_1", "lem1", etc.)
        # Also look for axiom references which are how lemmas are used in Lean
        # Look for lemma references in the combined text
        lemma_refs = set()
        # Match patterns like: lemma1, lemma_1, l1, lem1, axiom lemma1, etc.
        for match in re.finditer(r'\b(?:lemma|lem|l|axiom)\s*(?:_)?(\d+)', combined_text, re.IGNORECASE):
            try:
                lemma_num = int(match.group(1))
                if lemma_num != idx and 1 <= lemma_num <= len(lemmas):  # Don't self-reference
                    lemma_refs.add(lemma_num)
            except (ValueError, IndexError):
                pass

        lemma_deps[idx] = list(lemma_refs)

    # Extract which lemmas the theorem depends on using regex
    theorem_deps = []
    if theorem_text:
        theorem_refs = set()
        for match in re.finditer(r'\b(?:lemma|lem|l|axiom)\s*(?:_)?(\d+)', theorem_text, re.IGNORECASE):
            try:
                lemma_num = int(match.group(1))
                if 1 <= lemma_num <= len(lemmas):
                    theorem_refs.add(lemma_num)
            except (ValueError, IndexError):
                pass
        theorem_deps = list(theorem_refs)

    # Create tree structure with only relevant connections
    nodes = [{"id": "Theorem", "label": "Theorem", "level": 0}]
    edges = []

    # Add edges from Theorem ONLY to lemmas it actually depends on
    for dep in theorem_deps:
        if dep <= len(lemmas):  # Valid dependency
            lemma_id = f"L{dep}"
            edges.append({"source": "Theorem", "target": lemma_id})

    # Add dependency edges between lemmas
    for idx in range(1, len(lemmas) + 1):
        deps = lemma_deps.get(idx, [])
        lemma_id = f"L{idx}"
        for dep in deps:
            if dep <= len(lemmas):  # Valid dependency
                dep_id = f"L{dep}"
                # Add edge from lemma to its dependency
                edges.append({"source": lemma_id, "target": dep_id})

    # Calculate proper levels/depths using BFS from Theorem root
    # This ensures nested dependencies have depth > 1
    levels = {}
    visited = set()
    current_layer = ["Theorem"]
    level = 0

    while current_layer:
        for node_id in current_layer:
            if node_id not in visited:
                levels[node_id] = level
                visited.add(node_id)

        # Find next layer - nodes that current layer points to
        next_layer = []
        for node_id in current_layer:
            # Find all targets of edges from this node
            children = [e["target"] for e in edges if e["source"] == node_id]
            for child in children:
                if child not in visited and child not in next_layer:
                    next_layer.append(child)

        current_layer = next_layer
        level += 1

    # Add all lemmas as nodes with calculated levels
    for idx in range(1, len(lemmas) + 1):
        lemma_id = f"L{idx}"
        # Use calculated level, or default to 1 if not found (orphan lemmas)
        node_level = levels.get(lemma_id, 1)
        nodes.append({"id": lemma_id, "label": f"Lemma {idx}", "level": node_level})

    return "Theorem", nodes, edges
