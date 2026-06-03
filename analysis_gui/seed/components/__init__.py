"""Components for the analysis GUI.

This package contains modular components for the seed analysis GUI:
- enhanced_breakdown_details: Main orchestrator for breakdown visualization
- proof_status_component: Proof status metrics and axiom analysis
- breakdown_tree_component: Lemma tree and dependency visualization
- lemma_details_component: Lemma list and details
- theorem_viewer_component: Theorem proofs and attempts
- lemma_prover_component: Lemma prover results
- formalization_component: Formalization and compilation results
- utils_component: Utility functions for file operations
- combined_proof_viewer: Combined proof visualization
"""

# Export main entry point
from .enhanced_breakdown_details import (
    render_enhanced_breakdown_details,
    render_enhanced_breakdown_details_inner,
    render_parsed_breakdown_tab,
)

# Export component functions for direct use if needed
from .proof_status_component import render_proof_status_summary
from .breakdown_tree_component import render_parsed_lemma_tree, render_used_lemmas_tree
from .lemma_details_component import render_lemma_list, render_lemmas_section, render_lemma_formalization
from .theorem_viewer_component import render_theorem_prover_code_results, render_theorem_prover_attempts, render_theorem_formalization, render_lemma_prover_code_results, render_theorem_attempts_from_parsed_breakdown, render_lemma_attempts_from_parsed_breakdown
from .lemma_prover_component import render_lemma_prover_results
from .formalization_component import (
    render_compilation_result,
    render_parse_failure,
    render_theorem_from_parsed,
)
from .utils_component import _find_lean_file_path
from .combined_proof_viewer import render_combined_proof
from .recursive_breakdown_viewer import render_recursive_breakdown_viewer

__all__ = [
    # Main entry points
    "render_enhanced_breakdown_details",
    "render_enhanced_breakdown_details_inner",
    "render_parsed_breakdown_tab",
    # Component functions
    "render_proof_status_summary",
    "render_parsed_lemma_tree",
    "render_lemma_list",
    "render_lemmas_section",
    "render_lemma_formalization",
    "render_theorem_prover_code_results",
    "render_theorem_prover_attempts",
    "render_theorem_formalization",
    "render_lemma_prover_code_results",
    "render_lemma_prover_results",
    "render_compilation_result",
    "render_parse_failure",
    "render_theorem_from_parsed",
    "render_combined_proof",
    "render_recursive_breakdown_viewer",
]
