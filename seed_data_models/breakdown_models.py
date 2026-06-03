"""
Breakdown data models: ParsedBreakdown and Breakdown.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Set, Any, Union, Tuple, TYPE_CHECKING
from pathlib import Path
import re
from collections import deque

from .theorem import Theorem
from .lemma import Lemma
from .exceptions_and_helpers import ProverPathNode, SimulationFailure

if TYPE_CHECKING:
    from .simulations.attempt_tracker import ProofAttemptTracker

# Effective parameters for breakdown/parsing model (oss-20b)
BREAKDOWN_EFFECTIVE_PARAMS = 3.6


@dataclass
class ParsedBreakdown:
    """Represents a parsed breakdown with theorem and lemmas."""
    theorem: Theorem
    lemmas: Dict[int, Lemma] = field(default_factory=dict)
    detailed_cost: Optional[Dict[str, Any]] = None

    def is_fully_solved(self) -> bool:
        """
        Returns True if the entire breakdown is solved.
        A breakdown is fully solved if:
        1. Theorem has a passing & complete proof
        2. All lemmas used (transitively) in the proof are also solved
        """
        # Check if theorem itself has a passing & complete proof
        if not self.theorem.is_solved():
            return False

        # Get the best proof attempt for the theorem
        # Pass lemmas_dict so it can prioritize fully-solved attempts
        best_attempt = self.theorem.get_best_attempt(lemmas_dict=self.lemmas)
        if not best_attempt:
            return False

        # Get all lemmas used transitively in the proof
        used_lemma_ids = best_attempt.get_used_lemmas(lemmas_dict=self.lemmas, recursive=True)

        # Check if all used lemmas are solved
        for lemma_id in used_lemma_ids:
            if lemma_id in self.lemmas:
                if not self.lemmas[lemma_id].is_solved():
                    return False
            else:
                # Lemma not found in breakdown
                return False

        return True

    def get_solve_stats(self) -> tuple[bool, int, int]:
        """
        Returns solve status with lemma usage statistics for the theorem.

        Returns:
            Tuple of (is_solved, num_used_lemmas, num_proven_used_lemmas)
            - is_solved: Whether the breakdown theorem is proven
            - num_used_lemmas: Total number of lemmas used transitively in proof
            - num_proven_used_lemmas: How many of those used lemmas are also proven

        Uses a single recursive traversal through the dependency graph.
        """
        return self.theorem.get_solve_stats(self.lemmas)

    def is_formalized(self) -> bool:
        """
        Returns True if the entire breakdown has been successfully formalized.
        A breakdown is formalized if:
        1. Theorem has at least one formalization that compiled
        2. ALL lemmas have at least one formalization that compiled
        """
        # Check if theorem is formalized
        if not self.theorem.is_formalized():
            return False

        # Check if all lemmas are formalized
        for lemma in self.lemmas.values():
            if not lemma.is_formalized():
                return False

        return True

    def is_fully_validated(self) -> bool:
        """
        Returns True if the entire breakdown has been formalized AND validated.
        A breakdown is fully validated if:
        1. Theorem has at least one formalization that compiled
        2. ALL lemmas have at least one formalization that compiled AND validated
        """
        # Check if theorem is formalized
        if not self.theorem.is_formalized():
            return False

        # Check if all lemmas are formalized and validated
        for lemma in self.lemmas.values():
            if not lemma.has_valid_formalization():
                return False

        return True

    def get_formalization_summary(self) -> Dict[str, int]:
        """
        Returns a summary of formalization status for this breakdown.

        Returns:
            Dict with keys:
            - 'total_lemmas': Total number of lemmas
            - 'formalized_lemmas': Number of lemmas with at least one passing formalization
            - 'validated_lemmas': Number of lemmas with at least one passing+validated formalization
            - 'theorem_formalized': 1 if theorem is formalized, 0 otherwise
        """
        total_lemmas = len(self.lemmas)
        formalized_count = sum(1 for lemma in self.lemmas.values() if lemma.is_formalized())
        validated_count = sum(1 for lemma in self.lemmas.values() if lemma.has_valid_formalization())

        return {
            'total_lemmas': total_lemmas,
            'formalized_lemmas': formalized_count,
            'validated_lemmas': validated_count,
            'theorem_formalized': 1 if self.theorem.is_formalized() else 0
        }

    def get_theorem_used_lemmas(self, recursive: bool = False) -> Set[int]:
        """
        Get lemma IDs used by the theorem, with optional recursive traversal.

        Args:
            recursive: If True, include transitive lemma dependencies

        Returns:
            Set of lemma IDs (integers) used
        """
        best_attempt = self.theorem.get_best_attempt(lemmas_dict=self.lemmas)
        if not best_attempt:
            return set()
        return best_attempt.get_used_lemmas(lemmas_dict=self.lemmas, recursive=recursive)

    def get_cost(self, cost_type="cost") -> float:
        """Returns the cost of generating the parsed breakdown itself.

        For sflops cost types, calculates from tokens * effective_params if not pre-computed.
        ParsedBreakdown uses oss-20b with effective_params = 3.6.
        """
        if cost_type == "prover_calls":
            return 0.0

        if not self.detailed_cost:
            return 0.0

        # If sflops requested and not already in detailed_cost, calculate from tokens
        if cost_type in ['input_sflops', 'output_sflops']:
            if cost_type not in self.detailed_cost:
                tokens_key = 'input_tokens' if cost_type == 'input_sflops' else 'output_tokens'
                tokens = self.detailed_cost.get(tokens_key, 0)
                return float(tokens * BREAKDOWN_EFFECTIVE_PARAMS)

        return float(self.detailed_cost.get(cost_type, 0.0))

    def get_total_cost(self, cost_type="cost", exclude_prover_calls: bool = False) -> float:
        """Returns the sum of costs from theorem and all lemmas, plus the cost of generating the parsed breakdown itself.

        Args:
            cost_type: Type of cost to return (e.g., 'cost', 'output_sflops', 'input_tokens')
            exclude_prover_calls: If True, excludes proof attempt costs
        """
        parsed_breakdown_cost = self.get_cost(cost_type)

        theorem_cost = self.theorem.get_total_cost(cost_type, exclude_prover_calls=exclude_prover_calls)
        lemmas_cost = sum(lemma.get_total_cost(cost_type, exclude_prover_calls=exclude_prover_calls) for lemma in self.lemmas.values())
        return parsed_breakdown_cost + theorem_cost + lemmas_cost

    @classmethod
    def from_dict(cls, theorem_obj, lemmas_dict) -> 'ParsedBreakdown':
        """Reconstruct ParsedBreakdown from theorem and lemmas objects."""
        return cls(
            theorem=theorem_obj,
            lemmas=lemmas_dict
        )


@dataclass
class Breakdown:
    """Represents a single breakdown of a problem."""
    problem_id: str  # Unique: "aime_1983_p1_r0_b0"
    origin_problem_id: str
    round_id: int
    breakdown_id: int
    parent_problem_id: Optional[str] = None  # For round 1+: the lemma being recursively proven
    informal_breakdown: Optional[str] = None  # Lazy-load in detail phase
    breakdown_prompt: Optional[str] = None  # Lazy-load in detail phase
    informal_breakdown_reasoning: Optional[str] = None  # Lazy-load in detail phase
    parsed_breakdown: Optional[ParsedBreakdown] = None
    parse_failure: Optional[Dict[str, Any]] = None  # Error info if parsing failed
    formalized_breakdown: Optional[Union[ParsedBreakdown, Dict[str, Any]]] = None  # Formalized Lean code (OOP or dict)
    theorem_prover_results: Optional[Dict[str, Any]] = None  # Theorem proof attempts
    lemma_prover_results: Optional[Dict[str, Any]] = None  # Lemma proof attempts
    theorem_full_records: Optional[Dict[str, Any]] = None  # Theorem reasoning
    lemma_full_records: Optional[Dict[str, Any]] = None  # Lemma reasoning
    formal_statement: Optional[str] = None
    informal_prefix: Optional[str] = None
    informal_solution: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    detailed_cost: Optional[Dict[str, Any]] = None
    lean4_code: Optional[str] = None
    name: Optional[str] = None
    component_costs: Optional[Dict[str, Dict[str, int]]] = None  # Token costs by component: {"breakdown": {...}, "parser": {...}, ...}
    _parser_detailed_cost: Optional[Dict[str, Any]] = None  # Temp storage for parser cost until ParsedBreakdown is created

    def is_solved(self) -> bool:
        """Returns True if this breakdown is fully solved."""
        if not self.parsed_breakdown:
            return False
        return self.parsed_breakdown.is_fully_solved()

    def get_solve_stats(self) -> tuple[bool, int, int]:
        """
        Returns solve status with lemma usage statistics for the breakdown.

        Returns:
            Tuple of (is_solved, num_used_lemmas, num_proven_used_lemmas)
            - is_solved: Whether the breakdown theorem is proven
            - num_used_lemmas: Total number of lemmas used transitively in proof
            - num_proven_used_lemmas: How many of those used lemmas are also proven

        Uses a single recursive traversal through the dependency graph.
        """
        if not self.parsed_breakdown:
            return (False, 0, 0)
        return self.parsed_breakdown.get_solve_stats()

    def get_used_lemmas_count(self) -> tuple:
        """
        Get count of used lemmas and how many are proven.

        Returns:
            Tuple of (used_lemmas_count, proven_used_lemmas_count)
            - used_lemmas_count: Number of lemmas actually used in theorem proof
            - proven_used_lemmas_count: Number of those used lemmas that are proven
        """
        if not self.parsed_breakdown:
            return 0, 0

        # Get all transitively used lemma IDs from the theorem
        used_lemma_ids = self.parsed_breakdown.get_theorem_used_lemmas(recursive=True)

        if not used_lemma_ids:
            return 0, 0

        # Count how many of the used lemmas are proven
        proven_count = 0
        for lemma_id in used_lemma_ids:
            if lemma_id in self.parsed_breakdown.lemmas:
                if self.parsed_breakdown.lemmas[lemma_id].is_solved():
                    proven_count += 1

        return len(used_lemma_ids), proven_count
    
    def get_cost(self, cost_type="cost") -> float:
        """Returns the cost of generating the breakdown itself.

        For sflops cost types, calculates from tokens * effective_params if not pre-computed.
        Breakdown generation uses oss-20b with effective_params = 3.6.
        """
        if cost_type == "prover_calls":
            return 0.0

        if not self.detailed_cost:
            return 0.0

        # If sflops requested and not already in detailed_cost, calculate from tokens
        if cost_type in ['input_sflops', 'output_sflops']:
            if cost_type not in self.detailed_cost:
                tokens_key = 'input_tokens' if cost_type == 'input_sflops' else 'output_tokens'
                tokens = self.detailed_cost.get(tokens_key, 0)
                return float(tokens * BREAKDOWN_EFFECTIVE_PARAMS)

        return float(self.detailed_cost.get(cost_type, 0.0))

    def get_total_cost(self, cost_type="cost", exclude_prover_calls: bool = False) -> float:
        """Returns the cost for this breakdown (detailed_cost + proof attempt costs). Can be in form of cost (dollars), tokens, or sflops.

        Args:
            cost_type: One of 'cost', 'input_tokens', 'output_tokens', 'input_sflops', 'output_sflops', 'prover_calls'
            exclude_prover_calls: If True, excludes proof attempt costs (useful for seeing just breakdown/formalization costs)
        """
        # Cost from breakdown itself
        breakdown_cost = self.get_cost(cost_type)
        # Cost from all proof attempts in parsed/formalized breakdown
        proof_cost = 0.0
        if self.parsed_breakdown:
            proof_cost += self.parsed_breakdown.get_total_cost(cost_type, exclude_prover_calls=exclude_prover_calls)

        return breakdown_cost + proof_cost

    def get_component_costs(self) -> Dict[str, Dict[str, int]]:
        """
        Returns a dict of token costs broken down by component.

        Structure:
        {
            "breakdown": {"input_tokens": N, "output_tokens": N},
            "breakdown_parser": {"input_tokens": N, "output_tokens": N},
            "formalization": {"input_tokens": N, "output_tokens": N},
            "prover": {"input_tokens": N, "output_tokens": N}
        }

        Returns the stored component_costs if available, otherwise returns a dict with zeros.
        """
        if self.component_costs:
            return self.component_costs

        # Return zero costs for all components if not populated
        return {
            "breakdown": {"input_tokens": 0, "output_tokens": 0},
            "breakdown_parser": {"input_tokens": 0, "output_tokens": 0},
            "formalization": {"input_tokens": 0, "output_tokens": 0},
            "prover": {"input_tokens": 0, "output_tokens": 0}
        }

    def get_component_output_tokens(self, component: str) -> int:
        """Returns the output token count for a specific component."""
        costs = self.get_component_costs()
        return costs.get(component, {}).get("output_tokens", 0)

    def get_cumulative_output_tokens(self, up_to_component: Optional[str] = None) -> int:
        """
        Get cumulative output tokens up to and including a specific component.

        Component order: 'breakdown' -> 'breakdown_parser' -> 'formalization' -> 'prover'

        Args:
            up_to_component: Optional component name to stop at. If None, returns all costs.
                           Valid values: 'breakdown', 'breakdown_parser', 'formalization', 'prover'

        Returns:
            Sum of output tokens from start up to (and including) the specified component.

        Examples:
            get_cumulative_output_tokens()                           # All components
            get_cumulative_output_tokens('formalization')            # Up to formalization (exclude prover)
            get_cumulative_output_tokens('breakdown_parser')         # Breakdown + breakdown_parser only
        """
        component_order = ['breakdown', 'breakdown_parser', 'formalization', 'prover']

        if up_to_component and up_to_component not in component_order:
            raise ValueError(f"Unknown component: {up_to_component}. Valid: {component_order}")

        costs = self.get_component_costs()
        total = 0

        for component in component_order:
            component_tokens = costs.get(component, {}).get("output_tokens", 0)
            total += component_tokens
            if up_to_component and component == up_to_component:
                break

        return total

    def generate_proof(self, parent_problem: Optional['Problem'] = None) -> tuple:
        """
        Generate a complete proof for this breakdown by calling theorem.generate_proof().

        Args:
            parent_problem: Optional parent Problem object (unused - kept for compatibility)

        Returns:
            Tuple of (proof_code: str, is_complete: bool)
            - proof_code: The generated Lean code (complete or with sorries)
            - is_complete: True if all lemmas were found and proved, False if any sorries added
        """
        if not self.parsed_breakdown:
            return "", False

        from seed_prover.utils import remove_axiom_declarations, strip_preamble, simplify_theorem_name_in_code

        # Get theorem proof
        theorem = self.parsed_breakdown.theorem

        # Call theorem.generate_proof() to get complete proof with all used lemmas
        theorem_proof, is_complete = theorem.generate_proof(lemmas_dict=self.parsed_breakdown.lemmas)

        if not theorem_proof:
            return "", False

        # Build proof content with header
        lines = [
            "-- Automatically generated proof file",
            "-- Problem: " + self.origin_problem_id,
            "-- Round: " + str(self.round_id),
            "-- Breakdown: " + str(self.breakdown_id),
        ]

        if not is_complete:
            lines.append("-- WARNING: This proof contains sorry statements for unprovable lemmas")

        lines.append("")
        lines.append(theorem_proof)

        proof_string = "\n".join(lines)
        return proof_string, is_complete

    def _collect_transitive_lemmas(self, used_lemma_ids: set, visited: set = None, parent_problem: Optional['Problem'] = None) -> set:
        """
        Recursively collect all lemmas transitively used by the given lemmas.

        Handles both direct proofs (in parsed_breakdown.lemmas) and recursive proofs
        (in lemma.recursive_attempt).

        Args:
            used_lemma_ids: Initial set of lemma IDs to expand
            visited: Set of already visited lemma IDs (for cycle prevention)
            parent_problem: Optional parent Problem to search recursive attempts

        Returns:
            Set of all lemma IDs needed (directly and transitively)
        """
        if visited is None:
            visited = set()

        all_needed = set(used_lemma_ids)

        for lemma_id in used_lemma_ids:
            if lemma_id in visited:
                continue
            visited.add(lemma_id)

            proof_code = None

            # First check direct proof in parsed_breakdown
            if lemma_id in self.parsed_breakdown.lemmas:
                lemma = self.parsed_breakdown.lemmas[lemma_id]
                best_attempt = lemma.get_best_attempt()
                if best_attempt and best_attempt.code:
                    proof_code = best_attempt.code

            # If no direct proof, check recursive_attempt
            if not proof_code and lemma_id in self.parsed_breakdown.lemmas:
                lemma = self.parsed_breakdown.lemmas[lemma_id]
                if lemma.recursive_attempt:
                    # Find the best solved breakdown in recursive attempt
                    for breakdown in lemma.recursive_attempt.breakdowns.values():
                        if breakdown.is_solved() and breakdown.parsed_breakdown:
                            best_attempt = breakdown.parsed_breakdown.theorem.get_best_attempt()
                            if best_attempt and best_attempt.code:
                                proof_code = best_attempt.code
                                break

            # Extract used lemmas from whichever proof we found
            if proof_code:
                from seed_prover.utils import extract_axiom_names, check_if_axiom_used

                axiom_names = extract_axiom_names(proof_code)
                for axiom_name in axiom_names:
                    if check_if_axiom_used(proof_code, axiom_name):
                        match = re.search(r'lemma(\d+)(?:_f\d+)?', axiom_name)
                        if match:
                            sub_lemma_id = int(match.group(1))
                            if sub_lemma_id not in visited:
                                # Recursively collect lemmas of this lemma
                                sub_lemmas = self._collect_transitive_lemmas({sub_lemma_id}, visited, parent_problem)
                                all_needed.update(sub_lemmas)

        return all_needed

    def _get_proof_for_lemma(self, lemma_id: int, parent_problem: Optional['Problem'] = None) -> Optional[tuple]:
        """
        Get the best proof for a lemma, checking both direct proofs and recursive attempts.

        Args:
            lemma_id: The lemma ID
            parent_problem: Optional parent Problem to search recursive attempts if direct proof fails

        Returns:
            Tuple of (proof_code, round_id) or None if not found
        """
        if lemma_id not in self.parsed_breakdown.lemmas:
            return None

        lemma = self.parsed_breakdown.lemmas[lemma_id]
        best_attempt = lemma.get_best_attempt()

        if best_attempt and best_attempt.code:
            # Check if it's complete (no sorries)
            if best_attempt.compilation_result and best_attempt.compilation_result.complete:
                return (best_attempt.code, self.round_id)

        # If direct proof failed or incomplete, try to find it in recursive attempts
        if parent_problem:
            recursive_proof = self._search_recursive_lemma_proof(parent_problem, lemma_id)
            if recursive_proof:
                return recursive_proof

        return None

    def _search_recursive_lemma_proof(self, parent_problem: 'Problem', lemma_id: int) -> Optional[tuple]:
        """
        Search for a lemma proof in the parent problem's recursive attempts.

        This is used when a lemma failed to prove directly in round 0 but was successfully
        solved recursively in round 1+.

        Args:
            parent_problem: The parent Problem with recursive attempts
            lemma_id: The lemma ID to find

        Returns:
            Tuple of (proof_code, round_id) or None if not found
        """
        # Build the UID for the failed lemma from round 0
        # Lemmas that get recursively broken down have origin_problem_id = lemma_uid
        for recursive_attempt in parent_problem.recursive_attempts:
            # Check if this recursive attempt is trying to prove our failed lemma
            # The origin_problem_id for recursive attempts is the UID of the lemma being proved
            if recursive_attempt.origin_problem_id and f"_l{lemma_id}_" in recursive_attempt.origin_problem_id:
                # Found the recursive attempt for this lemma
                # Now find the best solved breakdown
                for bd_key, breakdown in recursive_attempt.breakdowns.items():
                    # Only use solved breakdowns
                    if breakdown.is_solved() and breakdown.parsed_breakdown:
                        # Get the theorem proof from this solved breakdown
                        theorem = breakdown.parsed_breakdown.theorem
                        best_attempt = theorem.get_best_attempt()
                        if best_attempt and best_attempt.code:
                            # Check if it's complete
                            if best_attempt.compilation_result and best_attempt.compilation_result.complete:
                                # Generate complete proof for this recursive breakdown
                                # This will include all lemmas used by the recursive breakdown
                                recursive_proof, _ = breakdown.generate_proof(parent_problem=recursive_attempt)
                                if recursive_proof:
                                    return (recursive_proof, breakdown.round_id)

            # Also recursively search in nested recursive attempts
            nested_result = self._search_recursive_lemma_proof(recursive_attempt, lemma_id)
            if nested_result:
                return nested_result

        return None

    def _topological_sort_lemmas(self, lemma_ids: set, proof_map: dict) -> list:
        """
        Sort lemmas in topological order (dependencies first).

        Args:
            lemma_ids: Set of lemma IDs to sort
            proof_map: Map of lemma_id -> proof_code

        Returns:
            List of lemma IDs in topological order
        """
        from seed_prover.utils import extract_axiom_names, check_if_axiom_used

        # Build dependency graph
        dependencies = {lid: set() for lid in lemma_ids}

        for lemma_id in lemma_ids:
            if lemma_id in proof_map and lemma_id in self.parsed_breakdown.lemmas:
                proof_code = proof_map[lemma_id]
                # Find which other lemmas this lemma uses
                axiom_names = extract_axiom_names(proof_code)
                for axiom_name in axiom_names:
                    if check_if_axiom_used(proof_code, axiom_name):
                        match = re.search(r'lemma(\d+)(?:_f\d+)?', axiom_name)
                        if match:
                            dep_id = int(match.group(1))
                            if dep_id in lemma_ids:
                                dependencies[lemma_id].add(dep_id)

        # Kahn's algorithm for topological sort
        in_degree = {lid: 0 for lid in lemma_ids}
        for lid in lemma_ids:
            for dep_id in dependencies[lid]:
                in_degree[dep_id] += 1

        queue = deque([lid for lid in lemma_ids if in_degree[lid] == 0])
        sorted_lemmas = []

        while queue:
            lid = queue.popleft()
            sorted_lemmas.append(lid)

            for other_id in lemma_ids:
                if lid in dependencies[other_id]:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        return sorted_lemmas

    def get_prover_path(self, recursive: bool = True) -> Optional[ProverPathNode]:
        """
        Build a prover path tree for this breakdown.

        Returns the prover path of the theorem in this breakdown, recursively including
        all used lemmas and their nested recursive attempts.

        Args:
            recursive: If True, follow recursive_attempts for unsolved lemmas

        Returns:
            ProverPathNode tree, or None if theorem has no proof
        """
        if not self.parsed_breakdown:
            return None

        # Get the theorem's prover path
        return self.parsed_breakdown.theorem.get_prover_path(
            lemmas_dict=self.parsed_breakdown.lemmas,
            recursive=recursive
        )

    def plot_prover_path(self, recursive: bool = True) -> Optional[Any]:
        """
        Plot the prover path tree as a matplotlib figure.

        Returns a matplotlib figure that can be displayed via st.pyplot().

        Args:
            recursive: If True, shows complete proof journey including all recursion depths.
                      If False, shows only lemmas directly proven in this round (no recursion).

        Colors:
        - Blue: Theorem node (main or recursive)
        - Green: Directly solved lemma (has passing+complete proof)
        - Red: Unsolved lemma (no direct proof, but may have recursive solution)

        Returns:
            matplotlib.figure.Figure, or None if no prover path available
        """
        import matplotlib.pyplot as plt

        path = self.get_prover_path(recursive=recursive)
        if not path:
            return None

        # Build layout using BFS from root node
        layers = {}
        visited = []  # Use list instead of set since ProverPathNode is not hashable
        current_layer = [path]
        layer_num = 0

        while current_layer:
            layers[layer_num] = current_layer
            visited.extend(current_layer)
            next_layer = []

            for node in current_layer:
                for child in node.children:
                    if child not in visited and child not in next_layer:
                        next_layer.append(child)

            current_layer = next_layer
            layer_num += 1

        # Assign positions based on layers
        num_layers = len(layers)
        max_width = max(len(nodes) for nodes in layers.values())
        # Tighter spacing for narrow/shallow trees
        vert_spacing = 0.6 if max_width <= 2 else 0.8
        horiz_spread = max(1.5, min(5, max_width * 1.2))

        positions = {}
        for layer_idx, layer_nodes in layers.items():
            y = -layer_idx * vert_spacing
            num_nodes = len(layer_nodes)

            for node_idx, node in enumerate(layer_nodes):
                if num_nodes == 1:
                    # Single node - center it under parent
                    parents = [p for p in visited if node in p.children]
                    if parents and parents[0] in positions:
                        x = positions[parents[0]][0]
                    else:
                        x = 0
                else:
                    x = -(horiz_spread / 2) + (node_idx * horiz_spread / (num_nodes - 1))
                positions[node] = (x, y)

        # Dynamic sizing: compact for small trees, larger for complex ones
        fig_height = max(3, min(12, 1.5 + num_layers * 1.2))
        fig_width = max(6, min(14, 6 + max_width * 1.0))

        # Create figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Draw edges
        edges = []
        def collect_edges(node):
            """Collect all edges from the tree."""
            for child in node.children:
                edges.append((node, child))
                collect_edges(child)
        collect_edges(path)

        for parent_node, child_node in edges:
            if parent_node in positions and child_node in positions:
                x1, y1 = positions[parent_node]
                x2, y2 = positions[child_node]

                dx = x2 - x1
                dy = y2 - y1
                dist = (dx**2 + dy**2)**0.5
                if dist > 0:
                    norm_dx = dx / dist
                    norm_dy = dy / dist
                    src_radius = 0.22
                    tgt_radius = 0.22
                    start_x = x1 + norm_dx * src_radius
                    start_y = y1 + norm_dy * src_radius
                    end_x = x2 - norm_dx * tgt_radius
                    end_y = y2 - norm_dy * tgt_radius
                    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.6))

        # Draw nodes with colors
        color_map = {
            'blue': '#4169E1',    # Royal blue for theorems
            'green': '#27AE60',   # Green for directly solved
            'gold': '#F39C12',    # Gold for solved via recursion
            'red': '#E74C3C'      # Red for unsolved
        }

        for node in visited:
            if node in positions:
                x, y = positions[node]
                color = color_map.get(node.color, '#808080')

                # Draw circle
                circle = plt.Circle((x, y), 0.22, color=color, ec='black', linewidth=1.5, zorder=3)
                ax.add_patch(circle)

                # Draw label
                ax.text(x, y, node.label, ha='center', va='center', fontsize=12, fontweight='bold',
                       color='white', zorder=4)

        # Set axis properties
        if positions:
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
            ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        return fig

    def simulate(
        self,
        seed: int,
        max_depth: Optional[int] = None,
        search_policy: str = "sequential",
        strategy: Optional[Any] = None,
        tracker: Optional["ProofAttemptTracker"] = None,
    ) -> "Breakdown":
        """
        Simulate breakdown solving: theorem + all lemmas.

        Args:
            seed: Random seed for reproducibility
            max_depth: Maximum depth/attempts to try (None = unlimited)
            search_policy: Either "exponential" or "sequential"
            strategy: Optional strategy object to guide simulation decisions (will be cloned)
            tracker: Optional ProofAttemptTracker for recording attempts

        Returns:
            New Breakdown with all lemmas but filtered proof attempts.
            The breakdown may or may not be solved - check is_solved() after.
        """
        # Check if breakdown is parsed - return empty breakdown if not
        if not self.parsed_breakdown:
            return Breakdown(
                origin_problem_id=self.origin_problem_id,
                round_id=self.round_id,
                breakdown_id=self.breakdown_id,
                problem_id=self.problem_id,
                parsed_breakdown=None,
                component_costs=self.component_costs,
                detailed_cost=self.detailed_cost,
            )

        import random
        from .formalization import Formalization

        rng = random.Random(seed)

        # Track breakdown attempt
        if tracker:
            tracker.set_breakdown_attempted()

        # Step 1: Simulate theorem (always returns a theorem, may be failing)
        theorem_seed = rng.randint(0, 2**31 - 1)
        new_theorem = self.parsed_breakdown.theorem.simulate(
            theorem_seed,
            max_depth,
            search_policy,
            strategy=strategy,
            tracker=tracker,
            breakdown_id=self.breakdown_id,
            lemma_id=-1,
        )

        # Step 2: Get used lemmas from theorem attempt (if theorem succeeded)
        used_lemma_ids = set()
        if new_theorem.is_solved():
            best_attempt = new_theorem.get_best_attempt()
            if best_attempt and best_attempt.used_lemma_ids:
                used_lemma_ids = best_attempt.used_lemma_ids

        # Step 3: Simulate ALL lemmas (always returns a lemma, may be failing)
        new_lemmas = {}
        for lemma_id, lemma in self.parsed_breakdown.lemmas.items():
            lemma_seed = rng.randint(0, 2**31 - 1)

            new_lemmas[lemma_id] = lemma.simulate(
                lemma_seed,
                max_depth,
                search_policy,
                strategy=strategy,
                tracker=tracker,
                breakdown_id=self.breakdown_id,
            )

        # Return new breakdown with all simulated components
        # is_solved() will naturally return False if theorem or used lemmas failed
        new_parsed_bd = ParsedBreakdown(
            theorem=new_theorem,
            lemmas=new_lemmas,
        )

        return Breakdown(
            origin_problem_id=self.origin_problem_id,
            round_id=self.round_id,
            breakdown_id=self.breakdown_id,
            problem_id=self.problem_id,
            parsed_breakdown=new_parsed_bd,
            formalized_breakdown=new_parsed_bd,  # Also set formalized_breakdown
            component_costs=self.component_costs,
            detailed_cost=self.detailed_cost,
        )

    def simulate_action(
        self,
        action: Any,
        tracker: Optional[Any] = None,
        seed: Optional[int] = None,
        max_depth: Optional[int] = None,
        search_policy: str = "sequential",
    ) -> dict:
        """
        Simulate a single agentic action on this breakdown.

        Supports:
        - ATTEMPT_8B: Try proving with 8b model
        - ATTEMPT_32B: Try proving with 32b model
        - CORRECTION_8B: Try correcting last 8b attempt
        - CORRECTION_32B: Try correcting last 32b attempt

        Args:
            action: ProofSearchAction
            tracker: Attempt tracker
            seed: Random seed
            max_depth: Max attempts for this action
            search_policy: "sequential" or "exponential"

        Returns:
            Dict with cost, success, proof_attempt, etc.
        """
        from seed_prover.simulations.rl.actions import ActionType

        if not hasattr(action, 'action_type'):
            raise ValueError("Action must have action_type attribute")

        action_type = action.action_type

        # Delegate to theorem simulate with appropriate params
        if action_type in (ActionType.ATTEMPT_8B, ActionType.ATTEMPT_32B, ActionType.CORRECTION_8B, ActionType.CORRECTION_32B):
            return self._simulate_theorem_action(
                action_type=action_type,
                tracker=tracker,
                seed=seed,
                max_depth=max_depth or 1,
                search_policy=search_policy,
            )

        raise ValueError(f"Breakdown.simulate_action does not support {action_type}")

    def _get_organized_attempts(self, seed: int) -> dict:
        """
        Organize and shuffle attempts for RL simulation.

        NOTE: This method ONLY organizes THEOREM attempts. For lemma attempts,
        use _simulate_theorem_action() which handles per-lemma attempt organization.

        Returns:
            dict with:
                - attempts_8b: List[ProofAttempt] (shuffled)
                - attempts_32b_paired: List[Tuple[ProofAttempt, List[ProofAttempt]]]
                  Each tuple is (base_32b_attempt, [corrections]) (shuffled)
        """
        # Cache key based on seed
        cache_key = f"_rl_attempts_{seed}"
        if hasattr(self, cache_key):
            return getattr(self, cache_key)

        import random

        if not self.parsed_breakdown or not self.parsed_breakdown.theorem:
            return {"attempts_8b": [], "attempts_32b_paired": []}

        # Get all proof attempts from all formalizations (theorem only)
        all_attempts = []
        for formalization in self.parsed_breakdown.theorem.formalizations:
            all_attempts.extend(formalization.proof_attempts)

        if not all_attempts:
            return {"attempts_8b": [], "attempts_32b_paired": []}

        rng = random.Random(seed)

        # Helper to check model size from model_config_path
        def is_8b(attempt):
            return attempt.model_config_path and "8b" in attempt.model_config_path.lower()

        def is_32b(attempt):
            return attempt.model_config_path and "32b" in attempt.model_config_path.lower()

        # 1. Get and shuffle 8b attempts
        attempts_8b = [a for a in all_attempts
                      if is_8b(a) and a.correction_round_id == 0]
        rng.shuffle(attempts_8b)

        # 2. Get base 32b attempts (correction_round_id == 0)
        base_32b_attempts = [a for a in all_attempts
                            if is_32b(a) and a.correction_round_id == 0]

        # 3. For each base 32b attempt, find and group its corrections
        attempts_32b_paired = []
        for base_attempt in base_32b_attempts:
            # Find all corrections for this attempt (matching initial_attempt_index, not attempt_id!)
            corrections = [a for a in all_attempts
                          if is_32b(a) and
                             a.correction_round_id > 0 and
                             getattr(a, 'initial_attempt_index', None) == base_attempt.attempt_id]

            # Sort corrections by correction_round_id to maintain order
            corrections.sort(key=lambda x: x.correction_round_id)

            attempts_32b_paired.append((base_attempt, corrections))

        # 4. Shuffle the pairs
        rng.shuffle(attempts_32b_paired)

        # Cache the result
        organized = {
            "attempts_8b": attempts_8b,
            "attempts_32b_paired": attempts_32b_paired,
        }
        setattr(self, cache_key, organized)
        return organized

    def _simulate_theorem_action(
        self,
        action_type: Any,
        tracker: Optional[Any],
        seed: Optional[int],
        max_depth: int,
        search_policy: str,
    ) -> dict:
        """
        Simulate a single proof attempt for the current target lemma/theorem.

        Uses the breakdown's current_target_lemma_id to determine what to prove.
        When successful, extracts used_lemma_ids and updates the proving queue.

        Attempts are organized per breakdown:
        - 8b attempts: shuffled list of (base, [corrections]) pairs
          - ATTEMPT_8B uses the base attempt from next pair
          - CORRECTION_8B uses next correction from CURRENT pair
        - 32b attempts: shuffled list of (base, [corrections]) pairs
          - ATTEMPT_32B uses the base attempt from next pair
          - CORRECTION_32B uses next correction from CURRENT pair
        """
        if not self.parsed_breakdown or not self.parsed_breakdown.theorem:
            return {
                "cost": 0.0,
                "success": False,
                "state": {"error": "No parsed breakdown or theorem"},
            }

        # Get current target from tracker
        current_breakdown = tracker.breakdowns.get(tracker.current_breakdown_id) if tracker else None
        target_lemma_id = current_breakdown.current_target_lemma_id if current_breakdown else -1

        from seed_prover.simulations.rl.actions import ActionType as AT

        # Get all proof attempts from all formalizations for target lemma/theorem
        all_attempts = []
        if target_lemma_id == -1:
            # Theorem attempts
            for formalization in self.parsed_breakdown.theorem.formalizations:
                all_attempts.extend([a for a in formalization.proof_attempts if a.lemma_id == -1])
        else:
            # Lemma attempts
            if target_lemma_id in self.parsed_breakdown.lemmas:
                lemma = self.parsed_breakdown.lemmas[target_lemma_id]
                for formalization in lemma.formalizations:
                    all_attempts.extend([a for a in formalization.proof_attempts if a.lemma_id == target_lemma_id])

        if not all_attempts:
            return {
                "cost": 0.0,
                "success": False,
                "state": {"error": f"No proof attempts available for lemma_id={target_lemma_id}"},
            }

        # Organize attempts by action type (using same logic as before)
        import random
        rng = random.Random(seed or 0)

        # Helper to check model size from model_config_path
        def is_8b(attempt):
            return attempt.model_config_path and "8b" in attempt.model_config_path.lower()

        def is_32b(attempt):
            return attempt.model_config_path and "32b" in attempt.model_config_path.lower()

        # 1. Get base 8b attempts (correction_round_id == 0) for this lemma
        base_8b_attempts = [a for a in all_attempts
                          if is_8b(a) and a.correction_round_id == 0]

        # 2. For each base 8b attempt, find and group its corrections
        attempts_8b_paired = []
        for base_attempt in base_8b_attempts:
            # Find all corrections for this attempt (matching initial_attempt_index, not attempt_id!)
            corrections = [a for a in all_attempts
                          if is_8b(a) and
                             a.correction_round_id > 0 and
                             getattr(a, 'initial_attempt_index', None) == base_attempt.attempt_id]

            # Sort corrections by correction_round_id to maintain order
            corrections.sort(key=lambda x: x.correction_round_id)

            attempts_8b_paired.append((base_attempt, corrections))

        # 3. Shuffle the 8b pairs
        rng.shuffle(attempts_8b_paired)

        # 4. Get base 32b attempts (correction_round_id == 0) for this lemma
        base_32b_attempts = [a for a in all_attempts
                            if is_32b(a) and a.correction_round_id == 0]

        # 5. For each base 32b attempt, find and group its corrections
        attempts_32b_paired = []
        for base_attempt in base_32b_attempts:
            # Find all corrections for this attempt (matching initial_attempt_index, not attempt_id!)
            corrections = [a for a in all_attempts
                          if is_32b(a) and
                             a.correction_round_id > 0 and
                             getattr(a, 'initial_attempt_index', None) == base_attempt.attempt_id]

            # Sort corrections by correction_round_id to maintain order
            corrections.sort(key=lambda x: x.correction_round_id)

            attempts_32b_paired.append((base_attempt, corrections))

        # 6. Shuffle the 32b pairs
        rng.shuffle(attempts_32b_paired)

        # Get current state from tracker
        current_breakdown = tracker.breakdowns.get(tracker.current_breakdown_id) if tracker else None

        if action_type == AT.ATTEMPT_8B:
            # Index = number of 8b attempts made so far FOR THIS LEMMA (determines which pair)
            # Count ATTEMPT_8B actions for the current target lemma
            pair_idx = 0
            if current_breakdown:
                pair_idx = sum(1 for a in current_breakdown.attempts
                              if a.action_type == AT.ATTEMPT_8B and a.lemma_id == target_lemma_id)

            if pair_idx >= len(attempts_8b_paired):
                return {
                    "cost": 0.0,
                    "success": False,
                    "state": {"error": f"No more 8b attempts available for lemma_id={target_lemma_id}"},
                }

            # Use the base attempt from this pair
            base_attempt, corrections = attempts_8b_paired[pair_idx]
            selected_attempt = base_attempt

        elif action_type == AT.ATTEMPT_32B:
            # Index = number of 32b attempts made so far FOR THIS LEMMA (determines which pair)
            # Count ATTEMPT_32B actions for the current target lemma
            pair_idx = 0
            if current_breakdown:
                pair_idx = sum(1 for a in current_breakdown.attempts
                              if a.action_type == AT.ATTEMPT_32B and a.lemma_id == target_lemma_id)

            if pair_idx >= len(attempts_32b_paired):
                return {
                    "cost": 0.0,
                    "success": False,
                    "state": {"error": f"No more 32b attempts available for lemma_id={target_lemma_id}"},
                }

            # Use the base attempt from this pair
            base_attempt, corrections = attempts_32b_paired[pair_idx]
            selected_attempt = base_attempt

        elif action_type == AT.CORRECTION_8B:
            # We need to correct the LAST 8b attempt made FOR THIS LEMMA
            # Count ATTEMPT_8B actions for the current target lemma
            num_8b_made = 0
            if current_breakdown:
                num_8b_made = sum(1 for a in current_breakdown.attempts
                                  if a.action_type == AT.ATTEMPT_8B and a.lemma_id == target_lemma_id)

            if num_8b_made == 0:
                return {
                    "cost": 0.0,
                    "success": False,
                    "state": {"error": f"No 8b attempt to correct yet for lemma_id={target_lemma_id}"},
                }

            # Get the last pair (the one we're currently correcting)
            last_pair_idx = num_8b_made - 1
            base_attempt, corrections = attempts_8b_paired[last_pair_idx]

            # Track corrections per pair using a cache on the breakdown object
            # Key format: "corrections_8b_for_pair_{breakdown_id}_{lemma_id}_{pair_idx}"
            corrections_key = f"_rl_corrections_8b_pair_{tracker.current_breakdown_id}_{target_lemma_id}_{last_pair_idx}"

            # Get or initialize the correction count for this specific pair
            if not hasattr(self, corrections_key):
                setattr(self, corrections_key, 0)

            correction_idx = getattr(self, corrections_key)

            if correction_idx >= len(corrections):
                return {
                    "cost": 0.0,
                    "success": False,
                    "state": {"error": f"No more 8b corrections available for lemma_id={target_lemma_id} attempt {last_pair_idx} ({correction_idx} tried, {len(corrections)} available)"},
                }

            selected_attempt = corrections[correction_idx]

            # Increment the correction count for this pair
            setattr(self, corrections_key, correction_idx + 1)

        elif action_type == AT.CORRECTION_32B:
            # We need to correct the LAST 32b attempt made FOR THIS LEMMA
            # Count ATTEMPT_32B actions for the current target lemma
            num_32b_made = 0
            if current_breakdown:
                num_32b_made = sum(1 for a in current_breakdown.attempts
                                  if a.action_type == AT.ATTEMPT_32B and a.lemma_id == target_lemma_id)

            if num_32b_made == 0:
                return {
                    "cost": 0.0,
                    "success": False,
                    "state": {"error": f"No 32b attempt to correct yet for lemma_id={target_lemma_id}"},
                }

            # Get the last pair (the one we're currently correcting)
            last_pair_idx = num_32b_made - 1
            base_attempt, corrections = attempts_32b_paired[last_pair_idx]

            # Track corrections per pair using a cache on the breakdown object
            # Key format: "corrections_for_pair_{breakdown_id}_{lemma_id}_{pair_idx}"
            corrections_key = f"_rl_corrections_pair_{tracker.current_breakdown_id}_{target_lemma_id}_{last_pair_idx}"

            # Get or initialize the correction count for this specific pair
            if not hasattr(self, corrections_key):
                setattr(self, corrections_key, 0)

            correction_idx = getattr(self, corrections_key)

            if correction_idx >= len(corrections):
                return {
                    "cost": 0.0,
                    "success": False,
                    "state": {"error": f"No more corrections available for lemma_id={target_lemma_id} attempt {last_pair_idx} ({correction_idx} tried, {len(corrections)} available)"},
                }

            selected_attempt = corrections[correction_idx]

            # Increment the correction count for this pair
            setattr(self, corrections_key, correction_idx + 1)

        else:
            return {
                "cost": 0.0,
                "success": False,
                "state": {"error": f"Unsupported action type: {action_type}"},
            }

        # Check if it succeeded
        success = (selected_attempt.compilation_result and
                  selected_attempt.compilation_result.is_successful())

        # If successful, update lemma proving state
        if success and current_breakdown:
            # Extract used lemma IDs - if not pre-computed, extract from code
            if selected_attempt.used_lemma_ids is not None:
                used_lemma_ids = selected_attempt.used_lemma_ids
            else:
                # Extract from code using get_used_lemmas (pass lemmas_dict for accurate extraction)
                used_lemma_ids = selected_attempt.get_used_lemmas(
                    lemmas_dict=self.parsed_breakdown.lemmas if self.parsed_breakdown else None,
                    recursive=False  # Only direct dependencies for now
                )
            current_breakdown.on_successful_proof(target_lemma_id, used_lemma_ids)

        # Get cost from the actual proof attempt
        cost = selected_attempt.get_cost("output_sflops")

        # Get num_errors from compilation_summary.total_errors
        # Default to None if not available to distinguish from "0 errors"
        num_errors = None
        if selected_attempt.compilation_summary:
            num_errors = selected_attempt.compilation_summary.get('total_errors')

        # Get proof_length (in lines) from reasoning_summary
        proof_length = 0
        if selected_attempt.reasoning_summary:
            proof_length = selected_attempt.reasoning_summary.get('proof_length', 0)

        # Check if fully solved (theorem + all lemmas proven)
        fully_solved = current_breakdown.is_fully_solved() if (success and current_breakdown) else False

        return {
            "cost": cost,
            "success": success,
            "fully_solved": fully_solved,
            "attempt": selected_attempt,
            "state": {
                "solved": success,
                "fully_solved": fully_solved,
                "proof_length": proof_length,
                "num_errors": num_errors,
                "target_lemma_id": target_lemma_id,
                "lemma_queue_size": len(current_breakdown.lemma_queue) if current_breakdown else 0,
                "num_proven_lemmas": len(current_breakdown.proven_lemmas) if current_breakdown else 0,
            },
        }

    def to_dict(self, minified: bool = False) -> dict:
        """Convert to dictionary representation (flat, no nested parsed_breakdown)."""
        result = {
            "metadata": {
                "origin_problem_id": self.origin_problem_id,
                "round_id": self.round_id,
                "breakdown_id": self.breakdown_id,
                "parent_problem_id": self.parent_problem_id
            },
            "problem_id": self.problem_id,
            "informal_breakdown": self.informal_breakdown,
            "parse_failure": self.parse_failure,
            "formal_statement": self.formal_statement,
            "informal_prefix": self.informal_prefix,
            "informal_solution": self.informal_solution,
            "tags": self.tags,
            "detailed_cost": self.detailed_cost,
            "component_costs": self.component_costs,
            "parser_detailed_cost": self.parsed_breakdown.detailed_cost if self.parsed_breakdown else None,
            "name": self.name
        }

        if not minified:
            result["breakdown_prompt"] = self.breakdown_prompt
            result["informal_breakdown_reasoning"] = self.informal_breakdown_reasoning
    
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'Breakdown':
        """Reconstruct Breakdown from dictionary representation (without parsed_breakdown)."""
        metadata = data["metadata"]
        return cls(
            problem_id=data["problem_id"],
            origin_problem_id=metadata["origin_problem_id"],
            round_id=metadata["round_id"],
            breakdown_id=metadata["breakdown_id"],
            parent_problem_id=metadata.get("parent_problem_id"),
            informal_breakdown=data.get("informal_breakdown"),
            breakdown_prompt=data.get("breakdown_prompt"),
            informal_breakdown_reasoning=data.get("informal_breakdown_reasoning"),
            parsed_breakdown=None,  # Loaded separately from lemmas/theorems
            parse_failure=data.get("parse_failure"),
            formal_statement=data.get("formal_statement"),
            informal_prefix=data.get("informal_prefix"),
            informal_solution=data.get("informal_solution"),
            tags=data.get("tags", []),
            detailed_cost=data.get("detailed_cost"),
            component_costs=data.get("component_costs"),
            name=data.get("name"),
            _parser_detailed_cost=data.get("parser_detailed_cost")
        )
