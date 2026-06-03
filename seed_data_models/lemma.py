"""
Lemma data model.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Set, TYPE_CHECKING, Any
import re
import random

from .formalization import Formalization
from .proof_attempt import ProofAttempt
from .exceptions_and_helpers import SimulationFailure, ProverPathNode, _try_formalization

if TYPE_CHECKING:
    from .simulations.attempt_tracker import ProofAttemptTracker


@dataclass
class Lemma:
    """Represents a lemma with statement and formalizations.

    The statement, assumptions, and proof_idea are shared across all formalizations.
    Different formalizations represent different samples from the formalizer model.

    For lemmas that failed direct proving, recursive_attempt can link to a separate Problem
    where the lemma is treated as the origin problem and recursively broken down and proven.
    """
    lemma_id: int
    statement: str
    assumptions: Optional[str] = None  # Natural language assumptions
    proof_idea: Optional[str] = None  # Proof strategy (from parsed breakdown)
    dependencies: List[int] = field(default_factory=list)  # List of lemma IDs this lemma depends on
    formalizations: List[Formalization] = field(default_factory=list)
    recursive_attempt: Optional['Problem'] = None  # For round 1+: recursive proving of this failed lemma

    def get_best_formalization(self) -> Optional[Formalization]:
        """
        Returns the best formalization with priority:
        1. Formalization with passing & complete proof attempt (proven)
        2. Formalization with passing & validated status
        3. Formalization with passing compilation
        4. First formalization (fallback)

        Returns None if no formalizations exist.
        """
        if not self.formalizations:
            return None

        # Priority 1: Proven formalization (has passing & complete proof)
        for form in self.formalizations:
            if form.is_proven():
                return form

        # Priority 2: Passing & validated
        for form in self.formalizations:
            if form.compilation_pass and form.validation_pass:
                return form

        # Priority 3: Passing compilation
        for form in self.formalizations:
            if form.compilation_pass:
                return form

        # Priority 4: Fallback to first
        return self.formalizations[0] if self.formalizations else None

    def get_best_attempt(self, lemmas_dict: Optional[Dict[int, 'Lemma']] = None) -> Optional[ProofAttempt]:
        """
        Returns the best proof attempt from the best formalization.

        Args:
            lemmas_dict: Optional dict of lemmas to check if dependencies are solved
        """
        best_form = self.get_best_formalization()
        if not best_form:
            return None
        return best_form.get_best_attempt(lemmas_dict=lemmas_dict)

    def is_solved(self) -> bool:
        """
        Returns True if lemma is solved through either:
        1. Direct formalization (formalization has passing & complete proof), OR
        2. Recursive attempt (lemma is recursively proven in round 1+)

        Checks formalization first, then falls back to recursive_attempt.
        In practice, only one path should succeed (recursive is used when formalization fails).
        """
        # Check formalization first - must be proven (passing & complete)
        best_form = self.get_best_formalization()
        if best_form and best_form.is_proven():
            return True

        # If formalization failed, check recursive attempt
        if self.recursive_attempt is not None:
            return self.recursive_attempt.is_solved()

        return False

    def get_solve_stats(self, lemmas_dict: Optional[Dict[int, 'Lemma']] = None, _visited: Optional[Set[int]] = None) -> tuple[bool, int, int]:
        """
        Returns solve status with lemma usage statistics.

        Returns:
            Tuple of (is_solved, num_used_lemmas, num_proven_used_lemmas)
            - is_solved: Whether this lemma is proven
            - num_used_lemmas: Total number of lemmas used transitively in proof
            - num_proven_used_lemmas: How many of those used lemmas are also proven

        Uses a single recursive traversal through the dependency graph.
        """
        if _visited is None:
            _visited = set()

        is_solved = self.is_solved()

        if not is_solved or not lemmas_dict:
            return (is_solved, 0, 0)

        # Prevent cycles
        if self.lemma_id in _visited:
            return (True, 0, 0)  # This lemma already counted
        _visited.add(self.lemma_id)

        # Get directly used lemmas (non-recursive to control recursion ourselves)
        used_lemma_ids = self.get_used_lemmas(lemmas_dict=lemmas_dict, recursive=False)

        total_used = 0
        total_proven = 0

        for lemma_id in used_lemma_ids:
            if lemma_id in lemmas_dict and lemma_id not in _visited:
                lemma = lemmas_dict[lemma_id]
                sub_is_solved, sub_used, sub_proven = lemma.get_solve_stats(lemmas_dict, _visited)
                total_used += 1 + sub_used  # Count this lemma + its dependencies
                total_proven += (1 if sub_is_solved else 0) + sub_proven

        return (is_solved, total_used, total_proven)

    def is_formalized(self) -> bool:
        """
        Returns True if this lemma has at least one formalization that compiled successfully.

        This checks if ANY formalization in the formalizations list has compilation_pass=True.
        This is the key method for determining if a lemma has been successfully formalized.
        """
        if not self.formalizations:
            return False
        return any(form.compilation_pass for form in self.formalizations)

    def has_valid_formalization(self) -> bool:
        """
        Returns True if this lemma has at least one formalization that both compiled and validated.

        This checks if ANY formalization has compilation_pass=True AND validation_pass=True.
        Use this when validation is enabled to ensure quality formalizations.
        """
        if not self.formalizations:
            return False
        return any(form.compilation_pass and form.validation_pass for form in self.formalizations)

    def get_formalization_status(self) -> str:
        """
        Returns a human-readable status of the lemma's formalization.

        Returns:
            'proven': Has a formalization with a passing & complete proof
            'valid': Has a formalization that compiled & validated but no proof yet
            'compiling': Has a formalization that compiled but didn't validate
            'no_compilation': No formalization compiled
            'no_formalizations': No formalizations exist
        """
        if not self.formalizations:
            return 'no_formalizations'

        # Check if any formalization is proven
        for form in self.formalizations:
            if form.is_proven():
                return 'proven'

        # Check if any formalization is valid (compiled & validated)
        for form in self.formalizations:
            if form.compilation_pass and form.validation_pass:
                return 'valid'

        # Check if any formalization compiled
        for form in self.formalizations:
            if form.compilation_pass:
                return 'compiling'

        # No formalization compiled
        return 'no_compilation'

    def get_used_lemmas(self, lemmas_dict: Optional[Dict[int, 'Lemma']] = None, recursive: bool = False) -> Set[int]:
        """
        Returns the set of lemma IDs used in this lemma's best proof.

        Args:
            lemmas_dict: Optional dictionary of lemmas for recursive lookup
            recursive: If True, include transitive lemma dependencies

        Returns:
            Set of lemma IDs (integers) used
        """
        best = self.get_best_attempt(lemmas_dict=lemmas_dict)
        if not best:
            return set()
        return best.get_used_lemmas(lemmas_dict=lemmas_dict, recursive=recursive)

    def get_total_cost(self, cost_type, exclude_prover_calls: bool = False) -> float:
        """Returns the sum of costs from all formalizations.

        Args:
            cost_type: Type of cost to return (e.g., 'cost', 'output_sflops', 'input_tokens')
            exclude_prover_calls: If True, excludes proof attempt costs
        """
        return sum(form.get_total_cost(cost_type, exclude_prover_calls=exclude_prover_calls) for form in self.formalizations)

    def generate_proof(self, lemmas_dict: Optional[Dict[int, 'Lemma']] = None, formalization_id: Optional[int] = None) -> str:
        """
        Generate complete proof for this lemma, including all lemmas it uses.

        This method generates the full proof code for a lemma by:
        1. Getting direct proof (if available) OR recursive_attempt proof
        2. Extracting all lemmas used in the proof
        3. Recursively generating proofs for used lemmas
        4. Combining them in topological order
        5. Appending this lemma's proof

        Args:
            lemmas_dict: Dictionary of all lemmas in the breakdown
            formalization_id: Specific formalization ID to use (if provided)

        Returns:
            Complete Lean code with all used lemmas and this lemma's proof
        """
        from seed_prover.utils import extract_axiom_names, check_if_axiom_used, remove_axiom_declarations, strip_preamble

        # Try to use specific formalization if requested
        lemma_code = None

        if formalization_id is not None and len(self.formalizations) > formalization_id:
            form = self.formalizations[formalization_id]
            best_attempt = form.get_best_attempt() if form else None
            if best_attempt and best_attempt.code and best_attempt.compilation_result and best_attempt.compilation_result.complete:
                lemma_code = best_attempt.code

        # Fallback: try best attempt
        if not lemma_code:
            best_attempt = self.get_best_attempt(lemmas_dict=lemmas_dict)
            if best_attempt and best_attempt.code and best_attempt.compilation_result and best_attempt.compilation_result.complete:
                lemma_code = best_attempt.code

        # If no direct proof, try recursive_attempt
        if not lemma_code and self.recursive_attempt:
            # Call generate_proof on the recursive problem to get the full proof
            recursive_proof = self.recursive_attempt.generate_proof()
            # Simplify the LAST (main) theorem name in the recursive proof to match parent references
            # Pattern: match theorem declarations with the full breakdown name format
            # theorem <prefix>_r<num>_b<num>_l<num>_f<num>
            pattern = r'\btheorem\s+(\w+)_r(\d+)_b(\d+)_l(\d+)_f(\d+)'
            matches = list(re.finditer(pattern, recursive_proof))
            if matches:
                # Simplify only the LAST match
                last_match = matches[-1]
                prefix = last_match.group(1)      # e.g., algebra_ineq_nto1onlt2m1on
                lemma_id = last_match.group(4)    # e.g., 3
                # Use requested formalization_id if provided, otherwise use the one from recursive proof
                final_form_id = formalization_id if formalization_id is not None else int(last_match.group(5))
                simple_name = f'{prefix}_lemma{lemma_id}_f{final_form_id}'
                # Replace the last occurrence
                old_name = last_match.group(0)
                print(f"[DEBUG] Simplifying recursive lemma theorem name:")
                print(f"  Before: {old_name}")
                print(f"  After:  theorem {simple_name} (formalization_id={formalization_id})")
                recursive_proof = recursive_proof[:last_match.start()] + f'theorem {simple_name}' + recursive_proof[last_match.end():]
            return recursive_proof

        # If still no proof found, return empty
        if not lemma_code:
            return ""

        # If no lemmas_dict provided, just return this lemma's proof without dependencies
        if not lemmas_dict:
            cleaned_code = remove_axiom_declarations(lemma_code)
            cleaned_code = strip_preamble(cleaned_code)
            return cleaned_code

        # Extract axiom names and check which are actually used
        axiom_names = extract_axiom_names(lemma_code)
        used_axiom_names = set()
        for axiom_name in axiom_names:
            if check_if_axiom_used(lemma_code, axiom_name):
                used_axiom_names.add(axiom_name)

        # Extract lemma IDs from used axiom names
        used_lemma_ids = set()
        for axiom_name in used_axiom_names:
            match = re.search(r'lemma(\d+)(?:_f\d+)?', axiom_name)
            if match:
                used_lemma_id = int(match.group(1))
                if used_lemma_id in lemmas_dict:
                    used_lemma_ids.add(used_lemma_id)

        # Recursively collect all transitive dependencies
        all_needed = set(used_lemma_ids)
        visited = set()

        def collect_deps(lemma_ids: set) -> None:
            for lid in lemma_ids:
                if lid in visited or lid not in lemmas_dict:
                    continue
                visited.add(lid)
                dep_lemma = lemmas_dict[lid]
                dep_attempt = dep_lemma.get_best_attempt(lemmas_dict=lemmas_dict)
                if dep_attempt and dep_attempt.code:
                    dep_axioms = extract_axiom_names(dep_attempt.code)
                    for axiom_name in dep_axioms:
                        if check_if_axiom_used(dep_attempt.code, axiom_name):
                            match = re.search(r'lemma(\d+)(?:_f\d+)?', axiom_name)
                            if match:
                                sub_id = int(match.group(1))
                                if sub_id in lemmas_dict and sub_id not in visited:
                                    all_needed.add(sub_id)
                                    collect_deps({sub_id})

        collect_deps(used_lemma_ids)

        # Build proof code for all needed lemmas
        lines = []
        processed = set()

        def add_lemma_proof(lid: int) -> None:
            """Add a lemma proof and all its dependencies first."""
            if lid in processed or lid not in lemmas_dict:
                return

            dep_lemma = lemmas_dict[lid]
            dep_attempt = dep_lemma.get_best_attempt(lemmas_dict=lemmas_dict)
            if not dep_attempt or not dep_attempt.code:
                return

            # First add dependencies of this lemma
            dep_code = dep_attempt.code
            dep_axioms = extract_axiom_names(dep_code)
            for axiom_name in dep_axioms:
                if check_if_axiom_used(dep_code, axiom_name):
                    match = re.search(r'lemma(\d+)(?:_f\d+)?', axiom_name)
                    if match:
                        sub_id = int(match.group(1))
                        add_lemma_proof(sub_id)

            # Then add this lemma
            processed.add(lid)
            cleaned_code = remove_axiom_declarations(dep_code)
            cleaned_code = strip_preamble(cleaned_code)
            lines.append(f"-- Lemma {lid}")
            lines.append(cleaned_code)
            lines.append("")

        # Add all needed lemmas in dependency order
        for lid in all_needed:
            add_lemma_proof(lid)

        # Finally add this lemma's proof
        cleaned_self_code = remove_axiom_declarations(lemma_code)
        cleaned_self_code = strip_preamble(cleaned_self_code)
        lines.append(f"-- Lemma {self.lemma_id}")
        lines.append(cleaned_self_code)

        return "\n".join(lines)

    def get_prover_path(self, lemmas_dict: Optional[Dict[int, 'Lemma']] = None, recursive: bool = True) -> Optional[ProverPathNode]:
        """
        Build a prover path tree for this lemma.

        If directly solved:
        - Returns green "L{i}" node with children from used lemmas

        If NOT directly solved but recursive_attempt has proven theorem:
        - Returns gold "L{i}" node with the theorem tree from recursive_attempt
        - The recursive attempt's theorem is proven (axioms defined, even if some aren't directly proven)

        If NOT directly solved and no proven recursive_attempt:
        - Returns red "L{i}" node (no children if recursive=False)
        - If recursive=True, tries to recurse into the attempt anyway

        Args:
            lemmas_dict: Dictionary of lemmas in the breakdown
            recursive: If True, follow recursive_attempts for unsolved lemmas

        Returns:
            ProverPathNode tree, or None if no proof found
        """
        # Check if directly solved
        best_attempt = self.get_best_attempt(lemmas_dict=lemmas_dict)
        if best_attempt:
            # Directly solved - green node
            lemma_node = ProverPathNode(label=f"L{self.lemma_id}", color="green")

            # Add children from used lemmas
            if lemmas_dict:
                used_lemma_ids = best_attempt.get_used_lemmas(lemmas_dict=lemmas_dict, recursive=False)
                for used_lemma_id in sorted(used_lemma_ids):
                    if used_lemma_id in lemmas_dict:
                        child_lemma = lemmas_dict[used_lemma_id]
                        child_node = child_lemma.get_prover_path(lemmas_dict=lemmas_dict, recursive=recursive)
                        if child_node:
                            lemma_node.add_child(child_node)

            return lemma_node

        # Not directly solved - check recursive_attempt
        if recursive and self.recursive_attempt is not None:
            # Check if the recursive attempt has a proven theorem
            recursive_theorem_node = self.recursive_attempt.get_prover_path(recursive=recursive)

            if recursive_theorem_node:
                # Recursive attempt has a proven theorem - use gold color
                lemma_node = ProverPathNode(label=f"L{self.lemma_id}", color="gold")
                lemma_node.add_child(recursive_theorem_node)
                return lemma_node
            else:
                # Recursive attempt has no proven theorem - use red color
                lemma_node = ProverPathNode(label=f"L{self.lemma_id}", color="red")
                return lemma_node

        # No proof found - return red node
        return ProverPathNode(label=f"L{self.lemma_id}", color="red")

    def simulate(
        self,
        seed: int,
        max_depth: Optional[int] = None,
        search_policy: str = "sequential",
        strategy: Optional[Any] = None,
        tracker: Optional["ProofAttemptTracker"] = None,
        breakdown_id: int = 0,
        iteration_id: int = 0,
        origin_problem_id: Optional[str] = None,
    ) -> "Lemma":
        """
        Simulate lemma proving with configurable strategy.

        Args:
            seed: Random seed for reproducibility
            max_depth: Maximum depth/attempts to try (None = unlimited)
            search_policy: Either "exponential" or "sequential"
            strategy: Strategy instance to use (will be cloned for each formalization)
            tracker: Optional ProofAttemptTracker for recording attempts
            breakdown_id: Breakdown ID for tracking
            iteration_id: Iteration depth (0=theorem, 1=direct lemmas, etc.)
            origin_problem_id: Origin problem ID for tracking

        Returns:
            New Lemma with only selected formalization & attempts (may be empty/failing).
        """
        rng = random.Random(seed)
        formalizations = self.formalizations.copy()
        rng.shuffle(formalizations)

        # Collect all attempted formalizations with their attempts
        all_tried_formalizations = []

        for formalization_id, formalization in enumerate(formalizations):
            form_seed = rng.randint(0, 2**31 - 1)

            # Try with specified strategy
            selected_attempts = _try_formalization(
                formalization,
                form_seed,
                max_depth,
                search_policy,
                strategy,
                tracker=tracker,
                breakdown_id=breakdown_id,
                lemma_id=self.lemma_id,
                formalization_id=formalization_id,
                iteration_id=iteration_id,
                origin_problem_id=origin_problem_id,
            )

            if selected_attempts:
                new_form = Formalization(
                    id=formalization.id,
                    compilation_pass=formalization.compilation_pass,
                    validation_pass=formalization.validation_pass,
                    is_selected=formalization.is_selected,
                    detailed_cost=formalization.detailed_cost,
                    proof_attempts=selected_attempts,
                )
                all_tried_formalizations.append(new_form)

                # Check if we got a passing attempt (last attempt should be passing)
                if selected_attempts[-1].is_passing():
                    # Success! Return new Lemma with only this successful formalization
                    return Lemma(
                        statement=self.statement,
                        lemma_id=self.lemma_id,
                        proof_idea=self.proof_idea,
                        dependencies=self.dependencies,
                        formalizations=[new_form],
                        assumptions=self.assumptions,
                        recursive_attempt=None,  # Don't include recursive attempts in simulation
                    )

        # All failed - return lemma with all tried formalizations (all failing)
        return Lemma(
            statement=self.statement,
            lemma_id=self.lemma_id,
            proof_idea=self.proof_idea,
            dependencies=self.dependencies,
            formalizations=all_tried_formalizations,
            assumptions=self.assumptions,
            recursive_attempt=None,  # Don't include recursive attempts in simulation
        )

    def to_dict(self, origin_problem_id: str, round_id: int, breakdown_id: int):
        """Convert to dictionary representation (flat, no nested formalizations).

        Args:
            origin_problem_id: Origin problem ID
            round_id: Round ID
            breakdown_id: Breakdown ID
        """
        return {
            "metadata": {
                "origin_problem_id": origin_problem_id,
                "round_id": round_id,
                "breakdown_id": breakdown_id,
                "lemma_id": self.lemma_id
            },
            "statement": self.statement,
            "assumptions": self.assumptions,
            "proof_idea": self.proof_idea,
            "dependencies": self.dependencies
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Lemma':
        """Reconstruct Lemma from dictionary representation (without formalizations)."""
        return cls(
            lemma_id=data["metadata"]["lemma_id"],
            statement=data["statement"],
            assumptions=data.get("assumptions"),
            proof_idea=data.get("proof_idea"),
            dependencies=data.get("dependencies", []),
            formalizations=[],  # Formalizations loaded separately
            recursive_attempt=None
        )
