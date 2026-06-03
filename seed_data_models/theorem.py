"""
Theorem data model.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Set, Tuple, TYPE_CHECKING, Any
import re
import random

from .formalization import Formalization
from .proof_attempt import ProofAttempt
from .exceptions_and_helpers import SimulationFailure, ProverPathNode, _try_formalization

if TYPE_CHECKING:
    from .lemma import Lemma
    from .simulations.attempt_tracker import ProofAttemptTracker

@dataclass
class Theorem:
    """Represents a theorem with statement and formalizations.

    The statement and proof_idea are shared across all formalizations.
    Different formalizations represent different samples from the formalizer model.
    """
    statement: str
    proof_idea: Optional[str] = None
    dependencies: List[int] = field(default_factory=list)  # List of lemma IDs this theorem depends on
    formalizations: List[Formalization] = field(default_factory=list)

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

    def get_used_lemmas(self, recursive: bool = False) -> Set[int]:
        """
        Returns the set of lemma IDs used in this theorem's best proof.

        Args:
            recursive: If True, include transitive lemma dependencies

        Returns:
            Set of lemma IDs (integers) used
        """
        best = self.get_best_attempt()
        if not best:
            return set()
        return best.get_used_lemmas(lemmas_dict=None, recursive=False)

    def is_solved(self) -> bool:
        """
        Returns True if theorem has a passing & complete proof attempt.

        Note: This only checks the theorem's own proof, not its lemma dependencies.
        To check if a breakdown is fully solved (including all used lemmas),
        use ParsedBreakdown.is_fully_solved() instead.
        """
        best_form = self.get_best_formalization()
        if not best_form:
            return False

        # Check if best formalization is proven (passing & complete)
        return best_form.is_proven()

    def get_solve_stats(self, lemmas_dict: Dict[int, 'Lemma']) -> tuple[bool, int, int]:
        """
        Returns solve status with lemma usage statistics.

        Returns:
            Tuple of (is_solved, num_used_lemmas, num_proven_used_lemmas)
            - is_solved: Whether this theorem is proven
            - num_used_lemmas: Total number of lemmas used transitively in proof
            - num_proven_used_lemmas: How many of those used lemmas are also proven

        Uses a single recursive traversal through the dependency graph.
        """
        is_solved = self.is_solved()

        if not is_solved:
            return (False, 0, 0)

        # Get lemma IDs used directly in the proof (non-recursive)
        # Pass lemmas_dict to use the same best attempt selection as is_fully_solved()
        best = self.get_best_attempt(lemmas_dict=lemmas_dict)
        if not best:
            return (False, 0, 0)

        used_lemma_ids = best.get_used_lemmas(lemmas_dict=lemmas_dict, recursive=False)

        total_used = 0
        total_proven = 0

        for lemma_id in used_lemma_ids:
            if lemma_id in lemmas_dict:
                lemma = lemmas_dict[lemma_id]
                sub_is_solved, sub_used, sub_proven = lemma.get_solve_stats(lemmas_dict)
                total_used += 1 + sub_used  # Count this lemma + its dependencies
                total_proven += (1 if sub_is_solved else 0) + sub_proven

        return (is_solved, total_used, total_proven)

    def is_formalized(self) -> bool:
        """
        Returns True if this theorem has at least one formalization that compiled successfully.

        This checks if ANY formalization in the formalizations list has compilation_pass=True.
        """
        if not self.formalizations:
            return False
        return any(form.compilation_pass for form in self.formalizations)

    def has_valid_formalization(self) -> bool:
        """
        Returns True if this theorem has at least one formalization that compiled.

        Note: Theorems are not validated (validation_pass is always False).
        """
        if not self.formalizations:
            return False
        return any(form.compilation_pass for form in self.formalizations)

    def get_formalization_status(self) -> str:
        """
        Returns a human-readable status of the theorem's formalization.

        Returns:
            'proven': Has a formalization with a passing & complete proof
            'compiling': Has a formalization that compiled
            'no_compilation': No formalization compiled
            'no_formalizations': No formalizations exist
        """
        if not self.formalizations:
            return 'no_formalizations'

        # Check if any formalization is proven
        for form in self.formalizations:
            if form.is_proven():
                return 'proven'

        # Check if any formalization compiled
        for form in self.formalizations:
            if form.compilation_pass:
                return 'compiling'

        # No formalization compiled
        return 'no_compilation'

    def get_total_cost(self, cost_type="cost", exclude_prover_calls: bool = False) -> float:
        """Returns the sum of costs from all formalizations.

        Args:
            cost_type: Type of cost to return (e.g., 'cost', 'output_sflops', 'input_tokens')
            exclude_prover_calls: If True, excludes proof attempt costs
        """
        return sum(form.get_total_cost(cost_type, exclude_prover_calls=exclude_prover_calls) for form in self.formalizations)

    def generate_proof(self, lemmas_dict: Optional[Dict[int, 'Lemma']] = None) -> Tuple[str, bool]:
        """
        Generate complete proof for this theorem, including all lemmas it uses.

        Args:
            lemmas_dict: Dictionary of lemmas in the breakdown

        Returns:
            Tuple of (proof_code: str, is_complete: bool)
            - proof_code: Complete Lean code with all used lemmas and theorem
            - is_complete: True if all lemmas were found, False if any sorries added
        """
        from seed_prover.utils import extract_axiom_names, check_if_axiom_used, remove_axiom_declarations, strip_preamble

        best_attempt = self.get_best_attempt(lemmas_dict=lemmas_dict)
        if not best_attempt or not best_attempt.code:
            return "", False

        theorem_code = best_attempt.code

        # Extract axiom names and check which are actually used
        axiom_names = extract_axiom_names(theorem_code)
        used_axiom_names = set()
        for axiom_name in axiom_names:
            if check_if_axiom_used(theorem_code, axiom_name):
                used_axiom_names.add(axiom_name)

        # Extract lemma IDs and formalization IDs from used axiom names
        # Map: lemma_id -> formalization_id
        used_lemma_form_ids = {}
        for axiom_name in used_axiom_names:
            # Pattern: lemma<id>_f<form_id>
            match = re.search(r'lemma(\d+)_f(\d+)', axiom_name)
            if match:
                used_lemma_id = int(match.group(1))
                form_id = int(match.group(2))
                if lemmas_dict and used_lemma_id in lemmas_dict:
                    used_lemma_form_ids[used_lemma_id] = form_id

        # Build complete proof by recursively generating lemma proofs
        lines = []
        is_complete = True
        processed_lemmas = set()

        def add_lemma_proof_recursive(lemma_id: int, form_id: int) -> None:
            """Add a lemma proof and all its dependencies first (depth-first)."""
            nonlocal is_complete
            if lemma_id in processed_lemmas or not lemmas_dict or lemma_id not in lemmas_dict:
                return

            processed_lemmas.add(lemma_id)
            lemma = lemmas_dict[lemma_id]

            # Recursively generate proof for this lemma (handles both direct and recursive)
            # Pass the formalization ID to enforce the correct formalization
            lemma_proof = lemma.generate_proof(lemmas_dict=lemmas_dict, formalization_id=form_id)
            if not lemma_proof:
                is_complete = False
                return

            lines.append(lemma_proof)
            lines.append("")

        # Add all used lemmas in dependency order
        for lemma_id, form_id in used_lemma_form_ids.items():
            add_lemma_proof_recursive(lemma_id, form_id)

        # Add theorem proof
        cleaned_theorem = remove_axiom_declarations(theorem_code)
        cleaned_theorem = strip_preamble(cleaned_theorem)

        # Extract only the theorem declaration and proof (remove helper definitions)
        # Find the theorem/lemma line and everything after it
        theorem_match = re.search(r'(theorem\s+\w+.*?:=\s*by\s*.*)$', cleaned_theorem, re.DOTALL)
        if theorem_match:
            cleaned_theorem = theorem_match.group(1)

        lines.append(cleaned_theorem)

        proof_code = "\n".join(lines)
        return proof_code, is_complete

    def get_prover_path(self, lemmas_dict: Optional[Dict[int, 'Lemma']] = None, recursive: bool = True) -> Optional[ProverPathNode]:
        """
        Build a prover path tree for this theorem.

        Returns a ProverPathNode tree with:
        - Blue "T" node for the theorem (if it has a passing+complete proof)
        - Children are recursively called get_prover_path() on each used lemma

        Args:
            lemmas_dict: Dictionary of lemmas in the breakdown (needed to recurse)
            recursive: If True, follow recursive_attempts for unsolved lemmas

        Returns:
            ProverPathNode tree, or None if theorem has no proof
        """
        best_attempt = self.get_best_attempt(lemmas_dict=lemmas_dict)
        if not best_attempt:
            return None

        # Create blue T node
        theorem_node = ProverPathNode(label="T", color="blue")

        # If we have lemmas_dict, recursively build children for used lemmas
        if lemmas_dict:
            used_lemma_ids = best_attempt.get_used_lemmas(lemmas_dict=lemmas_dict, recursive=False)
            for lemma_id in sorted(used_lemma_ids):
                if lemma_id in lemmas_dict:
                    lemma = lemmas_dict[lemma_id]
                    lemma_node = lemma.get_prover_path(lemmas_dict=lemmas_dict, recursive=recursive)
                    if lemma_node:
                        theorem_node.add_child(lemma_node)

        return theorem_node

    def simulate(
        self,
        seed: int,
        max_depth: Optional[int] = None,
        search_policy: str = "sequential",
        strategy: Optional[Any] = None,
        tracker: Optional["ProofAttemptTracker"] = None,
        breakdown_id: int = 0,
        lemma_id: int = -1,
        iteration_id: int = 0,
        origin_problem_id: Optional[str] = None,
    ) -> "Theorem":
        """
        Simulate theorem proving with configurable strategy.

        Args:
            seed: Random seed for reproducibility
            max_depth: Maximum depth/attempts to try (None = unlimited)
            search_policy: Either "exponential" or "sequential"
            strategy: Strategy instance to use (will be cloned for each formalization)
            tracker: Optional ProofAttemptTracker for recording attempts
            breakdown_id: Breakdown ID for tracking
            lemma_id: Lemma ID for tracking (-1 for theorem)
            iteration_id: Iteration depth (0=theorem, 1=direct lemmas, etc.)
            origin_problem_id: Origin problem ID for tracking

        Returns:
            New Theorem with only selected formalization & attempts (may be empty/failing).
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
                lemma_id=lemma_id,
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
                    # Success! Return new Theorem with only this successful formalization
                    return Theorem(
                        statement=self.statement,
                        proof_idea=self.proof_idea,
                        dependencies=self.dependencies,
                        formalizations=[new_form],
                    )

        # All failed - return theorem with all tried formalizations (all failing)
        return Theorem(
            statement=self.statement,
            proof_idea=self.proof_idea,
            dependencies=self.dependencies,
            formalizations=all_tried_formalizations,
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
                "lemma_id": -1  # Theorems always have lemma_id = -1
            },
            "statement": self.statement,
            "proof_idea": self.proof_idea,
            "dependencies": self.dependencies
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Theorem':
        """Reconstruct Theorem from dictionary representation (without formalizations)."""
        return cls(
            statement=data["statement"],
            proof_idea=data.get("proof_idea"),
            dependencies=data.get("dependencies", []),
            formalizations=[]  # Formalizations loaded separately
        )


