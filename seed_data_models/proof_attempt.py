"""
Proof attempt data model.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Set, Any
import re

from .model_config import get_effective_parameters

from .compilation import CompilationResult


@dataclass
class ProofAttempt:
    """Represents a single proof attempt with metadata and results."""

    # Metadata (structured fields)
    origin_problem_id: str
    round_id: int
    breakdown_id: int
    lemma_id: int  # -1 for theorem, 0+ for lemmas
    attempt_id: int

    # Compilation result (always loaded)
    compilation_result: CompilationResult

    # NEW: Pre-extracted lemma dependencies (populated during skeleton load)
    used_lemma_ids: Optional[Set[int]] = None  # Set of lemma IDs used in proof

    # Proof data (lazy-loaded - Optional for skeleton mode)
    model_reasoning: Optional[str] = None
    model_input: Optional[Any] = None  # Can be string or list of messages
    code: Optional[str] = None

    # Optional fields
    iteration_id: int = 0
    correction_round_id: int = 0
    formal_statement: Optional[str] = None
    detailed_cost: Optional[Dict[str, Any]] = None  # {"cost": float, "input_tokens": int, "output_tokens": int, "reasoning_tokens": int}
    model_config_path: Optional[str] = None  # Path to model config (e.g., "configs/models/goedel_prover_v2/32b.yaml")

    # Summary fields (always loaded - needed for decision logic)
    reasoning_summary: Optional[Dict[str, Any]] = None  # LLM-generated summary of reasoning trace
    compilation_summary: Optional[Dict[str, Any]] = None  # Classified error summary with error_counts

    # Correction round tracking
    initial_attempt_index: Optional[int] = None  # Index of the original failed attempt being corrected (chains across rounds)

    def is_passing(self) -> bool:
        """True if code compiled successfully with no sorries."""
        return self.compilation_result.is_successful()

    def get_cost(self, cost_type) -> float:
        """Returns the cost in dollars for this proof attempt."""
        if cost_type == 'prover_calls':
            return 1.0

        if self.detailed_cost is None:
            return 0.0

        if cost_type not in self.detailed_cost and cost_type in ['input_sflops', 'output_sflops']:
            effective_params = get_effective_parameters(self.model_config_path)
            if effective_params is None:
                return 0.0
            
            tokens_key = 'input_tokens' if cost_type == 'input_sflops' else 'output_tokens'

            tokens = self.detailed_cost.get(tokens_key, 0)

            sflops = tokens * effective_params
            return float(sflops)

        return float(self.detailed_cost.get(cost_type, 0.0))
    
    def get_total_cost(self, cost_type, exclude_prover_calls: bool = False) -> float:
        """Returns total cost for this proof attempt.

        Args:
            cost_type: Type of cost to return (e.g., 'cost', 'output_sflops', 'input_tokens')
            exclude_prover_calls: If True, returns 0 (excludes proof attempt costs from totals)
        """
        if exclude_prover_calls:
            return 0.0

        summary_cost = 0.0

        if self.reasoning_summary and cost_type != "prover_calls":
            detailed_cost = self.reasoning_summary.get('detailed_cost', None)

            if detailed_cost:
                summary_cost += detailed_cost.get(cost_type, 0,0)

        return self.get_cost(cost_type) + summary_cost

    def get_used_lemmas(self, lemmas_dict: Optional[Dict[int, 'Lemma']] = None, recursive: bool = False, _visited: Optional[Set[int]] = None) -> Set[int]:
        """
        Extract lemma IDs used in the proof body.
        Only returns lemmas that actually exist in the lemmas_dict.

        Now supports both skeleton mode (pre-extracted IDs) and full mode (extract from code).

        Args:
            lemmas_dict: Optional dictionary of lemmas for recursive lookup (required for accurate results)
            recursive: If True, include transitive lemma dependencies
            _visited: Internal tracking of visited lemmas to prevent cycles

        Returns:
            Set of lemma IDs (integers) that are used
        """

        if _visited is None:
            _visited = set()

        # NEW: If dependencies were pre-extracted during skeleton load, use them
        if self.used_lemma_ids is not None:
            used_lemma_ids = self.used_lemma_ids.copy()
        else:
            # OLD: Extract from code (for backward compatibility and full loads)
            used_lemma_ids = set()

            if not self.code:
                return used_lemma_ids

            # Find the proof body (after ":= by")
            proof_match = re.search(r':=\s+by\s+(.*)', self.code, re.DOTALL)
            if not proof_match:
                return used_lemma_ids

            proof_body = proof_match.group(1)

            # Look for axiom declarations and their usage
            # Match "axiom lemmaX" patterns
            axiom_pattern = r'\baxiom\s+(\w+)'
            axioms = re.findall(axiom_pattern, self.code)

            # For each axiom, check if it's actually used in the proof body
            for axiom_name in axioms:
                # Check if axiom appears as a word in the proof body (more than just the declaration)
                pattern = r'\b' + re.escape(axiom_name) + r'\b'
                matches = list(re.finditer(pattern, proof_body))

                # If there's at least one match in the proof body, it's used
                if matches:
                    # Try to extract lemma ID from axiom name (e.g., "lemma1" -> 1, "l1" -> 1)
                    lemma_id = self._extract_lemma_id_from_name(axiom_name)

                    if lemma_id is not None and lemmas_dict and lemma_id in lemmas_dict:
                        used_lemma_ids.add(lemma_id)

        # Handle recursive dependencies (works with both pre-extracted and code-extracted IDs)
        if recursive and lemmas_dict:
            for lemma_id in list(used_lemma_ids):
                if lemma_id not in _visited and lemma_id in lemmas_dict:
                    _visited.add(lemma_id)
                    lemma = lemmas_dict[lemma_id]
                    best_attempt = lemma.get_best_attempt(lemmas_dict=lemmas_dict)
                    if best_attempt:
                        transitive_lemmas = best_attempt.get_used_lemmas(
                            lemmas_dict=lemmas_dict,
                            recursive=True,
                            _visited=_visited
                        )
                        used_lemma_ids.update(transitive_lemmas)

        return used_lemma_ids

    @staticmethod
    def _extract_lemma_id_from_name(name: str) -> Optional[int]:
        """
        Extract lemma ID from lemma name.
        E.g., "lemma1" -> 1, "l2" -> 2, "l1" -> 1
        Also handles fully qualified names: "algebra_apbon2pownleqapownpbpowon2_lemma3" -> 3

        For formalized names like "parent_problem_id_lemma1_f0", extracts the lemma number (1).
        Returns None if not a valid lemma reference.
        """
        # First try: specific pattern for formalized lemmas with format "lemma{n}_f{j}"
        # This pattern is most common in formalized proofs and should be checked first
        matches = re.findall(r'_lemma(\d+)_f\d+', name)
        if matches:
            return int(matches[-1])

        # Second try: pattern "lemmaX" or "lX" anywhere in the name (not just at start)
        # This handles both simple names and fully qualified names with problem ID prefix
        match = re.search(r'_?(?:lemma|l)(\d+)(?:_|$)', name)
        if match:
            return int(match.group(1))

        return None

    def to_dict(self, formalization_id: int, minified: bool = False, include_code: bool = False):
        """Convert to dictionary representation.

        Args:
            formalization_id: Formalization ID to include in metadata
            minified: If True, exclude heavy fields (code, model_reasoning)
            include_code: If True and minified, still include code field
        """
        result = {
            "metadata": {
                "origin_problem_id": self.origin_problem_id,
                "round_id": self.round_id,
                "breakdown_id": self.breakdown_id,
                "lemma_id": self.lemma_id,
                "formalization_id": formalization_id,
                "attempt_id": self.attempt_id,
                "iteration_id": self.iteration_id,
                "correction_round_id": self.correction_round_id,
                "model_config_path": self.model_config_path,
                "initial_attempt_id": self.initial_attempt_index
            },
            "compilation_result": {
                "passed": self.compilation_result.passed,
                "complete": self.compilation_result.complete,
            },
            "used_lemma_ids": list(self.used_lemma_ids) if self.used_lemma_ids is not None else None,
            "detailed_cost": self.detailed_cost,
            "reasoning_summary": self.reasoning_summary,
            "compilation_summary": self.compilation_summary
        }

        if not minified:
            result["model_reasoning"] = self.model_reasoning
            result["model_input"] = self.model_input
            result["code"] = self.code
            result["formal_statement"] = self.formal_statement
            # Include full compilation result details
            result["compilation_result"] = {
                "passed": self.compilation_result.passed,
                "complete": self.compilation_result.complete,
                "errors": self.compilation_result.errors,
                "warnings": self.compilation_result.warnings,
                "system_errors": self.compilation_result.system_errors
            }
        elif include_code:
            result["code"] = self.code
            errors = [
                err.get("data", "").strip() if isinstance(err, dict) else str(err).strip()
                for err in self.compilation_result.errors
                if (isinstance(err, dict) and err.get("data")) or (isinstance(err, str) and err.strip())
            ]
            for w in self.compilation_result.warnings:
                data = w.get("data", "") if isinstance(w, dict) else str(w)
                if "declaration uses 'sorry'" in data or "failed" in data:
                    errors.append(data.strip())
            if self.compilation_result.system_errors:
                errors.append(self.compilation_result.system_errors)
            result["compilation_result"] = {
                "passed": self.compilation_result.passed,
                "complete": self.compilation_result.complete,
                "errors": errors,
            }

        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'ProofAttempt':
        """Reconstruct ProofAttempt from dictionary representation."""
        from .compilation import CompilationResult

        metadata = data["metadata"]
        compilation_data = data["compilation_result"]

        return cls(
            origin_problem_id=metadata["origin_problem_id"],
            round_id=metadata["round_id"],
            breakdown_id=metadata["breakdown_id"],
            lemma_id=metadata["lemma_id"],
            attempt_id=metadata["attempt_id"],
            iteration_id=metadata.get("iteration_id", 0),
            correction_round_id=metadata.get("correction_round_id", 0),
            compilation_result=CompilationResult(
                passed=compilation_data["passed"],
                complete=compilation_data["complete"],
                errors=compilation_data.get("errors", []),
                warnings=compilation_data.get("warnings", []),
                system_errors=compilation_data.get("system_errors")
            ),
            used_lemma_ids=set(data["used_lemma_ids"]) if data.get("used_lemma_ids") else None,
            model_reasoning=data.get("model_reasoning"),
            model_input=data.get("model_input"),
            code=data.get("code"),
            formal_statement=data.get("formal_statement"),
            detailed_cost=data.get("detailed_cost"),
            reasoning_summary=data.get("reasoning_summary"),
            compilation_summary=data.get("compilation_summary"),
            initial_attempt_index=metadata.get("initial_attempt_id"),
            model_config_path=metadata.get("model_config_path")
        )
