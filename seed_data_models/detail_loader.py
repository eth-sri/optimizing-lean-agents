"""
DetailLoader for on-demand loading of full proof details.

Populates skeleton models with:
- Full proof codes
- Model reasoning traces
- Formalization reasoning
- Validation results
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .models import Problem, Breakdown, Theorem, Lemma, ProofAttempt, Formalization

# Handle metadata_utils import from parent directory
try:
    from metadata_utils import generate_uid
except ImportError:
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from metadata_utils import generate_uid


class DetailLoader:
    """On-demand loader for full proof details."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def load_problem_details(self, problem: Problem) -> Problem:
        """
        Load full details for a problem and all its recursive attempts.

        Args:
            problem: Problem object with skeleton data

        Returns:
            Same Problem object with details populated
        """
        # Load details for main problem breakdowns
        for breakdown in problem.breakdowns.values():
            self._load_breakdown_details(breakdown)

        # Load details for recursive attempts
        for recursive_problem in problem.recursive_attempts:
            self.load_problem_details(recursive_problem)  # Recursive call

        return problem

    def _load_breakdown_details(self, breakdown: Breakdown):
        """Load full codes and reasoning for a breakdown."""
        if not breakdown.parsed_breakdown:
            return

        # Load full_records for this breakdown
        records_cache = self._load_full_records_file(breakdown.origin_problem_id)

        # Load theorem details
        self._load_theorem_details(breakdown.parsed_breakdown.theorem, breakdown, records_cache)

        # Load lemma details
        for lemma in breakdown.parsed_breakdown.lemmas.values():
            self._load_lemma_details(lemma, breakdown, records_cache)

    def _load_full_records_file(self, origin_problem_id: str) -> Dict:
        """Load and cache full_records file for a problem."""
        full_records_file = self.run_dir / 'full_records' / f'{origin_problem_id}.json'

        if not full_records_file.exists():
            return {}

        try:
            with open(full_records_file) as f:
                records = json.load(f)
                if not isinstance(records, list):
                    records = [records]

                # Index by metadata for fast lookup
                records_by_metadata = {}
                for record in records:
                    metadata = record.get('metadata', {})
                    key = (
                        metadata.get('origin_problem_id', ''),
                        metadata.get('round_id', 0),
                        metadata.get('breakdown_id', 0),
                        metadata.get('lemma_id', -1),
                        metadata.get('attempt_id', 0),
                        metadata.get('iteration_id', 0),
                        metadata.get('correction_round_id', 0)
                    )
                    records_by_metadata[key] = record

                return records_by_metadata

        except Exception as e:
            print(f"[DetailLoader] Error loading full_records for {origin_problem_id}: {e}")
            return {}

    def _load_theorem_details(self, theorem: Theorem, breakdown: Breakdown, records_cache: Dict):
        """Load full formalization and proof attempt details for theorem."""
        for formalization in theorem.formalizations:
            # Load proof attempt codes and reasoning
            for attempt in formalization.proof_attempts:
                self._load_proof_attempt_details(attempt, records_cache)

            # Note: formal_statement and formalization_reasoning would need to be loaded
            # from formalized.json if needed, but for now skeleton already has compilation status

    def _load_lemma_details(self, lemma: Lemma, breakdown: Breakdown, records_cache: Dict):
        """Load full formalization and proof attempt details for lemma."""
        for formalization in lemma.formalizations:
            # Load proof attempt codes and reasoning
            for attempt in formalization.proof_attempts:
                self._load_proof_attempt_details(attempt, records_cache)

    def _load_proof_attempt_details(self, attempt: ProofAttempt, records_cache: Dict):
        """
        Load code and reasoning for a specific proof attempt.

        Args:
            attempt: ProofAttempt with skeleton data (has metadata, compilation_result, used_lemma_ids)
            records_cache: Indexed full_records data
        """
        # Build lookup key from attempt metadata
        key = (
            attempt.origin_problem_id,
            attempt.round_id,
            attempt.breakdown_id,
            attempt.lemma_id,
            attempt.attempt_id,
            attempt.iteration_id,
            attempt.correction_round_id
        )

        # Find matching record
        if key in records_cache:
            record = records_cache[key]

            # Populate lazy-loaded fields
            attempt.code = record.get('full_code')
            # model_reasoning might be named 'model_output' in full_records
            attempt.model_reasoning = record.get('model_reasoning') or record.get('model_output')
            attempt.formal_statement = record.get('formal_statement')

            # Note: used_lemma_ids should already be populated from skeleton load
            # If not, we could extract here as fallback, but skeleton should have done it

    def load_formalization_details(self, formalization: Formalization, breakdown: Breakdown):
        """
        Load formalization reasoning and formal_statement for a formalization.

        This would load from formalized.json if needed. Currently skeleton loader
        doesn't populate formalization details, so this is a placeholder for future enhancement.
        """
        # TODO: Load from round{N}/formalizer/formalized.json
        # For now, skeleton has compilation_pass which is sufficient for browsing
        pass
