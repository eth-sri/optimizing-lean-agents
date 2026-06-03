"""
DataLoader for populating the OOP data models from filesystem results.

Loads all data from a run directory and constructs a Session object.
"""

import json
import re
import sys
import os
import glob
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Set, List
from loguru import logger

from .models import (
    CompilationResult, ProofAttempt, Formalization, Theorem, Lemma, ParsedBreakdown,
    Breakdown, Problem, Session
)

# Handle metadata_utils import from parent directory
try:
    from metadata_utils import (
        create_breakdown_problem_id, generate_uid, get_breakdown_key,
        get_origin_problem_id, get_breakdown_id, get_lemma_id
    )
except ImportError:
    # Add parent directory to path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from metadata_utils import (
        create_breakdown_problem_id, generate_uid, get_breakdown_key,
        get_origin_problem_id, get_breakdown_id, get_lemma_id
    )

# Lazy imports for on-the-fly summarization (will be imported when needed)
# This avoids dependency issues when loading data without summarization


class DataLoader:
    """Loads results from a pipeline run directory into model objects."""

    def __init__(
        self,
        run_dir: Path,
        skip_expensive_fields: bool = False,
        enable_reasoning_summary: bool = False,
        enable_compilation_summary: bool = False,
        summarization_config_path: Optional[str] = None
    ):
        self.run_dir = Path(run_dir)
        self.skip_expensive_fields = skip_expensive_fields
        self.enable_reasoning_summary = enable_reasoning_summary
        self.enable_compilation_summary = enable_compilation_summary
        self.summarization_config_path = summarization_config_path or "configs/seed_prover/on_the_fly_summarization.yaml"

    def load_session(self) -> Session:
        """
        Load all data from the run directory and return a populated Session.
        New data structure: all records consolidated in full_records/ at top level.
        Falls back to breakdown.json for problems that didn't make it to full_records.

        Cost loading strategy:
        - First checks if consolidated records already have detailed_cost
        - If yes: uses costs from full_records (new runs with seed_prover changes)
        - If no: loads costs from intermediate files as fallback (old runs without cost in full_records)
        """
        # Run on-the-fly summarization if enabled
        if self.enable_reasoning_summary or self.enable_compilation_summary:
            self._run_on_the_fly_summarization()

        session = Session(run_dir=self.run_dir)

        # Always load intermediate costs to ensure breakdown/parsing costs are included
        # New structure: combined cost in breakdown/parsed_breakdown.json
        # Old structure: parser costs in breakdown_parser/parsed_breakdown.json (NOT in full_records)
        cost_index = self._load_intermediate_costs()

        # Load component-specific costs
        component_costs_index = self._load_component_costs()

        # Load dataset.json for problem metadata (difficulty tags, etc.)
        dataset_index = self._load_dataset()

        # Load full records (consolidated at top level)
        full_records_dir = self.run_dir / 'full_records'
        if full_records_dir.exists():
            # Get all problem IDs from full_records files
            problem_files = list(full_records_dir.glob('*.json'))
            for problem_file in problem_files:
                origin_problem_id = problem_file.stem
                self._load_problem(session, origin_problem_id, cost_index, dataset_index, component_costs_index)

        # Load missing problems from breakdown.json (fallback for problems that didn't progress to full_records)
        # Load from all round directories that exist (dynamically discover rounds)
        round_num = 0
        while True:
            breakdown_file = self.run_dir / f'round{round_num}' / 'breakdown' / 'breakdown.json'
            if breakdown_file.exists():
                self._load_missing_problems_from_breakdown(session, breakdown_file, cost_index, component_costs_index, round_num)
                round_num += 1
            else:
                break  # Stop if round doesn't exist

        # Load validation results from validation_results.json if available
        # (older runs have validation_results.json; newer runs with selection workflow need this fallback)
        self._load_and_apply_validation_results(session)

        # Link recursive attempts to lemmas
        # This connects failed lemmas from round 0 to their recursive proving attempts in round 1+
        self._link_recursive_attempts(session)

        # Calculate initial_attempt_index and organize attempts by correction round
        # This tracks which failed attempts are being corrected in subsequent rounds
        self._calculate_initial_attempt_indices(session)

        return session

    def _check_if_costs_in_records(self) -> bool:
        """
        Check if consolidated records already have detailed_cost.
        Looks at the first record in the first full_records file.

        Returns:
            True if costs are present in consolidated records, False otherwise.
        """
        full_records_dir = self.run_dir / 'full_records'
        if not full_records_dir.exists():
            return False

        # Get the first problem file
        problem_files = sorted(full_records_dir.glob('*.json'))
        if not problem_files:
            return False

        try:
            with open(problem_files[0]) as f:
                records = json.load(f)
                if isinstance(records, list) and len(records) > 0:
                    return "detailed_cost" in records[0]
                elif isinstance(records, dict):
                    return "detailed_cost" in records
        except Exception:
            pass

        return False

    def _load_dataset(self) -> Dict[str, Any]:
        """
        Load dataset.json from the run directory and extract problem metadata.
        Returns a dictionary indexed by problem_id with problem data.

        Returns:
            Dict with keys like problem_id -> {problem_data}
        """
        dataset_index = {}
        dataset_file = self.run_dir / 'dataset.json'

        if not dataset_file.exists():
            return dataset_index

        try:
            with open(dataset_file) as f:
                records = json.load(f)
                if not isinstance(records, list):
                    records = [records]

                for record in records:
                    problem_id = record.get('problem_id') or record.get('name')
                    if problem_id:
                        dataset_index[problem_id] = record
        except Exception as e:
            print(f"Error loading dataset.json: {e}")

        return dataset_index

    def _load_intermediate_costs(self) -> Dict[Tuple, Dict[str, Any]]:
        """
        Load costs from intermediate files (breakdown.json, breakdown/parsed_breakdown.json, and formalized.json).
        New structure: Single LLM call, costs in breakdown/parsed_breakdown.json
        Old structure: Two-step process, costs in breakdown_parser/parsed_breakdown.json
        Returns a dictionary indexed by (origin_problem_id, round_id, breakdown_id, lemma_id).

        Returns:
            Dict with keys like (problem_id, round_id, breakdown_id, lemma_id) -> {cost_data}
        """
        cost_index = {}

        # Load breakdown costs from all rounds
        for round_dir in sorted(self.run_dir.glob('round*')):
            breakdown_file = round_dir / 'breakdown' / 'breakdown.json'
            if breakdown_file.exists():
                try:
                    with open(breakdown_file) as f:
                        records = json.load(f)
                        if not isinstance(records, list):
                            records = [records]

                        for record in records:
                            metadata = record.get('metadata', {})
                            origin_id = metadata.get('origin_problem_id') or record.get('problem_id')
                            round_id = metadata.get('round_id', 0)
                            breakdown_id = metadata.get('breakdown_id', 0)

                            if origin_id:
                                key = (origin_id, round_id, breakdown_id, -1)  # -1 for breakdown level
                                detailed_cost = record.get('detailed_cost')
                                if detailed_cost:
                                    cost_index[key] = detailed_cost
                except Exception as e:
                    print(f"Error loading costs from {breakdown_file}: {e}")

        # Load breakdown_parser costs (includes both breakdown and parser costs) from all rounds
        # New structure: breakdown/parsed_breakdown.json
        # Old structure: breakdown_parser/parsed_breakdown.json
        for round_dir in sorted(self.run_dir.glob('round*')):
            # Try new structure first, then fall back to old
            parser_candidates = [
                round_dir / 'breakdown' / 'parsed_breakdown.json',  # NEW
                round_dir / 'breakdown_parser' / 'parsed_breakdown.json',  # OLD
            ]
            parser_file = None
            for candidate in parser_candidates:
                if candidate.exists():
                    parser_file = candidate
                    break

            if parser_file:
                try:
                    with open(parser_file) as f:
                        records = json.load(f)
                        if not isinstance(records, list):
                            records = [records]

                        for record in records:
                            metadata = record.get('metadata', {})
                            origin_id = metadata.get('origin_problem_id') or record.get('problem_id')
                            round_id = metadata.get('round_id', 0)
                            breakdown_id = metadata.get('breakdown_id', 0)

                            if origin_id:
                                key = (origin_id, round_id, breakdown_id, -1)  # -1 for breakdown level

                                # Aggregate both breakdown and parser costs
                                if key not in cost_index:
                                    cost_index[key] = {
                                        'cost': 0.0,
                                        'input_tokens': 0,
                                        'output_tokens': 0
                                    }

                                # Add breakdown cost
                                detailed_cost = record.get('detailed_cost')
                                if detailed_cost:
                                    cost_index[key]['cost'] += float(detailed_cost.get('cost', 0.0))
                                    cost_index[key]['input_tokens'] += int(detailed_cost.get('input_tokens', 0))
                                    cost_index[key]['output_tokens'] += int(detailed_cost.get('output_tokens', 0))

                                # Add parser cost
                                parser_cost = record.get('parser_detailed_cost')
                                if parser_cost:
                                    cost_index[key]['cost'] += float(parser_cost.get('cost', 0.0))
                                    cost_index[key]['input_tokens'] += int(parser_cost.get('input_tokens', 0))
                                    cost_index[key]['output_tokens'] += int(parser_cost.get('output_tokens', 0))
                except Exception as e:
                    print(f"Error loading costs from {parser_file}: {e}")

        # Load formalization costs from all rounds
        for round_dir in sorted(self.run_dir.glob('round*')):
            formalized_file = round_dir / 'formalizer' / 'formalized.json'
            if formalized_file.exists():
                try:
                    with open(formalized_file) as f:
                        records = json.load(f)
                        if not isinstance(records, list):
                            records = [records]

                        for record in records:
                            metadata = record.get('metadata', {})
                            origin_id = metadata.get('origin_problem_id') or record.get('problem_id')
                            round_id = metadata.get('round_id', 0)
                            breakdown_id = metadata.get('breakdown_id', 0)
                            lemma_id = metadata.get('lemma_id', -1)

                            if origin_id:
                                key = (origin_id, round_id, breakdown_id, lemma_id)
                                detailed_cost = record.get('detailed_cost')
                                if detailed_cost:
                                    # For formalizations, accumulate costs
                                    if key not in cost_index:
                                        cost_index[key] = {
                                            'cost': 0.0,
                                            'input_tokens': 0,
                                            'output_tokens': 0
                                        }
                                    cost_index[key]['cost'] += float(detailed_cost.get('cost', 0.0))
                                    cost_index[key]['input_tokens'] += int(detailed_cost.get('input_tokens', 0))
                                    cost_index[key]['output_tokens'] += int(detailed_cost.get('output_tokens', 0))
                except Exception as e:
                    print(f"Error loading costs from {formalized_file}: {e}")

        return cost_index

    def _load_component_costs(self) -> Dict[Tuple, Dict[str, Dict[str, int]]]:
        """
        Load component-specific costs from intermediate files.
        Returns a dictionary indexed by (origin_problem_id, round_id, breakdown_id) -> {component_name -> {token counts}}.

        Returns:
            Dict with keys like (problem_id, round_id, breakdown_id) -> {
                "breakdown": {"input_tokens": N, "output_tokens": N},
                "breakdown_parser": {"input_tokens": N, "output_tokens": N},
                "formalization": {"input_tokens": N, "output_tokens": N},
                "prover": {"input_tokens": N, "output_tokens": N}
            }
        """
        component_costs = {}

        # Initialize structure for all problems
        def get_empty_component_costs():
            return {
                "breakdown": {"input_tokens": 0, "output_tokens": 0},
                "breakdown_parser": {"input_tokens": 0, "output_tokens": 0},
                "formalization": {"input_tokens": 0, "output_tokens": 0},
                "prover": {"input_tokens": 0, "output_tokens": 0}
            }

        # Load breakdown costs
        for round_dir in sorted(self.run_dir.glob('round*')):
            breakdown_file = round_dir / 'breakdown' / 'breakdown.json'
            if breakdown_file.exists():
                try:
                    with open(breakdown_file) as f:
                        records = json.load(f)
                        if not isinstance(records, list):
                            records = [records]

                        for record in records:
                            metadata = record.get('metadata', {})
                            origin_id = metadata.get('origin_problem_id') or record.get('problem_id')
                            round_id = metadata.get('round_id', 0)
                            breakdown_id = metadata.get('breakdown_id', 0)

                            if origin_id:
                                key = (origin_id, round_id, breakdown_id)
                                if key not in component_costs:
                                    component_costs[key] = get_empty_component_costs()

                                detailed_cost = record.get('detailed_cost')
                                if detailed_cost:
                                    component_costs[key]["breakdown"]["input_tokens"] += int(detailed_cost.get('input_tokens', 0))
                                    component_costs[key]["breakdown"]["output_tokens"] += int(detailed_cost.get('output_tokens', 0))
                except Exception as e:
                    print(f"Error loading component costs from {breakdown_file}: {e}")

        # Load breakdown_parser costs
        # New structure: breakdown/parsed_breakdown.json
        # Old structure: breakdown_parser/parsed_breakdown.json
        for round_dir in sorted(self.run_dir.glob('round*')):
            # Try new structure first, then fall back to old
            parser_candidates = [
                round_dir / 'breakdown' / 'parsed_breakdown.json',  # NEW
                round_dir / 'breakdown_parser' / 'parsed_breakdown.json',  # OLD
            ]
            parser_file = None
            for candidate in parser_candidates:
                if candidate.exists():
                    parser_file = candidate
                    break

            if parser_file:
                try:
                    with open(parser_file) as f:
                        records = json.load(f)
                        if not isinstance(records, list):
                            records = [records]

                        for record in records:
                            metadata = record.get('metadata', {})
                            origin_id = metadata.get('origin_problem_id') or record.get('problem_id')
                            round_id = metadata.get('round_id', 0)
                            breakdown_id = metadata.get('breakdown_id', 0)

                            if origin_id:
                                key = (origin_id, round_id, breakdown_id)
                                if key not in component_costs:
                                    component_costs[key] = get_empty_component_costs()

                                parser_cost = record.get('parser_detailed_cost')
                                if parser_cost:
                                    component_costs[key]["breakdown_parser"]["input_tokens"] += int(parser_cost.get('input_tokens', 0))
                                    component_costs[key]["breakdown_parser"]["output_tokens"] += int(parser_cost.get('output_tokens', 0))
                except Exception as e:
                    print(f"Error loading component costs from {parser_file}: {e}")

        # Load formalization costs from selected_formalizations in parsed_breakdown
        # Each selected formalization has its own detailed_cost
        for round_dir in sorted(self.run_dir.glob('round*')):
            formalized_file = round_dir / 'formalizer' / 'formalized.json'
            if formalized_file.exists():
                try:
                    with open(formalized_file) as f:
                        records = json.load(f)
                        if not isinstance(records, list):
                            records = [records]

                        for record in records:
                            metadata = record.get('metadata', {})
                            origin_id = metadata.get('origin_problem_id') or record.get('problem_id')
                            round_id = metadata.get('round_id', 0)
                            breakdown_id = metadata.get('breakdown_id', 0)

                            if origin_id:
                                key = (origin_id, round_id, breakdown_id)
                                if key not in component_costs:
                                    component_costs[key] = get_empty_component_costs()

                                # Get formalization costs from selected_formalizations in parsed_breakdown
                                parsed_breakdown = record.get('parsed_breakdown', {})
                                selected_formalizations = parsed_breakdown.get('selected_formalizations', [])

                                for formalization in selected_formalizations:
                                    detailed_cost = formalization.get('detailed_cost')
                                    if detailed_cost:
                                        component_costs[key]["formalization"]["input_tokens"] += int(detailed_cost.get('input_tokens', 0))
                                        component_costs[key]["formalization"]["output_tokens"] += int(detailed_cost.get('output_tokens', 0))
                except Exception as e:
                    print(f"Error loading component costs from {formalized_file}: {e}")

        # Load prover costs (from theorem and lemma proof attempts in full_records)
        full_records_dir = self.run_dir / 'full_records'
        if full_records_dir.exists():
            for problem_file in full_records_dir.glob('*.json'):
                try:
                    with open(problem_file) as f:
                        records = json.load(f)
                        if not isinstance(records, list):
                            records = [records]

                        for record in records:
                            metadata = record.get('metadata', {})
                            origin_id = metadata.get('origin_problem_id')
                            round_id = metadata.get('round_id', 0)
                            breakdown_id = metadata.get('breakdown_id', 0)

                            if origin_id:
                                key = (origin_id, round_id, breakdown_id)
                                if key not in component_costs:
                                    component_costs[key] = get_empty_component_costs()

                                detailed_cost = record.get('detailed_cost')
                                if detailed_cost:
                                    component_costs[key]["prover"]["input_tokens"] += int(detailed_cost.get('input_tokens', 0))
                                    component_costs[key]["prover"]["output_tokens"] += int(detailed_cost.get('output_tokens', 0))
                except Exception as e:
                    print(f"Error loading component costs from {problem_file}: {e}")

        return component_costs

    def _load_problem(self, session: Session, origin_problem_id: str, cost_index: Dict = None, dataset_index: Dict = None, component_costs_index: Dict = None):
        """Load a single problem and all its breakdowns."""
        if cost_index is None:
            cost_index = {}
        if dataset_index is None:
            dataset_index = {}
        if component_costs_index is None:
            component_costs_index = {}
        problem = Problem(origin_problem_id=origin_problem_id)

        # Load difficulty from dataset if available
        if origin_problem_id in dataset_index:
            dataset_record = dataset_index[origin_problem_id]
            problem.difficulty = dataset_record.get('difficulty')

        # All records for this problem are in a single file
        full_records_file = self.run_dir / 'full_records' / f'{origin_problem_id}.json'
        if not full_records_file.exists():
            return

        try:
            with open(full_records_file) as f:
                records = json.load(f)
                if not isinstance(records, list):
                    records = [records]

            # Group records by breakdown
            breakdowns_found = {}
            for record in records:
                metadata = record.get('metadata', {})
                round_id = metadata.get('round_id', 0)
                breakdown_id = metadata.get('breakdown_id', 0)
                # Use parent_problem_id if available (for round 1+), otherwise use origin_problem_id
                parent_problem_id = metadata.get('parent_problem_id') or metadata.get('origin_problem_id', origin_problem_id)

                key = (parent_problem_id, round_id, breakdown_id)
                if key not in breakdowns_found:
                    breakdowns_found[key] = {
                        'parent_problem_id': parent_problem_id,
                        'round_id': round_id,
                        'breakdown_id': breakdown_id,
                        'theorem_records': [],
                        'lemma_records': {},
                        'informal_prefix': None,
                        'formal_statement': None,
                        'breakdown_prompt': None,
                        'informal_breakdown_reasoning': None
                    }

                # Extract informal and formal data from first record
                if breakdowns_found[key]['informal_prefix'] is None:
                    breakdowns_found[key]['informal_prefix'] = record.get('informal_prefix')
                if breakdowns_found[key]['formal_statement'] is None:
                    breakdowns_found[key]['formal_statement'] = record.get('formal_statement')
                if breakdowns_found[key]['breakdown_prompt'] is None:
                    breakdowns_found[key]['breakdown_prompt'] = record.get('breakdown_prompt')
                if breakdowns_found[key]['informal_breakdown_reasoning'] is None:
                    breakdowns_found[key]['informal_breakdown_reasoning'] = record.get('informal_breakdown_reasoning')

                # Separate theorem and lemma records
                lemma_id = metadata.get('lemma_id', -1)
                if lemma_id == -1:
                    breakdowns_found[key]['theorem_records'].append(record)
                else:
                    if lemma_id not in breakdowns_found[key]['lemma_records']:
                        breakdowns_found[key]['lemma_records'][lemma_id] = []
                    breakdowns_found[key]['lemma_records'][lemma_id].append(record)

            # Now create Breakdown objects for each breakdown
            for (parent_problem_id, round_id, breakdown_id), bd_info in breakdowns_found.items():
                breakdown_problem_id = f"{origin_problem_id}_r{round_id}_b{breakdown_id}"

                breakdown = Breakdown(
                    problem_id=breakdown_problem_id,
                    origin_problem_id=origin_problem_id,
                    round_id=round_id,
                    breakdown_id=breakdown_id,
                    parent_problem_id=parent_problem_id,
                    name=origin_problem_id
                )

                # Attach breakdown costs from intermediate files
                breakdown_cost_key = (origin_problem_id, round_id, breakdown_id, -1)
                if breakdown_cost_key in cost_index:
                    breakdown.detailed_cost = cost_index[breakdown_cost_key]

                # Attach component costs for this breakdown
                component_cost_key = (origin_problem_id, round_id, breakdown_id)
                if component_cost_key in component_costs_index:
                    breakdown.component_costs = component_costs_index[component_cost_key]

                # Set informal and formal data from records
                # Note: components expect 'informal_breakdown' field
                # Conditionally skip expensive text fields in skeleton mode
                if not self.skip_expensive_fields:
                    breakdown.informal_breakdown = bd_info['informal_prefix']
                    breakdown.breakdown_prompt = bd_info['breakdown_prompt']
                    breakdown.informal_breakdown_reasoning = bd_info['informal_breakdown_reasoning']
                breakdown.informal_prefix = bd_info['informal_prefix']
                breakdown.formal_statement = bd_info['formal_statement']

                # Load parsed breakdown from records
                # For round 1+, use parent_problem_id if available (the lemma being recursively proven)
                problem_id_for_lookup = parent_problem_id if parent_problem_id else origin_problem_id
                parsed_bd, parse_error = self._load_parsed_breakdown_from_records(
                    problem_id_for_lookup, round_id, breakdown_id,
                    bd_info['theorem_records'],
                    bd_info['lemma_records'],
                    origin_problem_id  # Pass origin for error reporting
                )
                breakdown.parsed_breakdown = parsed_bd

                # Track parse failure with diagnostic information if parsing failed
                if not parsed_bd:
                    # Always set parse_failure when parsing fails, even if we don't have error details
                    if parse_error:
                        breakdown.parse_failure = parse_error
                    else:
                        # Fallback: create a generic error dict with diagnostic info
                        breakdown.parse_failure = {
                            'error': 'unknown_parse_failure',
                            'message': 'Parsed breakdown could not be loaded but no specific error was recorded'
                        }

                    # Ensure error dict always has context fields
                    if 'origin_problem_id' not in breakdown.parse_failure:
                        breakdown.parse_failure['origin_problem_id'] = origin_problem_id
                    if 'round_id' not in breakdown.parse_failure:
                        breakdown.parse_failure['round_id'] = round_id
                    if 'breakdown_id' not in breakdown.parse_failure:
                        breakdown.parse_failure['breakdown_id'] = breakdown_id

                # Load formalized breakdown (formalized.json contains the Lean code for each lemma)
                # Store as ParsedBreakdown to maintain consistent structure
                formalized_json = self._load_formalized_json(origin_problem_id, round_id, breakdown_id)
                if formalized_json and 'lemmas' in formalized_json:
                    # Create a ParsedBreakdown object for formalized data
                    formalized_lemmas = {}

                    # Check if we have selected_formalizations at the breakdown level (new structure)
                    # This is a flat list of all formalized lemmas
                    # NOTE: The formalized.json file may contain selected_formalizations from multiple breakdowns
                    # in a single record, so we need to filter by breakdown_id to get only this breakdown's formalizations
                    all_selected_formalizations = formalized_json.get('selected_formalizations', [])

                    # Filter selected_formalizations to only those for THIS breakdown
                    # Handle different structure variants - some have metadata, some don't
                    formalizations_for_this_breakdown = []
                    for form in all_selected_formalizations:
                        if form is None:
                            continue
                        # Try different ways to find breakdown_id
                        val_result = form.get('validation_result')
                        # Check if validation_result has metadata with matching breakdown_id
                        if val_result and isinstance(val_result, dict) and val_result.get('metadata', {}).get('breakdown_id') == breakdown_id:
                            formalizations_for_this_breakdown.append(form)
                        # Check if form itself has metadata with matching breakdown_id
                        elif form.get('metadata', {}).get('breakdown_id') == breakdown_id:
                            formalizations_for_this_breakdown.append(form)
                        # New format - selected_formalizations don't have breakdown_id directly
                        # They just have lemma id, so if we can't match by breakdown_id, include them
                        # This handles the case where validation_result is None or not a dict
                        elif 'metadata' not in form or not isinstance(form.get('validation_result'), dict):
                            # Include formalization if it doesn't have explicit breakdown metadata
                            # (assumes all formalizations in this record are for this breakdown)
                            formalizations_for_this_breakdown.append(form)

                    # Group formalizations by lemma_id for easier lookup
                    # Also track which formalization_ids are selected for each lemma
                    formalizations_by_id = {}
                    selected_ids_by_lemma = {}  # Track which formalization_ids are selected
                    for form_data in formalizations_for_this_breakdown:
                        form_id = form_data.get('id')
                        form_idx = form_data.get('formalization_id')
                        if form_id not in formalizations_by_id:
                            formalizations_by_id[form_id] = []
                            selected_ids_by_lemma[form_id] = set()
                        formalizations_by_id[form_id].append(form_data)
                        selected_ids_by_lemma[form_id].add(form_idx)

                    if 'lemmas' in formalized_json:
                        for formal_lemma_data in formalized_json['lemmas']:
                            # Use the explicit id from formal_lemma_data if available, otherwise skip
                            lemma_id = formal_lemma_data.get('id')
                            if lemma_id is None:
                                continue

                            # Create a Lemma with shared data
                            formal_lemma = Lemma(
                                lemma_id=lemma_id,
                                statement=formal_lemma_data.get('statement', ''),
                                assumptions=formal_lemma_data.get('assumptions', None)
                            )

                            # Check if we have formalization data for this lemma in selected_formalizations
                            if lemma_id in formalizations_by_id:
                                # New behavior: create multiple Formalization objects from selected_formalizations list
                                # Mark which ones are selected
                                selected_idx_set = selected_ids_by_lemma.get(lemma_id, set())

                                for i, selected_form in enumerate(formalizations_by_id[lemma_id]):
                                    form_idx = selected_form.get('formalization_id', i)

                                    # Extract formalization reasoning from formal_statement_raw if available
                                    formalization_reasoning = self._extract_thinking_from_raw(selected_form.get('formal_statement_raw', ''))
                                    if not formalization_reasoning:
                                        formalization_reasoning = formal_lemma_data.get('model_reasoning', '') or formal_lemma_data.get('reasoning', '')

                                    # Create a Formalization object for each selected formalization
                                    formalization = Formalization(
                                        id=form_idx,
                                        formal_statement=selected_form.get('formal_statement', '') if not self.skip_expensive_fields else None,
                                        formalization_reasoning=formalization_reasoning if not self.skip_expensive_fields else None,
                                        compilation_pass=selected_form.get('compilation_pass', False),
                                        compilation_result=selected_form.get('compilation_result') if not self.skip_expensive_fields else None,
                                        compilation_errors=selected_form.get('compilation_errors') if not self.skip_expensive_fields else None,
                                        validation_pass=selected_form.get('validation_pass', False),
                                        validation_result=selected_form.get('validation_result') if not self.skip_expensive_fields else None,
                                        validation_reasoning=selected_form.get('validation_reasoning') if not self.skip_expensive_fields else None,
                                        is_selected=form_idx in selected_idx_set,  # Mark as selected
                                        detailed_cost=selected_form.get('detailed_cost')  # Load detailed cost from selected_formalizations
                                    )

                                    # Fallback: Attach formalization costs from intermediate files if not in selected_form
                                    if not formalization.detailed_cost:
                                        formalization_cost_key = (origin_problem_id, round_id, breakdown_id, lemma_id)
                                        if formalization_cost_key in cost_index:
                                            formalization.detailed_cost = cost_index[formalization_cost_key]

                                    formal_lemma.formalizations.append(formalization)

                            # Note: It's OK if there are no formalizations - the lemma will just have an empty list

                            formalized_lemmas[lemma_id] = formal_lemma

                    # Create a theorem for formalized data if available
                    formal_theorem_data = formalized_json.get('theorem', {})
                    # Extract formalization reasoning from formal_statement_raw if available
                    theorem_reasoning = self._extract_thinking_from_raw(formal_theorem_data.get('formal_statement_raw', ''))
                    if not theorem_reasoning:
                        theorem_reasoning = formal_theorem_data.get('model_reasoning', '') or formal_theorem_data.get('reasoning', '')

                    # Create a Formalization object for theorem with formal data
                    theorem_formalization = Formalization(
                        id=0,
                        formal_statement=formal_theorem_data.get('formal_statement', '') if not self.skip_expensive_fields else None,
                        formalization_reasoning=theorem_reasoning if not self.skip_expensive_fields else None,
                        compilation_pass=formal_theorem_data.get('compilation_pass', False),
                        compilation_result=formal_theorem_data.get('compilation_result') if not self.skip_expensive_fields else None,
                        compilation_errors=formal_theorem_data.get('compilation_errors') if not self.skip_expensive_fields else None,
                        validation_pass=formal_theorem_data.get('validation_pass', False),  # Theorems may now be validated
                        validation_result=formal_theorem_data.get('validation_result') if not self.skip_expensive_fields else None,
                        validation_reasoning=formal_theorem_data.get('validation_reasoning') if not self.skip_expensive_fields else None,
                        detailed_cost=formal_theorem_data.get('detailed_cost')  # Load detailed cost from theorem data
                    )

                    # Create a Theorem with shared data
                    formal_theorem = Theorem(
                        statement=formal_theorem_data.get('statement', ''),
                        proof_idea=formal_theorem_data.get('proof_idea', None),
                        dependencies=formal_theorem_data.get('dependencies', [])
                    )
                    formal_theorem.formalizations.append(theorem_formalization)

                    formalized_bd = ParsedBreakdown(theorem=formal_theorem, lemmas=formalized_lemmas)
                    breakdown.formalized_breakdown = formalized_bd

                # Store theorem and lemma prover results for backward compatibility
                if bd_info['theorem_records']:
                    breakdown.theorem_prover_results = {
                        'breakdown_id': breakdown_problem_id,
                        'attempts': [
                            {
                                'data': self._record_to_dict(record) if not self.skip_expensive_fields else self._record_to_dict_skeleton(record),
                                'correction_round': record.get('correction_round', 0)
                            }
                            for record in bd_info['theorem_records']
                        ]
                    }

                if bd_info['lemma_records']:
                    breakdown.lemma_prover_results = {
                        'breakdown_id': breakdown_problem_id,
                        'lemmas': {}
                    }
                    for lemma_id, records in bd_info['lemma_records'].items():
                        breakdown.lemma_prover_results['lemmas'][lemma_id] = {
                            'attempts': [
                                {
                                    'data': self._record_to_dict(record) if not self.skip_expensive_fields else self._record_to_dict_skeleton(record),
                                    'correction_round': record.get('correction_round', 0)
                                }
                                for record in records
                            ]
                        }

                # Use composite key (parent_problem_id, round_id, breakdown_id) to avoid collisions across rounds and parent problems
                breakdown_key = (parent_problem_id, round_id, breakdown_id)
                problem.breakdowns[breakdown_key] = breakdown

            # Explicitly delete the records list to free memory
            del records
            del breakdowns_found

        except Exception as e:
            print(f"Error loading problem {origin_problem_id}: {e}")
            return

        if problem.breakdowns:
            session.problems[origin_problem_id] = problem

    def _load_parsed_breakdown_from_records(
        self,
        lookup_problem_id: str,
        round_id: int,
        breakdown_id: int,
        theorem_records: list,
        lemma_records: Dict[int, list],
        origin_problem_id: Optional[str] = None
    ) -> Tuple[Optional[ParsedBreakdown], Optional[Dict[str, Any]]]:
        """Load parsed breakdown from parsed_breakdown.json and combine with other sources.

        Args:
            lookup_problem_id: Problem ID to look for in JSON files (parent_problem_id for round 1+, origin_problem_id for round 0)
            round_id: Round number
            breakdown_id: Breakdown ID
            theorem_records: Theorem proof attempt records
            lemma_records: Lemma proof attempt records by lemma_id
            origin_problem_id: Original problem ID (for error reporting in round 1+)

        Returns:
            Tuple of (parsed_breakdown, error_info) where error_info is None if successful
            or contains diagnostic error information if parsing failed.
        """
        # Use origin_problem_id for error reporting if provided, otherwise use lookup_problem_id
        error_problem_id = origin_problem_id if origin_problem_id else lookup_problem_id

        # Load lemma definitions from parsed_breakdown.json
        parsed_bd_data, parse_error, parser_detailed_cost = self._load_parsed_breakdown_json(lookup_problem_id, round_id, breakdown_id)

        # If parsing failed, return None with error info so we can track this as an error
        if not parsed_bd_data:
            return None, parse_error

        # Load formalization data
        formalized_data = self._load_formalized_json(lookup_problem_id, round_id, breakdown_id)

        # Extract formal theorem data if available
        formal_theorem_data = None
        if formalized_data and 'theorem' in formalized_data:
            formal_theorem_data = formalized_data['theorem']

        # Extract parsed theorem data for proof_idea and dependencies
        parsed_theorem_data = None
        if parsed_bd_data and 'theorem' in parsed_bd_data:
            parsed_theorem_data = parsed_bd_data['theorem']

        # Create theorem from theorem records, optionally enriched with formalized data
        theorem = self._create_theorem_from_records(theorem_records, formal_theorem_data, parsed_theorem_data)

        # Build a map of lemma_id -> selected formalizations from formalized_data
        # For round 1+, selected_formalizations contains the actual formalization data we need
        formalizations_by_lemma_id = {}
        if formalized_data and 'selected_formalizations' in formalized_data:
            all_selected = formalized_data.get('selected_formalizations', [])
            for form_data in all_selected:
                # Extract lemma_id from metadata first, fall back to 'id' field
                metadata = form_data.get('metadata')
                lemma_id = metadata.get('lemma_id') if metadata else None
                if lemma_id is None:
                    # For round 1+ data, the lemma_id might be directly in 'id' field
                    lemma_id = form_data.get('id')

                if lemma_id is not None:
                    if lemma_id not in formalizations_by_lemma_id:
                        formalizations_by_lemma_id[lemma_id] = []
                    formalizations_by_lemma_id[lemma_id].append(form_data)

        # Create lemmas from parsed breakdown JSON combined with proof records
        lemmas = {}
        if parsed_bd_data and 'lemmas' in parsed_bd_data:
            for idx, parsed_lemma in enumerate(parsed_bd_data['lemmas']):
                # Use the explicit id from parsed_lemma if available, otherwise fall back to enumerate index
                lemma_id = parsed_lemma.get('id', idx)

                # Get proof records for this lemma using the actual lemma_id from metadata
                lemma_records_for_id = lemma_records.get(lemma_id, [])

                # Get formalization data for this lemma if available
                # First try: use selected_formalizations if available (for round 1+)
                formal_lemma_list = formalizations_by_lemma_id.get(lemma_id)

                # Second try: fall back to the legacy formalized.json structure (index-based)
                if not formal_lemma_list and formalized_data and 'lemmas' in formalized_data:
                    if idx < len(formalized_data['lemmas']):
                        formal_lemma_list = [formalized_data['lemmas'][idx]]

                lemma = self._create_lemma_from_parsed_and_formal_list(
                    lemma_id, parsed_lemma, lemma_records_for_id, formal_lemma_list
                )
                if lemma:
                    lemmas[lemma_id] = lemma

        return ParsedBreakdown(theorem=theorem, lemmas=lemmas, detailed_cost=parser_detailed_cost), None

    def _create_theorem_from_records(self, theorem_records: list, formal_theorem_data: Optional[Dict[str, Any]] = None, parsed_theorem_data: Optional[Dict[str, Any]] = None) -> Theorem:
        """Create a Theorem object from theorem records, optionally enriched with formalized and parsed data.

        Creates a Theorem with statement/proof_idea and a Formalization object containing
        formal_statement, compilation/validation data, and proof attempts.
        """
        # Get statement from first record if available
        statement = ''
        if theorem_records:
            statement = theorem_records[0].get('informal_prefix', '')

        # Get proof_idea and dependencies from parsed breakdown if available
        proof_idea = None
        dependencies = []
        if parsed_theorem_data:
            # Note: The field is called 'proof' in the parsed_breakdown.json
            proof_idea = parsed_theorem_data.get('proof', None)
            dependencies = parsed_theorem_data.get('dependencies', [])

        # Create the Theorem with statement only (shared across formalizations)
        theorem = Theorem(
            statement=statement,
            proof_idea=proof_idea,
            dependencies=dependencies
        )

        # Create a Formalization object if we have formal theorem data
        if formal_theorem_data:
            # Extract formalization reasoning
            formalization_reasoning = self._extract_thinking_from_raw(formal_theorem_data.get('formal_statement_raw', ''))
            if not formalization_reasoning:
                formalization_reasoning = formal_theorem_data.get('model_reasoning', '') or formal_theorem_data.get('reasoning', '')

            # Extract compilation and validation data
            compilation_pass = formal_theorem_data.get('compilation_pass', False)
            compilation_result = formal_theorem_data.get('compilation_result')
            compilation_errors = formal_theorem_data.get('compilation_errors')
            validation_pass = formal_theorem_data.get('validation_pass', False)
            validation_result = formal_theorem_data.get('validation_result')
            validation_reasoning = formal_theorem_data.get('validation_reasoning')

            formalization = Formalization(
                id=0,
                formal_statement=formal_theorem_data.get('formal_statement', ''),
                formalization_reasoning=formalization_reasoning,
                compilation_pass=compilation_pass,
                compilation_result=compilation_result,
                compilation_errors=compilation_errors,
                validation_pass=validation_pass,
                validation_result=validation_result,
                validation_reasoning=validation_reasoning,
                detailed_cost=formal_theorem_data.get('detailed_cost')  # Load detailed cost from theorem data
            )

            # Add proof attempts to the formalization
            for record in theorem_records:
                attempt = self._record_to_proof_attempt(record)
                if attempt:
                    formalization.proof_attempts.append(attempt)

            theorem.formalizations.append(formalization)

        return theorem

    def _load_parsed_breakdown_json(
        self,
        origin_problem_id: str,
        round_id: int,
        breakdown_id: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Load lemma definitions from parsed_breakdown.json for the given breakdown.

        Also checks failed_parses.json for parsing errors.
        The file is an array of problem records; we find the one matching origin_problem_id.
        Note: breakdown_id in parsed_breakdown.json may differ from full_records breakdown_id,
        so we fall back to any valid parsed_breakdown for this problem.

        Returns:
            Tuple of (parsed_breakdown_dict, error_info, parser_detailed_cost) where:
            - parsed_breakdown_dict: The parsed breakdown data if found, None otherwise
            - error_info: None if successful, or dict with diagnostic info if failed:
                - {"error": "parsing_failed", "message": "..."}
                - {"error": "file_missing", "paths_tried": [...]}
                - {"error": "no_matching_record", "found_problem_ids": [...]}
                - {"error": "exception", "message": str}
            - parser_detailed_cost: Cost dict from parsing, or None if not available
        """
        # Path depends on whether run_dir is already in round{N}/prover or at root level
        # Try multiple possible locations
        # New structure: breakdown/parsed_breakdown.json
        # Old structure: breakdown_parser/parsed_breakdown.json
        candidates = [
            # NEW STRUCTURE: If run_dir is round{N}/prover, go up to round{N} and find breakdown/
            self.run_dir.parent / 'breakdown' / 'parsed_breakdown.json',
            # NEW STRUCTURE: If run_dir is at root
            self.run_dir / f'round{round_id}' / 'breakdown' / 'parsed_breakdown.json',
            # OLD STRUCTURE: If run_dir is round{N}/prover, go up to round{N} and find breakdown_parser/
            self.run_dir.parent / 'breakdown_parser' / 'parsed_breakdown.json',
            # OLD STRUCTURE: If run_dir is at root (historical structure)
            self.run_dir / f'round{round_id}' / 'breakdown_parser' / 'parsed_breakdown.json',
        ]

        parsed_bd_file = None
        for candidate in candidates:
            if candidate.exists():
                parsed_bd_file = candidate
                break

        # First check failed_parses.json for this specific breakdown
        # New structure: breakdown/failed_parses.json
        # Old structure: breakdown_parser/failed_parses.json
        failed_parses_candidates = [
            # NEW STRUCTURE
            self.run_dir.parent / 'breakdown' / 'failed_parses.json',
            self.run_dir / f'round{round_id}' / 'breakdown' / 'failed_parses.json',
            # OLD STRUCTURE
            self.run_dir.parent / 'breakdown_parser' / 'failed_parses.json',
            self.run_dir / f'round{round_id}' / 'breakdown_parser' / 'failed_parses.json',
        ]
        for candidate in failed_parses_candidates:
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        failed_data = json.load(f)
                        if isinstance(failed_data, list):
                            for record in failed_data:
                                if record.get('problem_id') == origin_problem_id:
                                    metadata = record.get('metadata', {})
                                    if metadata.get('breakdown_id') == breakdown_id:
                                        parsed_bd = record.get('parsed_breakdown', {})
                                        if isinstance(parsed_bd, dict) and 'error' in parsed_bd:
                                            return None, {
                                                'error': 'parsing_failed',
                                                'message': parsed_bd['error'],
                                                'breakdown_error': True
                                            }, None
                except Exception:
                    pass  # If failed_parses.json can't be read, continue with parsed_breakdown.json

        if not parsed_bd_file:
            return None, {
                'error': 'file_missing',
                'paths_tried': [str(c) for c in candidates]
            }, None

        try:
            with open(parsed_bd_file) as f:
                data = json.load(f)

                # parsed_breakdown.json is an array of problem records
                if isinstance(data, list):
                    found_problem_ids = []

                    # First try: exact match using metadata (check parent_problem_id for round 1+, origin_problem_id for round 0) AND breakdown_id
                    for record in data:
                        if isinstance(record, dict):
                            metadata = record.get('metadata', {})
                            # Use parent_problem_id if available (round 1+), otherwise use origin_problem_id (round 0)
                            record_problem_id = metadata.get('parent_problem_id') or metadata.get('origin_problem_id')

                            if record_problem_id == origin_problem_id:
                                if record_problem_id not in found_problem_ids:
                                    found_problem_ids.append(record_problem_id)
                                # Check if this record matches our breakdown_id
                                if metadata.get('breakdown_id') == breakdown_id:
                                    # Extract the parsed_breakdown from this record
                                    if 'parsed_breakdown' in record and record['parsed_breakdown']:
                                        parsed_bd = record['parsed_breakdown']
                                        # Check if it has a parsing error
                                        if isinstance(parsed_bd, dict) and 'error' in parsed_bd:
                                            return None, {
                                                'error': 'parsing_failed',
                                                'message': parsed_bd['error'],
                                                'breakdown_error': True
                                            }, None
                                        # Extract parser cost from record
                                        parser_cost = record.get('parser_detailed_cost')
                                        return parsed_bd, None, parser_cost

                    # Fallback: if exact match failed, find ANY valid parsed_breakdown for this problem
                    # This handles cases where breakdown_id differs between full_records and parsed_breakdown.json
                    for record in data:
                        if isinstance(record, dict):
                            metadata = record.get('metadata', {})
                            # Use parent_problem_id if available (round 1+), otherwise use origin_problem_id (round 0)
                            record_problem_id = metadata.get('parent_problem_id') or metadata.get('origin_problem_id')

                            if record_problem_id == origin_problem_id:
                                if 'parsed_breakdown' in record and record['parsed_breakdown']:
                                    parsed_bd = record['parsed_breakdown']
                                    # Check if it has a parsing error
                                    if isinstance(parsed_bd, dict) and 'error' in parsed_bd:
                                        return None, {
                                            'error': 'parsing_failed',
                                            'message': parsed_bd['error'],
                                            'breakdown_error': True
                                        }, None
                                    # Verify it has valid structure (has 'theorem' and/or 'lemmas')
                                    if isinstance(parsed_bd, dict) and ('theorem' in parsed_bd or 'lemmas' in parsed_bd):
                                        parser_cost = record.get('parser_detailed_cost')
                                        return parsed_bd, None, parser_cost

                    # No matching records found
                    if found_problem_ids:
                        return None, {
                            'error': 'no_valid_parsed_breakdown',
                            'problem_id': origin_problem_id,
                            'breakdown_id': breakdown_id,
                            'message': f'Found record for {origin_problem_id} but parsed_breakdown was invalid or missing'
                        }, None
                    else:
                        # Collect all problem IDs that exist in the file for debugging
                        all_problem_ids = []
                        for record in data:
                            if isinstance(record, dict) and 'problem_id' in record:
                                all_problem_ids.append(record['problem_id'])

                        return None, {
                            'error': 'no_matching_record',
                            'problem_id': origin_problem_id,
                            'breakdown_id': breakdown_id,
                            'found_problem_ids': list(set(all_problem_ids))[:10]  # First 10 unique IDs
                        }, None

                # Fallback: handle dict-based structure (older format)
                if isinstance(data, dict) and 'parsed_breakdown' in data:
                    parser_cost = data.get('parser_detailed_cost')
                    return data['parsed_breakdown'], None, parser_cost

                return data, None, None
        except json.JSONDecodeError as e:
            return None, {
                'error': 'json_parse_error',
                'message': f'Failed to parse JSON: {str(e)}'
            }, None
        except Exception as e:
            return None, {
                'error': 'exception',
                'message': str(e),
                'type': type(e).__name__
            }, None

    def _load_formalized_json(
        self,
        origin_problem_id: str,
        round_id: int,
        breakdown_id: int
    ) -> Optional[Dict[str, Any]]:
        """Load formalization data from formalized.json for the given breakdown.

        For round 0: Matches by origin_problem_id AND breakdown_id
        For round 1+: Matches by parent_problem_id (which is the origin_problem_id passed in)

        The file is an array of problem records; we find the one matching the appropriate ID.
        """
        # Path depends on whether run_dir is already in round{N}/prover or at root level
        # Try multiple possible locations
        candidates = [
            # If run_dir is round{N}/prover, go up to round{N} and find formalizer
            self.run_dir.parent / 'formalizer' / 'formalized.json',
            # If run_dir is at root (historical structure)
            self.run_dir / f'round{round_id}' / 'formalizer' / 'formalized.json',
        ]

        formalized_file = None
        for candidate in candidates:
            if candidate.exists():
                formalized_file = candidate
                break

        if not formalized_file:
            return None

        try:
            with open(formalized_file) as f:
                data = json.load(f)

                # formalized.json is an array of problem records
                if isinstance(data, list):
                    # For round 1+, the origin_problem_id we're looking for might be in parent_problem_id
                    # Match strategy:
                    # 1. Try exact match with both ID and breakdown_id
                    # 2. Fall back to any match on ID alone

                    for record in data:
                        if isinstance(record, dict):
                            metadata = record.get('metadata', {})

                            # Check both parent_problem_id (round 1+) and origin_problem_id (round 0)
                            record_parent_id = metadata.get('parent_problem_id')
                            record_origin_id = metadata.get('origin_problem_id')
                            record_breakdown_id = metadata.get('breakdown_id')

                            # Match if:
                            # - parent_problem_id matches (round 1+), OR
                            # - origin_problem_id matches (round 0)
                            if record_parent_id == origin_problem_id or record_origin_id == origin_problem_id:
                                # Check if this record matches our breakdown_id (prefer exact match)
                                if record_breakdown_id == breakdown_id:
                                    if 'parsed_breakdown' in record:
                                        return record['parsed_breakdown']

                    # Fallback: find ANY valid formalized breakdown for this problem ID
                    for record in data:
                        if isinstance(record, dict):
                            metadata = record.get('metadata', {})
                            record_parent_id = metadata.get('parent_problem_id')
                            record_origin_id = metadata.get('origin_problem_id')

                            if record_parent_id == origin_problem_id or record_origin_id == origin_problem_id:
                                if 'parsed_breakdown' in record and record['parsed_breakdown']:
                                    parsed_bd = record['parsed_breakdown']
                                    # Verify it has valid structure (has 'lemmas' and/or 'theorem')
                                    if isinstance(parsed_bd, dict) and ('lemmas' in parsed_bd or 'theorem' in parsed_bd or 'selected_formalizations' in parsed_bd):
                                        return parsed_bd

                    return None

                # Fallback: handle dict-based structure (older format)
                if isinstance(data, dict) and 'parsed_breakdown' in data:
                    return data['parsed_breakdown']

                return data
        except Exception as e:
            print(f"Error loading formalized JSON: {e}")
            return None

    def _create_lemma_from_parsed_and_formal_list(
        self,
        lemma_id: int,
        parsed_lemma: Dict[str, Any],
        lemma_records: list,
        formal_lemma_list: Optional[list] = None
    ) -> Optional[Lemma]:
        """Create a Lemma object by combining parsed, formalized (from selected_formalizations), and proof records.

        Creates a Lemma with statement/assumptions/proof_idea from parsed breakdown,
        and wraps formalization data (formal_statement, compilation/validation) in
        multiple Formalization objects with proof attempts.

        Args:
            lemma_id: The lemma ID
            parsed_lemma: Parsed lemma data from parsed_breakdown.json
            lemma_records: Proof attempt records for this lemma
            formal_lemma_list: List of selected formalization data (from selected_formalizations)
        """
        # Get statement and assumptions from parsed lemma (shared across formalizations)
        statement = parsed_lemma.get('statement', '')
        assumptions = parsed_lemma.get('assumptions', None)
        # Note: The field is called 'proof' in the parsed_breakdown.json, map it to 'proof_idea'
        proof_idea = parsed_lemma.get('proof', None)

        # Get dependencies from the parsed breakdown JSON (already extracted during parsing)
        dependencies = parsed_lemma.get('dependencies', [])

        # Create the Lemma with statement/assumptions/proof_idea (shared across formalizations)
        lemma = Lemma(
            lemma_id=lemma_id,
            statement=statement,
            assumptions=assumptions,
            proof_idea=proof_idea,
            dependencies=dependencies
        )

        # Create Formalization objects for each selected formalization
        if formal_lemma_list:
            for i, formal_lemma in enumerate(formal_lemma_list):
                # Extract formalization reasoning
                formalization_reasoning = self._extract_thinking_from_raw(formal_lemma.get('formal_statement_raw', ''))
                if not formalization_reasoning:
                    formalization_reasoning = formal_lemma.get('model_reasoning', '') or formal_lemma.get('reasoning', '')

                # Extract formal statement
                formal_statement = formal_lemma.get('formal_statement', '') or formal_lemma.get('statement', '')

                # Extract compilation and validation data
                compilation_pass = formal_lemma.get('compilation_pass', False)
                compilation_result = formal_lemma.get('compilation_result')
                compilation_errors = formal_lemma.get('compilation_errors')
                validation_pass = formal_lemma.get('validation_pass', False)
                validation_result = formal_lemma.get('validation_result')
                validation_reasoning = formal_lemma.get('validation_reasoning')

                # Get formalization_id from the data, fall back to index
                form_id = formal_lemma.get('formalization_id', i)

                formalization = Formalization(
                    id=form_id,
                    formal_statement=formal_statement,
                    formalization_reasoning=formalization_reasoning,
                    compilation_pass=compilation_pass,
                    compilation_result=compilation_result,
                    compilation_errors=compilation_errors,
                    validation_pass=validation_pass,
                    validation_result=validation_result,
                    validation_reasoning=validation_reasoning,
                    is_selected=True,  # All items in formal_lemma_list are selected
                    detailed_cost=formal_lemma.get('detailed_cost')  # Load detailed cost from selected_formalizations
                )

                # Add proof attempts to the formalization
                for record in lemma_records:
                    attempt = self._record_to_proof_attempt(record)
                    if attempt:
                        formalization.proof_attempts.append(attempt)

                lemma.formalizations.append(formalization)

        return lemma

    def _create_lemma_from_parsed_and_formal(
        self,
        lemma_id: int,
        parsed_lemma: Dict[str, Any],
        lemma_records: list,
        formal_lemma: Optional[Dict[str, Any]] = None
    ) -> Optional[Lemma]:
        """Create a Lemma object by combining parsed, formalized, and proof records.

        Creates a Lemma with statement/assumptions/proof_idea from parsed breakdown,
        and wraps formalization data (formal_statement, compilation/validation) in
        a Formalization object with proof attempts.

        DEPRECATED: Use _create_lemma_from_parsed_and_formal_list instead for round 1+.
        """
        # Get statement and assumptions from parsed lemma (shared across formalizations)
        statement = parsed_lemma.get('statement', '')
        assumptions = parsed_lemma.get('assumptions', None)
        # Note: The field is called 'proof' in the parsed_breakdown.json, map it to 'proof_idea'
        proof_idea = parsed_lemma.get('proof', None)

        # Get dependencies from the parsed breakdown JSON (already extracted during parsing)
        dependencies = parsed_lemma.get('dependencies', [])

        # Create the Lemma with statement/assumptions/proof_idea (shared across formalizations)
        lemma = Lemma(
            lemma_id=lemma_id,
            statement=statement,
            assumptions=assumptions,
            proof_idea=proof_idea,
            dependencies=dependencies
        )

        # Create a Formalization object if we have formal lemma data
        if formal_lemma:
            # Extract formalization reasoning
            formalization_reasoning = self._extract_thinking_from_raw(formal_lemma.get('formal_statement_raw', ''))
            if not formalization_reasoning:
                formalization_reasoning = formal_lemma.get('model_reasoning', '') or formal_lemma.get('reasoning', '')

            # Extract formal statement
            formal_statement = formal_lemma.get('formal_statement', '') or formal_lemma.get('statement', '')

            # Extract compilation and validation data
            compilation_pass = formal_lemma.get('compilation_pass', False)
            compilation_result = formal_lemma.get('compilation_result')
            compilation_errors = formal_lemma.get('compilation_errors')
            validation_pass = formal_lemma.get('validation_pass', False)
            validation_result = formal_lemma.get('validation_result')
            validation_reasoning = formal_lemma.get('validation_reasoning')

            formalization = Formalization(
                id=0,
                formal_statement=formal_statement,
                formalization_reasoning=formalization_reasoning,
                compilation_pass=compilation_pass,
                compilation_result=compilation_result,
                compilation_errors=compilation_errors,
                validation_pass=validation_pass,
                validation_result=validation_result,
                validation_reasoning=validation_reasoning,
                detailed_cost=formal_lemma.get('detailed_cost')  # Load detailed cost from formal lemma data
            )

            # Add proof attempts to the formalization
            for record in lemma_records:
                attempt = self._record_to_proof_attempt(record)
                if attempt:
                    formalization.proof_attempts.append(attempt)

            lemma.formalizations.append(formalization)

        return lemma

    def _create_lemma_from_records(self, lemma_id: int, lemma_records: list) -> Optional[Lemma]:
        """Create a Lemma object from lemma records."""
        # Get statement and formal statement from first record if available
        statement = ''
        formal_statement = ''
        assumptions = None

        if lemma_records:
            first_record = lemma_records[0]
            statement = first_record.get('informal_prefix', '')
            formal_statement = first_record.get('formal_statement', '')

        lemma = Lemma(
            lemma_id=lemma_id,
            statement=statement,
            formal_statement=formal_statement,
            assumptions=assumptions
        )

        # Add proof attempts from all lemma records
        for record in lemma_records:
            attempt = self._record_to_proof_attempt(record)
            if attempt:
                lemma.proof_attempts.append(attempt)

        return lemma

    @staticmethod
    def _record_to_dict(record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a full_records entry to a dict for backward compatibility."""
        return record

    @staticmethod
    def _record_to_dict_skeleton(record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a full_records entry to a skeleton dict with only essential fields.

        This keeps only metadata and compilation results, dropping expensive fields like:
        - full_code, code
        - model_output, model_reasoning
        - model_input, messages_history_for_this_attempt
        - any other large text fields

        This significantly reduces memory usage while keeping enough data for browsing.
        """
        # Only keep essential fields
        skeleton = {
            'metadata': record.get('metadata', {}),
            'compilation_result': record.get('compilation_result', {}),
            'name': record.get('name'),
            'uid': record.get('uid'),
            'split': record.get('split'),
            'iteration': record.get('iteration'),
            'correction_round': record.get('correction_round'),
            'verify_time': record.get('verify_time'),
        }
        return {k: v for k, v in skeleton.items() if v is not None}

    def _record_to_proof_attempt(self, record: Dict[str, Any]) -> Optional[ProofAttempt]:
        """Convert a full_records entry to a ProofAttempt object."""
        try:
            metadata = record.get('metadata', {})
            compilation_result_data = record.get('compilation_result', {})

            compilation_result = CompilationResult(
                passed=compilation_result_data.get('pass', False),
                complete=compilation_result_data.get('complete', False),
                errors=compilation_result_data.get('errors', []),
                warnings=compilation_result_data.get('warnings', []),
                system_errors=compilation_result_data.get('system_errors', '').splitlines()[0] if compilation_result_data.get('system_errors') else None
            )

            # Extract lemma dependencies from code (do this even in skeleton mode to populate used_lemma_ids)
            full_code = record.get('full_code', '')
            used_lemma_ids = self._extract_lemma_dependencies_from_code(full_code) if full_code else None

            attempt = ProofAttempt(
                origin_problem_id=metadata.get('origin_problem_id', ''),
                round_id=metadata.get('round_id', 0),
                breakdown_id=metadata.get('breakdown_id', 0),
                lemma_id=metadata.get('lemma_id', -1),
                attempt_id=metadata.get('attempt_id', 0),
                iteration_id=record.get('iteration', metadata.get('iteration_id', 0)),
                correction_round_id=record.get('correction_round', metadata.get('correction_round_id', 0)),
                model_reasoning=(record.get('model_reasoning', '') or record.get('model_output', '')) if not self.skip_expensive_fields else None,
                model_input=record.get('model_input') if not self.skip_expensive_fields else None,
                code=record.get('full_code', '') if not self.skip_expensive_fields else None,
                compilation_result=compilation_result,
                formal_statement=record.get('formal_statement') if not self.skip_expensive_fields else None,
                detailed_cost=record.get('detailed_cost'),
                reasoning_summary=record.get('reasoning_summary'),  # Always load - LLM summary of reasoning trace
                compilation_summary=record.get('compilation_summary'),  # Always load - classified error summary
                used_lemma_ids=used_lemma_ids,  # Pre-extracted dependencies
                model_config_path=record.get('model_config_path') or metadata.get('model_config_path'),
            )

            return attempt
        except Exception as e:
            print(f"Error converting record to ProofAttempt: {e}")
            return None

    def _load_missing_problems_from_breakdown(self, session: Session, breakdown_file: Path, cost_index: Dict = None, component_costs_index: Dict = None, round_num: int = 0):
        """
        Load problems from breakdown.json that didn't make it to full_records,
        or supplement problems in full_records that are missing parsed_breakdown or additional breakdowns.
        This shows all original problems with available data, even those that failed at later pipeline stages.

        Args:
            session: Session object to populate
            breakdown_file: Path to breakdown.json file
            cost_index: Dict of costs indexed by (origin_problem_id, round_id, breakdown_id, lemma_id)
            round_num: Round number this breakdown.json is from (default 0)
        """
        try:
            with open(breakdown_file) as f:
                breakdown_records = json.load(f)
                if not isinstance(breakdown_records, list):
                    return

            # Group by (origin_problem_id, breakdown_id)
            problems_and_breakdowns = {}
            for record in breakdown_records:
                metadata = record.get('metadata', {})
                origin_id = metadata.get('origin_problem_id') or record.get('problem_id')
                breakdown_id = metadata.get('breakdown_id', 0)

                # For Round 1+ breakdowns in breakdown.json, extract parent_problem_id from UID
                parent_problem_id = metadata.get('parent_problem_id')
                if not parent_problem_id and round_num > 0:
                    # Round 1+ breakdown.json uses UIDs like "lemma_id_r{round}_b{breakdown}"
                    # Extract the lemma_id (parent_problem_id) from the UID
                    uid = record.get('uid', '')
                    if uid:
                        # Remove the trailing _r{round}_b{breakdown} part
                        match = re.search(r'_r\d+_b\d+$', uid)
                        if match:
                            parent_problem_id = uid[:match.start()]

                if origin_id:
                    key = (origin_id, breakdown_id)
                    if key not in problems_and_breakdowns:
                        problems_and_breakdowns[key] = []
                    record_with_parent = dict(record)
                    if parent_problem_id:
                        record_with_parent['_extracted_parent_problem_id'] = parent_problem_id
                    problems_and_breakdowns[key].append(record_with_parent)

            # Process: load missing problems or supplement existing ones with missing breakdowns
            for (origin_id, breakdown_id), records in problems_and_breakdowns.items():
                first_record = records[0]
                parent_problem_id = first_record.get('_extracted_parent_problem_id')

                if origin_id not in session.problems:
                    # Problem not in full_records at all - create from breakdown.json
                    problem = Problem(origin_problem_id=origin_id)
                    # Use extracted parent_problem_id if available (from Round 1+ breakdown), else use origin_id
                    final_parent_problem_id = parent_problem_id or origin_id
                    breakdown = Breakdown(
                        problem_id=f"{origin_id}_r{round_num}_b{breakdown_id}",
                        origin_problem_id=origin_id,
                        round_id=round_num,
                        breakdown_id=breakdown_id,
                        parent_problem_id=final_parent_problem_id,
                        name=origin_id,
                        informal_breakdown=first_record.get('informal_breakdown') if not self.skip_expensive_fields else None,
                        breakdown_prompt=first_record.get('breakdown_prompt') if not self.skip_expensive_fields else None,
                        informal_breakdown_reasoning=first_record.get('informal_breakdown_reasoning') if not self.skip_expensive_fields else None,
                        informal_prefix=first_record.get('informal_prefix'),
                        formal_statement=first_record.get('formal_statement'),
                        lean4_code=first_record.get('lean4_code'),
                        tags=first_record.get('tags', [])
                    )

                    # Try to load parsed breakdown and check for parse errors
                    # Call the full parsing method to get a proper ParsedBreakdown object
                    parsed_bd, parse_error = self._load_parsed_breakdown_from_records(
                        origin_id, 0, breakdown_id, [], {}
                    )
                    breakdown.parsed_breakdown = parsed_bd
                    if parse_error:
                        # Ensure error dict has context fields
                        if 'origin_problem_id' not in parse_error:
                            parse_error['origin_problem_id'] = origin_id
                        if 'round_id' not in parse_error:
                            parse_error['round_id'] = 0
                        if 'breakdown_id' not in parse_error:
                            parse_error['breakdown_id'] = breakdown_id
                        breakdown.parse_failure = parse_error

                    # Also load formalized breakdown (formalization data)
                    formalized_json = self._load_formalized_json(origin_id, 0, breakdown_id)
                    if formalized_json and 'lemmas' in formalized_json:
                        # Create a ParsedBreakdown object for formalized data
                        formalized_lemmas = {}
                        if 'lemmas' in formalized_json:
                            for formal_lemma_data in formalized_json['lemmas']:
                                # Use the explicit id from formal_lemma_data if available, otherwise skip
                                lemma_id = formal_lemma_data.get('id')
                                if lemma_id is None:
                                    continue

                                # Extract formalization reasoning from formal_statement_raw if available
                                formalization_reasoning = self._extract_thinking_from_raw(formal_lemma_data.get('formal_statement_raw', ''))
                                if not formalization_reasoning:
                                    formalization_reasoning = formal_lemma_data.get('model_reasoning', '') or formal_lemma_data.get('reasoning', '')

                                # Create a Formalization object with formal data
                                formalization = Formalization(
                                    id=0,
                                    formal_statement=formal_lemma_data.get('formal_statement', ''),
                                    formalization_reasoning=formalization_reasoning,
                                    compilation_pass=formal_lemma_data.get('compilation_pass', False),
                                    compilation_result=formal_lemma_data.get('compilation_result'),
                                    compilation_errors=formal_lemma_data.get('compilation_errors'),
                                    validation_pass=formal_lemma_data.get('validation_pass', False),
                                    validation_result=formal_lemma_data.get('validation_result'),
                                    validation_reasoning=formal_lemma_data.get('validation_reasoning'),
                                    detailed_cost=formal_lemma_data.get('detailed_cost')  # Load detailed cost from formal lemma data
                                )

                                # Create a Lemma with shared data
                                formal_lemma = Lemma(
                                    lemma_id=lemma_id,
                                    statement=formal_lemma_data.get('statement', ''),
                                    assumptions=formal_lemma_data.get('assumptions', None)
                                )
                                formal_lemma.formalizations.append(formalization)
                                formalized_lemmas[lemma_id] = formal_lemma

                        # Create a theorem for formalized data if available
                        formal_theorem_data = formalized_json.get('theorem', {})
                        # Extract formalization reasoning from formal_statement_raw if available
                        theorem_reasoning = self._extract_thinking_from_raw(formal_theorem_data.get('formal_statement_raw', ''))
                        if not theorem_reasoning:
                            theorem_reasoning = formal_theorem_data.get('model_reasoning', '') or formal_theorem_data.get('reasoning', '')

                        # Create a Formalization object for theorem with formal data
                        theorem_formalization = Formalization(
                            id=0,
                            formal_statement=formal_theorem_data.get('formal_statement', ''),
                            formalization_reasoning=theorem_reasoning,
                            compilation_pass=formal_theorem_data.get('compilation_pass', False),
                            compilation_result=formal_theorem_data.get('compilation_result'),
                            compilation_errors=formal_theorem_data.get('compilation_errors'),
                            validation_pass=False,  # Theorems are not validated
                            validation_result=None,
                            validation_reasoning=None,
                            detailed_cost=formal_theorem_data.get('detailed_cost')  # Load detailed cost from theorem data
                        )

                        # Create a Theorem with shared data
                        formal_theorem = Theorem(
                            statement=formal_theorem_data.get('statement', ''),
                            proof_idea=formal_theorem_data.get('proof_idea', None),
                            dependencies=formal_theorem_data.get('dependencies', [])
                        )
                        formal_theorem.formalizations.append(theorem_formalization)

                        formalized_bd = ParsedBreakdown(theorem=formal_theorem, lemmas=formalized_lemmas)
                        breakdown.formalized_breakdown = formalized_bd

                    # Attach component costs for this breakdown
                    component_cost_key = (origin_id, round_num, breakdown_id)
                    if component_costs_index and component_cost_key in component_costs_index:
                        breakdown.component_costs = component_costs_index[component_cost_key]

                    # Attach breakdown costs
                    breakdown_cost_key = (origin_id, round_num, breakdown_id, -1)
                    if cost_index and breakdown_cost_key in cost_index:
                        breakdown.detailed_cost = cost_index[breakdown_cost_key]

                    # Use composite key (parent_problem_id, round_id, breakdown_id) to avoid collisions across rounds and parent problems
                    breakdown_key = (final_parent_problem_id, round_num, breakdown_id)
                    # Only add if this breakdown doesn't already exist (from full_records)
                    if breakdown_key not in problem.breakdowns:
                        problem.breakdowns[breakdown_key] = breakdown
                    session.problems[origin_id] = problem
                else:
                    # Problem exists in full_records
                    problem = session.problems[origin_id]
                    # Use extracted parent_problem_id if available (from Round 1+ breakdown), else use origin_id
                    final_parent_problem_id = parent_problem_id or origin_id
                    breakdown_key = (final_parent_problem_id, round_num, breakdown_id)
                    if breakdown_key not in problem.breakdowns:
                        # Add missing breakdown from breakdown.json
                        breakdown = Breakdown(
                            problem_id=f"{origin_id}_r{round_num}_b{breakdown_id}",
                            origin_problem_id=origin_id,
                            round_id=round_num,
                            breakdown_id=breakdown_id,
                            parent_problem_id=final_parent_problem_id,
                            name=origin_id,
                            informal_breakdown=first_record.get('informal_breakdown') if not self.skip_expensive_fields else None,
                            breakdown_prompt=first_record.get('breakdown_prompt') if not self.skip_expensive_fields else None,
                            informal_breakdown_reasoning=first_record.get('informal_breakdown_reasoning') if not self.skip_expensive_fields else None,
                            informal_prefix=first_record.get('informal_prefix'),
                            formal_statement=first_record.get('formal_statement'),
                            lean4_code=first_record.get('lean4_code'),
                            tags=first_record.get('tags', [])
                        )

                        # Try to load parsed breakdown and check for parse errors
                        # Call the full parsing method to get a proper ParsedBreakdown object
                        parsed_bd, parse_error = self._load_parsed_breakdown_from_records(
                            origin_id, round_num, breakdown_id, [], {}
                        )
                        breakdown.parsed_breakdown = parsed_bd
                        if parse_error:
                            # Ensure error dict has context fields
                            if 'origin_problem_id' not in parse_error:
                                parse_error['origin_problem_id'] = origin_id
                            if 'round_id' not in parse_error:
                                parse_error['round_id'] = round_num
                            if 'breakdown_id' not in parse_error:
                                parse_error['breakdown_id'] = breakdown_id
                            breakdown.parse_failure = parse_error

                        # Also load formalized breakdown (formalization data)
                        formalized_json = self._load_formalized_json(origin_id, round_num, breakdown_id)
                        if formalized_json and 'lemmas' in formalized_json:
                            # Create a ParsedBreakdown object for formalized data
                            formalized_lemmas = {}
                            if 'lemmas' in formalized_json:
                                for formal_lemma_data in formalized_json['lemmas']:
                                    # Use the explicit id from formal_lemma_data if available, otherwise skip
                                    lemma_id = formal_lemma_data.get('id')
                                    if lemma_id is None:
                                        continue

                                    # Extract formalization reasoning from formal_statement_raw if available
                                    formalization_reasoning = self._extract_thinking_from_raw(formal_lemma_data.get('formal_statement_raw', ''))
                                    if not formalization_reasoning:
                                        formalization_reasoning = formal_lemma_data.get('model_reasoning', '') or formal_lemma_data.get('reasoning', '')

                                    # Create a Formalization object with formal data
                                    formalization = Formalization(
                                        id=0,
                                        formal_statement=formal_lemma_data.get('formal_statement', ''),
                                        formalization_reasoning=formalization_reasoning,
                                        compilation_pass=formal_lemma_data.get('compilation_pass', False),
                                        compilation_result=formal_lemma_data.get('compilation_result'),
                                        compilation_errors=formal_lemma_data.get('compilation_errors'),
                                        validation_pass=formal_lemma_data.get('validation_pass', False),
                                        validation_result=formal_lemma_data.get('validation_result'),
                                        validation_reasoning=formal_lemma_data.get('validation_reasoning'),
                                        detailed_cost=formal_lemma_data.get('detailed_cost')  # Load detailed cost from formal lemma data
                                    )

                                    # Create a Lemma with shared data
                                    formal_lemma = Lemma(
                                        lemma_id=lemma_id,
                                        statement=formal_lemma_data.get('statement', ''),
                                        assumptions=formal_lemma_data.get('assumptions', None)
                                    )
                                    formal_lemma.formalizations.append(formalization)
                                    formalized_lemmas[lemma_id] = formal_lemma

                            # Create a theorem for formalized data if available
                            formal_theorem_data = formalized_json.get('theorem', {})
                            # Extract formalization reasoning from formal_statement_raw if available
                            theorem_reasoning = self._extract_thinking_from_raw(formal_theorem_data.get('formal_statement_raw', ''))
                            if not theorem_reasoning:
                                theorem_reasoning = formal_theorem_data.get('model_reasoning', '') or formal_theorem_data.get('reasoning', '')

                            # Create a Formalization object for theorem with formal data
                            theorem_formalization = Formalization(
                                id=0,
                                formal_statement=formal_theorem_data.get('formal_statement', ''),
                                formalization_reasoning=theorem_reasoning,
                                compilation_pass=formal_theorem_data.get('compilation_pass', False),
                                compilation_result=formal_theorem_data.get('compilation_result'),
                                compilation_errors=formal_theorem_data.get('compilation_errors'),
                                validation_pass=False,  # Theorems are not validated
                                validation_result=None,
                                validation_reasoning=None,
                                detailed_cost=formal_theorem_data.get('detailed_cost')  # Load detailed cost from theorem data
                            )

                            # Create a Theorem with shared data
                            formal_theorem = Theorem(
                                statement=formal_theorem_data.get('statement', ''),
                                proof_idea=formal_theorem_data.get('proof_idea', None),
                                dependencies=formal_theorem_data.get('dependencies', [])
                            )
                            formal_theorem.formalizations.append(theorem_formalization)

                            formalized_bd = ParsedBreakdown(theorem=formal_theorem, lemmas=formalized_lemmas)
                            breakdown.formalized_breakdown = formalized_bd

                        # Attach component costs for this breakdown
                        component_cost_key = (origin_id, round_num, breakdown_id)
                        if component_costs_index and component_cost_key in component_costs_index:
                            breakdown.component_costs = component_costs_index[component_cost_key]

                        # Attach breakdown costs
                        breakdown_cost_key = (origin_id, round_num, breakdown_id, -1)
                        if cost_index and breakdown_cost_key in cost_index:
                            breakdown.detailed_cost = cost_index[breakdown_cost_key]

                        problem.breakdowns[breakdown_key] = breakdown
                    else:
                        # Breakdown exists - check if it needs supplementing
                        breakdown = problem.breakdowns[breakdown_key]
                        # If breakdown is missing informal/formal data, fill from breakdown.json (but respect skip_expensive_fields)
                        if not breakdown.informal_breakdown and not self.skip_expensive_fields:
                            breakdown.informal_breakdown = first_record.get('informal_breakdown')
                        if not breakdown.breakdown_prompt and not self.skip_expensive_fields:
                            breakdown.breakdown_prompt = first_record.get('breakdown_prompt')
                        if not breakdown.informal_breakdown_reasoning and not self.skip_expensive_fields:
                            breakdown.informal_breakdown_reasoning = first_record.get('informal_breakdown_reasoning')
                        if not breakdown.informal_prefix:
                            breakdown.informal_prefix = first_record.get('informal_prefix')
                        if not breakdown.formal_statement:
                            breakdown.formal_statement = first_record.get('formal_statement')
                        if not breakdown.lean4_code:
                            breakdown.lean4_code = first_record.get('lean4_code')

        except Exception as e:
            print(f"Error loading missing problems from breakdown.json: {e}")

    def _load_and_apply_validation_results(self, session: Session):
        """Load validation_results.json and enrich formalization validation data in the session.

        Validation results contain the reasoning even when verdict is 'no', which is important
        for understanding why validation failed.
        """
        # Try to find validation_results.json
        validation_file = self.run_dir / 'round0' / 'formalizer' / 'validation_results.json'
        if not validation_file.exists():
            return

        try:
            with open(validation_file) as f:
                validation_data = json.load(f)
                if not isinstance(validation_data, list):
                    return

            # Build a map of (origin_problem_id, breakdown_id, lemma_id, formalization_id) -> validation result
            # This ensures we match the exact formalization that was validated
            validation_map = {}
            for item in validation_data:
                # Try to get metadata first (new format), fall back to top-level fields (old format)
                metadata = item.get('metadata', {})
                origin_problem_id = metadata.get('origin_problem_id') or item.get('problem_id')
                breakdown_id = metadata.get('breakdown_id', 0)
                lemma_id = metadata.get('lemma_id', item.get('lemma_id'))
                formalization_id = metadata.get('formalization_id', item.get('sample_idx', 0))

                if origin_problem_id and lemma_id is not None:
                    key = (origin_problem_id, breakdown_id, lemma_id, formalization_id)
                    # Store validation result with the exact formalization it belongs to
                    validation_map[key] = {
                        'verdict': item.get('verdict'),
                        'reasoning': item.get('reasoning', ''),
                        'raw_response': item.get('raw_response', '')
                    }

            # Apply validation data to formalizations in the session
            for problem in session.problems.values():
                for breakdown in problem.breakdowns.values():
                    # Extract origin problem ID and breakdown ID from the breakdown object
                    origin_id = breakdown.origin_problem_id
                    bd_id = breakdown.breakdown_id

                    # Update formalized lemmas with validation data
                    if breakdown.formalized_breakdown and breakdown.formalized_breakdown.lemmas:
                        for lemma_id, lemma in breakdown.formalized_breakdown.lemmas.items():
                            # Try to apply validation for each formalization using formalization_id
                            for form_idx, formalization in enumerate(lemma.formalizations):
                                key = (origin_id, bd_id, lemma_id, form_idx)
                                if key in validation_map:
                                    val_result = validation_map[key]
                                    formalization.validation_pass = val_result['verdict'] == 'yes'
                                    formalization.validation_result = val_result['raw_response']
                                    formalization.validation_reasoning = val_result['reasoning']

        except Exception as e:
            print(f"Error loading validation results: {e}")

    @staticmethod
    def _extract_thinking_from_raw(raw_text: str) -> str:
        """Extract thinking/reasoning from formal_statement_raw which may contain <think> tags."""
        if not raw_text:
            return ''

        import re
        # Try to extract content between <think> and </think> tags
        match = re.search(r'<think>(.*?)</think>', raw_text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            # Clean up the thinking text
            lines = thinking.split('\n')
            cleaned = [line.strip() for line in lines if line.strip()]
            return '\n'.join(cleaned)

        return ''

    def _link_recursive_attempts(self, session: 'Session') -> None:
        """
        Organize problems into round 0 breakdowns and recursive attempts.

        After all problems are loaded, this method:
        1. Separates round 0 breakdowns from round 1+ breakdowns in each problem
        2. Creates separate Problem objects for round 1+ recursive attempts
        3. Links recursive Problems to the lemmas they're attempting to solve
        4. Stores recursive Problems in problem.recursive_attempts list

        This ensures:
        - problem.breakdowns contains only round 0 (for problem browser)
        - problem.recursive_attempts contains separate Problems for round 1+
        - Lemma.recursive_attempt points to the Problem attempting to solve it
        - All aggregation methods (is_solved, get_cost, etc.) work across both
        """
        # Build a map of lemma UIDs to their lemmas (from all rounds)
        # Format: lemma_uid (e.g., "mathd_algebra_209_r0_b0_l1") -> (problem, lemma)
        # This enables nested recursion where round 2+ can link to round 1 lemmas
        lemma_uid_map: Dict[str, Tuple[Problem, Lemma]] = {}

        # First pass: collect all lemmas (all rounds) and separate round 0 from round 1+
        recursive_breakdowns_by_parent: Dict[str, Dict[tuple, Breakdown]] = {}

        for origin_problem_id, problem in session.problems.items():
            round0_breakdowns = {}

            # Separate round 0 from round 1+ breakdowns
            for (parent_problem_id, round_id, breakdown_id), breakdown in problem.breakdowns.items():
                if round_id == 0:
                    # Keep round 0 breakdowns
                    round0_breakdowns[(parent_problem_id, round_id, breakdown_id)] = breakdown

                    # Collect lemmas for UID mapping
                    if breakdown.parsed_breakdown:
                        for lemma_id, lemma in breakdown.parsed_breakdown.lemmas.items():
                            base_uid = f"{origin_problem_id}_r0_b{breakdown_id}_l{lemma_id}"
                            lemma_uid_map[base_uid] = (problem, lemma)
                else:
                    # Store round 1+ breakdowns for later organization as recursive attempts
                    if parent_problem_id not in recursive_breakdowns_by_parent:
                        recursive_breakdowns_by_parent[parent_problem_id] = {}
                    recursive_breakdowns_by_parent[parent_problem_id][(parent_problem_id, round_id, breakdown_id)] = breakdown

            # Replace breakdowns with only round 0
            # Round 1+ breakdowns will be organized as recursive attempts linked to lemmas
            problem.breakdowns = round0_breakdowns

        # Second pass: create recursive Problem objects for each parent_problem_id
        processed_parents = set()

        for parent_problem_id, breakdowns_dict in recursive_breakdowns_by_parent.items():
            if parent_problem_id in processed_parents:
                continue
            processed_parents.add(parent_problem_id)

            # Extract base UID from parent_problem_id (handle formalization_id suffix)
            match = re.match(r'(.+_r\d+_b\d+_l\d+)', parent_problem_id)
            base_uid = match.group(1) if match else parent_problem_id

            # Find the source lemma this recursive attempt is trying to prove
            if base_uid in lemma_uid_map:
                source_problem, source_lemma = lemma_uid_map[base_uid]

                # Create a recursive Problem object
                recursive_problem = Problem(
                    origin_problem_id=parent_problem_id,
                    breakdowns=breakdowns_dict,  # All round 1+ breakdowns for this parent
                    difficulty=None
                )

                # Link to lemma and add to source problem's recursive attempts
                source_lemma.recursive_attempt = recursive_problem
                source_problem.recursive_attempts.append(recursive_problem)

                # Collect lemmas from this recursive problem for nested recursion (round 2+)
                # This allows round 2+ recursion to link to lemmas from round 1 recursive problems
                for (_, rec_round_id, rec_breakdown_id), breakdown in breakdowns_dict.items():
                    if breakdown.parsed_breakdown:
                        for lemma_id, lemma in breakdown.parsed_breakdown.lemmas.items():
                            # Create UID for this recursive problem's lemma
                            rec_lemma_uid = f"{parent_problem_id}_r{rec_round_id}_b{rec_breakdown_id}_l{lemma_id}"
                            lemma_uid_map[rec_lemma_uid] = (recursive_problem, lemma)

    def _calculate_initial_attempt_indices(self, session: 'Session') -> None:
        """
        Calculate initial_attempt_index for all proof attempts and organize by correction round.

        For each formalization:
        - In correction_round 0: initial_attempt_index = attempt_id
        - In correction_round N (N > 0): initial_attempt_index chains from previous round
          - Maps failed attempts from round N-1 to positions in round N

        This allows tracking which failed attempts are being corrected in subsequent rounds.
        """
        for problem in session.problems.values():
            self._calculate_indices_for_problem(problem)

    def _calculate_indices_for_problem(self, problem: 'Problem') -> None:
        """
        Recursively calculate initial_attempt_indices for a problem and its recursive attempts.
        """
        # Process round 0 breakdowns
        for breakdown in problem.breakdowns.values():
            if breakdown.parsed_breakdown:
                # Process theorem formalizations
                self._calculate_indices_for_formalizations(breakdown.parsed_breakdown.theorem.formalizations)

                # Process lemma formalizations
                for lemma in breakdown.parsed_breakdown.lemmas.values():
                    self._calculate_indices_for_formalizations(lemma.formalizations)

        # Process recursive attempts (round 1+)
        for recursive_problem in problem.recursive_attempts:
            self._calculate_indices_for_problem(recursive_problem)

    def _calculate_indices_for_formalizations(self, formalizations: List['Formalization']) -> None:
        """
        For a list of formalizations, calculate initial_attempt_index for all attempts.
        """
        for formalization in formalizations:
            self._calculate_indices_for_formalization(formalization)

    def _calculate_indices_for_formalization(self, formalization: 'Formalization') -> None:
        """
        For a single formalization, organize attempts by correction round and calculate indices.

        Algorithm:
        - Round 0: initial_attempt_index = attempt_id
        - Round N>0: Map failed attempts from round N-1 to positions in round N
        """
        if not formalization.proof_attempts:
            return

        # Organize attempts by correction_round
        attempts_by_round: Dict[int, List['ProofAttempt']] = {}
        for attempt in formalization.proof_attempts:
            corr_round = attempt.correction_round_id
            if corr_round not in attempts_by_round:
                attempts_by_round[corr_round] = []
            attempts_by_round[corr_round].append(attempt)

        # Build proof_attempts_by_round dict
        formalization.proof_attempts_by_round = attempts_by_round

        # Calculate initial_attempt_index for each round
        sorted_rounds = sorted(attempts_by_round.keys())

        for corr_round in sorted_rounds:
            current_attempts = attempts_by_round[corr_round]

            if corr_round == 0:
                # Round 0: initial_attempt_index = attempt_id
                for attempt in current_attempts:
                    attempt.initial_attempt_index = attempt.attempt_id
            else:
                # Round N: Map failed attempts from round N-1
                previous_attempts = attempts_by_round.get(corr_round - 1, [])

                # Collect initial_attempt_index of failed attempts from previous round
                failed_initial_indices = []
                for prev_attempt in previous_attempts:
                    if not prev_attempt.is_passing():
                        failed_initial_indices.append(prev_attempt.initial_attempt_index)

                # Map current attempts to failed attempts from previous round
                for i, current_attempt in enumerate(current_attempts):
                    if i < len(failed_initial_indices):
                        current_attempt.initial_attempt_index = failed_initial_indices[i]
                    # If more attempts than failed indices, something is inconsistent
                    # but we handle it gracefully

    def _extract_lemma_dependencies_from_code(self, code: str) -> Optional[Set[int]]:
        """
        Extract lemma dependencies from proof code without storing the code.

        Returns Set of lemma IDs used in the proof, or None if code is empty.
        """
        if not code:
            return None

        # Find proof body (after ":= by")
        proof_match = re.search(r':=\s+by\s+(.*)', code, re.DOTALL)
        if not proof_match:
            return None

        proof_body = proof_match.group(1)

        # Extract axiom names
        axiom_pattern = r'\baxiom\s+(\w+)'
        axioms = re.findall(axiom_pattern, code)

        # Check which axioms are actually used
        used_lemma_ids = set()
        for axiom_name in axioms:
            pattern = r'\b' + re.escape(axiom_name) + r'\b'
            if re.search(pattern, proof_body):
                # Extract lemma ID from name
                lemma_id = self._extract_lemma_id_from_name(axiom_name)
                if lemma_id is not None:
                    used_lemma_ids.add(lemma_id)

        return used_lemma_ids if used_lemma_ids else None

    def _find_project_root(self) -> Optional[Path]:
        """
        Find the project root by searching up the directory tree.

        Looks for a directory containing both 'configs' and 'seed_prover' directories.

        Returns:
            Path to project root, or None if not found.
        """
        current = Path.cwd()

        # Try starting from current working directory and moving up
        for _ in range(10):  # Limit search depth
            if (current / 'configs').exists() and (current / 'seed_prover').exists():
                return current

            parent = current.parent
            if parent == current:
                # Reached filesystem root
                break
            current = parent

        # Also try from the module's location
        module_dir = Path(__file__).parent.parent
        current = module_dir
        for _ in range(5):
            if (current / 'configs').exists() and (current / 'seed_prover').exists():
                return current

            parent = current.parent
            if parent == current:
                break
            current = parent

        return None

    def _run_on_the_fly_summarization(self):
        """
        Run reasoning trace and/or compilation summarization on-the-fly before loading data.

        This discovers all rounds (0, 1, 2, ...) and all iterations (iter0, iter1, ...),
        runs the AttemptSummarizerComponent on the full_records and code_compilation files,
        and then consolidates the updated records into the top-level full_records directory.
        """
        # Lazy import to avoid dependency issues
        try:
            # Ensure parent directory is in path for imports
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            from prover.components.attempt_summarizer import AttemptSummarizerComponent
            from seed_prover.consolidation_utils import consolidate_to_base_full_records
        except ImportError as e:
            logger.warning(f"Cannot import summarization components: {e}. Skipping on-the-fly summarization.")
            return

        logger.info("Starting on-the-fly summarization...")

        # Find project root to resolve config paths
        # Search up the directory tree for a directory containing 'configs' and 'seed_prover'
        project_root = self._find_project_root()
        if not project_root:
            logger.warning("Could not find project root, skipping on-the-fly summarization")
            return

        # Resolve absolute path to model config
        model_config_path = project_root / 'configs' / 'models' / 'openai' / 'oss-20b-low.yaml'
        if not model_config_path.exists():
            logger.warning(f"Model config not found at {model_config_path}, skipping on-the-fly summarization")
            return

        # Create a modified config for this run with absolute paths
        config = {
            'summarization': {
                'override_existing_summaries': True,  # Always override when explicitly running summarization
                'reasoning_trace': {
                    'enabled': self.enable_reasoning_summary,
                    'model_config': str(model_config_path),
                    'template': str(project_root / 'prover' / 'template' / 'reasoning_summarizer' / 'reasoning_summarizer.md')
                },
                'compilation': {
                    'enabled': self.enable_compilation_summary
                }
            }
        }

        # Discover all rounds and iterations
        rounds_to_process = []
        round_num = 0
        while True:
            round_dir = self.run_dir / f'round{round_num}'
            if round_dir.exists():
                # Discover iterations in this round
                prover_dir = round_dir / 'prover'
                if prover_dir.exists():
                    rounds_to_process.append((round_num, prover_dir))
                round_num += 1
            else:
                break

        if not rounds_to_process:
            logger.warning(f"No rounds found in {self.run_dir}")
            return

        # Process each round/iteration combination
        total_summaries = 0
        failed_summaries = 0

        for round_num, prover_dir in rounds_to_process:
            # Discover iterations in this round
            iter_dirs = sorted([
                d for d in prover_dir.glob('iter*')
                if d.is_dir()
            ])

            if not iter_dirs:
                # If no explicit iter directories, check for full_records directly
                iter_dirs = [prover_dir]

            for iter_dir in iter_dirs:
                try:
                    logger.info(f"Processing round{round_num} {iter_dir.name}...")

                    # Create summarizer with this iteration's output dir
                    summarizer = AttemptSummarizerComponent(
                        name='on_the_fly_summarizer',
                        component_config=config.get('summarization', {}),
                        global_config={'output_dir': str(iter_dir)}
                    )

                    # Process this iteration
                    logger.info(f"  - reasoning_trace enabled: {summarizer.reasoning_trace_enabled}")
                    logger.info(f"  - compilation enabled: {summarizer.compilation_enabled}")

                    # Process initial round (round_num=0)
                    summarizer.process([], round_num=0)
                    total_summaries += 1

                    # Process correction rounds (round_num=1, 2, 3, ...)
                    corr_round = 1
                    while True:
                        # Check if correction round files exist
                        full_records_corr = iter_dir / f'full_records_corr{corr_round}.json'
                        if full_records_corr.exists():
                            logger.info(f"  Processing correction round {corr_round}...")
                            summarizer.process([], round_num=corr_round)
                            total_summaries += 1
                            corr_round += 1
                        else:
                            break

                    logger.info(f"  ✓ Completed round{round_num} {iter_dir.name}")

                except Exception as e:
                    logger.warning(f"Failed to summarize round{round_num} {iter_dir.name}: {e}")
                    failed_summaries += 1

        # After all rounds are summarized, consolidate iter-level records to round-level first
        try:
            logger.info("Consolidating iter-level records to round-level...")
            for round_num, prover_dir in rounds_to_process:
                self._consolidate_iter_to_round(round_num, prover_dir)
        except Exception as e:
            logger.error(f"Failed to consolidate iter-level records: {e}")

        # Then consolidate from round-level to top-level full_records
        try:
            logger.info("Consolidating summarized records to top-level...")
            consolidate_to_base_full_records(
                base_output_dir=str(self.run_dir),
                round_num=None,  # Consolidate all rounds
                verbosity=1
            )
        except Exception as e:
            logger.error(f"Failed to consolidate records: {e}")

        # Log results
        logger.info(f"Summarization complete: {total_summaries} successful, {failed_summaries} failed")

    def _consolidate_iter_to_round(self, round_num: int, prover_dir: Path):
        """
        Consolidate full_records from iter directories to round-level full_records.

        Exactly replicates RecursiveProverComponent._consolidate_results() logic including:
        - Merging full_records with compilation results and costs
        - Adding iteration metadata
        - Grouping by origin_problem_id
        - Saving to round{N}/prover/full_records/ split by problem
        """
        from collections import defaultdict
        from jload import jload, jsave

        verbosity = 1
        consolidated_records = []

        # Helper function to merge full_records with compilation results and costs
        # (exact copy from RecursiveProverComponent)
        def merge_records(full_records_file, compilation_file, iteration, correction_round):
            if not full_records_file.exists():
                return []

            full_records = jload(str(full_records_file))

            # Load compilation results if available
            compilation_map = {}
            if compilation_file.exists():
                compilation_results = jload(str(compilation_file))
                for comp in compilation_results:
                    if isinstance(comp, dict) and "uid" in comp:
                        compilation_map[comp["uid"]] = comp

            # Load inference costs if available
            cost_map = {}
            to_inference_file = full_records_file.parent / "to_inference_codes.json"
            if to_inference_file.exists():
                try:
                    inference_records = jload(str(to_inference_file))
                    for record in inference_records:
                        if isinstance(record, dict) and "uid" in record:
                            detailed_cost = record.get("detailed_cost")
                            if detailed_cost:
                                cost_map[record["uid"]] = detailed_cost
                except Exception as e:
                    if verbosity >= 2:
                        logger.warning(f"Error loading inference costs from {to_inference_file}: {e}")

            # Merge compilation results and costs into full_records
            merged = []
            for record in full_records:
                # Add iteration metadata
                record["iteration"] = iteration
                record["correction_round"] = correction_round

                # Add compilation result if available (match by uid)
                uid = record.get("uid")
                if uid and uid in compilation_map:
                    comp = compilation_map[uid]
                    record["compilation_result"] = comp.get("compilation_result", {})
                    record["compilation_summary"] = comp.get("compilation_summary")
                    record["verify_time"] = comp.get("verify_time")
                    record["code"] = comp.get("code")

                # Add detailed cost if available (match by uid)
                if uid and uid in cost_map:
                    record["detailed_cost"] = cost_map[uid]

                merged.append(record)

            return merged

        # Load all records from all iterations (matching RecursiveProverComponent logic)
        for iteration in range(100):  # Reasonable upper limit
            iter_dir = prover_dir / f'iter{iteration}'
            if not iter_dir.exists():
                continue

            # Load initial round
            full_records_file = iter_dir / 'full_records.json'
            compilation_file = iter_dir / 'code_compilation_repl.json'

            merged_records = merge_records(full_records_file, compilation_file, iteration, 0)
            consolidated_records.extend(merged_records)

            if verbosity >= 2 and merged_records:
                logger.info(f"  Iteration {iteration}: Loaded {len(merged_records)} records from initial round")

            # Load correction rounds
            corr_round = 1
            while True:
                full_records_corr = iter_dir / f'full_records_corr{corr_round}.json'
                compilation_corr = iter_dir / f'code_compilation_repl_corr{corr_round}.json'

                if full_records_corr.exists():
                    merged_records = merge_records(full_records_corr, compilation_corr, iteration, corr_round)
                    consolidated_records.extend(merged_records)

                    if verbosity >= 2 and merged_records:
                        logger.info(f"  Iteration {iteration}: Loaded {len(merged_records)} records from correction round {corr_round}")
                    corr_round += 1
                else:
                    break

        # Group records by origin_problem_id (same as RecursiveProverComponent)
        records_by_problem = defaultdict(list)

        for record in consolidated_records:
            metadata = record.get('metadata')
            if metadata:
                origin_problem_id = get_origin_problem_id(metadata)
            else:
                origin_problem_id = record.get('name', 'unknown')
                if verbosity >= 1:
                    logger.warning(f"Record missing metadata, using fallback: {origin_problem_id}")

            records_by_problem[origin_problem_id].append(record)

        # Create round-level full_records directory
        round_full_records_dir = prover_dir / 'full_records'
        round_full_records_dir.mkdir(parents=True, exist_ok=True)

        # Save records for each problem (same as RecursiveProverComponent)
        if verbosity >= 1:
            logger.info(f"Saving {len(consolidated_records)} records split by origin_problem_id...")

        for origin_problem_id, records in records_by_problem.items():
            safe_filename = origin_problem_id.replace('/', '_').replace('\\', '_').replace(':', '_')
            problem_file = round_full_records_dir / f'{safe_filename}.json'
            try:
                jsave(records, str(problem_file))
                if verbosity >= 2:
                    logger.info(f"  {origin_problem_id}: {len(records)} records → {safe_filename}.json")
            except Exception as e:
                logger.error(f"Failed to save {problem_file}: {e}")

        if verbosity >= 1:
            logger.info(f"Consolidated {len(consolidated_records)} records into {len(records_by_problem)} problems")

    @staticmethod
    def _extract_lemma_id_from_name(name: str) -> Optional[int]:
        """Extract lemma ID from lemma name."""
        # Pattern 1: formalized lemmas with format "lemma{n}_f{j}"
        matches = re.findall(r'_lemma(\d+)_f\d+', name)
        if matches:
            return int(matches[-1])

        # Pattern 2: "lemmaX" or "lX" anywhere in the name (with optional trailing underscore)
        match = re.search(r'_?(?:lemma|l)(\d+)(?:_|$|(?=\s))', name)
        if match:
            return int(match.group(1))

        return None

