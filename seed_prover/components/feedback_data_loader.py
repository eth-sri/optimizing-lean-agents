import os
import json
from loguru import logger
from collections import defaultdict

from seed_prover.core.component import Component
from metadata_utils import get_origin_problem_id, create_lemma_key, create_formalization_key, generate_uid
from seed_prover.utils import strip_preamble

class FeedbackDataLoaderComponent(Component):
    """
    Loads failed lemmas from a previous round to be used as input for a new round of breakdown and proving.
    """

    def __init__(self, name, config, global_config):
        super().__init__(name, config, global_config)
        # Proportion threshold for recursive breakdown
        # If prop_failed_recursive is 0.3, then lemmas with >= 30% non-complete attempts are included
        # Default is 1.0 (only include lemmas where ALL attempts failed - preserves old behavior)
        self.prop_failed_recursive = config.get('prop_failed_recursive', 1.0)

    def process(self, data, round_num):
        """
        Load failed lemmas from the previous round.

        Args:
            data (list): The data from the previous component (should be empty for the first component in a round).
            round_num (int): The current round number.

        Returns:
            list: A list of problems (failed lemmas) to be processed in the current round.
        """
        logger.info(f"Running FeedbackDataLoader for round {round_num}")
        if round_num == 0:
            logger.info("FeedbackDataLoaderComponent is passing through input data unchanged for round 0.")
            return data

        output_dir = self.global_config.get("output_dir")
        prev_round_num = round_num - 1

        # Load attempted lemmas (now a list of metadata dicts)
        attempted_lemmas_path = os.path.join(output_dir, f"round{prev_round_num}", "prover", "attempted_lemmas.json")
        if not os.path.exists(attempted_lemmas_path):
            logger.warning(f"Attempted lemmas file not found at: {attempted_lemmas_path}")
            return []

        with open(attempted_lemmas_path, 'r') as f:
            attempted_metadata_list = json.load(f)

        logger.info(f"Found {len(attempted_metadata_list)} attempted lemmas in round {prev_round_num}")

        # Load full_records to get compilation results
        full_records_dir = os.path.join(output_dir, "full_records")
        if not os.path.exists(full_records_dir):
            logger.warning(f"Full records directory not found at: {full_records_dir}")
            return []

        # Load all records
        all_records = []
        for filename in os.listdir(full_records_dir):
            if filename.endswith(".json"):
                with open(os.path.join(full_records_dir, filename), 'r') as f:
                    all_records.extend(json.load(f))

        # Build a map from lemma_key to records for quick lookup
        records_by_lemma_key = defaultdict(list)
        for record in all_records:
            metadata = record.get("metadata", {})
            lemma_key = create_lemma_key(metadata)
            if lemma_key:
                records_by_lemma_key[lemma_key].append(record)

        # For each attempted lemma, check if it was solved
        failed_lemmas = []
        for attempted_metadata in attempted_metadata_list:
            lemma_key = create_lemma_key(attempted_metadata)
            if not lemma_key:
                continue

            # Get all attempts for this lemma
            attempts = records_by_lemma_key.get(lemma_key, [])
            if not attempts:
                logger.warning(f"No records found for attempted lemma: {lemma_key}")
                continue

            # Filter out correction round attempts - only count regular attempts
            regular_attempts = []
            for attempt in attempts:
                attempt_metadata = attempt.get("metadata", {})
                # Include attempts without correction_round_id or with correction_round_id == 0
                if attempt_metadata.get("correction_round_id", 0) == 0:
                    regular_attempts.append(attempt)

            if not regular_attempts:
                logger.warning(f"No regular (non-correction) attempts found for lemma: {lemma_key}")
                continue

            # Count attempts with complete=False
            incomplete_count = 0
            for attempt in regular_attempts:
                comp_res = attempt.get("compilation_result", {})
                if not comp_res.get("complete", False):
                    incomplete_count += 1

            # Calculate proportion of incomplete attempts
            total_attempts = len(regular_attempts)
            prop_incomplete = incomplete_count / total_attempts if total_attempts > 0 else 0

            # Include lemma if proportion of incomplete attempts meets or exceeds threshold
            should_include = prop_incomplete >= self.prop_failed_recursive

            logger.debug(f"Lemma {lemma_key}: {incomplete_count}/{total_attempts} incomplete ({prop_incomplete:.2%}), "
                        f"threshold={self.prop_failed_recursive:.2%}, include={should_include}")

            if should_include:
                # Find the formalization record (has formalization_id but no attempt_id)
                formalization_record = None
                for attempt in attempts:
                    attempt_metadata = attempt.get("metadata", {})
                    # Look for records with formalization_id but no attempt_id
                    if "formalization_id" in attempt_metadata and "attempt_id" not in attempt_metadata:
                        formalization_record = attempt
                        break

                if formalization_record:
                    failed_lemmas.append(formalization_record)
                else:
                    # Fallback to last attempt if no formalization found (shouldn't happen normally)
                    logger.warning(f"No formalization record found for {lemma_key}, using last attempt as fallback")
                    failed_lemmas.append(attempts[-1])

        logger.info(f"Found {len(failed_lemmas)} lemmas for recursive breakdown from round {prev_round_num} "
                   f"(using threshold prop_failed_recursive={self.prop_failed_recursive:.2%})")

        # Load parsed breakdowns from previous round to get informal lemma statements
        parsed_breakdown_path = os.path.join(output_dir, f"round{prev_round_num}", "breakdown_parser", "parsed_breakdown.json")
        parsed_breakdowns = {}
        if os.path.exists(parsed_breakdown_path):
            with open(parsed_breakdown_path, 'r') as f:
                for item in json.load(f):
                    item_metadata = item.get("metadata", {})
                    # Create a lookup key using origin_problem_id and breakdown_id
                    from metadata_utils import get_breakdown_key
                    breakdown_key = get_breakdown_key(item_metadata)
                    parsed_breakdowns[breakdown_key] = item.get("parsed_breakdown", {})

        logger.info(f"Loaded {len(parsed_breakdowns)} parsed breakdowns from round {prev_round_num}")

        new_problems = []
        for lemma in failed_lemmas:
            metadata = lemma["metadata"]
            origin_problem_id = get_origin_problem_id(metadata)

            # For round 1+, parent_problem_id is the UID of the failed lemma from previous round
            # This ensures consistent metadata structure across all rounds
            new_metadata = {
                "origin_problem_id": origin_problem_id,
                "parent_problem_id": create_formalization_key(metadata),  # UID of failed lemma
                "round_id": round_num,
            }

            # Get the informal statement from the parsed breakdown
            from metadata_utils import get_breakdown_key
            breakdown_key = get_breakdown_key(metadata)
            parsed_breakdown = parsed_breakdowns.get(breakdown_key, {})

            informal_stmt = None
            lemma_id = metadata.get("lemma_id")

            if parsed_breakdown:
                # If lemma_id is -1, it's the theorem
                if lemma_id == -1:
                    theorem_data = parsed_breakdown.get("theorem", {})
                    informal_stmt = theorem_data.get("statement", "")
                else:
                    # Find the lemma with matching id
                    lemmas = parsed_breakdown.get("lemmas", [])
                    for lemma_data in lemmas:
                        if lemma_data.get("id") == lemma_id:
                            # Combine statement, assumptions, and proof for fuller context
                            stmt_parts = []
                            if lemma_data.get("statement"):
                                stmt_parts.append(lemma_data.get("statement"))
                            if lemma_data.get("assumption"):
                                stmt_parts.append(f"Assumptions: {lemma_data.get('assumption')}")
                            # if lemma_data.get("proof"):
                            #     stmt_parts.append(f"Proof idea: {lemma_data.get('proof')}")
                            informal_stmt = "\n\n".join(stmt_parts)
                            break

            # Fallback to existing fields if not found in parsed breakdown
            if not informal_stmt:
                logger.warning(f"Could not find informal statement in parsed breakdown for {breakdown_key}, lemma_id={lemma_id}")
                informal_stmt = lemma.get("informal_prefix") or lemma.get("statement") or ""

            # Get formal statement
            formal_stmt = lemma.get("formal_statement") or lemma.get("lean4_code") or lemma.get("full_code")

            # Strip axioms and preamble from formal statement to give model a clean slate
            # This removes dependency axioms from round 0, allowing the model to find new approaches
            if formal_stmt:
                formal_stmt = strip_preamble(formal_stmt)

            new_problem = {
                "metadata": new_metadata,
                "statement": informal_stmt,
                "formal_statement": formal_stmt,
                "lean4_code": formal_stmt,
                "original_statement": "",  # Skip original statement for now
            }
            new_problems.append(new_problem)

        return new_problems
