import os
import json
from typing import Dict, List, Any, Set
from loguru import logger
from jload import jload, jsave
from core.component import Component
from utils import extract_axiom_names, check_if_axiom_used
from metadata_utils import add_lemma, generate_uid, get_breakdown_key, create_lemma_key, create_formalization_key


class DependencyFilterComponent(Component):
    """
    Component to extract and filter lemma dependencies based on actually used axioms in proofs.

    This component:
    1. Loads compilation results from the previous iteration
    2. Identifies proofs with pass=True AND complete=True
    3. Extracts axioms actually used in these proofs
    4. Maps axiom names to actual lemma IDs
    5. Filters out already-attempted lemmas
    6. Returns only new lemmas to prove
    """

    def __init__(self, name, component_config, global_config):
        super().__init__(name, component_config, global_config)

    def process(
        self,
        data_list: List[Dict[str, Any]],
        round_num: int = 0,
        iteration: int = 1,
        all_lemmas: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract dependencies from successful proofs and filter lemmas.

        Args:
            data_list: Original data (passed through)
            round_num: Current round number
            iteration: Previous iteration number (we look at this iteration's results)
            all_lemmas: Complete list of all available lemmas with their metadata

        Returns:
            List of lemmas to prove in the next iteration
        """
        verbosity = self.global_config.get('verbosity', 3)
        base_output_dir = self.global_config.get("output_dir")
        prover_dir = os.path.join(base_output_dir, f"round{round_num}", "prover")

        if verbosity >= 1:
            logger.info(f"DependencyFilterComponent: Extracting dependencies from iteration {iteration}")

        # Load attempted lemmas tracker (list of metadata dicts)
        attempted_file = os.path.join(prover_dir, "attempted_lemmas.json")
        attempted_lemmas = []  # List of metadata dicts
        attempted_keys = set()  # Set of formalization_keys for quick lookup
        if os.path.exists(attempted_file):
            with open(attempted_file, 'r') as f:
                attempted_lemmas = json.load(f)
                # Build lookup set from formalization keys (to distinguish different formalizations)
                for metadata in attempted_lemmas:
                    formalization_key = create_formalization_key(metadata)
                    if formalization_key:
                        attempted_keys.add(formalization_key)

        if verbosity >= 2:
            logger.info(f"  Previously attempted lemmas: {len(attempted_lemmas)}")

        # Load compilation results from the specified iteration
        iter_dir = os.path.join(prover_dir, f"iter{iteration}")
        if not os.path.exists(iter_dir):
            logger.warning(f"Iteration directory not found: {iter_dir}")
            return []

        # Helper function to merge full_records with compilation results
        def merge_records(full_records_file, compilation_file):
            """Merge full_records (with full_code and metadata) and compilation results."""
            merged = []

            # Load full_records if available
            full_records = []
            if os.path.exists(full_records_file):
                full_records = jload(full_records_file)

            # Load compilation results if available
            compilation_map = {}
            if os.path.exists(compilation_file):
                compilation_results = jload(compilation_file)
                for comp in compilation_results:
                    if isinstance(comp, dict) and "name" in comp:
                        compilation_map[comp["name"]] = comp

            # Merge compilation results into full_records using uid as key
            for record in full_records:
                uid = record.get("uid")
                if not uid:
                    logger.warning(f"Record missing uid, skipping: {record.get('name', 'unknown')}")
                    continue

                if uid in compilation_map:
                    comp = compilation_map[uid]
                    # Add compilation result fields to the record
                    record["compilation_result"] = comp.get("compilation_result", {})
                    record["verify_time"] = comp.get("verify_time")

                merged.append(record)

            return merged

        # Load and merge all records (including correction rounds)
        compilation_results = []

        # Load initial round
        full_records_file = os.path.join(iter_dir, "full_records.json")
        comp_file = os.path.join(iter_dir, "code_compilation_repl.json")

        initial_results = merge_records(full_records_file, comp_file)
        compilation_results.extend(initial_results)
        if verbosity >= 2:
            logger.info(f"  Loaded {len(initial_results)} merged records from initial round")

        # Load correction rounds
        corr_round = 1
        while True:
            full_records_corr = os.path.join(iter_dir, f"full_records_corr{corr_round}.json")
            corr_file = os.path.join(iter_dir, f"code_compilation_repl_corr{corr_round}.json")

            if os.path.exists(full_records_corr) or os.path.exists(corr_file):
                corr_results = merge_records(full_records_corr, corr_file)
                compilation_results.extend(corr_results)
                if verbosity >= 2:
                    logger.info(f"  Loaded {len(corr_results)} merged records from correction round {corr_round}")
                corr_round += 1
            else:
                break

        if not compilation_results:
            logger.warning(f"No compilation results found in {iter_dir}")
            return []

        if verbosity >= 1:
            logger.info(f"  Total merged records: {len(compilation_results)}")

        # Extract used axioms from successful proofs and map them to lemmas
        # Process each successful proof individually to handle breakdown-specific mappings
        successful_count = 0
        filtered_lemmas = []
        lemmas_to_add = {}  # Use dict to deduplicate by problem_id

        for comp in compilation_results:
            if not isinstance(comp, dict):
                continue

            compilation_result = comp.get("compilation_result", {})
            is_pass = compilation_result.get("pass", False)
            is_complete = compilation_result.get("complete", False)

            # Only process proofs that are both passing and complete
            if not (is_pass and is_complete):
                continue

            successful_count += 1

            # Try multiple code field names (full_records have full_code)
            code = comp.get("full_code", "") or comp.get("code", "") or comp.get("lean4_code", "")
            if not code:
                continue

            # Extract breakdown key from metadata
            metadata = comp.get("metadata")
            if not metadata:
                if verbosity >= 2:
                    logger.warning(f"  Record missing metadata: {comp.get('uid', 'unknown')}")
                continue

            breakdown_key = get_breakdown_key(metadata)
            if breakdown_key == "unknown":
                if verbosity >= 2:
                    logger.warning(f"  Could not extract breakdown_key from metadata: {comp.get('uid', 'unknown')}")
                continue

            # Extract axiom declarations from the code
            axiom_names = extract_axiom_names(code)

            # Check which axioms are actually used in the proof
            used_axioms = []
            for axiom_name in axiom_names:
                if check_if_axiom_used(code, axiom_name):
                    used_axioms.append(axiom_name)

            if not used_axioms:
                continue

            # Map used axioms to lemmas by direct name matching
            for axiom_name in used_axioms:
                matching_lemma = None

                # Direct lookup by lemma_name
                for lemma in all_lemmas:
                    if lemma.get("lemma_name") == axiom_name:
                        # Also verify it's from the same breakdown for safety
                        lemma_metadata = lemma.get("metadata", {})
                        lemma_breakdown_key = get_breakdown_key(lemma_metadata)

                        if lemma_breakdown_key == breakdown_key:
                            matching_lemma = lemma
                            break

                if matching_lemma:
                    # Use formalization key to uniquely identify this specific formalization
                    lemma_uid = create_formalization_key(matching_lemma.get("metadata", {}))

                    if not lemma_uid:
                        if verbosity >= 2:
                            logger.warning(f"  Lemma missing uid: {axiom_name}")
                        continue

                    # Check if already attempted
                    if lemma_uid in attempted_keys:
                        if verbosity >= 2:
                            logger.debug(f"  Skipping {lemma_uid} (already attempted)")
                        continue

                    # Add to dict (deduplicates automatically)
                    if lemma_uid not in lemmas_to_add:
                        lemmas_to_add[lemma_uid] = matching_lemma
                        if verbosity >= 2:
                            logger.info(f"  Adding {lemma_uid} (used as {axiom_name} in breakdown {breakdown_key})")
                else:
                    if verbosity >= 2:
                        logger.warning(f"  Axiom {axiom_name} used in breakdown {breakdown_key} but not found in lemmas data")

        # Convert dict to list and mark all as attempted
        filtered_lemmas = list(lemmas_to_add.values())

        # Add new lemmas to attempted list (store metadata, not UIDs)
        for lemma in filtered_lemmas:
            lemma_metadata = lemma.get("metadata", {})
            if lemma_metadata:
                # Extract only core metadata fields (no derived fields like UIDs)
                # parent_problem_id should always exist now (consistent across all rounds)
                core_metadata = {
                    "origin_problem_id": lemma_metadata.get("origin_problem_id"),
                    "parent_problem_id": lemma_metadata.get("parent_problem_id"),
                    "round_id": lemma_metadata.get("round_id"),
                    "breakdown_id": lemma_metadata.get("breakdown_id"),
                    "lemma_id": lemma_metadata.get("lemma_id"),
                    "formalization_id": lemma_metadata.get("formalization_id")
                }

                attempted_lemmas.append(core_metadata)

        if verbosity >= 1:
            logger.info(f"  Successful proofs (pass=True AND complete=True): {successful_count}")
            logger.info(f"  Lemmas found from dependencies: {len(filtered_lemmas)}")

        if not filtered_lemmas:
            logger.info("  No new lemmas found - recursion complete")
            return []

        # Save updated attempted lemmas tracker (list of metadata dicts)
        with open(attempted_file, 'w') as f:
            json.dump(attempted_lemmas, f, indent=2)

        # Save versioned copy for easy resuming from any iteration
        versioned_file = os.path.join(prover_dir, f"attempted_lemmas_iter{iteration}.json")
        with open(versioned_file, 'w') as f:
            json.dump(attempted_lemmas, f, indent=2)

        if verbosity >= 1:
            logger.info(f"  Filtered lemmas for next iteration: {len(filtered_lemmas)}")
            logger.info(f"  Total attempted lemmas so far: {len(attempted_lemmas)}")
            logger.info(f"  Saved versioned checkpoint: attempted_lemmas_iter{iteration}.json")

        return filtered_lemmas
