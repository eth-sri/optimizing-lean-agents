import os
import shutil
from typing import Dict, List, Any
from loguru import logger
from jload import jload, jsave
from core.component import Component
from seed_prover.components.unified_prover import UnifiedProverComponent
from seed_prover.components.dependency_filter import DependencyFilterComponent
from metadata_utils import get_origin_problem_id
from seed_prover.consolidation_utils import consolidate_to_base_full_records


class RecursiveProverComponent(Component):
    """
    Orchestrates recursive depth-first proving of theorems and their dependencies.

    This component:
    1. Starts by proving theorems (iteration 1)
    2. Extracts actually used axioms from successful proofs
    3. Proves those lemmas (iteration 2)
    4. Recursively continues until no more dependencies are found
    """

    def __init__(self, name, component_config, global_config):
        super().__init__(name, component_config, global_config)

        self.max_iterations = self.config.get("max_iterations", 20)
        self.recursive_model_config = self.config.get("recursive_model_config")

        # Create sub-components
        self.unified_prover = UnifiedProverComponent(
            name="unified_prover",
            component_config=self.config,
            global_config=global_config
        )

        self.dependency_filter = DependencyFilterComponent(
            name="dependency_filter",
            component_config={},
            global_config=global_config
        )

    def _copy_iterations_from_base_run(
        self,
        base_run_path: str,
        prover_dir: str,
        start_from_iteration: int,
        round_num: int
    ) -> tuple[bool, int]:
        """
        Copy iteration directories and attempted_lemmas files from base run.

        Note: This method is only called for iteration resume (start_from_iteration > 0).
        For component resume (start_from_iteration == 0), the pipeline's
        _copy_round_for_component_resume handles copying theorems.json and lemmas.json.

        Args:
            base_run_path: Path to the base run directory
            prover_dir: Path to current run's prover directory
            start_from_iteration: Iteration to resume from (must be > 0, copies iter0 through iter{J-1})
            round_num: Current round number

        Returns:
            Tuple of (success, max_iteration_copied):
                - success: True if copy succeeded, False otherwise
                - max_iteration_copied: The highest iteration number that was copied (-1 if none)
        """
        verbosity = self.global_config.get('verbosity', 1)

        if verbosity >= 1:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"Resuming from iteration {start_from_iteration}")
            logger.info("=" * 80)

        base_prover_dir = os.path.join(base_run_path, f"round{round_num}", "prover")

        if not os.path.exists(base_prover_dir):
            logger.error(f"Base run prover directory not found: {base_prover_dir}")
            return False, -1

        # Detect maximum iteration that exists in base run
        max_existing_iteration = -1
        for i in range(100):  # Reasonable upper bound
            iter_dir = os.path.join(base_prover_dir, f"iter{i}")
            if os.path.exists(iter_dir):
                max_existing_iteration = i
            else:
                break

        if max_existing_iteration == -1:
            logger.error(f"No iteration directories found in base run: {base_prover_dir}")
            return False, -1

        if verbosity >= 1:
            logger.info(f"Found iterations 0 through {max_existing_iteration} in base run")

        # If start_from_iteration exceeds what exists, copy everything and signal to skip prover
        if start_from_iteration > max_existing_iteration + 1:
            if verbosity >= 1:
                logger.info(f"start_from_iteration ({start_from_iteration}) exceeds max available iteration ({max_existing_iteration})")
                logger.info(f"Copying all available iterations (0-{max_existing_iteration}) and will skip to proof builder")
            iterations_to_copy = max_existing_iteration + 1
        else:
            # Validate that all required iterations exist
            for i in range(start_from_iteration):
                iter_dir = os.path.join(base_prover_dir, f"iter{i}")
                if not os.path.exists(iter_dir):
                    logger.error(f"Required iteration directory not found in base run: iter{i}")
                    logger.error(f"Cannot resume from iteration {start_from_iteration}")
                    return False, -1
            iterations_to_copy = start_from_iteration

        if verbosity >= 1:
            logger.info(f"Copying iterations 0 through {iterations_to_copy-1} from base run...")

        # Copy iteration directories
        for i in range(iterations_to_copy):
            source_iter = os.path.join(base_prover_dir, f"iter{i}")
            dest_iter = os.path.join(prover_dir, f"iter{i}")

            if os.path.exists(dest_iter):
                logger.warning(f"Destination iter{i}/ already exists, skipping copy")
                continue

            shutil.copytree(source_iter, dest_iter)
            if verbosity >= 1:
                logger.info(f"  Copied iter{i}/")

        # Copy versioned attempted_lemmas files (for record-keeping)
        for i in range(max(0, iterations_to_copy - 1)):  # 0 through J-2
            source_file = os.path.join(base_prover_dir, f"attempted_lemmas_iter{i}.json")
            dest_file = os.path.join(prover_dir, f"attempted_lemmas_iter{i}.json")

            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)
                if verbosity >= 2:
                    logger.info(f"  Copied attempted_lemmas_iter{i}.json")

        # Reconstruct the working attempted_lemmas.json file
        # IMPORTANT: Must explicitly reset to ensure no stale data from previous failed attempts
        dest_file = os.path.join(prover_dir, "attempted_lemmas.json")
        if iterations_to_copy > 1:
            # Copy attempted_lemmas_iter{J-2}.json → attempted_lemmas.json
            source_file = os.path.join(base_prover_dir, f"attempted_lemmas_iter{iterations_to_copy-2}.json")

            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)
                if verbosity >= 1:
                    logger.info(f"  Reconstructed attempted_lemmas.json from iter{iterations_to_copy-2}")
            else:
                logger.warning(f"  attempted_lemmas_iter{iterations_to_copy-2}.json not found, creating empty attempted_lemmas.json")
                jsave([], dest_file)
        else:
            # Starting from iteration 1: explicitly create empty attempted_lemmas.json
            # (ensures no stale data from previous failed runs)
            if verbosity >= 1:
                logger.info(f"  Starting from iteration 1, creating empty attempted_lemmas.json")
            jsave([], dest_file)

        if verbosity >= 1:
            logger.info("Iteration resume setup complete")
            logger.info("=" * 80)

        return True, iterations_to_copy - 1

    def _consolidate_results(self, prover_dir: str, total_iterations: int) -> str:
        """
        Consolidate all full_records from all iterations and split by origin_problem_id.
        Also merges compilation results into each record.

        Creates:
        - full_records/ directory with separate JSON files per origin_problem_id
        - consolidated_full_records.json with all records (for backward compatibility)

        Args:
            prover_dir: Path to the prover directory (contains iter0/, iter1/, etc.)
            total_iterations: Total number of iterations that were run

        Returns:
            Path to the full_records directory
        """
        verbosity = self.global_config.get('verbosity', 3)

        if verbosity >= 1:
            logger.info("")
            logger.info("=" * 80)
            logger.info("Consolidating results from all iterations")
            logger.info("=" * 80)

        consolidated_records = []

        # Iterate through all iterations (0-based)
        for iteration in range(total_iterations):
            iter_dir = os.path.join(prover_dir, f"iter{iteration}")
            if not os.path.exists(iter_dir):
                if verbosity >= 2:
                    logger.warning(f"Iteration directory not found: {iter_dir}")
                continue

            # Helper function to merge full_records with compilation results and costs
            def merge_records(full_records_file, compilation_file, iteration, correction_round):
                if not os.path.exists(full_records_file):
                    return []

                full_records = jload(full_records_file)

                # Load compilation results if available
                compilation_map = {}
                if os.path.exists(compilation_file):
                    compilation_results = jload(compilation_file)
                    for comp in compilation_results:
                        if isinstance(comp, dict) and "name" in comp:
                            compilation_map[comp["name"]] = comp

                # Load inference costs if available
                cost_map = {}
                iter_dir = os.path.dirname(full_records_file)

                # Determine the correct to_inference file based on correction_round
                if correction_round == 0:
                    to_inference_file = os.path.join(iter_dir, "to_inference_codes.json")
                else:
                    to_inference_file = os.path.join(iter_dir, f"to_inference_codes_corr{correction_round}.json")

                if os.path.exists(to_inference_file):
                    try:
                        inference_records = jload(to_inference_file)
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
                        record["code"] = comp.get("code")  # Add the compiled code too

                    # Add detailed cost if available (match by uid)
                    if uid and uid in cost_map:
                        record["detailed_cost"] = cost_map[uid]

                    merged.append(record)

                return merged

            # Load initial round
            full_records_file = os.path.join(iter_dir, "full_records.json")
            compilation_file = os.path.join(iter_dir, "code_compilation_repl.json")

            merged_records = merge_records(full_records_file, compilation_file, iteration, 0)
            consolidated_records.extend(merged_records)

            if verbosity >= 2 and merged_records:
                logger.info(f"  Iteration {iteration}: Loaded {len(merged_records)} records from initial round")

            # Load correction rounds
            corr_round = 1
            while True:
                full_records_corr = os.path.join(iter_dir, f"full_records_corr{corr_round}.json")
                compilation_corr = os.path.join(iter_dir, f"code_compilation_repl_corr{corr_round}.json")

                if os.path.exists(full_records_corr):
                    merged_records = merge_records(full_records_corr, compilation_corr, iteration, corr_round)
                    consolidated_records.extend(merged_records)

                    if verbosity >= 2 and merged_records:
                        logger.info(f"  Iteration {iteration}: Loaded {len(merged_records)} records from correction round {corr_round}")
                    corr_round += 1
                else:
                    break

        # Group records by origin_problem_id
        from collections import defaultdict
        records_by_problem = defaultdict(list)

        for record in consolidated_records:
            # Extract origin_problem_id from metadata
            metadata = record.get("metadata")
            if metadata:
                origin_problem_id = get_origin_problem_id(metadata)
            else:
                # Fallback for records that might not have metadata yet
                origin_problem_id = record.get("name", "unknown")
                logger.warning(f"Record missing metadata, using fallback: {origin_problem_id}")

            records_by_problem[origin_problem_id].append(record)

        # Create full_records directory
        full_records_dir = os.path.join(prover_dir, "full_records")
        os.makedirs(full_records_dir, exist_ok=True)

        # Save records for each problem to separate files
        if verbosity >= 1:
            logger.info(f"Saving records split by origin_problem_id...")

        for origin_problem_id, records in records_by_problem.items():
            # Create safe filename from origin_problem_id
            safe_filename = origin_problem_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            problem_file = os.path.join(full_records_dir, f"{safe_filename}.json")
            jsave(records, problem_file)

            if verbosity >= 2:
                logger.info(f"  {origin_problem_id}: {len(records)} records → {safe_filename}.json")

        # Also save the full consolidated file for backward compatibility
        consolidated_file = os.path.join(prover_dir, "consolidated_full_records.json")
        jsave(consolidated_records, consolidated_file)

        if verbosity >= 1:
            logger.info(f"Consolidated {len(consolidated_records)} total records across {len(records_by_problem)} problems")
            logger.info(f"  Split records saved to: {full_records_dir}")
            logger.info(f"  Full consolidated file saved to: {consolidated_file}")

        # Also consolidate to base-level full_records/ for next round's feedback loop
        # This is critical: FeedbackDataLoaderComponent in round N+1 needs access to
        # results from rounds 0 through N via the base-level full_records/ directory
        base_output_dir = self.global_config.get("output_dir")

        # Extract round number from prover_dir path (e.g., "round0/prover" -> 0)
        round_num = None
        prover_dir_parts = prover_dir.split(os.sep)
        for part in prover_dir_parts:
            if part.startswith("round"):
                try:
                    round_num = int(part[5:])  # Extract number after "round"
                    break
                except ValueError:
                    pass

        if round_num is not None:
            if verbosity >= 1:
                logger.info("")
                logger.info("Updating base-level full_records for feedback loop...")

            consolidate_to_base_full_records(
                base_output_dir=base_output_dir,
                round_num=round_num,
                verbosity=verbosity
            )

            # Also save proof attempts in hierarchical structure
        else:
            if verbosity >= 1:
                logger.warning("Could not determine round number, skipping base-level consolidation")

        return full_records_dir

    def process(self, data_list: List[Dict[str, Any]], round_num: int = 0) -> List[Dict[str, Any]]:
        """
        Execute recursive proving process.

        Args:
            data_list: Pipeline data (should contain path to theorems and lemmas)
            round_num: Current round number

        Returns:
            Original data_list (unchanged)
        """
        verbosity = self.global_config.get('verbosity', 3)
        base_output_dir = self.global_config.get("output_dir")

        logger.info("=" * 80)
        logger.info("RecursiveProverComponent: Starting recursive proving process")
        logger.info("=" * 80)

        # Load theorems and lemmas from the prover directory
        prover_dir = os.path.join(base_output_dir, f"round{round_num}", "prover")
        theorems_file = os.path.join(prover_dir, "theorems.json")
        lemmas_file = os.path.join(prover_dir, "lemmas.json")

        if not os.path.exists(theorems_file):
            logger.error(f"Theorems file not found: {theorems_file}")
            return data_list

        if not os.path.exists(lemmas_file):
            logger.error(f"Lemmas file not found: {lemmas_file}")
            return data_list

        # Load data
        theorems = jload(theorems_file)
        all_lemmas = jload(lemmas_file)

        if verbosity >= 1:
            logger.info(f"Loaded {len(theorems)} theorems")
            logger.info(f"Loaded {len(all_lemmas)} total lemmas")

        # Check if we should resume from a specific iteration or proving round
        # Note: global_config is the full config dict, so we need to access pipeline.run_reuse
        pipeline_config = self.global_config.get('pipeline', {})
        run_reuse_config = pipeline_config.get('run_reuse', {})
        start_from_round = run_reuse_config.get('start_from_round', 0)
        start_from_iteration = run_reuse_config.get('start_from_iteration', 0)
        start_from_proving_round = run_reuse_config.get('start_from_proving_round', 0)
        base_run_path = run_reuse_config.get('_resolved_base_run_path')

        if verbosity >= 1:
            logger.info(f"Iteration resume check:")
            logger.info(f"  enabled: {run_reuse_config.get('enabled', False)}")
            logger.info(f"  round_num: {round_num}, start_from_round: {start_from_round}")
            logger.info(f"  start_from_iteration: {start_from_iteration}")
            logger.info(f"  start_from_proving_round: {start_from_proving_round}")
            logger.info(f"  base_run_path: {base_run_path}")

        # Initialize statistics tracking
        iteration_stats = []

        # Determine starting iteration and items to prove
        iteration = 0
        items_to_prove = theorems
        proving_round_override = None  # Will be set if we're resuming from a specific proving round

        # Handle iteration resume (only when start_from_iteration > 0)
        # Note: When start_from_iteration == 0, component resume handles copying files
        is_in_place_resume = run_reuse_config.get('_in_place_resume', False)
        if (run_reuse_config.get('enabled', False) and
            round_num == start_from_round and
            start_from_iteration > 0 and
            base_run_path):

            if is_in_place_resume:
                # In-place resume: iterations already exist, skip copying
                logger.info(f"In-place resume: skipping iteration copy (iterations already in {prover_dir})")
                max_iteration_copied = start_from_iteration - 1
                success = True

                # CRITICAL: Reset attempted_lemmas.json to the correct state for resume.
                # The issue: attempted_lemmas.json contains lemmas that were IDENTIFIED as
                # dependencies (output from dependency filter), not lemmas that were PROVED.
                # When dependency filter runs at the end of iter N, it adds lemmas to
                # attempted_lemmas.json BEFORE they are proved in iter N+1.
                # For in-place resume from iter N, we need to reset to the state BEFORE
                # the dependency filter for iter N-1 ran.
                attempted_file = os.path.join(prover_dir, "attempted_lemmas.json")
                if start_from_iteration == 1:
                    # Resuming from iter 1: reset to empty (state before iter0's dep filter)
                    if verbosity >= 1:
                        logger.info(f"In-place resume: resetting attempted_lemmas.json to empty (iter 1 start)")
                    jsave([], attempted_file)
                elif start_from_iteration > 1:
                    # Resuming from iter N (N > 1): use attempted_lemmas_iter{N-2}.json
                    # This is the state after iter N-2's dep filter ran (lemmas for iter N-1)
                    source_file = os.path.join(prover_dir, f"attempted_lemmas_iter{start_from_iteration - 2}.json")
                    if os.path.exists(source_file):
                        if verbosity >= 1:
                            logger.info(f"In-place resume: resetting attempted_lemmas.json from iter{start_from_iteration - 2}")
                        shutil.copy2(source_file, attempted_file)
                    else:
                        if verbosity >= 1:
                            logger.warning(f"In-place resume: attempted_lemmas_iter{start_from_iteration - 2}.json not found, resetting to empty")
                        jsave([], attempted_file)
            else:
                # Copy iterations from base run
                success, max_iteration_copied = self._copy_iterations_from_base_run(
                    base_run_path=base_run_path,
                    prover_dir=prover_dir,
                    start_from_iteration=start_from_iteration,
                    round_num=round_num
                )

            if not success:
                logger.error("Failed to copy iterations from base run, starting from iteration 0 instead")
                start_from_iteration = 0
            else:
                # Check if start_from_iteration exceeds what was available
                if start_from_iteration > max_iteration_copied + 1:
                    # All iterations already copied, skip prover and go to consolidation
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info(f"All available iterations (0-{max_iteration_copied}) copied")
                    logger.info("Skipping prover and proceeding to proof builder")
                    logger.info("=" * 80)

                    # Collect statistics from copied iterations
                    for iter_num in range(max_iteration_copied + 1):
                        iter_dir = os.path.join(prover_dir, f"iter{iter_num}")
                        stats = self._collect_iteration_stats(iter_dir, iter_num)
                        iteration_stats.append(stats)

                        if verbosity >= 2:
                            logger.info(f"  Loaded stats from iter{iter_num}: {stats['successful']}/{stats['total']} successful")

                    # Set iteration and items_to_prove to signal we should skip the prover loop
                    iteration = max_iteration_copied + 1
                    items_to_prove = []
                else:
                    # Run dependency filter on the last copied iteration to determine what to prove next
                    last_copied_iteration = start_from_iteration - 1

                    logger.info("")
                    logger.info("=" * 80)
                    logger.info(f"Running dependency filter on iteration {last_copied_iteration} to determine next items to prove...")
                    logger.info("=" * 80)

                    next_lemmas = self.dependency_filter.process(
                        data_list=data_list,
                        round_num=round_num,
                        iteration=last_copied_iteration,
                        all_lemmas=all_lemmas
                    )

                    # Collect statistics from copied iterations
                    for iter_num in range(start_from_iteration):
                        iter_dir = os.path.join(prover_dir, f"iter{iter_num}")
                        stats = self._collect_iteration_stats(iter_dir, iter_num)
                        iteration_stats.append(stats)

                        if verbosity >= 2:
                            logger.info(f"  Loaded stats from iter{iter_num}: {stats['successful']}/{stats['total']} successful")

                    if next_lemmas:
                        # Start proving from start_from_iteration
                        iteration = start_from_iteration
                        items_to_prove = next_lemmas

                        if verbosity >= 1:
                            logger.info(f"Will start proving at iteration {iteration} with {len(items_to_prove)} lemmas")
                    else:
                        logger.info(f"No dependencies found from iteration {last_copied_iteration}, recursion complete")
                        # Still need to consolidate the copied results
                        iteration = start_from_iteration
                        items_to_prove = []

        # Handle proving round resume (when start_from_proving_round > 0)
        # This is independent of iteration resume and applies to the current/starting iteration
        if (run_reuse_config.get('enabled', False) and
            round_num == start_from_round and
            start_from_proving_round > 0 and
            base_run_path):

            # Import the copy function
            from seed_prover.run_reuse_utils import copy_proving_rounds_for_iteration

            logger.info("")
            logger.info("=" * 80)
            logger.info(f"Proving round resume enabled: Starting from proving round {start_from_proving_round}")
            logger.info("=" * 80)

            # Copy proving rounds for the starting iteration
            success = copy_proving_rounds_for_iteration(
                base_run_path=base_run_path,
                output_dir=base_output_dir,
                round_num=round_num,
                iteration_num=iteration,
                start_from_proving_round=start_from_proving_round,
                verbosity=verbosity
            )

            if success:
                # Set proving_round_override to tell unified_prover where to start
                proving_round_override = start_from_proving_round
                logger.info(f"Successfully copied proving rounds 0-{start_from_proving_round-1}")
                logger.info(f"Will start proving at iteration {iteration}, proving round {start_from_proving_round}")
            else:
                logger.error("Failed to copy proving rounds, starting from proving round 0 instead")
                proving_round_override = None

        while items_to_prove and iteration < self.max_iterations:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"ITERATION {iteration}: Starting")
            logger.info("=" * 80)

            if verbosity >= 1:
                if iteration == 0:
                    logger.info(f"  Type: Theorems")
                else:
                    logger.info(f"  Type: Lemmas (depth {iteration})")
                logger.info(f"  Items to prove: {len(items_to_prove)}")

            # Determine if we should use recursive model config
            model_config_override = None
            cleanup_after = False
            if iteration >= 1 and self.recursive_model_config is not None:
                model_config_override = self.recursive_model_config
                if verbosity >= 1:
                    logger.info(f"  Using recursive model config for iteration {iteration}: {model_config_override}")

            # Cleanup GPU memory after iteration 0 if we're going to switch models
            if iteration == 0 and self.recursive_model_config is not None:
                cleanup_after = True

            # Run the prover
            self.unified_prover.process(
                data=data_list,
                round_num=round_num,
                iteration=iteration,
                input_data=items_to_prove,
                start_from_proving_round=proving_round_override,
                model_config_override=model_config_override,
                cleanup_after=cleanup_after
            )

            # Reset proving_round_override after first iteration
            # (only the first iteration should use the proving round resume)
            proving_round_override = None

            logger.info(f"ITERATION {iteration}: Proving completed")

            # Collect statistics from this iteration
            iter_dir = os.path.join(prover_dir, f"iter{iteration}")
            stats = self._collect_iteration_stats(iter_dir, iteration)
            iteration_stats.append(stats)

            if verbosity >= 1:
                logger.info(f"  Results: {stats['successful']} successful / {stats['total']} total")

            # Check if we should continue
            if stats['successful'] == 0:
                logger.info(f"ITERATION {iteration}: No successful proofs, stopping recursion")
                break

            # Extract dependencies from this iteration
            logger.info(f"ITERATION {iteration}: Extracting dependencies...")
            next_lemmas = self.dependency_filter.process(
                data_list=data_list,
                round_num=round_num,
                iteration=iteration,
                all_lemmas=all_lemmas
            )

            if not next_lemmas:
                logger.info(f"ITERATION {iteration}: No more dependencies found, recursion complete")
                break

            if verbosity >= 1:
                logger.info(f"  Dependencies found: {len(next_lemmas)} new lemmas to prove")

            # Prepare for next iteration
            items_to_prove = next_lemmas
            iteration += 1

        # Save summary statistics
        summary = {
            "total_iterations": len(iteration_stats),
            "max_iterations_reached": iteration >= self.max_iterations,
            "iterations": iteration_stats
        }

        summary_file = os.path.join(prover_dir, "summary.json")
        jsave(summary, summary_file)

        # Consolidate all results from all iterations
        if len(iteration_stats) > 0:
            full_records_dir = self._consolidate_results(prover_dir, len(iteration_stats))
        else:
            full_records_dir = None

        # Print final summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("RECURSIVE PROVING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total iterations: {len(iteration_stats)}")

        total_attempted = sum(s['total'] for s in iteration_stats)
        total_successful = sum(s['successful'] for s in iteration_stats)

        logger.info(f"Total items attempted: {total_attempted}")
        logger.info(f"Total successful proofs: {total_successful}")

        if verbosity >= 1:
            logger.info("")
            logger.info("Per-iteration breakdown:")
            for stat in iteration_stats:
                iter_num = stat['iteration']
                iter_type = "Theorems" if iter_num == 0 else f"Lemmas (depth {iter_num})"
                logger.info(f"  Iteration {iter_num} ({iter_type}): {stat['successful']}/{stat['total']} successful")

        logger.info(f"Summary saved to: {summary_file}")
        if full_records_dir:
            logger.info(f"Full records split by problem saved to: {full_records_dir}")
            logger.info(f"Consolidated records also saved to: {os.path.join(prover_dir, 'consolidated_full_records.json')}")
        logger.info("=" * 80)

        return data_list

    def _collect_iteration_stats(self, iter_dir: str, iteration: int) -> Dict[str, Any]:
        """
        Collect statistics from an iteration's compilation results.

        Args:
            iter_dir: Path to iteration directory
            iteration: Iteration number

        Returns:
            Dictionary with statistics
        """
        # Load all compilation results
        compilation_results = []

        # Initial round
        comp_file = os.path.join(iter_dir, "code_compilation_repl.json")
        if os.path.exists(comp_file):
            compilation_results.extend(jload(comp_file))

        # Correction rounds
        corr_round = 1
        while True:
            corr_file = os.path.join(iter_dir, f"code_compilation_repl_corr{corr_round}.json")
            if os.path.exists(corr_file):
                compilation_results.extend(jload(corr_file))
                corr_round += 1
            else:
                break

        # Count successful proofs (pass=True AND complete=True)
        total = len(compilation_results)
        successful = 0
        passing = 0
        complete = 0

        for comp in compilation_results:
            if not isinstance(comp, dict):
                continue

            comp_result = comp.get("compilation_result", {})
            is_pass = comp_result.get("pass", False)
            is_complete = comp_result.get("complete", False)

            if is_pass:
                passing += 1
            if is_complete:
                complete += 1
            if is_pass and is_complete:
                successful += 1

        return {
            "iteration": iteration,
            "total": total,
            "passing": passing,
            "complete": complete,
            "successful": successful
        }
