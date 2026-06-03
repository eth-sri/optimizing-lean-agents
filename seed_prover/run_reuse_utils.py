"""
Run Reuse Utilities - Helper functions for resuming from previous runs.

This module contains all the logic for:
- Resolving base run paths
- Copying previous rounds
- Copying round data for iteration/component resume
- Determining which components to skip
- Loading data for component resume
"""

import os
import json
import shutil
from loguru import logger


def resolve_base_run_path(run_reuse_config, output_dir):
    """
    Resolve the base_run_path from run_reuse config.

    Args:
        run_reuse_config: The run_reuse configuration dict
        output_dir: Current run's output directory (used for relative path resolution)

    Returns:
        str: Absolute path to the base run directory

    Raises:
        ValueError: If base_run_path not specified
        FileNotFoundError: If the base run path doesn't exist
    """
    base_run_path = run_reuse_config.get('base_run_path')

    if not base_run_path:
        raise ValueError("base_run_path not specified in run_reuse config")

    # Handle relative timestamp paths (e.g., "2025/11/07/120609" or "/2025/11/07/120609")
    if not os.path.isabs(base_run_path) or (base_run_path.startswith('/') and base_run_path.count('/') <= 4):
        # This looks like a timestamp path, not a full absolute path
        # Extract base job directory from current output_dir
        # E.g., /scratch/results/seed_prover_chunk_0/2025/11/07/150000 -> /scratch/results/seed_prover_chunk_0
        base_job_dir = os.path.abspath(os.path.join(output_dir, '../../../..'))
        # Remove leading slash from base_run_path if present
        timestamp_path = base_run_path.lstrip('/')
        base_run_path = os.path.join(base_job_dir, timestamp_path)
        logger.info(f"Inferred base_run_path from timestamp: {base_run_path}")
    else:
        # Full absolute path provided - use it directly
        base_run_path = os.path.expandvars(base_run_path)  # Expand env vars like $SCRATCH
        logger.info(f"Using explicit base_run_path: {base_run_path}")

    if not os.path.exists(base_run_path):
        raise FileNotFoundError(f"Specified base_run_path does not exist: {base_run_path}")

    return base_run_path


def copy_round_for_component_resume(base_run_path, output_dir, round_num, resume_from_component, verbosity=1):
    """
    Copy round data for component resume (only copy output from completed components).

    When resuming from a specific component, we need outputs from all previous components.

    Args:
        base_run_path: Path to the base run directory
        output_dir: Current run's output directory
        round_num: Round number to copy
        resume_from_component: Component to resume from
        verbosity: Logging verbosity level
    """
    if verbosity >= 1:
        logger.info(f"Copying round{round_num} data for component resume from {resume_from_component}...")

    # Copy dataset.json if it exists
    source_dataset = os.path.join(base_run_path, "dataset.json")
    dest_dataset = os.path.join(output_dir, "dataset.json")
    if os.path.exists(source_dataset):
        shutil.copy2(source_dataset, dest_dataset)
        if verbosity >= 1:
            logger.info(f"  Copied dataset.json")

    source_round = os.path.join(base_run_path, f"round{round_num}")
    dest_round = os.path.join(output_dir, f"round{round_num}")

    if not os.path.exists(source_round):
        logger.error(f"Source round directory not found: {source_round}")
        raise FileNotFoundError(f"Source round directory not found: {source_round}")

    # Create destination round directory
    os.makedirs(dest_round, exist_ok=True)

    # Map components to their output directories
    component_dirs = {
        'breakdown': 'breakdown',
        'breakdown_json': 'breakdown',
        'breakdown_parser': 'breakdown_parser',
        'breakdown_json_parser': 'breakdown_parser',
        'formalizer': 'formalizer',
        'recursive_prover': 'prover'
    }

    # Determine which directories to copy based on resume point
    dirs_to_copy = []
    component_order = ['breakdown', 'breakdown_parser', 'formalizer', 'recursive_prover']

    if resume_from_component in component_order:
        resume_idx = component_order.index(resume_from_component)
        # Copy all component outputs before the resume point
        for comp in component_order[:resume_idx]:
            if comp in component_dirs:
                dirs_to_copy.append(component_dirs[comp])

    # Copy the directories
    for dir_name in dirs_to_copy:
        source_dir = os.path.join(source_round, dir_name)
        dest_dir = os.path.join(dest_round, dir_name)

        if os.path.exists(source_dir):
            shutil.copytree(source_dir, dest_dir)
            if verbosity >= 1:
                logger.info(f"  Copied {dir_name}/")
        else:
            logger.warning(f"  {dir_name}/ not found in base run, skipping")

    # Special case: when resuming from recursive_prover, also copy theorems.json and lemmas.json
    # These files are created by the formalizer but written to the prover directory
    if resume_from_component == 'recursive_prover':
        source_prover_dir = os.path.join(source_round, 'prover')
        dest_prover_dir = os.path.join(dest_round, 'prover')
        os.makedirs(dest_prover_dir, exist_ok=True)

        # Copy theorems.json
        source_theorems = os.path.join(source_prover_dir, 'theorems.json')
        dest_theorems = os.path.join(dest_prover_dir, 'theorems.json')
        if os.path.exists(source_theorems):
            shutil.copy2(source_theorems, dest_theorems)
            if verbosity >= 1:
                logger.info(f"  Copied prover/theorems.json")
        else:
            logger.warning(f"  prover/theorems.json not found in base run")

        # Copy lemmas.json
        source_lemmas = os.path.join(source_prover_dir, 'lemmas.json')
        dest_lemmas = os.path.join(dest_prover_dir, 'lemmas.json')
        if os.path.exists(source_lemmas):
            shutil.copy2(source_lemmas, dest_lemmas)
            if verbosity >= 1:
                logger.info(f"  Copied prover/lemmas.json")
        else:
            logger.warning(f"  prover/lemmas.json not found in base run")

    if verbosity >= 1:
        logger.info(f"Finished copying round{round_num} for component resume")


def copy_previous_rounds(base_run_path, output_dir, start_from_round):
    """
    Copy all previous rounds (0 through start_from_round-1) from base run.

    Args:
        base_run_path: Path to the base run directory
        output_dir: Current run's output directory
        start_from_round: Round number to start from (will copy rounds 0..start_from_round-1)
    """
    logger.info(f"Copying rounds 0 through {start_from_round-1} from base run...")

    # Copy dataset.json if it exists (needed for comparison/tracking)
    source_dataset = os.path.join(base_run_path, "dataset.json")
    dest_dataset = os.path.join(output_dir, "dataset.json")

    if os.path.exists(source_dataset):
        shutil.copy2(source_dataset, dest_dataset)
        logger.info(f"  Copied dataset.json")
    else:
        logger.debug("  dataset.json not found in base run")

    # Copy each round directory
    for round_num in range(start_from_round):
        source_round = os.path.join(base_run_path, f"round{round_num}")
        dest_round = os.path.join(output_dir, f"round{round_num}")

        if os.path.exists(source_round):
            shutil.copytree(source_round, dest_round)
            logger.info(f"  Copied round{round_num}/")
        else:
            logger.warning(f"  round{round_num}/ not found in base run, skipping")

    # Copy consolidated full_records if it exists
    source_full_records = os.path.join(base_run_path, "full_records")
    dest_full_records = os.path.join(output_dir, "full_records")

    if os.path.exists(source_full_records):
        shutil.copytree(source_full_records, dest_full_records)
        logger.info(f"  Copied full_records/")
    else:
        logger.debug("  full_records/ not found in base run (may not have been generated yet)")

    logger.info("Finished copying previous rounds")


def copy_proving_rounds_for_iteration(base_run_path, output_dir, round_num, iteration_num, start_from_proving_round, verbosity=1):
    """
    Copy proving/correction round data within a specific iteration.

    When resuming from proving round J at a specific iteration, we need:
    - All proving round outputs (summary_round_0 through summary_round_{J-1})
    - Previous round's full_records.json to know what to correct

    Args:
        base_run_path: Path to the base run directory
        output_dir: Current run's output directory
        round_num: Breakdown round number
        iteration_num: Prover iteration number
        start_from_proving_round: Proving/correction round to resume from (rounds >= this will not be copied)
        verbosity: Logging verbosity level

    Returns:
        bool: True if successful, False otherwise
    """
    if start_from_proving_round == 0:
        # No need to copy anything - starting from scratch
        return True

    if verbosity >= 1:
        logger.info(f"Copying proving rounds 0 through {start_from_proving_round-1} for iteration {iteration_num}...")

    source_iter = os.path.join(base_run_path, f"round{round_num}", "prover", f"iter{iteration_num}")
    dest_iter = os.path.join(output_dir, f"round{round_num}", "prover", f"iter{iteration_num}")

    if not os.path.exists(source_iter):
        logger.error(f"Source iteration directory not found: {source_iter}")
        return False

    # Create destination iteration directory
    os.makedirs(dest_iter, exist_ok=True)

    # Copy proving round summary directories (summary_round_0, summary_round_1, ...)
    for proving_round in range(start_from_proving_round):
        source_summary = os.path.join(source_iter, f"summary_round_{proving_round}")
        dest_summary = os.path.join(dest_iter, f"summary_round_{proving_round}")

        if os.path.exists(source_summary):
            shutil.copytree(source_summary, dest_summary)
            if verbosity >= 1:
                logger.info(f"  Copied summary_round_{proving_round}/")
        else:
            logger.warning(f"  summary_round_{proving_round}/ not found in base run")

    # Copy attempt_summary directory if it exists
    source_attempt_summary = os.path.join(source_iter, "attempt_summary")
    dest_attempt_summary = os.path.join(dest_iter, "attempt_summary")
    if os.path.exists(source_attempt_summary):
        shutil.copytree(source_attempt_summary, dest_attempt_summary)
        if verbosity >= 1:
            logger.info(f"  Copied attempt_summary/")

    # Copy essential files (input.json, full_records.json from last proving round)
    # Input stays the same across proving rounds
    source_input = os.path.join(source_iter, "input.json")
    dest_input = os.path.join(dest_iter, "input.json")
    if os.path.exists(source_input):
        shutil.copy2(source_input, dest_input)
        if verbosity >= 1:
            logger.info(f"  Copied input.json")

    # Copy the most recent full_records.json and code_compilation_repl.json
    # (from the last completed proving round)
    source_full_records = os.path.join(source_iter, "full_records.json")
    dest_full_records = os.path.join(dest_iter, "full_records.json")
    if os.path.exists(source_full_records):
        shutil.copy2(source_full_records, dest_full_records)
        if verbosity >= 1:
            logger.info(f"  Copied full_records.json")

    source_compilation = os.path.join(source_iter, "code_compilation_repl.json")
    dest_compilation = os.path.join(dest_iter, "code_compilation_repl.json")
    if os.path.exists(source_compilation):
        shutil.copy2(source_compilation, dest_compilation)
        if verbosity >= 1:
            logger.info(f"  Copied code_compilation_repl.json")

    if verbosity >= 1:
        logger.info(f"Finished copying proving rounds for iteration {iteration_num}")

    return True


def copy_round_for_iteration_resume(base_run_path, output_dir, round_num, start_from_iteration, verbosity=1):
    """
    Copy round data for iteration resume (excludes prover iterations >= start_from_iteration).

    When resuming from iteration J at a specific round, we need:
    - All pipeline stage outputs (breakdown, parser, formalization)
    - Prover metadata (theorems.json, lemmas.json)
    - Prover iterations 0 through J-1 (handled by RecursiveProverComponent)

    Args:
        base_run_path: Path to the base run directory
        output_dir: Current run's output directory
        round_num: Round number to copy
        start_from_iteration: Iteration to resume from (iterations >= this will not be copied)
        verbosity: Logging verbosity level

    Returns:
        bool: True if successful, False otherwise
    """
    if verbosity >= 1:
        logger.info(f"Copying round{round_num} data for iteration resume from iteration {start_from_iteration}...")

    source_round = os.path.join(base_run_path, f"round{round_num}")
    dest_round = os.path.join(output_dir, f"round{round_num}")

    if not os.path.exists(source_round):
        logger.error(f"Source round directory not found: {source_round}")
        return False

    # Create destination round directory
    os.makedirs(dest_round, exist_ok=True)

    # Copy all non-prover subdirectories (breakdown, parser, formalization, etc.)
    for item in os.listdir(source_round):
        source_item = os.path.join(source_round, item)
        dest_item = os.path.join(dest_round, item)

        if item == "prover":
            # Handle prover directory specially - copy structure but not iterations
            if os.path.isdir(source_item):
                os.makedirs(dest_item, exist_ok=True)

                # Copy prover files (theorems.json, lemmas.json, etc.)
                for prover_item in os.listdir(source_item):
                    source_prover_item = os.path.join(source_item, prover_item)
                    dest_prover_item = os.path.join(dest_item, prover_item)

                    # Skip iteration directories and attempted_lemmas files (will be handled by RecursiveProverComponent)
                    if prover_item.startswith("iter") or prover_item.startswith("attempted_lemmas"):
                        continue

                    # Skip consolidated outputs (will be regenerated)
                    if prover_item in ["consolidated_full_records.json", "full_records", "summary.json"]:
                        continue

                    # Copy file or directory
                    if os.path.isfile(source_prover_item):
                        shutil.copy2(source_prover_item, dest_prover_item)
                        if verbosity >= 2:
                            logger.info(f"  Copied {item}/{prover_item}")
                    elif os.path.isdir(source_prover_item):
                        shutil.copytree(source_prover_item, dest_prover_item)
                        if verbosity >= 2:
                            logger.info(f"  Copied {item}/{prover_item}/")
        else:
            # Copy other directories/files as-is
            if os.path.isfile(source_item):
                shutil.copy2(source_item, dest_item)
                if verbosity >= 1:
                    logger.info(f"  Copied {item}")
            elif os.path.isdir(source_item):
                shutil.copytree(source_item, dest_item)
                if verbosity >= 1:
                    logger.info(f"  Copied {item}/")

    if verbosity >= 1:
        logger.info(f"Finished copying round{round_num} for iteration resume")

    return True


def setup_base_run_resume(config, output_dir, run_reuse_config, start_from_round, start_from_iteration):
    """
    Handle base run path resolution and data copying for run reuse.

    Args:
        config: Full pipeline configuration
        output_dir: Current run's output directory
        run_reuse_config: The run_reuse section of the config
        start_from_round: Round to start from
        start_from_iteration: Iteration to start from

    Returns:
        tuple: (updated_start_from_round, updated_start_from_iteration, success)
    """
    logger.info(f"Run reuse enabled, starting from round {start_from_round}")

    # Resolve base_run_path
    try:
        base_run_path = resolve_base_run_path(run_reuse_config, output_dir)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Failed to resolve base_run_path: {e}")
        if start_from_round >= 1:
            logger.info(f"Falling back to start from round 0 instead of round {start_from_round}")
            start_from_round = 0
        if start_from_iteration > 0:
            logger.info(f"Falling back to start from iteration 0 instead of iteration {start_from_iteration}")
            start_from_iteration = 0
        return start_from_round, start_from_iteration, False

    # Get verbosity from config
    verbosity = config.get('verbosity', 1)

    # Copy previous rounds if starting from round >= 1
    if start_from_round >= 1:
        copy_previous_rounds(base_run_path, output_dir, start_from_round)

    # Copy the target round data if using iteration resume
    if start_from_iteration > 0:
        logger.info(f"Iteration resume enabled: will start from iteration {start_from_iteration} at round {start_from_round}")
        success = copy_round_for_iteration_resume(
            base_run_path=base_run_path,
            output_dir=output_dir,
            round_num=start_from_round,
            start_from_iteration=start_from_iteration,
            verbosity=verbosity
        )
        if not success:
            logger.error(f"Failed to copy round data for iteration resume")
            logger.info(f"Falling back to start from iteration 0")
            start_from_iteration = 0
            config['pipeline']['run_reuse']['start_from_iteration'] = 0

    # Store base_run_path in config for components to use
    config['pipeline']['run_reuse']['_resolved_base_run_path'] = base_run_path
    return start_from_round, start_from_iteration, True


def get_components_to_skip_for_resume(resume_from_component):
    """
    Determine which components to skip based on resume point.

    Args:
        resume_from_component: Component to resume from (e.g., "formalizer", "recursive_prover")

    Returns:
        List of component names to skip
    """
    # Define component order (breakdown_parser is deprecated, breakdown outputs parsed_breakdown.json directly)
    component_order = ['data_loader', 'feedback_data_loader', 'breakdown', 'breakdown_json', 'formalizer', 'recursive_prover']

    if not resume_from_component or resume_from_component not in component_order:
        return []

    # Find index of resume component
    resume_idx = component_order.index(resume_from_component)

    # Skip all components before the resume point
    return component_order[:resume_idx]


def load_data_for_component_resume(resume_from_component, round_num, base_run_path, verbosity=1):
    """
    Load data from the last completed component before resume point.

    Args:
        resume_from_component: Component to resume from
        round_num: Current round number
        base_run_path: Path to the base run to load from
        verbosity: Logging verbosity level

    Returns:
        Loaded data list
    """
    # Determine which file to load
    # Note: breakdown_parser is deprecated, output is always breakdown/parsed_breakdown.json
    if resume_from_component == 'formalizer':
        # Load breakdown output (parsed_breakdown.json)
        load_file = os.path.join(base_run_path, f'round{round_num}/breakdown/parsed_breakdown.json')
    elif resume_from_component == 'recursive_prover':
        # Load formalizer output
        load_file = os.path.join(base_run_path, f'round{round_num}/formalizer/formalized.json')
    else:
        logger.warning(f"No data loading logic for resume_from_component={resume_from_component}")
        return []

    if not os.path.exists(load_file):
        logger.error(f"Cannot resume: required file not found: {load_file}")
        raise FileNotFoundError(f"Required file for resume not found: {load_file}")

    # Load the data
    with open(load_file, 'r') as f:
        data = json.load(f)

    if verbosity >= 1:
        logger.info(f"Loaded {len(data)} items from {load_file} for component resume")

    return data
