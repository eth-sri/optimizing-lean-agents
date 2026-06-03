"""
Auto-Resume Utilities - Automatic checkpoint detection and resume for the seed prover pipeline.

This module provides functionality to:
- Detect existing checkpoints in an output directory
- Compute the optimal resume point
- Clean up partial/corrupted state
"""

import os
import shutil
from typing import Optional
from loguru import logger


def _detect_round_status(output_dir: str, round_num: int) -> dict:
    """
    Check completion status of each component in a round.

    Args:
        output_dir: Base output directory
        round_num: Round number to check

    Returns:
        Dictionary with completion status for each component
    """
    round_dir = os.path.join(output_dir, f"round{round_num}")

    status = {
        'round_exists': os.path.exists(round_dir),
        'breakdown_complete': False,
        'formalizer_complete': False,
        'prover_started': False,
        'prover_iterations_complete': 0,  # Number of fully completed iterations
        'prover_has_partial_iteration': False,  # Has an incomplete iteration
        'prover_partial_iteration_num': -1,  # Which iteration is partial
        'prover_fully_complete': False,  # Has consolidated full_records/
    }

    if not status['round_exists']:
        return status

    # Check breakdown - output is always breakdown/parsed_breakdown.json
    breakdown_file = os.path.join(round_dir, "breakdown", "parsed_breakdown.json")
    if os.path.exists(breakdown_file) and os.path.getsize(breakdown_file) > 10:
        status['breakdown_complete'] = True

    # Check formalizer (needs all 3 files)
    formalizer_file = os.path.join(round_dir, "formalizer", "formalized.json")
    theorems_file = os.path.join(round_dir, "prover", "theorems.json")
    lemmas_file = os.path.join(round_dir, "prover", "lemmas.json")

    formalizer_complete = all([
        os.path.exists(formalizer_file) and os.path.getsize(formalizer_file) > 10,
        os.path.exists(theorems_file) and os.path.getsize(theorems_file) > 10,
        os.path.exists(lemmas_file) and os.path.getsize(lemmas_file) > 10,
    ])
    status['formalizer_complete'] = formalizer_complete

    # Check prover iterations
    prover_dir = os.path.join(round_dir, "prover")
    if os.path.exists(prover_dir):
        status['prover_started'] = True

        # Count completed iterations and detect partial ones
        for i in range(100):  # Max iterations
            iter_dir = os.path.join(prover_dir, f"iter{i}")
            if not os.path.exists(iter_dir):
                break

            full_records_file = os.path.join(iter_dir, "full_records.json")
            compilation_file = os.path.join(iter_dir, "code_compilation_repl.json")

            # Check if iteration has required output files
            if (os.path.exists(full_records_file) and os.path.getsize(full_records_file) > 10 and
                os.path.exists(compilation_file) and os.path.getsize(compilation_file) > 10):
                status['prover_iterations_complete'] = i + 1
            else:
                # This iteration exists but is incomplete
                status['prover_has_partial_iteration'] = True
                status['prover_partial_iteration_num'] = i
                break

        # Check for consolidated results (indicates prover fully done for this round)
        full_records_dir = os.path.join(prover_dir, "full_records")
        if os.path.exists(full_records_dir) and os.listdir(full_records_dir):
            status['prover_fully_complete'] = True

    return status


def _cleanup_partial_iteration(prover_dir: str, iteration: int) -> None:
    """
    Delete a partial iteration directory to allow clean restart.

    Args:
        prover_dir: Path to the prover directory
        iteration: Iteration number to clean up
    """
    iter_dir = os.path.join(prover_dir, f"iter{iteration}")
    if os.path.exists(iter_dir):
        logger.warning(f"Cleaning up partial iteration: {iter_dir}")
        shutil.rmtree(iter_dir)


def detect_checkpoint(output_dir: str, max_rounds: int = 10) -> dict:
    """
    Detect the last checkpoint in the output directory and compute the resume point.

    Args:
        output_dir: Base output directory to scan
        max_rounds: Maximum number of rounds to check

    Returns:
        Dictionary containing:
        - has_previous_run: bool - Whether there's anything to resume from
        - start_from_round: int - Round to start from
        - resume_from_component: str | None - Component to resume from (None = start of round)
        - start_from_iteration: int - Iteration to resume from (for prover)
        - last_completed_round: int - Last fully completed round (-1 if none)
        - last_completed_component: str | None - Last completed component in current round
    """
    result = {
        'has_previous_run': False,
        'start_from_round': 0,
        'resume_from_component': None,
        'start_from_iteration': 0,
        'last_completed_round': -1,
        'last_completed_component': None,
    }

    # Quick check: if output_dir doesn't exist or is empty, nothing to resume
    if not os.path.exists(output_dir):
        return result

    # Check each round in order
    for round_num in range(max_rounds):
        status = _detect_round_status(output_dir, round_num)

        if not status['round_exists']:
            # Round doesn't exist - start from this round
            if round_num > 0:
                result['has_previous_run'] = True
                result['last_completed_round'] = round_num - 1
            result['start_from_round'] = round_num
            return result

        # Track last completed component for reporting
        if status['breakdown_complete']:
            result['last_completed_component'] = 'breakdown'
        if status['formalizer_complete']:
            result['last_completed_component'] = 'formalizer'
        if status['prover_iterations_complete'] > 0:
            result['last_completed_component'] = f"prover_iter{status['prover_iterations_complete'] - 1}"
        if status['prover_fully_complete']:
            result['last_completed_component'] = 'prover_complete'

        # Check for incomplete components in this round
        if not status['breakdown_complete']:
            # Resume from breakdown (or start fresh if round 0)
            result['has_previous_run'] = (round_num > 0)
            result['start_from_round'] = round_num
            if round_num > 0:
                result['last_completed_round'] = round_num - 1
            # resume_from_component = None means start from beginning of round
            return result

        if not status['formalizer_complete']:
            result['has_previous_run'] = True
            result['start_from_round'] = round_num
            result['resume_from_component'] = 'formalizer'
            if round_num > 0:
                result['last_completed_round'] = round_num - 1
            return result

        if not status['prover_fully_complete']:
            result['has_previous_run'] = True
            result['start_from_round'] = round_num
            result['resume_from_component'] = 'recursive_prover'
            result['start_from_iteration'] = status['prover_iterations_complete']

            # Clean up partial iteration if exists
            if status['prover_has_partial_iteration']:
                prover_dir = os.path.join(output_dir, f"round{round_num}", "prover")
                _cleanup_partial_iteration(prover_dir, status['prover_partial_iteration_num'])

            if round_num > 0:
                result['last_completed_round'] = round_num - 1
            return result

        # This round is fully complete
        result['last_completed_round'] = round_num

    # All checked rounds are complete - run proof_builder only
    result['has_previous_run'] = True
    result['start_from_round'] = max_rounds  # Will skip all rounds, just run proof_builder
    result['last_completed_component'] = 'all_rounds_complete'
    return result


def format_checkpoint_info(checkpoint: dict) -> str:
    """
    Format checkpoint information for logging.

    Args:
        checkpoint: Checkpoint dictionary from detect_checkpoint()

    Returns:
        Formatted string for display
    """
    lines = []
    lines.append("=" * 60)
    lines.append("AUTO-RESUME: Detected previous run")
    lines.append(f"  Last completed round: {checkpoint['last_completed_round']}")
    lines.append(f"  Last completed component: {checkpoint['last_completed_component']}")

    if checkpoint['start_from_round'] >= 100:  # max_rounds
        lines.append("  Action: All rounds complete, will run proof_builder only")
    else:
        resume_info = f"round {checkpoint['start_from_round']}"
        if checkpoint['resume_from_component']:
            resume_info += f", component: {checkpoint['resume_from_component']}"
        if checkpoint['start_from_iteration'] > 0:
            resume_info += f", iteration: {checkpoint['start_from_iteration']}"
        lines.append(f"  Resuming from: {resume_info}")

    lines.append("=" * 60)
    return "\n".join(lines)
