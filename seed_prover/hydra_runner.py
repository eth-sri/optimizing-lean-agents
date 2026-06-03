#!/usr/bin/env python3
"""
Hydra-enabled pipeline runner using decorator mode.
Uses @hydra.main for clean configuration management.

Usage:
    # Basic usage (uses default config)
    python seed_prover/hydra_runner.py

    # Override config file
    python seed_prover/hydra_runner.py --config-name=seed_prover_training

    # Override specific values
    python seed_prover/hydra_runner.py \
        components.data_loader.config.input_path=dataset/test.jsonl \
        output.dir=/scratch/results \
        pipeline.max_refine_rounds=2

    # Override config path (for configs in different location)
    python seed_prover/hydra_runner.py \
        --config-path=/absolute/path/to/configs \
        --config-name=my_config
"""
import gc
import os
import sys
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# Add current directory and project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

from core.pipeline import Pipeline
from run_reuse_utils import setup_base_run_resume
from auto_resume import detect_checkpoint, format_checkpoint_info


def setup_environment():
    """Setup environment variables."""
    scratch_dir = os.environ.get('SCRATCH', '/tmp')
    os.environ['HF_HOME'] = os.path.join(scratch_dir, 'huggingface')
    os.environ['TRANSFORMERS_VERBOSITY'] = 'info'


def run_pipeline(cfg: DictConfig) -> int:
    """
    Run the pipeline with the given configuration.

    Args:
        cfg: Resolved configuration dict

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Convert OmegaConf to regular dict for Pipeline compatibility
    config = OmegaConf.to_container(cfg, resolve=True)

    # Expand environment variables in output directory
    if 'output' in config and 'dir' in config['output']:
        config['output']['dir'] = os.path.expandvars(config['output']['dir'])

    logger.info("Configuration loaded:")
    if config.get('verbosity', 0) >= 2:
        print(OmegaConf.to_yaml(cfg, resolve=True))

    try:
        # Create output directory once (all rounds will use the same dir)
        logger.info("Initializing pipeline...")
        first_pipeline = Pipeline(config)
        output_dir = first_pipeline.get_output_dir()

        # Clean up the first pipeline (we only needed it to create the output dir)
        del first_pipeline
        gc.collect()

        logger.info("Starting pipeline execution...")

        # Run each round with a fresh pipeline instance for maximum memory reclamation
        max_refine_rounds = config.get('pipeline', {}).get('max_refine_rounds', 1)
        run_reuse_config = config.get('pipeline', {}).get('run_reuse', {})

        # Resume logic:
        # - enabled: false → No resume, start fresh
        # - enabled: true + base_run_path set → Manual resume from specified path
        # - enabled: true + auto_resume: true + base_run_path: null → Auto-resume (detect checkpoint)

        if run_reuse_config.get('enabled', False):
            base_run_path = run_reuse_config.get('base_run_path')
            auto_resume_enabled = run_reuse_config.get('auto_resume', True)

            # Auto-resume: detect checkpoint if no base_run_path specified
            if not base_run_path and auto_resume_enabled:
                checkpoint = detect_checkpoint(output_dir, max_rounds=max_refine_rounds)

                if checkpoint['has_previous_run']:
                    logger.info(format_checkpoint_info(checkpoint))

                    # Update run_reuse config for in-place resume
                    run_reuse_config['_in_place_resume'] = True  # Flag to skip file copying
                    run_reuse_config['base_run_path'] = output_dir
                    run_reuse_config['_resolved_base_run_path'] = output_dir
                    run_reuse_config['start_from_round'] = checkpoint['start_from_round']
                    run_reuse_config['resume_from_component'] = checkpoint['resume_from_component']
                    run_reuse_config['start_from_iteration'] = checkpoint['start_from_iteration']

                    # Update config dict
                    if 'pipeline' not in config:
                        config['pipeline'] = {}
                    if 'run_reuse' not in config['pipeline']:
                        config['pipeline']['run_reuse'] = {}
                    config['pipeline']['run_reuse'].update(run_reuse_config)
                else:
                    # No previous run found, disable resume and start fresh
                    logger.info("Auto-resume enabled but no previous run detected, starting fresh")
                    run_reuse_config['enabled'] = False

        if run_reuse_config.get('enabled', False):
            start_from_round = run_reuse_config.get('start_from_round', 0)
        else:
            start_from_round = 0

        # Check if we need to access base run (for round copying or iteration resume)
        start_from_iteration = run_reuse_config.get('start_from_iteration', 0)

        # Skip setup_base_run_resume for in-place resume (files already exist)
        is_in_place_resume = run_reuse_config.get('_in_place_resume', False)
        needs_base_run = (run_reuse_config.get('enabled', False) and
                         not is_in_place_resume and
                         (start_from_round >= 1 or start_from_iteration > 0))

        # Resolve base_run_path and copy previous rounds if needed
        if needs_base_run:
            start_from_round, start_from_iteration, _ = setup_base_run_resume(
                config=config,
                output_dir=output_dir,
                run_reuse_config=run_reuse_config,
                start_from_round=start_from_round,
                start_from_iteration=start_from_iteration
            )

        for round_num in range(start_from_round, max_refine_rounds):
            logger.info(f"\n{'='*60}")
            logger.info(f"ROUND {round_num}/{max_refine_rounds - 1}")
            logger.info(f"{'='*60}")

            # Create fresh pipeline for this round
            logger.info(f"Creating fresh pipeline instance for round {round_num}...")
            pipeline = Pipeline(config, base_output_dir=output_dir)

            # Run this round
            success = pipeline.run_single_round(round_num)

            if not success:
                logger.error(f"Round {round_num} failed!")
                del pipeline
                gc.collect()
                return 1

            # Cleanup pipeline to free memory before next round
            logger.info(f"Cleaning up pipeline for round {round_num}...")
            del pipeline
            gc.collect()

            # Extra cleanup for GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("GPU memory cleared")
            except ImportError:
                pass

        # Run proof_builder once after all rounds complete
        logger.info(f"\n{'='*60}")
        logger.info("Running proof builder on all rounds...")
        logger.info(f"{'='*60}")

        final_pipeline = Pipeline(config, base_output_dir=output_dir)
        if 'proof_builder' in final_pipeline.components:
            final_pipeline._run_component('proof_builder', [], round_num=None)

        # Final cleanup
        del final_pipeline
        gc.collect()

        logger.info("\nPipeline completed successfully!")
        logger.info(f"Results available in: {output_dir}")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


# Hydra configuration
# - config_path: relative path from this file to config directory
# - config_name: default config file name (without .yaml extension)
# - version_base: None uses latest Hydra defaults
@hydra.main(
    version_base=None,
    config_path="../configs/hydra/seed_prover",
    config_name="config"
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point with Hydra configuration.

    The configuration is automatically loaded by Hydra based on:
    - Default config: configs/seed_prover/seed_prover.yaml
    - Command-line overrides: key=value format

    Hydra automatically saves:
    - .hydra/config.yaml - resolved config
    - .hydra/hydra.yaml - Hydra settings
    - .hydra/overrides.yaml - command-line overrides

    Examples:
        # Use different prover config
        python hydra_runner.py prover=unified_fast

        # Override values
        python hydra_runner.py output.dir=/my/output pipeline.max_refine_rounds=3

        # Override nested values
        python hydra_runner.py components.data_loader.config.input_path=my_data.jsonl
    """
    print("=" * 60)
    print("Formal Proof Pipeline Runner (Hydra Mode)")
    print("=" * 60)

    setup_environment()

    exit_code = run_pipeline(cfg)

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
