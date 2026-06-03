#!/usr/bin/env python3
"""
Hydra-enabled pipeline runner for the prover module.
Uses @hydra.main for clean configuration management.

Usage:
    # Basic usage (uses default config)
    python prover/hydra_runner.py

    # Override config file
    python prover/hydra_runner.py --config-name=custom_prover

    # Override specific values
    python prover/hydra_runner.py \
        components.data_loader.config.input_path=dataset/test.jsonl \
        output.dir=/scratch/results \
        pipeline.max_correction_rounds=2

    # Override model selection
    python prover/hydra_runner.py \
        model.prover=models/goedel_prover_v2/32b
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

from prover.core.pipeline import Pipeline


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
        # Create and run pipeline
        logger.info("Initializing pipeline...")
        pipeline = Pipeline(config)

        logger.info("Starting pipeline execution...")
        success = pipeline.run()

        if success:
            logger.info("\nPipeline completed successfully!")
            logger.info(f"Results available in: {pipeline.get_output_dir()}")
            return 0
        else:
            logger.error("\nPipeline failed!")
            return 1

    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# Hydra configuration
# - config_path: relative path from this file to config directory
# - config_name: default config file name (without .yaml extension)
# - version_base: None uses latest Hydra defaults
@hydra.main(
    version_base=None,
    config_path="../configs/hydra/prover",
    config_name="config"
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point with Hydra configuration.

    The configuration is automatically loaded by Hydra based on:
    - Default config: configs/hydra/prover/config.yaml
    - Command-line overrides: key=value format

    Hydra automatically saves:
    - .hydra/config.yaml - resolved config
    - .hydra/hydra.yaml - Hydra settings
    - .hydra/overrides.yaml - command-line overrides

    Examples:
        # Use different model
        python hydra_runner.py model.prover=models/goedel_prover_v2/32b

        # Override values
        python hydra_runner.py output.dir=/my/output pipeline.max_correction_rounds=3

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
