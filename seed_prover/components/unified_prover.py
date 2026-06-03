import os
import sys
from pathlib import Path
from core.component import Component
from jload import jload, jsave
from loguru import logger
import yaml

# Import the prover pipeline
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from prover.core.pipeline import Pipeline as ProverPipeline


class UnifiedProverComponent(Component):
    """
    Unified component that can prove both theorems and lemmas.

    This component wraps the core prover and handles iteration-based output directory
    structure (iter1/, iter2/, etc.) for recursive proving.
    """

    def __init__(self, name, component_config, global_config):
        super().__init__(name, component_config, global_config)

        if isinstance(component_config["prover_config"], str):
            prover_config_path = component_config["prover_config"]
            with open(prover_config_path, "r") as f:
                self.prover_config = yaml.safe_load(f)
        elif isinstance(component_config["prover_config"], dict):
            self.prover_config = component_config["prover_config"]
        else:
            raise ValueError("prover_config must be either a file path or a dict")

    def process(self, data, round_num=0, iteration=1, input_data=None, start_from_proving_round=None, model_config_override=None, cleanup_after=False):
        """
        Run the prover on the provided input data.

        Args:
            data: Pipeline data (passed through unchanged)
            round_num: Current round number
            iteration: Current iteration number (1=theorems, 2+=lemmas)
            input_data: List of items to prove (theorems or lemmas)
            start_from_proving_round: Optional proving/correction round to start from (for resume)
            model_config_override: Optional model config path to override the default prover model
            cleanup_after: If True, cleanup GPU memory after this iteration (for model switching)

        Returns:
            Original data (unchanged)
        """
        verbosity = self.global_config.get('verbosity', 3)
        base_output_dir = self.global_config.get("output_dir")

        # Create iteration-specific output directory
        output_dir = os.path.join(base_output_dir, f"round{round_num}", "prover", f"iter{iteration}")
        os.makedirs(output_dir, exist_ok=True)

        if verbosity >= 1:
            logger.info(f"UnifiedProverComponent: Running iteration {iteration}")
            logger.info(f"  Output directory: {output_dir}")

        if not input_data:
            logger.warning(f"No input data provided for iteration {iteration}, skipping")
            return data

        if verbosity >= 1:
            logger.info(f"  Items to prove: {len(input_data)}")

        # Save input data to iteration directory for reference
        input_file = os.path.join(output_dir, "input.json")
        jsave(input_data, input_file)

        # Configure the prover
        self.prover_config["output"] = {
            "dir": output_dir,
            "create_subdirs": False
        }
        self.prover_config["components"]["data_loader"]["config"]["input_path"] = input_file

        # Override model config if provided
        if model_config_override is not None:
            if verbosity >= 1:
                logger.info(f"  Overriding prover model config with: {model_config_override}")
            self.prover_config["components"]["prover"]["config"]["model_config"] = model_config_override

        # Set start_correction_round if provided
        if start_from_proving_round is not None:
            if verbosity >= 1:
                logger.info(f"  Setting start_correction_round to {start_from_proving_round}")
            self.prover_config["pipeline"]["start_correction_round"] = start_from_proving_round

        # Run the prover pipeline
        if verbosity >= 1:
            logger.info(f"  Starting proof generation and compilation...")

        prover_pipeline = ProverPipeline(self.prover_config)
        prover_pipeline.run()

        if verbosity >= 1:
            logger.info(f"  Iteration {iteration} completed")

        # Cleanup GPU memory if requested (for model switching)
        if cleanup_after:
            if verbosity >= 1:
                logger.info(f"  Cleaning up GPU memory before switching models...")

            # Delete the pipeline to release references
            del prover_pipeline

            # Force garbage collection
            import gc
            gc.collect()

            # Clear GPU cache if torch is available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if verbosity >= 2:
                        logger.info(f"  GPU memory cleared")
            except ImportError:
                pass

        # Return original data so the pipeline can continue
        return data
