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

class ProverWrapperComponent(Component):
    """
    This component wraps the core prover
    takes the lemmas directory as input and call the core prover on them
    """
    def __init__(self, name, component_config, global_config):
        super().__init__(name, component_config, global_config)

        print(component_config)

        prover_config_path = component_config["prover_config"]
        with open(prover_config_path, "r") as f:
            self.prover_config = yaml.safe_load(f)

        # Initialize the prover pipeline once to reuse across rounds
        self.prover_pipeline = None
        self.reuse_pipeline = component_config.get("reuse_pipeline", True)

        if self.reuse_pipeline:
            logger.info("ProverWrapper: Pipeline will be created on first use and reused across rounds")
        else:
            logger.info("ProverWrapper: Pipeline will be recreated for each round")

    def process(self, data, round_num=0):
        base_output_dir = self.global_config.get("output_dir")

        # Get output subdirectory from config (e.g., "theorem_prover" or "lemma_prover")
        output_subdir = self.config.get("output_subdir", "lemmas")
        output_dir = os.path.join(base_output_dir, f"round{round_num}", output_subdir)

        # Get input filename from config (default: "lemmas.json")
        input_filename = self.config.get("input_filename", "lemmas.json")
        input_path = os.path.join(output_dir, input_filename)

        # Check if input file exists
        if not os.path.exists(input_path):
            logger.warning(f"Input file not found at {input_path}, skipping prover")
            return data

        logger.info(f"ProverWrapper: Using input file {input_filename} from {output_subdir}/")

        # Update config for this round
        self.prover_config["output"] = dict()
        self.prover_config["output"]["dir"] = output_dir
        self.prover_config["output"]["create_subdirs"] = False
        self.prover_config["components"]["data_loader"]["config"]["input_path"] = input_path

        if self.reuse_pipeline and self.prover_pipeline is not None:
            # Reuse existing pipeline, just update the data loader config
            logger.info(f"ProverWrapper: Reusing pipeline from previous round")
            self.prover_pipeline.config["output"]["dir"] = output_dir
            self.prover_pipeline.config["components"]["data_loader"]["config"]["input_path"] = input_path
            self.prover_pipeline.output_dir = output_dir
        else:
            # Create new pipeline (first run or reuse_pipeline=False)
            if self.prover_pipeline is None:
                logger.info(f"ProverWrapper: Creating prover pipeline for the first time")
            else:
                logger.info(f"ProverWrapper: Recreating prover pipeline for round {round_num}")
            self.prover_pipeline = ProverPipeline(self.prover_config)

        self.prover_pipeline.run()

        # Return the original data so the pipeline can continue
        # The prover results are saved to disk and don't need to flow through the pipeline
        return data