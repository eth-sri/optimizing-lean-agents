"""
ProofBuilderComponent - Builds comprehensive proofs by recursively extracting used lemmas.
"""

import os
from typing import Optional, Any, List, Dict

from loguru import logger
from jload import jsave

from core.component import Component
from lean_compiler.repl_scheduler import scheduler

# Import seed_data_models (models and loader)
from seed_data_models import Session, DataLoader


class ProofBuilderComponent(Component):
    """
    Component that builds comprehensive proofs with full round context.

    Builds complete .lean files by:
    1. Loading all rounds into memory via analysis_gui Session model
    2. Identifying fully solved breakdowns
    3. Recursively extracting used lemmas across all rounds
    4. Generating flat .lean files in topological order
    5. Distinguishing between complete and incomplete proofs
    """

    def __init__(self, name, component_config, global_config):
        super().__init__(name, component_config, global_config)
        self.session: Optional[Session] = None
        self.fully_solved_breakdowns: List[Dict[str, Any]] = []
        self.combined_proofs_metadata: List[Dict[str, Any]] = []
        self.compilation_results: List[Dict[str, Any]] = []

        # Set run_dir from global_config output_dir
        # The proof builder will look for full_records/ in this directory
        self.run_dir = os.path.expandvars(global_config.get('output_dir', '.'))

    def _load_all_rounds(self) -> None:
        """
        Load all rounds from results directory into a Session object.

        Uses analysis_gui's Session model to consolidate all round data into
        a unified in-memory representation that can be queried across rounds.
        """
        logger.info("Loading all rounds into Session object...")
        logger.info(f"Reading from run_dir: {self.run_dir}")

        # Check if full_records exists
        full_records_path = os.path.join(self.run_dir, 'full_records')
        if not os.path.exists(full_records_path):
            logger.warning(f"full_records directory not found at: {full_records_path}")
            logger.info(f"Contents of run_dir: {os.listdir(self.run_dir) if os.path.exists(self.run_dir) else 'run_dir does not exist'}")

        # Load session from run directory using DataLoader
        # The DataLoader will read from full_records/ and build the object model
        loader = DataLoader(self.run_dir)
        self.session = loader.load_session()

        logger.info(f"Loaded Session with {len(self.session.problems)} problems")

    def process(self, data_list, output_dir, round_num=0, additional_data=None):
        """
        Main entry point for the proof builder component.

        Orchestrates the full workflow:
        1. Load all rounds into Session object
        2. For each breakdown with a proof:
           a. Determine if complete or incomplete
           b. Generate .lean file content
           c. Write to complete/ or incomplete/ directory
        3. Compile only complete proofs
        4. Save metadata and results

        Args:
            data_list: List of problem dictionaries (passed through unchanged)
            output_dir: Output directory path (not used - loads from run_dir)
            round_num: Current round number (not used)
            additional_data: Additional data for correction rounds (not used)

        Returns:
            data_list: The input data_list unchanged
        """
        logger.info("Starting ProofBuilder component...")

        # Phase 1: Load data
        self._load_all_rounds()

        if not self.session:
            logger.error("Failed to load session")
            return data_list

        # Phase 2: Process all breakdowns and generate proofs
        logger.info("Building proofs for all breakdowns...")

        combined_dir = os.path.join(self.run_dir, "combined")
        os.makedirs(combined_dir, exist_ok=True)

        complete_dir = os.path.join(combined_dir, "lean_files", "complete")
        incomplete_dir = os.path.join(combined_dir, "lean_files", "incomplete")
        os.makedirs(complete_dir, exist_ok=True)
        os.makedirs(incomplete_dir, exist_ok=True)

        complete_proofs = []

        for problem_key, problem in self.session.problems.items():
            for breakdown_key, breakdown in problem.breakdowns.items():
                # Generate proof using the breakdown's generate_proof() method
                # Breakdown.generate_proof() recursively handles all lemmas and recursive attempts
                try:
                    proof_code, is_complete = breakdown.generate_proof()
                except Exception as e:
                    logger.warning(f"Failed to generate proof for {breakdown_key}: {e}")
                    continue

                if not proof_code:
                    # No proof could be generated
                    continue

                # Determine file location
                file_name = f"{breakdown.origin_problem_id}_r{breakdown.round_id}_b{breakdown.breakdown_id}.lean"

                if is_complete:
                    file_path = os.path.join(complete_dir, file_name)
                    status = "complete"
                else:
                    file_path = os.path.join(incomplete_dir, file_name)
                    status = "incomplete"

                # Write file
                with open(file_path, 'w') as f:
                    f.write(proof_code)

                logger.info(f"Generated {status} proof: {file_name}")

                # Store metadata
                self.combined_proofs_metadata.append({
                    "breakdown_key": str(breakdown_key),
                    "problem_id": breakdown.origin_problem_id,
                    "round_id": breakdown.round_id,
                    "breakdown_id": breakdown.breakdown_id,
                    "status": status,
                    "file_path": file_path,
                })

                if is_complete:
                    complete_proofs.append(file_path)

        # Phase 3: Compile only complete proofs
        logger.info(f"Compiling {len(complete_proofs)} complete proofs...")

        if complete_proofs:
            # Prepare proofs for scheduler: convert file paths to dicts with code
            proof_dicts = []
            for file_path in complete_proofs:
                with open(file_path, 'r') as f:
                    code = f.read()
                proof_name = os.path.basename(file_path)
                proof_dicts.append({
                    "code": code,
                    "name": proof_name,
                    "file_path": file_path
                })

            # Get number of workers from config, default to 64
            num_workers = self.config.get('cpu', 64)
            compilation_results = scheduler(proof_dicts, num_workers=num_workers)
            self.compilation_results = compilation_results
            jsave(
                compilation_results,
                os.path.join(combined_dir, "compilation_results.json")
            )

            # Count compilation results
            passed = sum(1 for r in compilation_results if r.get('compilation_result', {}).get('pass', False))
            failed = len(compilation_results) - passed
            logger.info(f"Compilation results: {passed} passed, {failed} failed")

        # Phase 4: Save metadata
        jsave(
            [bp for bp in self.fully_solved_breakdowns],
            os.path.join(combined_dir, "fully_solved.json")
        )

        jsave(
            self.combined_proofs_metadata,
            os.path.join(combined_dir, "combined_proofs.json")
        )

        # Summary statistics
        logger.info(f"Generated {len(self.combined_proofs_metadata)} total proofs")
        complete_count = sum(1 for p in self.combined_proofs_metadata if p.get('status') == 'complete')
        incomplete_count = sum(1 for p in self.combined_proofs_metadata if p.get('status') == 'incomplete')
        logger.info(f"  - {complete_count} complete proofs (no sorries)")
        logger.info(f"  - {incomplete_count} incomplete proofs (with sorries)")

        # Phase 5: Create minified dump using the loaded session
        logger.info("Creating minified dump from session...")
        self._create_minified_dump()

        logger.info("ProofBuilder component completed successfully")
        return data_list

    def _create_minified_dump(self) -> None:
        """
        Create minified and hierarchical dumps from the loaded session.

        Creates a "dump" subfolder in the run directory:
        - dump/minified/: Minified flat JSON data
        - dump/hierarchical/: Hierarchical JSON data
        - dump/combined/: Copy of combined proofs and lean files
        - dump/configs/: Copy of configs
        - dump/dataset.json: Copy of dataset
        """
        import shutil
        from pathlib import Path

        try:
            if not self.session:
                logger.warning("No session loaded, skipping dump creation")
                return

            # Create dump directory in the run_dir
            dump_base_dir = os.path.join(self.run_dir, "dump")
            os.makedirs(dump_base_dir, exist_ok=True)

            logger.info(f"Creating dump at: {dump_base_dir}")

            # Dump minified data
            minified_dir = os.path.join(dump_base_dir, "minified")
            logger.info(f"Dumping minified data to {minified_dir}...")
            self.session.dump_minified(minified_dir)

            # Dump hierarchical data
            hierarchical_dir = os.path.join(dump_base_dir, "hierarchical")
            logger.info(f"Dumping hierarchical data to {hierarchical_dir}...")
            self.session.dump_hierarchical(hierarchical_dir)

            # Copy combined folder
            combined_src = os.path.join(self.run_dir, "combined")
            if os.path.exists(combined_src):
                combined_dst = os.path.join(dump_base_dir, "combined")
                if os.path.exists(combined_dst):
                    shutil.rmtree(combined_dst)
                shutil.copytree(combined_src, combined_dst)
                logger.info(f"Copied combined folder to {combined_dst}")
            else:
                logger.warning(f"Combined folder not found at {combined_src}")

            # Copy configs
            configs_src = os.path.join(self.run_dir, "configs")
            if os.path.exists(configs_src):
                configs_dst = os.path.join(dump_base_dir, "configs")
                if os.path.exists(configs_dst):
                    shutil.rmtree(configs_dst)
                shutil.copytree(configs_src, configs_dst)
                logger.info(f"Copied configs to {configs_dst}")

            # Copy dataset.json
            dataset_src = os.path.join(self.run_dir, "dataset.json")
            if os.path.exists(dataset_src):
                dataset_dst = os.path.join(dump_base_dir, "dataset.json")
                shutil.copy2(dataset_src, dataset_dst)
                logger.info(f"Copied dataset.json to {dataset_dst}")

            logger.info(f"Minified dump created successfully at {dump_base_dir}")

        except Exception as e:
            logger.error(f"Error creating minified dump: {e}")
            import traceback
            logger.error(traceback.format_exc())
