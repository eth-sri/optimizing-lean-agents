import os
import json
import shutil
from datetime import datetime
from .component import create_component, Component
from seed_prover.components.feedback_data_loader import FeedbackDataLoaderComponent
from run_reuse_utils import (
    resolve_base_run_path,
    copy_round_for_component_resume,
    get_components_to_skip_for_resume,
    load_data_for_component_resume
)
from loguru import logger


class Pipeline:
    """
    Main pipeline orchestrator that executes components in sequence.
    """
    
    def __init__(self, config, base_output_dir=None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config (dict): Full pipeline configuration from YAML
            base_output_dir (str, optional): Override output directory
        """
        self.config = config
        self.components = {}
        self.repeat_components_list = []
        
        # Setup output directory
        if base_output_dir:
            self.output_dir = base_output_dir
        else:
            self.output_dir = self._create_output_dir()
        
        # Extract global config (everything except 'components')
        self.global_config = config.get('pipeline', {})
        self.config['output_dir'] = self.output_dir  # Add computed output dir
        
        # Create components
        self._create_components()

        # Add feedback data loader
        # Check if there's a config for feedback_data_loader in the YAML
        feedback_config = {}
        if 'components' in self.config and 'feedback_data_loader' in self.config['components']:
            feedback_config = self.config['components']['feedback_data_loader'].get('config', {})

        self.components['feedback_data_loader'] = FeedbackDataLoaderComponent(
            name="feedback_data_loader",
            config=feedback_config,
            global_config=self.config
        )

        logger.info(f"Pipeline initialized. Output directory: {self.output_dir}")

    def _create_output_dir(self):
        """Create timestamped output directory like the bash script."""
        output_config = self.config.get('output', {})
        base_dir = output_config.get('dir', 'results')
        job_name = output_config.get('job_name', 'pipeline-job')

        if output_config.get('create_timestamp_dirs', True):
            timestamp = datetime.now().strftime('%Y/%m/%d/%H%M%S')
            output_dir = os.path.join(base_dir, job_name, timestamp)
        else:
            output_dir = os.path.join(base_dir, job_name)

        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _create_components(self):
        """Create all components from configuration."""
        if 'components' not in self.config:
            raise ValueError("No components defined in configuration")
        
        for name, comp_info in self.config['components'].items():
            if 'type' not in comp_info:
                raise ValueError(f"Component '{name}' missing 'type' field")
            
            component_type = comp_info['type']
            component_config = comp_info.get('config', {})
            
            self.components[name] = create_component(
                component_type, name, component_config, self.config
            )
            self.repeat_components_list.append(name)
            logger.info(f"Created component: {name} ({component_type})")

        logger.info(f"Repeat components: {self.repeat_components_list}")

    def _cleanup_unused_components_for_iteration_resume(self, round_num):
        """
        Clean up components that won't be used for iteration resume.
        This is critical for freeing CUDA memory from models like the formalizer.

        Args:
            round_num: The round being executed
        """
        run_reuse_config = self.global_config.get('run_reuse', {})
        start_from_round = run_reuse_config.get('start_from_round', 0)
        start_from_iteration = run_reuse_config.get('start_from_iteration', 0)

        # Only cleanup if doing iteration resume at this specific round
        if not (run_reuse_config.get('enabled', False) and
                round_num == start_from_round and
                start_from_iteration > 0):
            return

        # Components that won't be used during iteration resume
        components_to_cleanup = ['breakdown', 'breakdown_json', 'breakdown_parser', 'breakdown_json_parser', 'formalizer', 'data_loader']

        logger.info("Iteration resume active: cleaning up unused components to free memory...")

        for comp_name in components_to_cleanup:
            if comp_name in self.components:
                logger.info(f"  Deleting {comp_name} component...")
                del self.components[comp_name]

                # Remove from repeat list if present
                if comp_name in self.repeat_components_list:
                    self.repeat_components_list.remove(comp_name)

        # Force garbage collection to free memory (especially CUDA memory)
        import gc
        gc.collect()

        # Try to clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("  Cleared CUDA cache")
        except ImportError:
            pass

        logger.info("Component cleanup complete")

    def run_single_round(self, round_num):
        """
        Run a single round of the pipeline.

        Args:
            round_num: The round number to execute

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Starting Round {round_num}")

            # Check if we're doing iteration resume at this round
            # Note: self.global_config is the pipeline config section, config contains full config
            # But we set it in __init__ as self.global_config = config.get('pipeline', {})
            # So self.global_config should have run_reuse directly
            run_reuse_config = self.global_config.get('run_reuse', {})
            start_from_round = run_reuse_config.get('start_from_round', 0)
            start_from_iteration = run_reuse_config.get('start_from_iteration', 0)
            resume_from_component = run_reuse_config.get('resume_from_component', None)
            is_iteration_resume = (run_reuse_config.get('enabled', False) and
                                   round_num == start_from_round and
                                   start_from_iteration > 0)
            is_component_resume = (run_reuse_config.get('enabled', False) and
                                   round_num == start_from_round and
                                   resume_from_component is not None and
                                   start_from_iteration == 0)  # Component resume only when not doing iteration resume

            # Determine component list for this round
            current_round_components = self.repeat_components_list.copy()

            # If doing iteration resume, skip early pipeline stages
            if is_iteration_resume:
                logger.info(f"Iteration resume active: skipping early pipeline stages (breakdown, parser, formalization)")

                # Clean up unused components to free memory (especially CUDA memory)
                self._cleanup_unused_components_for_iteration_resume(round_num)

                # Only run recursive_prover
                components_to_skip = ['data_loader', 'feedback_data_loader', 'breakdown', 'breakdown_json', 'breakdown_parser', 'breakdown_json_parser', 'formalizer']
                current_round_components = [c for c in current_round_components if c not in components_to_skip]
                data = []  # Empty data, prover will load from saved files
            elif is_component_resume:
                # Component resume: skip to specified component
                logger.info(f"Component resume active: resuming from {resume_from_component}")

                verbosity = self.global_config.get('verbosity', 1)
                is_in_place_resume = run_reuse_config.get('_in_place_resume', False)

                if is_in_place_resume:
                    # In-place resume: use output_dir directly, no copying needed
                    base_run_path = self.output_dir
                    logger.info("In-place resume: using output_dir, skipping file copy")
                else:
                    # Manual resume: resolve base_run_path and copy files
                    base_run_path = resolve_base_run_path(run_reuse_config, self.output_dir)
                    copy_round_for_component_resume(base_run_path, self.output_dir, round_num, resume_from_component, verbosity)

                # Determine which components to skip
                components_to_skip = get_components_to_skip_for_resume(resume_from_component)
                logger.info(f"Skipping components: {components_to_skip}")

                # Clean up skipped components to free memory
                for comp_name in components_to_skip:
                    if comp_name in self.components and comp_name not in ['data_loader', 'feedback_data_loader']:
                        logger.info(f"  Deleting {comp_name} component to free memory...")
                        del self.components[comp_name]
                        if comp_name in self.repeat_components_list:
                            self.repeat_components_list.remove(comp_name)

                # Force garbage collection
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.info("  Cleared CUDA cache")
                except ImportError:
                    pass

                # Filter component list
                current_round_components = [c for c in current_round_components if c not in components_to_skip]

                # Load data from previous component
                data = load_data_for_component_resume(resume_from_component, round_num, base_run_path, verbosity)
            else:
                # Normal execution: load data for this round
                if round_num == 0:
                    # Initial data loading
                    data = self._run_component('data_loader', [], round_num)
                else:
                    # Load failed lemmas from previous round
                    data = self._run_component('feedback_data_loader', [], round_num)

                if not data:
                    logger.info(f"No data to process for round {round_num}. Skipping round.")
                    return True

                # Remove data_loader from round 1+ (already called feedback_data_loader)
                if round_num >= 1 and 'data_loader' in current_round_components:
                    current_round_components.remove('data_loader')

            # Run components for the current round
            for component_name in current_round_components:
                data = self._run_component(component_name, data, round_num)

            logger.info(f"Round {round_num} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Round {round_num} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _run_component(self, component_name, data, round_num):

        if component_name not in self.components:
            logger.warning(f"Component '{component_name}' not found, skipping")
            return data
        
        component = self.components[component_name]

        logger.info(f"\n--- Running {component_name.title()} (Round {round_num}) ---")

        try:
            component.load_model()
            updated_data = component.process(data, round_num)
            component.unload_model()
            logger.info(f"Completed {component_name}: {len(data)} -> {len(updated_data)} problems")
            if hasattr(component, 'cleanup'):
                component.cleanup()
            return updated_data
            
        except Exception as e:
            logger.error(f"Error in {component_name}: {e}")
            import traceback
            traceback.print_exc()
            return data

    def get_component(self, name):
        """Get a component by name."""
        return self.components.get(name)
    
    def list_components(self):
        """List all available components."""
        return list(self.components.keys())
    
    def get_output_dir(self):
        """Get the pipeline output directory."""
        return self.output_dir