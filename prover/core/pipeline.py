import os
from datetime import datetime
from prover.core.component import create_component
import yaml


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
        if config['output'].get('create_subdirs', True):
            self.output_dir = self._create_output_dir()
        else:
            self.output_dir = config['output']['dir']
        
        # Extract global config (everything except 'components')
        self.global_config = config.get('pipeline', {})
        self.config['output_dir'] = self.output_dir  # Add computed output dir

        self.start_correction_round = self.global_config.get('start_correction_round', 0)
        self.max_rounds = self.global_config.get('max_correction_rounds', 0)
        
        # Create components
        self._create_components()
        
        print(f"Pipeline initialized. Output directory: {self.output_dir}")
    
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

        config_path = os.path.join(output_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

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
            if component_type != "DataLoaderComponent":
                self.repeat_components_list.append(name)
            print(f"Created component: {name} ({component_type})")

        print(self.repeat_components_list)
    
    def run(self):
        """
        Main pipeline execution method.
        Replaces the bash script's main loop logic.
        """

        print("Starting Pipeline Execution")

        # Step 1: Load initial data
        data = self._run_data_loader()
        
        # Step 2: Run correction rounds (replaces bash for loop)
        for round_num in range(self.start_correction_round, self.max_rounds + 1):  # 0 to max_rounds
            if not data:
                print("No data to process, stopping pipeline")
                break
            print(f"\n{'='*20} Starting Round {round_num} {'='*20}")
            
            # Run inference
            for component_name in self.repeat_components_list:
                print(f"Running component: {component_name}")
                if component_name == "informal_summary" and round_num == 0:
                    continue
                data = self._run_component(component_name, data, round_num)

        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print(f"Results saved in: {self.output_dir}")
        print("=" * 60)
        
        return True
    
    def _run_data_loader(self):

        if 'data_loader' not in self.components:
            raise ValueError("No data_loader component defined")
        
        print("\n--- Loading Initial Data ---")
        component = self.components['data_loader']
        data = component.process([], self.output_dir, 0)
        print(f"Loaded {len(data)} problems")
        return data
    
    def _run_component(self, component_name, data, round_num):

        if component_name not in self.components:
            print(f"Warning: Component '{component_name}' not found, skipping")
            return data
        
        component = self.components[component_name]
        
        print(f"\n--- Running {component_name.title()} (Round {round_num}) ---")
        
        updated_data = component.process(data, round_num)
        print(f"Completed {component_name}: {len(data)} -> {len(updated_data)} problems")
        return updated_data

    
    def get_component(self, name):
        """Get a component by name."""
        return self.components.get(name)
    
    def list_components(self):
        """List all available components."""
        return list(self.components.keys())
    
    def get_output_dir(self):
        """Get the pipeline output directory."""
        return self.output_dir