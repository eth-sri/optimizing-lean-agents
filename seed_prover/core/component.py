import yaml
from prover.query.api import APIQuery

class Component:
    """
    Base class for all pipeline components.
    Simple interface: process a list of problems and return updated list.
    """
    
    def __init__(self, name, component_config, global_config):
        """
        Initialize component with name and configurations.
        
        Args:
            name (str): Component name
            component_config (dict): Component-specific configuration
            global_config (dict): Whole configuration
        """
        self.name = name
        self.config = component_config      # Component-specific settings
        self.global_config = global_config  # Shared settings (output, pipeline, etc.)
        if self.config.get('model_config', None):
            if isinstance(self.config.get('model_config'), str):
                model_config_path = self.config.get('model_config')
                if model_config_path:
                    self.model_config = yaml.safe_load(open(model_config_path, 'r'))
            else:
                self.model_config = self.config.get('model_config')


    def process(self, data_list, output_dir, round_num=0, additional_data=None):
        """
        Process a list of problem dictionaries.
        
        Args:
            data_list (List[Dict]): List of problem dictionaries
            output_dir (str): Output directory path
            round_num (int): Current correction round (0 = initial)
            additional_data (dict, optional): Additional data for correction rounds
            
        Returns:
            List[Dict]: Updated list of problem dictionaries
        """
        raise NotImplementedError(f"Component {self.name} must implement process() method")
    
    def _load_template(self):
        
        template_path = self.config.get("template_path")

        if not template_path or not os.path.exists(template_path):
            raise ValueError(f"Template path {template_path} does not exist or is not set in config")

        with open(template_path, "r") as f:
            template = f.read()

        return template

    def cleanup(self):
        """
        Clean up resources used by this component.
        This should be called after the component finishes processing.
        """
        # If component has a querier attribute (APIQuery), clean it up
        if hasattr(self, 'querier') and self.querier is not None:
            if hasattr(self.querier, 'cleanup'):
                self.querier.cleanup()

    def can_process(self, data_list, round_num=0):
        """
        Check if this component should run for the given data and round.
        Override this for conditional execution.
        
        Args:
            data_list (List[Dict]): List of problem dictionaries  
            round_num (int): Current correction round
            
        Returns:
            bool: True if component should process, False to skip
        """
        return True
    
    def get_output_files(self, output_dir, round_num=0):
        """
        Get expected output file paths for this component.
        Override this to specify output files.
        
        Args:
            output_dir (str): Output directory path
            round_num (int): Current correction round
            
        Returns:
            dict: Dictionary of file types to file paths
        """
        return {}

    def load_model(self):
        if hasattr(self, 'model_config'):
            model = self.model_config.get('model')
            api = self.model_config.get('api')
            max_tokens = self.model_config.get('max_tokens', 40960)
            temperature = self.model_config.get('temperature', 1.0)

            structured_output = self.model_config.get('structured_output', None)
            if structured_output == "json":
                if api == "google":
                    self.model_config['config']['response_mime_type'] = "application/json"
                    self.model_config['config']['response_json_schema'] = self.output_schema
                elif api == "together":
                    self.model_config["response_format"]={
                        "type": "json_schema",
                        "schema": self.output_schema,
                    }

            kwargs = self.model_config.copy()
            if "structured_output" in kwargs:
                del kwargs["structured_output"]
            del kwargs["model"]
            del kwargs["api"]
            if 'date' in kwargs:
                del kwargs['date']
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

            self.querier = APIQuery(
                model=model,
                api=api,
                **kwargs
            )
        

    def unload_model(self):
        """
        Clean up resources used by this component.
        This should be called after the component finishes processing.
        """
        # If component has a querier attribute (APIQuery), clean it up
        if hasattr(self, 'querier') and self.querier is not None:
            if hasattr(self.querier, 'cleanup'):
                self.querier.cleanup()


def create_component(component_type, name, component_config, global_config):
    """
    Factory function to create component instances by type.
    
    Args:
        component_type (str): Type of component to create
        name (str): Component name
        component_config (dict): Component-specific configuration
        global_config (dict): Global/shared configuration
        
    Returns:
        Component: Instance of the requested component type
    """
    # Import components here to avoid circular imports
    from components.breakdown import BreakdownComponent
    from components.data_loader import DataLoaderComponent
    from components.breakdown_parser import BreakdownParserComponent
    from components.formalization import FormalizationComponent
    from components.prover_wrapper import ProverWrapperComponent
    from components.proof_builder import ProofBuilderComponent
    from components.recursive_prover import RecursiveProverComponent
    from components.unified_prover import UnifiedProverComponent
    from components.dependency_filter import DependencyFilterComponent
    from components.feedback_data_loader import FeedbackDataLoaderComponent
    from components.breakdown_json import BreakdownJsonComponent
    from components.breakdown_json_parser import BreakdownJsonParserComponent

    component_registry = {
        "DataLoaderComponent": DataLoaderComponent,
        "FeedbackDataLoaderComponent": FeedbackDataLoaderComponent,
        "BreakdownComponent": BreakdownComponent,
        "BreakdownParserComponent": BreakdownParserComponent,
        "FormalizationComponent": FormalizationComponent,
        "ProverWrapperComponent": ProverWrapperComponent,
        "ProofBuilderComponent": ProofBuilderComponent,
        "RecursiveProverComponent": RecursiveProverComponent,
        "UnifiedProverComponent": UnifiedProverComponent,
        "DependencyFilterComponent": DependencyFilterComponent,
        "BreakdownJsonComponent": BreakdownJsonComponent,
        "BreakdownJsonParserComponent": BreakdownJsonParserComponent
    }
    
    if component_type not in component_registry:
        raise ValueError(f"Unknown component type: {component_type}")
    
    component_class = component_registry[component_type]
    return component_class(name, component_config, global_config)