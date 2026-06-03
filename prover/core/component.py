import yaml

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
        # Handle model_config as either a file path (str) or resolved dict (from Hydra)
        if self.config.get('model_config', None):
            if isinstance(self.config.get('model_config'), str):
                model_config_path = self.config.get('model_config')
                if model_config_path:
                    self.model_config = yaml.safe_load(open(model_config_path, 'r'))
            else:
                # Hydra-resolved dict config
                self.model_config = self.config.get('model_config')

    
    def process(self, data_list, output_dir, round_num=0, additional_data=None):
        raise NotImplementedError(f"Component {self.name} must implement process() method")




def create_component(component_type, name, component_config, global_config):
    # Import components here to avoid circular imports
    from prover.components.data_loader import DataLoaderComponent
    from prover.components.summarization_component import SummarizationComponent
    from prover.components.prover import ProverComponent
    from prover.components.informal_summary import InformalSummaryComponent
    from prover.components.attempt_summarizer import AttemptSummarizerComponent

    component_registry = {
        "DataLoaderComponent": DataLoaderComponent,
        "SummarizationComponent": SummarizationComponent,
        "ProverComponent": ProverComponent,
        "InformalSummaryComponent": InformalSummaryComponent,
        "AttemptSummarizerComponent": AttemptSummarizerComponent,
    }
    
    if component_type not in component_registry:
        raise ValueError(f"Unknown component type: {component_type}")
    
    component_class = component_registry[component_type]
    return component_class(name, component_config, global_config)