"""
Data Loader Component - Loads problems from input files
"""
from prover.core.component import Component
from prover.utils import InferenceHandler, load_data_for_correction
import yaml


class DataLoaderComponent(Component):
    """
    Component to load initial problem data from jsonl files.
    """
    
    def process(self, data_list, output_dir, round_num=0):
        verbosity = self.global_config.get('verbosity', 3)

        if verbosity >= 1:
            print(f"DataLoaderComponent: Loading data from {self.config.get('input_path', 'default.jsonl')}")

        previous_output_dir = self.config.get('previous_output_dir', None)
        if previous_output_dir:
            start_correction_round = (
                self.global_config
                    .get('pipeline', {})
                    .get('start_correction_round', 0)
            )
            assert start_correction_round > 0, "start_correction_round must be non-negative when have previous output"
            next_component = self.config.get('next_component', None)
            model_config_path = (
                self.global_config
                    .get('components', {})
                    .get('prover', {})
                    .get('config', {})
                    .get('model_config', None)
            )

            model_config = yaml.safe_load(open(model_config_path, 'r'))
            base_output_template = (
                model_config.get('base_output_template', '')
            )
            
            # Get use_metadata flag from prover component config
            use_metadata = (
                self.global_config
                    .get('components', {})
                    .get('prover', {})
                    .get('config', {})
                    .get('use_metadata', False)
            )

            # Get correct_all_failed_attempts from pipeline config
            correct_all_failed_attempts = self.global_config.get('pipeline', {}).get('correct_all_failed_attempts', False)

            return load_data_for_correction(
                previous_output_dir, start_correction_round, base_output_template, use_metadata, correct_all_failed_attempts
            )
        else:
            input_path = self.config.get('input_path')
            split = self.config.get('split', 'none')

            if not input_path:
                print("Error: input_path not specified in configuration")
                return []
        
            handler = InferenceHandler()
            
            # Load initial data using handler
            initial_data_list = handler.load_split(input_path, split)

            for item in initial_data_list:
                item["origin_problem_id"] = item.get("problem_id", item.get("name"))
                problem_id = item.get("problem_id")
                problem_id = f"{problem_id}_r{round_num}"
                item["problem_id"] = problem_id
                if item.get("id_maps") is None:
                    item["id_maps"] = [
                        {"origin_problem_id": item["origin_problem_id"]}
                    ]

            if verbosity >= 1:
                print(f"Loaded {len(initial_data_list)} problems from {input_path}")
            return initial_data_list
        