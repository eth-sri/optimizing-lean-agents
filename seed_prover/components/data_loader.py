"""
Data Loader Component - Loads problems from input files
"""
import os
import json
from core.component import Component
from prover.utils import InferenceHandler
from metadata_utils import create_metadata, generate_uid


class DataLoaderComponent(Component):
    """
    Component to load initial problem data from jsonl files.
    """
    
    def process(self, data_list, round_num=0):
        """
        Load problems from input file.
        
        Args:
            data_list: Empty list (initial load)
            round_num: Current round (should be 0 for data loading)
            
        Returns:
            List[Dict]: List of problem dictionaries loaded from file
        """
        print(f"DataLoaderComponent: Loading data from {self.config.get('input_path', 'default.jsonl')}")
        
        # Get configuration
        input_path = self.config.get('input_path')
        split = self.config.get('split', 'none')

        if not input_path:
            print("Error: input_path not specified in configuration")
            return []
        
        # Create handler to load data (using any handler since load_split is the same)
        handler = InferenceHandler()
        
        try:
            # Load initial data using handler
            initial_data_list = handler.load_split(input_path, split)

            # Save dataset for reuse purposes
            output_dir = self.global_config.get("output_dir")
            dataset_path = os.path.join(output_dir, "dataset.json")
            with open(dataset_path, 'w') as f:
                json.dump(initial_data_list, f, indent=2)

            # Initialize metadata for each problem
            for item in initial_data_list:
                # Get the base problem ID from the data
                origin_problem_id = item.get("problem_id", item.get("name"))

                # Create metadata with origin_problem_id and round_id
                metadata = create_metadata(
                    origin_problem_id=origin_problem_id,
                    round_id=round_num
                )

                # Add metadata and uid to item
                item["metadata"] = metadata
                item["uid"] = generate_uid(metadata)

            print(f"Loaded {len(initial_data_list)} problems from {input_path}")
            return initial_data_list
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return []