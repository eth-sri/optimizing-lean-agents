import argparse
import sys
import os
import yaml
import shutil
from pathlib import Path

# Add current directory and project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

from core.pipeline import Pipeline


def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file: {e}")
        sys.exit(1)


def find_referenced_configs(config_data, base_path='.', visited=None):
    """
    Recursively find all YAML config files referenced in a configuration.

    Args:
        config_data: Configuration data (dict, list, or other)
        base_path: Base path for resolving relative config paths
        visited: Set of already visited config paths to avoid circular references

    Returns:
        set: Set of config file paths referenced in the configuration
    """
    if visited is None:
        visited = set()

    referenced_configs = set()

    def process_value(value):
        """Process a value to find config references."""
        if isinstance(value, str):
            # Check if this looks like a config path
            if value.endswith('.yaml') or value.endswith('.yml'):
                # Resolve relative to base_path
                config_path = os.path.join(base_path, value)
                if os.path.exists(config_path) and config_path not in visited:
                    visited.add(config_path)
                    referenced_configs.add(config_path)

                    # Recursively load and process this config
                    try:
                        with open(config_path, 'r') as f:
                            nested_config = yaml.safe_load(f)
                        nested_configs = find_referenced_configs(
                            nested_config,
                            base_path=base_path,
                            visited=visited
                        )
                        referenced_configs.update(nested_configs)
                    except Exception as e:
                        print(f"Warning: Could not load referenced config {config_path}: {e}")
        elif isinstance(value, dict):
            for v in value.values():
                process_value(v)
        elif isinstance(value, list):
            for item in value:
                process_value(item)

    process_value(config_data)
    return referenced_configs


def save_all_configs(main_config_path, config_data, output_dir):
    """
    Save the main config and all referenced configs to the output directory.

    Args:
        main_config_path: Path to the main configuration file
        config_data: The loaded configuration data
        output_dir: Directory to save configs to
    """
    configs_dir = os.path.join(output_dir, 'configs')
    os.makedirs(configs_dir, exist_ok=True)

    print("Saving configuration files to output directory...")

    # Save the main config
    main_config_name = os.path.basename(main_config_path)
    main_config_dest = os.path.join(configs_dir, main_config_name)
    shutil.copy2(main_config_path, main_config_dest)
    print(f"  Saved main config: {main_config_name}")

    # Find and save all referenced configs
    # Use current working directory as base path since config references are relative to project root
    base_path = os.getcwd()

    referenced_configs = find_referenced_configs(config_data, base_path=base_path)

    for config_path in referenced_configs:
        # Preserve directory structure relative to project root
        rel_path = os.path.relpath(config_path, '.')
        dest_path = os.path.join(configs_dir, rel_path)

        # Create necessary subdirectories
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Copy the config file
        shutil.copy2(config_path, dest_path)
        print(f"  Saved referenced config: {rel_path}")

    print(f"Total configs saved: {1 + len(referenced_configs)}")
    return configs_dir


def setup_environment():
    """Setup environment variables like the bash script."""
    # Set environment variables (like in pipeline.sh)
    scratch_dir = os.environ.get('SCRATCH')
    os.environ['HF_HOME'] = os.path.join(scratch_dir, 'huggingface')
    os.environ['TRANSFORMERS_VERBOSITY'] = 'info'


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Formal Proof Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Required arguments
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='Path to YAML configuration file (default: configs/default.yaml)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    print("Formal Proof Pipeline Runner")
    print("Replaces pipeline.sh with Python implementation\n")
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    setup_environment()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Expand environment variables in output directory
    if 'output' in config and 'dir' in config['output']:
        config['output']['dir'] = os.path.expandvars(config['output']['dir'])

    # Create and run pipeline
    print("\nInitializing pipeline...")
    pipeline = Pipeline(config)

    # Save all configs to the output directory
    save_all_configs(args.config, config, pipeline.get_output_dir())

    print("Starting pipeline execution...")
    success = pipeline.run()

    if success:
        print(f"Pipeline completed successfully!")
        print(f"Results available in: {pipeline.get_output_dir()}")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)