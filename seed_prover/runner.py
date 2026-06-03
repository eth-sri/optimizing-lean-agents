#!/usr/bin/env python3
"""
Pipeline Runner - CLI entry point that replaces pipeline.sh
Handles configuration loading, argument parsing, and pipeline execution.
"""
import argparse
import sys
import os
import yaml
import shutil
import gc
from pathlib import Path
from loguru import logger

# Add current directory and project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent  # Go up one level to formal_proof_agent/
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    level="INFO",
    colorize=True,
)

from core.pipeline import Pipeline
from run_reuse_utils import setup_base_run_resume


def load_config(config_path):
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to YAML config file

    Returns:
        dict: Parsed configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Error: Configuration file not found: {config_path}")
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
                        logger.warning(f"Could not load referenced config {config_path}: {e}")
        elif isinstance(value, dict):
            for v in value.values():
                process_value(v)
        elif isinstance(value, list):
            for item in value:
                process_value(item)

    process_value(config_data)
    return referenced_configs


def expand_config_references(config_data, base_path='.', visited=None):
    """
    Recursively expand all YAML config file references in a configuration.

    This function replaces config file path strings with the actual loaded config content,
    creating a fully expanded configuration with no external references.

    Args:
        config_data: Configuration data (dict, list, or other)
        base_path: Base path for resolving relative config paths
        visited: Set of already visited config paths to avoid circular references

    Returns:
        Expanded configuration with all references inlined
    """
    if visited is None:
        visited = set()

    def expand_value(value):
        """Recursively expand a value."""
        if isinstance(value, str):
            # Check if this looks like a config path
            if (value.endswith('.yaml') or value.endswith('.yml')) and not value.startswith('$'):
                # Resolve relative to base_path
                config_path = os.path.join(base_path, value)
                if os.path.exists(config_path):
                    if config_path in visited:
                        logger.warning(f"Circular reference detected: {config_path}")
                        return value  # Return reference as-is to avoid infinite loop

                    visited.add(config_path)

                    # Load and recursively expand the referenced config
                    try:
                        with open(config_path, 'r') as f:
                            nested_config = yaml.safe_load(f)

                        # Recursively expand nested references
                        expanded_nested = expand_config_references(
                            nested_config,
                            base_path=base_path,
                            visited=visited
                        )
                        return expanded_nested
                    except Exception as e:
                        logger.warning(f"Could not load referenced config {config_path}: {e}")
                        return value  # Return reference as-is on error
            return value
        elif isinstance(value, dict):
            return {k: expand_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand_value(item) for item in value]
        else:
            return value

    return expand_value(config_data)


def merge_cli_overrides(config, args):
    """
    Merge command-line argument overrides into config.

    Args:
        config (dict): Base configuration
        args (Namespace): Parsed command-line arguments

    Returns:
        dict: Updated configuration with CLI overrides
    """
    # Override global settings if provided
    if args.output_dir:
        if 'output' not in config:
            config['output'] = {}
        config['output']['dir'] = args.output_dir

    if args.job_name:
        if 'output' not in config:
            config['output'] = {}
        config['output']['job_name'] = args.job_name

    if args.no_timestamp_dirs:
        if 'output' not in config:
            config['output'] = {}
        config['output']['create_timestamp_dirs'] = False
    
    if args.max_rounds is not None:
        if 'pipeline' not in config:
            config['pipeline'] = {}
        config['pipeline']['max_correction_rounds'] = args.max_rounds
    
    # Override component-specific settings
    if args.input_path:
        if 'components' in config and 'data_loader' in config['components']:
            config['components']['data_loader']['config']['input_path'] = args.input_path
    
    if args.model_path:
        if 'components' in config and 'inference' in config['components']:
            config['components']['inference']['config']['model_path'] = args.model_path
    
    if args.gpu is not None:
        if 'components' in config and 'inference' in config['components']:
            config['components']['inference']['config']['gpu'] = args.gpu
    
    if args.cpu is not None:
        if 'components' in config and 'compilation' in config['components']:
            config['components']['compilation']['config']['cpu'] = args.cpu
    
    return config


def setup_environment():
    """Setup environment variables like the bash script."""
    # Set environment variables (like in pipeline.sh)
    scratch_dir = os.environ.get('SCRATCH', '/tmp')
    os.environ['HF_HOME'] = os.path.join(scratch_dir, 'huggingface')
    os.environ['TRANSFORMERS_VERBOSITY'] = 'info'


def print_config_summary(config):
    """Print a summary of the pipeline configuration."""
    print("\n" + "=" * 60)
    print("PIPELINE CONFIGURATION SUMMARY")
    print("=" * 60)
    
    # Output settings
    output_config = config.get('output', {})
    print(f"Job Name: {output_config.get('job_name', 'default')}")
    print(f"Output Dir: {output_config.get('dir', 'results')}")
    
    # Pipeline settings
    pipeline_config = config.get('pipeline', {})
    max_rounds = pipeline_config.get('max_correction_rounds', 0)
    print(f"Max Correction Rounds: {max_rounds}")
    
    # Component settings
    components = config.get('components', {})
    print(f"Components: {', '.join(components.keys())}")
    
    # Key component settings
    if 'data_loader' in components:
        input_path = components['data_loader'].get('config', {}).get('input_path', 'N/A')
        print(f"Input Data: {input_path}")
    
    if 'inference' in components:
        inf_config = components['inference'].get('config', {})
        model_path = inf_config.get('model_path', 'N/A')
        gpu = inf_config.get('gpu', 'N/A')
        print(f"Model: {model_path}")
        print(f"GPUs: {gpu}")
    
    if 'compilation' in components:
        comp_config = components['compilation'].get('config', {})
        cpu = comp_config.get('cpu', 'N/A')
        print(f"CPU Cores: {cpu}")
    
    print("=" * 60)


def save_all_configs(main_config_path, config_data, output_dir):
    """
    Save both the original configuration and the effective configuration to the output directory.

    Creates two configs:
    1. original_config.yaml - Copy of the main config file used to run the pipeline (unchanged)
    2. effective_config.yaml - Final config after CLI overrides and reference expansion

    This prevents loss of the original pipeline configuration.

    Args:
        main_config_path: Path to the main configuration file (for reference)
        config_data: The loaded and merged configuration data (may contain config file references)
        output_dir: Directory to save configs to
    """
    configs_dir = os.path.join(output_dir, 'configs')
    os.makedirs(configs_dir, exist_ok=True)

    print("Saving configurations to output directory...")

    # Save a copy of the original config file (unchanged, for preservation)
    main_config_name = os.path.basename(main_config_path)
    original_config_path = os.path.join(configs_dir, 'original_config.yaml')
    try:
        shutil.copy2(main_config_path, original_config_path)
        print(f"  Saved original config: original_config.yaml")
    except Exception as e:
        logger.warning(f"Could not copy original config file: {e}")

    # Expand all config file references to create a fully self-contained config
    base_path = os.getcwd()
    expanded_config = expand_config_references(config_data, base_path=base_path)

    # Save the effective config (after all CLI overrides and with all references expanded)
    effective_config_path = os.path.join(configs_dir, 'effective_config.yaml')
    with open(effective_config_path, 'w') as f:
        yaml.dump(expanded_config, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved effective config (with expanded references): effective_config.yaml")

    # Save metadata about the configs
    with open(os.path.join(configs_dir, 'config_info.txt'), 'w') as f:
        f.write(f"Original config file: {main_config_path}\n")
        f.write(f"Original config name: {main_config_name}\n")
        f.write(f"\nFiles:\n")
        f.write(f"  original_config.yaml   - Exact copy of the config used to run the pipeline\n")
        f.write(f"  effective_config.yaml  - Final config after CLI overrides and reference expansion\n")

    print(f"Configurations saved to: {configs_dir}")
    return configs_dir


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Formal Proof Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python prover/runner.py --config configs/default.yaml
    
    # Override specific settings
    python prover/runner.py --config configs/default.yaml --gpu 8 --max-rounds 5
    
    # Use different input data
    python prover/runner.py --config configs/default.yaml --input-path dataset/test.jsonl
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='Path to YAML configuration file (default: configs/default.yaml)'
    )
    
    # Global overrides
    parser.add_argument(
        '--output-dir', 
        type=str,
        help='Override output directory'
    )
    
    parser.add_argument(
        '--job-name', 
        type=str,
        help='Override job name'
    )
    
    parser.add_argument(
        '--max-rounds', 
        type=int,
        help='Override maximum correction rounds'
    )
    
    # Component-specific overrides
    parser.add_argument(
        '--input-path', 
        type=str,
        help='Override input data path'
    )
    
    parser.add_argument(
        '--model-path', 
        type=str,
        help='Override model path'
    )
    
    parser.add_argument(
        '--gpu', 
        type=int,
        help='Override number of GPUs'
    )
    
    parser.add_argument(
        '--cpu', 
        type=int,
        help='Override number of CPU cores'
    )
    
    # Utility flags
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration and exit without running pipeline'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--no-timestamp-dirs',
        action='store_true',
        help='Disable automatic timestamp directory creation (use output-dir exactly as specified)'
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
    
    # Apply CLI overrides
    config = merge_cli_overrides(config, args)

    # Expand environment variables in output directory
    if 'output' in config and 'dir' in config['output']:
        config['output']['dir'] = os.path.expandvars(config['output']['dir'])

    # Print configuration summary
    if args.verbose or args.dry_run:
        print_config_summary(config)
    
    # Dry run - just show config and exit
    if args.dry_run:
        print("\nDry run completed. Use --verbose to see full configuration.")
        return 0
    
    try:
        import gc

        # Create output directory once (all rounds will use the same dir)
        print("\nInitializing pipeline...")
        first_pipeline = Pipeline(config)
        output_dir = first_pipeline.get_output_dir()

        # Save all configs to the output directory
        save_all_configs(args.config, config, output_dir)

        # Clean up the first pipeline (we only needed it to create the output dir)
        del first_pipeline
        gc.collect()

        print("Starting pipeline execution...")

        # Run each round with a fresh pipeline instance for maximum memory reclamation
        max_refine_rounds = config.get('pipeline', {}).get('max_refine_rounds', 1)
        run_reuse_config = config.get('pipeline', {}).get('run_reuse', {})

        # Only use start_from_round if run_reuse is enabled
        if run_reuse_config.get('enabled', False):
            start_from_round = run_reuse_config.get('start_from_round', 0)
        else:
            start_from_round = 0

        # Check if we need to access base run (for round copying or iteration resume)
        start_from_iteration = run_reuse_config.get('start_from_iteration', 0)
        needs_base_run = run_reuse_config.get('enabled', False) and (start_from_round >= 1 or start_from_iteration > 0)

        # Resolve base_run_path and copy previous rounds if needed
        if needs_base_run:
            start_from_round, start_from_iteration, _ = setup_base_run_resume(
                config=config,
                output_dir=output_dir,
                run_reuse_config=run_reuse_config,
                start_from_round=start_from_round,
                start_from_iteration=start_from_iteration
            )

        for round_num in range(start_from_round, max_refine_rounds):
            logger.info(f"\n{'='*60}")
            logger.info(f"ROUND {round_num}/{max_refine_rounds - 1}")
            logger.info(f"{'='*60}")

            # Create fresh pipeline for this round
            logger.info(f"Creating fresh pipeline instance for round {round_num}...")
            pipeline = Pipeline(config, base_output_dir=output_dir)

            # Run this round
            success = pipeline.run_single_round(round_num)

            if not success:
                logger.error(f"Round {round_num} failed!")
                # Cleanup before exit
                del pipeline
                gc.collect()
                return 1

            # Cleanup pipeline to free memory before next round
            logger.info(f"Cleaning up pipeline for round {round_num}...")
            del pipeline
            gc.collect()

            # Extra cleanup for GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("GPU memory cleared")
            except ImportError:
                pass

        # Run proof_builder once after all rounds complete
        logger.info(f"\n{'='*60}")
        logger.info("Running proof builder on all rounds...")
        logger.info(f"{'='*60}")

        final_pipeline = Pipeline(config, base_output_dir=output_dir)
        if 'proof_builder' in final_pipeline.components:
            final_pipeline._run_component('proof_builder', [], round_num=None)

        # Final cleanup
        del final_pipeline
        gc.collect()

        print(f"\n🎉 Pipeline completed successfully!")
        print(f"Results available in: {output_dir}")
        return 0
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)