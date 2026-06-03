"""
Config Viewer Tab Component.

Displays the YAML configuration used for the RL training run.
"""
import streamlit as st
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def find_config_file(run_dir: Path) -> Optional[Path]:
    """
    Find the config YAML file for a run.

    Searches in various locations where configs might be stored.
    """
    # Common config file names
    config_names = [
        "config.yaml",
        "config.yml",
        "iterative_rl_training.yaml",
        "iterative_rl_training_production.yaml",
        "training_config.yaml",
    ]

    # Search locations
    search_paths = [
        run_dir,
        run_dir / "configs",
        run_dir / "config",
        run_dir.parent,  # Sometimes config is one level up
    ]

    for search_path in search_paths:
        if not search_path.exists():
            continue
        for config_name in config_names:
            config_file = search_path / config_name
            if config_file.exists():
                return config_file

    # Also search for any .yaml file in run_dir
    if run_dir.exists():
        yaml_files = list(run_dir.glob("*.yaml")) + list(run_dir.glob("*.yml"))
        if yaml_files:
            return yaml_files[0]

    return None


def load_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse YAML config file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None


def render_config_tab(run_dir: Path, run_metadata: Dict[str, Any]):
    """
    Render the Config tab showing the training configuration.

    Args:
        run_dir: Path to the RL training run directory
        run_metadata: Metadata about the run (from iteration_results.json)
    """
    st.subheader("Training Configuration")

    # Try to find and load the config file
    config_file = find_config_file(run_dir)

    if config_file is None:
        st.warning("No configuration file found for this run.")

        # Show what we can extract from iteration_results
        if run_metadata.get('iteration_results'):
            st.info("Showing configuration from iteration_results.json:")
            iter_results = run_metadata['iteration_results']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Training Settings**")
                st.write(f"- Num rounds: {iter_results.get('num_rounds', 'N/A')}")
                st.write(f"- Num problems: {iter_results.get('num_problems', 'N/A')}")
                st.write(f"- Num seeds: {iter_results.get('num_seeds', 'N/A')}")
                st.write(f"- Max steps: {iter_results.get('max_steps', 'N/A')}")

            with col2:
                st.markdown("**Hyperparameters**")
                budgets = iter_results.get('budgets', [])
                budget_strs = ["unlimited" if b is None else f"{b/1e6:.0f}M" for b in budgets]
                st.write(f"- Budgets: {', '.join(budget_strs)}")

                lambdas = iter_results.get('lambda_values', [])
                lambda_strs = ["0" if l == 0 else f"{l:.0e}" for l in lambdas]
                st.write(f"- Lambda values: {', '.join(lambda_strs)}")
        return

    st.success(f"Config file: `{config_file}`")

    # Load the config
    config = load_config(config_file)

    if config is None:
        return

    # Display in organized sections
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Data Paths")
        st.code(f"""minified_dir: {config.get('minified_dir', 'N/A')}
output_base_dir: {config.get('output_base_dir', 'N/A')}
baseline_rollouts_dir: {config.get('baseline_rollouts_dir', 'N/A')}""")

        st.markdown("### Training Settings")
        training_info = f"""num_rounds: {config.get('num_rounds', 'N/A')}
num_problems: {config.get('num_problems', 'all')}
num_seeds: {config.get('num_seeds', 'N/A')}
max_steps: {config.get('max_steps', 'N/A')}
seed: {config.get('seed', 'N/A')}"""
        st.code(training_info)

        st.markdown("### Budgets & Lambda")
        budgets = config.get('budgets', [])
        lambdas = config.get('lambda_values', [])
        budget_strs = ["unlimited" if b is None else f"{b:,}" for b in (budgets or [])]
        # Handle lambda values that might be strings (e.g., "1e-7") or floats
        lambda_strs = []
        for l in (lambdas or []):
            l_float = float(l) if isinstance(l, str) else l
            lambda_strs.append("0" if l_float == 0 else f"{l_float:.0e}")
        st.code(f"""budgets: {budget_strs}
lambda_values: {lambda_strs}""")

    with col2:
        st.markdown("### Active Features")
        features = config.get('active_features', [])
        if features:
            for f in features:
                st.write(f"- `{f}`")
        else:
            st.write("_No features configured_")

        st.markdown("### Action Space")
        action_space = config.get('action_space', 'N/A')
        if isinstance(action_space, list):
            for action in action_space:
                st.write(f"- `{action}`")
        else:
            st.write(f"`{action_space}`")

        st.markdown("### Model Training")
        model_info = f"""min_samples_per_action: {config.get('min_samples_per_action', 'N/A')}
balance_rounds: {config.get('balance_rounds', 'N/A')}
weight_by_trajectory_length: {config.get('weight_by_trajectory_length', 'N/A')}
separate_models_per_lambda: {config.get('separate_models_per_lambda', 'N/A')}
cumulate_training_data: {config.get('cumulate_training_data', 'N/A')}"""
        st.code(model_info)

    # Warm start settings
    warm_start = config.get('warm_start', {})
    random_warm = config.get('random_rollout_warm_start', {})

    if warm_start or random_warm:
        st.markdown("### Warm Start Settings")
        col3, col4 = st.columns(2)

        with col3:
            if warm_start:
                st.markdown("**Fixed warm start:**")
                for k, v in warm_start.items():
                    st.write(f"- {k}: {v}")

        with col4:
            if random_warm:
                st.markdown("**Random rollout warm start:**")
                st.write(f"- Enabled: {random_warm.get('enabled', False)}")
                if random_warm.get('enabled'):
                    st.write(f"- prob_continue_8b: {random_warm.get('prob_continue_8b')}")
                    st.write(f"- prob_upgrade_32b: {random_warm.get('prob_upgrade_32b')}")
                    st.write(f"- prob_terminate_8b: {random_warm.get('prob_terminate_8b')}")

    # Full YAML view (collapsible)
    with st.expander("View Full YAML Config"):
        st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")
