"""
Component for viewing and displaying run configurations.

Discovers all config files in a run directory and displays them in a user-friendly way.
Also detects dataset information from the problems in the run.
"""
import streamlit as st
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any, Optional
import json
import yaml
from collections import Counter


def get_config_files(run_dir: str) -> List[Path]:
    """
    Find all config files in a run directory (recursively).

    Handles both:
    - Legacy structure: run_dir/configs/
    - New dump structure: run_dir/../configs/ (when run_dir is dump/minified)

    Args:
        run_dir: Path to the run directory

    Returns:
        List of Path objects for all config files found
    """
    run_path = Path(run_dir).expanduser().resolve()

    # Check if we're in a minified directory (new dump structure)
    # If so, configs are in the parent (dump) directory
    if run_path.name == "minified":
        config_dir = run_path.parent / "configs"
    else:
        config_dir = run_path / "configs"

    if not config_dir.exists():
        return []

    # Find all files (both .yaml/.yml and .json)
    config_files = []
    config_files.extend(sorted(config_dir.glob("**/*.yaml")))
    config_files.extend(sorted(config_dir.glob("**/*.yml")))
    config_files.extend(sorted(config_dir.glob("**/*.json")))

    return sorted(set(config_files))  # Remove duplicates and sort


def load_config_file(file_path: Path) -> Tuple[str, Union[Dict, str]]:
    """
    Load a config file and parse it based on file extension.

    Args:
        file_path: Path to the config file

    Returns:
        Tuple of (file_content_str, parsed_data)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Parse based on file extension
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                parsed = yaml.safe_load(content)
                return content, parsed
            except yaml.YAMLError as e:
                return content, f"Error parsing YAML: {str(e)}"

        elif file_path.suffix.lower() == '.json':
            try:
                parsed = json.loads(content)
                return content, parsed
            except json.JSONDecodeError as e:
                return content, f"Error parsing JSON: {str(e)}"

        else:
            return content, "Unknown file format"

    except Exception as e:
        return "", f"Error reading file: {str(e)}"


def detect_dataset_info(run_dir: str) -> Optional[Dict[str, Any]]:
    """
    Detect dataset information from the problems in a run.

    Extracts the dataset name and size from the breakdown.json file.

    Args:
        run_dir: Path to the run directory

    Returns:
        Dictionary with keys: dataset, num_problems, sample_problem_ids
        Or None if dataset info cannot be determined
    """
    try:
        run_path = Path(run_dir).expanduser().resolve()
        breakdown_file = run_path / "round0" / "breakdown" / "breakdown.json"

        if not breakdown_file.exists():
            return None

        with open(breakdown_file, 'r') as f:
            breakdowns = json.load(f)

        if not isinstance(breakdowns, list) or len(breakdowns) == 0:
            return None

        # Extract dataset from origin_problem_id patterns
        # Examples: imo_1969_p2, putnam_1965_a6, aimo_2024_i5
        problem_ids = [b.get('origin_problem_id', '') for b in breakdowns]
        dataset_prefixes = Counter()

        for problem_id in problem_ids:
            if problem_id:
                # Extract prefix (e.g., "imo", "putnam", "aimo" from "imo_1969_p2")
                parts = problem_id.split('_')
                if len(parts) >= 2:
                    prefix = parts[0]
                    dataset_prefixes[prefix] += 1

        if not dataset_prefixes:
            return None

        # Get the most common dataset prefix
        most_common_dataset = dataset_prefixes.most_common(1)[0][0]
        dataset_names = {
            'imo': 'International Mathematical Olympiad (IMO)',
            'putnam': 'Putnam Mathematical Competition',
            'aimo': 'American Invitational Mathematics Olympiad (AIMO)',
        }

        dataset_name = dataset_names.get(most_common_dataset, most_common_dataset.upper())
        num_problems = len(breakdowns)
        sample_ids = problem_ids[:5]  # Get first 5 as samples

        return {
            'dataset': dataset_name,
            'prefix': most_common_dataset,
            'num_problems': num_problems,
            'sample_problem_ids': sample_ids,
            'distribution': dict(dataset_prefixes),
        }

    except Exception as e:
        return None


def render_config_viewer(run_dir: str) -> None:
    """
    Render a config viewer interface showing all configs in a run.

    Args:
        run_dir: Path to the run directory
    """
    st.header("⚙️ Run Configuration")

    # Detect dataset information
    dataset_info = detect_dataset_info(run_dir)

    if dataset_info:
        st.markdown("---")
        st.subheader("📊 Dataset Information")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Primary Dataset", dataset_info['dataset'])

        with col2:
            st.metric("Total Problems", dataset_info['num_problems'])

        # Show distribution
        dist = dataset_info['distribution']
        if len(dist) > 1:
            st.markdown("**Dataset Distribution:**")
            dist_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            dist_str = ", ".join([f"**{k}**: {v}" for k, v in dist_items])
            st.markdown(dist_str)
        else:
            st.success("Homogeneous dataset - all problems from same source")

        if dataset_info['sample_problem_ids']:
            with st.expander("👀 View Sample Problem IDs"):
                st.code("\n".join(dataset_info['sample_problem_ids']), language="text")

    st.markdown("---")

    # Get config files
    config_files = get_config_files(run_dir)

    if not config_files:
        st.warning(f"No config files found in {run_dir}/configs")
        return

    # Display info
    st.info(f"Found {len(config_files)} config file(s)")

    # Create tabs for each config file
    if len(config_files) == 1:
        # Single file - just display it
        file_path = config_files[0]
        relative_path = file_path.relative_to(Path(run_dir).expanduser().resolve())

        st.subheader(f"📄 {file_path.name}")
        st.caption(f"Path: {relative_path}")

        content, _ = load_config_file(file_path)
        st.code(content, language="yaml" if file_path.suffix.lower() in ['.yaml', '.yml'] else "json")

    else:
        # Multiple files - create tabs
        tab_names = [f.name for f in config_files]
        tabs = st.tabs(tab_names)

        for tab, file_path in zip(tabs, config_files):
            with tab:
                relative_path = file_path.relative_to(Path(run_dir).expanduser().resolve())
                st.caption(f"Path: {relative_path}")

                content, _ = load_config_file(file_path)
                st.code(content, language="yaml" if file_path.suffix.lower() in ['.yaml', '.yml'] else "json")
