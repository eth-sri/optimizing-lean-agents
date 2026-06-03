"""
Baseline Rollouts Analysis viewer.

Analyzes tree rollout baselines and compares them to full_proof baselines.
Shows scatter plot of average cost vs success rate with filtering controls.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from scipy import interpolate


def calculate_auc(x_values: List[float], y_values: List[float], x_min: float, x_max: float) -> float:
    """Calculate AUC using trapezoidal rule over specified x range.

    Args:
        x_values: X coordinates (must be sorted)
        y_values: Y coordinates
        x_min: Start of x range
        x_max: End of x range

    Returns:
        Area under curve (normalized by x range width)
    """
    if len(x_values) < 2:
        return 0.0

    # Filter points within range
    points = [(x, y) for x, y in zip(x_values, y_values) if x_min <= x <= x_max]
    if len(points) < 2:
        return 0.0

    points = sorted(points, key=lambda p: p[0])
    xs, ys = zip(*points)

    # Trapezoidal integration
    auc = 0.0
    for i in range(len(xs) - 1):
        auc += (xs[i+1] - xs[i]) * (ys[i] + ys[i+1]) / 2

    return auc


def compute_upper_envelope(curves: List[Dict[str, List[float]]], x_min: float, x_max: float, num_samples: int = 200) -> Tuple[List[float], List[float]]:
    """Compute upper convex hull of multiple curves using Andrew's monotone chain.

    This computes the upper convex hull of all points, which represents
    the best achievable success rate at each cost level (including
    interpolation between configurations).

    Args:
        curves: List of dicts with 'x' and 'y' keys
        x_min: Start of x range
        x_max: End of x range
        num_samples: Number of sample points (unused, kept for compatibility)

    Returns:
        Tuple of (x_values, y_values) for the upper convex hull
    """
    if not curves:
        return [], []

    # Collect all points from all curves
    all_points = []
    for curve in curves:
        x, y = curve['x'], curve['y']
        for xi, yi in zip(x, y):
            if x_min <= xi <= x_max:
                all_points.append((xi, yi))

    if not all_points:
        return [], []

    # Remove duplicates and sort by x, then by y descending (for upper hull)
    all_points = sorted(set(all_points), key=lambda p: (p[0], -p[1]))

    # Andrew's monotone chain - upper hull
    # Cross product to determine turn direction
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build upper hull (we want points that make right turns or go straight)
    upper = []
    for p in all_points:
        # For upper hull: remove points that make left turns (cross > 0)
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) > 0:
            upper.pop()
        upper.append(p)

    if upper:
        xs, ys = zip(*upper)
        xs, ys = list(xs), list(ys)

        # Prepend linear segment from origin to first point (not part of convex hull)
        if xs[0] > 0.0 and x_min <= 0.0:
            xs.insert(0, 0.0)
            ys.insert(0, 0.0)

        # Extend envelope to x_max if hull ends earlier (horizontal extension)
        # More cost shouldn't reduce max achievable success rate
        if xs[-1] < x_max:
            xs.append(x_max)
            ys.append(ys[-1])

        return xs, ys

    return [], []


def interpolate_baseline_auc(baseline_x: List[float], baseline_y: List[float],
                              x_min: float, x_max: float) -> Optional[float]:
    """Interpolate baseline curve and calculate AUC over specified range.

    Args:
        baseline_x: X coordinates of baseline (sorted)
        baseline_y: Y coordinates of baseline
        x_min: Start of x range
        x_max: End of x range

    Returns:
        AUC if baseline covers the range, None otherwise
    """
    if len(baseline_x) < 2:
        return None

    # Check if baseline covers the range
    bl_min, bl_max = min(baseline_x), max(baseline_x)

    # Clamp to baseline range
    effective_min = max(x_min, bl_min)
    effective_max = min(x_max, bl_max)

    if effective_min >= effective_max:
        return None

    # Create interpolation function
    f = interpolate.interp1d(baseline_x, baseline_y, kind='linear', fill_value='extrapolate')

    # Sample at fine resolution for accurate AUC
    num_samples = 100
    xs = np.linspace(effective_min, effective_max, num_samples)
    ys = f(xs)

    # Trapezoidal integration
    auc = np.trapezoid(ys, xs)

    return auc


# Get project root (lean-breakdown directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Default paths - relative to project root
DEFAULT_FULL_PROOF_DIR = PROJECT_ROOT / "outputs/rl/baselines/full_proof/putnam"
DEFAULT_AGENTIC_DIR = PROJECT_ROOT / "outputs/rl/baselines/agentic"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset/combined_with_complexity.json"


@st.cache_data
def load_problem_difficulties(dataset_path: str = None) -> Dict[str, str]:
    """Load problem difficulties from dataset.json.

    Returns:
        Dict mapping problem_id -> difficulty ('easy', 'medium', 'hard')
    """
    if dataset_path is None:
        dataset_path = str(DEFAULT_DATASET_PATH)

    path = Path(dataset_path)
    if not path.exists():
        return {}

    try:
        with open(path, 'r') as f:
            dataset = json.load(f)

        difficulties = {}
        for record in dataset:
            problem_id = record.get('problem_id') or record.get('name')
            difficulty = record.get('difficulty')
            if problem_id and difficulty:
                difficulties[problem_id] = difficulty

        return difficulties
    except Exception as e:
        return {}


@st.cache_data
def load_full_proof_summaries(full_proof_dir: str) -> List[Dict[str, Any]]:
    """Load summaries from full_proof baselines.

    Supports two formats:
    1. combined_summary.json - single file with all configs
    2. Directory structure with FULL_PROOF_8B/max_N/summary.json
    """
    fp_path = Path(full_proof_dir)
    summaries = []

    if not fp_path.exists():
        return summaries

    # First, try to load from combined_summary.json
    combined_file = fp_path / "combined_summary.json"
    if combined_file.exists():
        try:
            with open(combined_file, 'r') as f:
                combined_data = json.load(f)

            for entry in combined_data:
                summary = entry.copy()
                baseline = summary.get('baseline', 'UNKNOWN')
                max_attempts = summary.get('max_attempts', 0)
                summary['config_name'] = f"{baseline}_max_{max_attempts}"
                summary['source'] = 'full_proof'
                summaries.append(summary)

            return summaries
        except Exception as e:
            st.warning(f"Failed to load {combined_file}: {e}")

    # Fallback: Look for baseline type directories (FULL_PROOF_8B, FULL_PROOF_32B)
    for baseline_dir in fp_path.iterdir():
        if not baseline_dir.is_dir() or baseline_dir.name.startswith('.'):
            continue

        baseline_name = baseline_dir.name  # e.g., FULL_PROOF_8B, FULL_PROOF_32B

        # Look for max_* subdirectories
        for max_dir in baseline_dir.iterdir():
            if not max_dir.is_dir() or not max_dir.name.startswith('max_'):
                continue

            summary_file = max_dir / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)

                    summary['baseline'] = baseline_name
                    summary['config_name'] = f"{baseline_name}_{max_dir.name}"
                    summary['source'] = 'full_proof'
                    summaries.append(summary)
                except Exception as e:
                    pass

    return summaries


@st.cache_data
def load_agentic_summaries(agentic_dir: str, subdir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load summaries from agentic baseline configurations.

    Reads from individual config directories with summary.json files.

    Args:
        agentic_dir: Base agentic directory
        subdir: Optional subdirectory (e.g., '8b_theorem', 'sft8b_theorem')
    """
    agentic_path = Path(agentic_dir)
    if subdir:
        agentic_path = agentic_path / subdir

    summaries = []

    if not agentic_path.exists():
        return summaries

    # Look for config directories with summary.json
    for config_dir in agentic_path.iterdir():
        if not config_dir.is_dir() or config_dir.name.startswith('.'):
            continue

        summary_file = config_dir / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)

                # Parse config params from summary or name
                config_params = summary.get('config_params', {})
                summary['config_name'] = config_dir.name
                summary['n1'] = config_params.get('n1', 0)
                summary['n2'] = config_params.get('n2', 0)
                summary['n3'] = config_params.get('n3', 0)
                summary['n4'] = config_params.get('n4', 0)
                summary['n5'] = config_params.get('n5', 0)
                summary['use_corrections'] = config_params.get('use_corrections_8b', False) or config_params.get('use_corrections_32b', False)
                summary['use_corrections_8b'] = config_params.get('use_corrections_8b', False)
                summary['use_corrections_32b'] = config_params.get('use_corrections_32b', False)
                summary['source'] = 'agentic'
                summary['subdir'] = subdir or ''

                summaries.append(summary)
            except Exception as e:
                st.warning(f"Failed to load {summary_file}: {e}")

    return summaries


def get_available_agentic_subdirs(agentic_dir: str) -> List[str]:
    """Get list of available agentic subdirectories that have data."""
    agentic_path = Path(agentic_dir)
    subdirs = ['']  # Root directory

    if not agentic_path.exists():
        return subdirs

    for subdir in agentic_path.iterdir():
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue
        # Check if this subdir has config subdirectories with summary.json
        if any((d / "summary.json").exists() for d in subdir.iterdir() if d.is_dir()):
            subdirs.append(subdir.name)

    return sorted(subdirs)


@st.cache_data
def load_agentic_per_problem(agentic_dir: str, subdir: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load per-problem metrics from agentic baseline configurations.

    Args:
        agentic_dir: Base agentic directory
        subdir: Optional subdirectory (e.g., '8b_theorem', 'sft8b_theorem')

    Returns:
        Dict mapping config_name -> problem_id -> {success_rate, avg_cost, num_rollouts}
    """
    agentic_path = Path(agentic_dir)
    if subdir:
        agentic_path = agentic_path / subdir

    result = {}

    if not agentic_path.exists():
        return result

    for config_dir in agentic_path.iterdir():
        if not config_dir.is_dir() or config_dir.name.startswith('.'):
            continue

        rollouts_file = config_dir / "rollouts_summary.json"
        if not rollouts_file.exists():
            continue

        try:
            with open(rollouts_file, 'r') as f:
                rollouts = json.load(f)

            # Group by problem_id
            per_problem = {}
            for rollout in rollouts:
                pid = rollout.get('problem_id')
                if not pid:
                    continue

                if pid not in per_problem:
                    per_problem[pid] = {
                        'costs': [],
                        'successes': [],
                    }
                per_problem[pid]['costs'].append(rollout.get('final_cost', 0))
                per_problem[pid]['successes'].append(rollout.get('success', False))

            # Compute metrics per problem
            config_metrics = {}
            for pid, data in per_problem.items():
                num_rollouts = len(data['costs'])
                num_successful = sum(data['successes'])
                config_metrics[pid] = {
                    'avg_cost': np.mean(data['costs']) if data['costs'] else 0,
                    'success_rate': num_successful / num_rollouts if num_rollouts > 0 else 0,
                    'num_rollouts': num_rollouts,
                    'num_successful': num_successful,
                }

            config_name = f"{subdir}/{config_dir.name}" if subdir else config_dir.name
            result[config_name] = config_metrics
        except Exception as e:
            pass

    return result


@st.cache_data
def load_full_proof_per_problem(full_proof_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load per-problem metrics from full_proof baselines.

    Supports:
    1. per_problem_summary.json - pre-computed summary per problem
    2. rollouts_summary.json - raw rollouts to aggregate

    Returns:
        Dict mapping config_name (e.g., FULL_PROOF_8B_max_4) -> problem_id -> metrics
    """
    fp_path = Path(full_proof_dir)
    result = {}

    if not fp_path.exists():
        return result

    for baseline_dir in fp_path.iterdir():
        if not baseline_dir.is_dir() or baseline_dir.name.startswith('.'):
            continue

        baseline_name = baseline_dir.name

        for max_dir in baseline_dir.iterdir():
            if not max_dir.is_dir() or not max_dir.name.startswith('max_'):
                continue

            config_name = f"{baseline_name}_{max_dir.name}"

            # First try per_problem_summary.json (preferred, more efficient)
            per_problem_file = max_dir / "per_problem_summary.json"
            if per_problem_file.exists():
                try:
                    with open(per_problem_file, 'r') as f:
                        per_problem_data = json.load(f)

                    config_metrics = {}
                    for pid, metrics in per_problem_data.items():
                        config_metrics[pid] = {
                            'success_rate': metrics.get('success_rate', 0),
                            'num_rollouts': metrics.get('num_rollouts', 0),
                            'num_successful': metrics.get('num_successful', 0),
                            'avg_cost': 0,  # Not in per_problem_summary
                        }
                    result[config_name] = config_metrics
                    continue
                except Exception as e:
                    pass

            # Fallback: Load from rollouts_summary.json
            rollouts_file = max_dir / "rollouts_summary.json"
            if not rollouts_file.exists():
                continue

            try:
                with open(rollouts_file, 'r') as f:
                    rollouts = json.load(f)

                # Group by problem_id
                per_problem = {}
                for rollout in rollouts:
                    pid = rollout.get('problem_id')
                    if not pid:
                        continue

                    if pid not in per_problem:
                        per_problem[pid] = {
                            'costs': [],
                            'successes': [],
                        }
                    per_problem[pid]['costs'].append(rollout.get('final_cost', 0))
                    per_problem[pid]['successes'].append(rollout.get('success', False))

                # Compute metrics per problem
                config_metrics = {}
                for pid, data in per_problem.items():
                    num_rollouts = len(data['costs'])
                    num_successful = sum(data['successes'])
                    config_metrics[pid] = {
                        'avg_cost': np.mean(data['costs']) if data['costs'] else 0,
                        'success_rate': num_successful / num_rollouts if num_rollouts > 0 else 0,
                        'num_rollouts': num_rollouts,
                        'num_successful': num_successful,
                    }

                result[config_name] = config_metrics
            except Exception as e:
                pass

    return result


def render_baseline_rollouts_viewer():
    """Main entry point for baseline rollouts analysis."""
    col1, col2 = st.columns([6, 1])
    with col1:
        st.header("📊 Analyze Baseline Rollouts")
    with col2:
        if st.button("🔄 Refresh", key="refresh_baseline_cache", help="Clear cached data and reload"):
            st.cache_data.clear()
            st.rerun()

    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(["📈 Cost vs Success", "🔍 Problem-level", "💰 Action Costs"])

    with tab1:
        render_cost_success_analysis()

    with tab2:
        render_problem_level_analysis()

    with tab3:
        render_action_cost_analysis()


def render_cost_success_analysis():
    """Render the cost vs success rate analysis tab."""
    st.markdown("Compare agentic baselines (varying n1-n5 parameters) against full proof baselines.")

    # Path configuration in expander
    with st.expander("📁 Data Paths", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            full_proof_path = st.text_input(
                "Full Proof Directory",
                value=str(DEFAULT_FULL_PROOF_DIR),
                key="full_proof_path"
            )
        with col2:
            agentic_path = st.text_input(
                "Agentic Directory",
                value=str(DEFAULT_AGENTIC_DIR),
                key="agentic_path"
            )

    # Get available agentic subdirectories
    available_subdirs = get_available_agentic_subdirs(agentic_path)

    # Subdirectory selector for agentic
    if len(available_subdirs) > 1:
        selected_subdir = st.selectbox(
            "Agentic Baseline Set",
            options=available_subdirs,
            format_func=lambda x: "(root)" if x == '' else x,
            key="agentic_subdir_selector"
        )
    else:
        selected_subdir = available_subdirs[0] if available_subdirs else ''

    # Load data
    fp_summaries = load_full_proof_summaries(full_proof_path)
    comparison_summaries = load_agentic_summaries(agentic_path, selected_subdir if selected_subdir else None)

    if not comparison_summaries and not fp_summaries:
        st.error("No data found. Check directory paths.")
        return

    # Filtering controls in expander
    with st.expander("🔧 Filters", expanded=False):
        # Agentic baseline filters
        all_n1 = sorted(set(s.get('n1') for s in comparison_summaries if s.get('n1') is not None))
        all_n2 = sorted(set(s.get('n2') for s in comparison_summaries if s.get('n2') is not None))
        all_n3 = sorted(set(s.get('n3') for s in comparison_summaries if s.get('n3') is not None))
        all_n4 = sorted(set(s.get('n4') for s in comparison_summaries if s.get('n4') is not None))
        all_n5 = sorted(set(s.get('n5') for s in comparison_summaries if s.get('n5') is not None))

        # Create 3 columns for filter groups
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**n1 (full_proof_8b)**")
            selected_n1 = []
            cols = st.columns(min(len(all_n1), 4)) if all_n1 else []
            for i, n in enumerate(all_n1):
                with cols[i % len(cols)]:
                    if st.checkbox(str(n), value=True, key=f"n1_{n}"):
                        selected_n1.append(n)

            st.markdown("**n2 (full_proof_32b)**")
            selected_n2 = []
            cols = st.columns(min(len(all_n2), 4)) if all_n2 else []
            for i, n in enumerate(all_n2):
                with cols[i % len(cols)]:
                    if st.checkbox(str(n), value=True, key=f"n2_{n}"):
                        selected_n2.append(n)

        with col2:
            st.markdown("**n3 (breakdowns)**")
            selected_n3 = []
            cols = st.columns(min(len(all_n3), 4)) if all_n3 else []
            for i, n in enumerate(all_n3):
                with cols[i % len(cols)]:
                    if st.checkbox(str(n), value=True, key=f"n3_{n}"):
                        selected_n3.append(n)

            st.markdown("**n4 (attempt_8b)**")
            selected_n4 = []
            cols = st.columns(min(len(all_n4), 4)) if all_n4 else []
            for i, n in enumerate(all_n4):
                with cols[i % len(cols)]:
                    if st.checkbox(str(n), value=True, key=f"n4_{n}"):
                        selected_n4.append(n)

        with col3:
            st.markdown("**n5 (attempt_32b)**")
            selected_n5 = []
            cols = st.columns(min(len(all_n5), 4)) if all_n5 else []
            for i, n in enumerate(all_n5):
                with cols[i % len(cols)]:
                    if st.checkbox(str(n), value=True, key=f"n5_{n}"):
                        selected_n5.append(n)

            st.markdown("**Use Corrections**")
            show_corrections = st.checkbox("With corrections", value=True, key="show_corrections")
            show_no_corrections = st.checkbox("Without corrections", value=True, key="show_no_corrections")

        # Filter agentic summaries
        filtered_comparison = [
            s for s in comparison_summaries
            if s.get('n1') in selected_n1
            and s.get('n2') in selected_n2
            and s.get('n3') in selected_n3
            and s.get('n4') in selected_n4
            and s.get('n5') in selected_n5
            and ((s.get('use_corrections') and show_corrections) or (not s.get('use_corrections') and show_no_corrections))
        ]

    # Create scatter plot
    st.subheader("Cost vs Success Rate")

    flip_axes = st.checkbox(
        "Flip axes (success rate on x, cost on y)",
        value=False,
        key="flip_axes",
        help="Swap the X and Y axes. AUC is then computed as ∫ cost d(success rate).",
    )

    # Helper: swap (x, y) for plotly traces when flipping axes.
    def _xy(x_orig, y_orig):
        return (y_orig, x_orig) if flip_axes else (x_orig, y_orig)

    # Helper: project a curve dict {'x': cost, 'y': accuracy} into the
    # current axis convention (returns (axis_x, axis_y)).
    def _current_axes(curve):
        return (curve['y'], curve['x']) if flip_axes else (curve['x'], curve['y'])

    # Format the X Range column in AUC tables.
    def _fmt_xrange(v):
        return f"0 - {v*100:.1f}%" if flip_axes else f"0 - {v:.2f}M"

    # Interpolate baseline AUC in the current axis convention.
    def _baseline_auc_current(bl_data, x_max_val):
        x_arr, y_arr = _current_axes(bl_data)
        pairs = sorted(set(zip(x_arr, y_arr)))
        if len(pairs) < 2:
            return None
        sx, sy = zip(*pairs)
        return interpolate_baseline_auc(list(sx), list(sy), 0.0, x_max_val)

    # Compute upper-envelope of agentic curves projected to current axes.
    # We always compute the envelope in (cost, accuracy) frame (since it is
    # monotone non-decreasing there), then transpose for the flipped view.
    def _envelope_current(envelope_curves_orig, x_max_current):
        if not flip_axes:
            return compute_upper_envelope(envelope_curves_orig, 0.0, x_max_current)
        # x_max_current is an accuracy bound. Build envelope over full cost
        # range, then truncate points to accuracy <= bound.
        all_cost_max = 0.0
        for c in envelope_curves_orig:
            if c['x']:
                all_cost_max = max(all_cost_max, max(c['x']))
        env_cost, env_acc = compute_upper_envelope(envelope_curves_orig, 0.0, all_cost_max)
        pairs = [(c, a) for c, a in zip(env_cost, env_acc) if a <= x_max_current]
        if len(pairs) < 2:
            return [], []
        env_cost_t, env_acc_t = zip(*pairs)
        # Return in current-axis order: x=accuracy, y=cost
        return list(env_acc_t), list(env_cost_t)

    # Color-by selector for agentic baselines
    selected_color_values = None
    if filtered_comparison:
        color_by_options = ["use_corrections", "n1", "n2", "n3", "n4", "n5"]
        color_by_labels = {
            "use_corrections": "Use Corrections",
            "n1": "n1 (full_proof_8b)",
            "n2": "n2 (full_proof_32b)",
            "n3": "n3 (breakdowns)",
            "n4": "n4 (attempt_8b)",
            "n5": "n5 (attempt_32b)",
        }

        col1, col2 = st.columns([1, 3])
        with col1:
            color_by = st.selectbox(
                "Color points by",
                options=color_by_options,
                format_func=lambda x: color_by_labels.get(x, x),
                key="agentic_color_by"
            )

        # Get unique values for the selected color_by attribute
        df_temp = pd.DataFrame(filtered_comparison)
        if color_by in df_temp.columns:
            unique_values = sorted(df_temp[color_by].unique())

            with col2:
                st.markdown(f"**Show/hide {color_by_labels.get(color_by, color_by)} values:**")
                selected_color_values = []
                cols = st.columns(min(len(unique_values), 6))
                for i, val in enumerate(unique_values):
                    with cols[i % len(cols)]:
                        # Format label for display
                        if color_by == "use_corrections":
                            label = "corr" if val else "no corr"
                        else:
                            label = str(val)
                        if st.checkbox(label, value=True, key=f"color_val_{color_by}_{val}"):
                            selected_color_values.append(val)
        else:
            selected_color_values = None
    else:
        color_by = None

    fig = go.Figure()

    # Store data for AUC calculations
    auc_data = {}  # {label: {'x': [...], 'y': [...], 'auc': float}}
    baseline_curves = {}  # {'FULL_PROOF_8B': {'x': [...], 'y': [...]}, ...}

    # Add agentic baseline data
    if filtered_comparison:
        df_comp = pd.DataFrame(filtered_comparison)

        if 'avg_cost' in df_comp.columns and 'success_rate' in df_comp.columns:
            df_comp['avg_cost_millions'] = df_comp['avg_cost'] / 1e6

            # Color by selected attribute for agentic
            if 'avg_steps' not in df_comp.columns:
                df_comp['avg_steps'] = 0

            # Define color palette
            color_palette = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]

            if color_by and color_by in df_comp.columns:
                # Get unique values for color_by attribute
                unique_values = sorted(df_comp[color_by].unique())

                for i, val in enumerate(unique_values):
                    # Skip if this value is not selected in checkboxes
                    if selected_color_values is not None and val not in selected_color_values:
                        continue

                    subset_df = df_comp[df_comp[color_by] == val].copy()
                    if subset_df.empty:
                        continue

                    # Sort by cost for line connection
                    subset_df = subset_df.sort_values('avg_cost_millions')

                    color = color_palette[i % len(color_palette)]

                    # Format legend label
                    if color_by == "use_corrections":
                        legend_label = "with corrections" if val else "no corrections"
                    else:
                        legend_label = f"{color_by}={val}"

                    # Prepare data starting from (0, 0)
                    x_data = [0.0] + subset_df['avg_cost_millions'].tolist()
                    y_data = [0.0] + subset_df['success_rate'].tolist()

                    # Store for AUC calculation
                    auc_data[legend_label] = {'x': x_data, 'y': y_data, 'color': color}

                    # Prepare customdata (add None row for origin point)
                    customdata_list = [[None]*8] + subset_df[['n1', 'n2', 'n3', 'n4', 'n5', 'use_corrections', 'unique_problems_solved', 'avg_steps']].values.tolist()

                    plot_x, plot_y = _xy(x_data, y_data)
                    fig.add_trace(go.Scatter(
                        x=plot_x,
                        y=plot_y,
                        mode='markers+lines',
                        name=legend_label,
                        marker=dict(size=10, color=color, symbol='circle'),
                        line=dict(color=color, width=2),
                        hovertemplate=(
                            '<b>Agentic</b><br>'
                            'n1=%{customdata[0]}, n2=%{customdata[1]}, n3=%{customdata[2]}<br>'
                            'n4=%{customdata[3]}, n5=%{customdata[4]}, corr=%{customdata[5]}<br>'
                            f'Success rate: %{{{"x" if flip_axes else "y"}:.3f}}<br>'
                            f'Avg cost: %{{{"y" if flip_axes else "x"}:.2f}}M<br>'
                            'Unique solved: %{customdata[6]}<br>'
                            'Avg steps: %{customdata[7]:.1f}<extra></extra>'
                        ),
                        customdata=customdata_list,
                    ))
            else:
                # No color_by, plot all as one group
                df_comp_sorted = df_comp.sort_values('avg_cost_millions')
                x_data = [0.0] + df_comp_sorted['avg_cost_millions'].tolist()
                y_data = [0.0] + df_comp_sorted['success_rate'].tolist()

                auc_data['Agentic'] = {'x': x_data, 'y': y_data, 'color': '#1f77b4'}

                customdata_list = [['']] + (df_comp_sorted[['config_name']].values.tolist() if 'config_name' in df_comp_sorted.columns else [['']] * len(df_comp_sorted))

                plot_x, plot_y = _xy(x_data, y_data)
                fig.add_trace(go.Scatter(
                    x=plot_x,
                    y=plot_y,
                    mode='markers+lines',
                    name='Agentic',
                    marker=dict(size=10, color='#1f77b4', symbol='circle'),
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate=(
                        '<b>Agentic</b><br>'
                        'Config: %{customdata[0]}<br>'
                        f'Success rate: %{{{"x" if flip_axes else "y"}:.3f}}<br>'
                        f'Avg cost: %{{{"y" if flip_axes else "x"}:.2f}}M<extra></extra>'
                    ),
                    customdata=customdata_list,
                ))

    # Add full proof baselines (always shown)
    if fp_summaries:
        df_fp = pd.DataFrame(fp_summaries)

        if 'avg_cost' in df_fp.columns and 'success_rate' in df_fp.columns:
            df_fp['avg_cost_millions'] = df_fp['avg_cost'] / 1e6

            baseline_colors = {'FULL_PROOF_8B': 'orange', 'FULL_PROOF_32B': 'purple'}

            baselines_to_plot = df_fp['baseline'].unique().tolist() if 'baseline' in df_fp.columns else []

            for baseline in baselines_to_plot:
                bl_df = df_fp[df_fp['baseline'] == baseline].copy()
                bl_df = bl_df.sort_values('avg_cost')
                color = baseline_colors.get(baseline, 'gray')

                # Prepare data starting from (0, 0)
                x_data = [0.0] + bl_df['avg_cost_millions'].tolist()
                y_data = [0.0] + bl_df['success_rate'].tolist()

                # Store baseline curve for AUC comparison
                baseline_curves[baseline] = {'x': x_data, 'y': y_data}

                # Build customdata with max_attempts and unique_problems_solved
                customdata_cols = []
                if 'max_attempts' in bl_df.columns:
                    customdata_cols.append([None] + bl_df['max_attempts'].tolist())
                else:
                    customdata_cols.append([None] + [0] * len(bl_df))
                if 'unique_problems_solved' in bl_df.columns:
                    customdata_cols.append([None] + bl_df['unique_problems_solved'].tolist())
                else:
                    customdata_cols.append([None] + [0] * len(bl_df))
                customdata = list(zip(*customdata_cols))

                plot_x, plot_y = _xy(x_data, y_data)
                fig.add_trace(go.Scatter(
                    x=plot_x,
                    y=plot_y,
                    mode='markers+lines',
                    name=f'{baseline}',
                    marker=dict(size=10, color=color, symbol='diamond'),
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        f'<b>{baseline}</b><br>'
                        'Max attempts: %{customdata[0]}<br>'
                        f'Success rate: %{{{"x" if flip_axes else "y"}:.3f}}<br>'
                        f'Avg cost: %{{{"y" if flip_axes else "x"}:.2f}}M<br>'
                        'Unique solved: %{customdata[1]}<extra></extra>'
                    ),
                    customdata=customdata,
                ))

    # Add upper envelope of agentic curves as gray dotted line
    if auc_data and len(auc_data) > 0:
        envelope_curves = [{'x': d['x'], 'y': d['y']} for d in auc_data.values() if len(d['x']) >= 2]
        if envelope_curves:
            # Find max x across all agentic curves
            all_x_max = max(max(d['x']) for d in auc_data.values() if len(d['x']) >= 2)
            env_x, env_y = compute_upper_envelope(envelope_curves, 0.0, all_x_max)

            if env_x and env_y:
                env_plot_x, env_plot_y = _xy(env_x, env_y)
                envelope_name = 'Cost-Efficient Frontier' if flip_axes else 'Upper Envelope'
                fig.add_trace(go.Scatter(
                    x=env_plot_x,
                    y=env_plot_y,
                    mode='lines',
                    name=envelope_name,
                    line=dict(color='#00FFFF', width=3, dash='dot'),  # Cyan
                    hovertemplate=(
                        f'<b>{envelope_name}</b><br>'
                        f'Success rate: %{{{"x" if flip_axes else "y"}:.3f}}<br>'
                        f'Avg cost: %{{{"y" if flip_axes else "x"}:.2f}}M<extra></extra>'
                    ),
                ))

    # Update layout
    if flip_axes:
        xaxis_title = 'Success Rate'
        yaxis_title = 'Average Cost per Problem (M SFLOPs)'
        xaxis_tickformat = '.0%'
        yaxis_tickformat = None
    else:
        xaxis_title = 'Average Cost per Problem (M SFLOPs)'
        yaxis_title = 'Success Rate'
        xaxis_tickformat = None
        yaxis_tickformat = '.0%'
    fig.update_layout(
        title='Cost vs Success Rate Trade-off',
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_tickformat=xaxis_tickformat,
        yaxis_tickformat=yaxis_tickformat,
        hovermode='closest',
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Calculate and display AUC comparison
    if auc_data and baseline_curves:
        st.subheader("AUC Comparison")
        st.markdown("Area Under Curve (AUC) measures overall performance. Higher AUC = better cost-effectiveness.")

        # Calculate combined upper envelope AUC with separate comparisons for 8B and 32B
        # Get max x (in current axes) for agentic curves
        agentic_x_maxes = [max(_current_axes(d)[0]) for d in auc_data.values() if len(d['x']) >= 2]

        if agentic_x_maxes:
            envelope_x_max = max(agentic_x_maxes)  # Envelope extends to furthest agentic point
            # Envelope is always computed in (cost, accuracy) frame; helpers handle the projection.
            envelope_curves = [{'x': d['x'], 'y': d['y']} for d in auc_data.values() if len(d['x']) >= 2]

            envelope_section_label = (
                "**Combined Cost-Efficient Frontier Comparison**" if flip_axes
                else "**Combined Upper Envelope Comparison**"
            )
            st.markdown(envelope_section_label)
            combined_rows = []

            for baseline_name, label in [('FULL_PROOF_8B', '8B'), ('FULL_PROOF_32B', '32B')]:
                if baseline_name not in baseline_curves:
                    continue
                bl_data = baseline_curves[baseline_name]
                bl_x_arr, _ = _current_axes(bl_data)
                bl_x_max = max(bl_x_arr) if len(bl_x_arr) >= 2 else 0
                common_x_max = min(envelope_x_max, bl_x_max)
                if common_x_max <= 0:
                    continue
                env_x, env_y = _envelope_current(envelope_curves, common_x_max)
                if not env_x or not env_y:
                    continue
                envelope_auc = np.trapezoid(env_y, env_x)
                bl_auc = _baseline_auc_current(bl_data, common_x_max)
                if bl_auc is None or bl_auc <= 0:
                    continue
                diff = ((envelope_auc / bl_auc) - 1) * 100
                combined_rows.append({
                    'Comparison': f'Envelope vs {label}',
                    'X Range': _fmt_xrange(common_x_max),
                    'Envelope AUC': envelope_auc,
                    'Baseline AUC': bl_auc,
                    'Difference': f"{diff:+.1f}%",
                })

            if combined_rows:
                df_combined = pd.DataFrame(combined_rows)
                for col in ['Envelope AUC', 'Baseline AUC']:
                    df_combined[col] = df_combined[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
                st.dataframe(df_combined, use_container_width=True, hide_index=True)

            st.markdown("---")

        # Per-curve AUC table
        st.markdown("**Per-Curve AUC** (each curve uses its own x-range)")

        auc_rows = []
        for label, data in auc_data.items():
            x_vals, y_vals = _current_axes(data)
            if len(x_vals) < 2:
                continue

            x_max = max(x_vals)

            # Calculate AUC for this curve (from 0 to x_max)
            curve_auc = calculate_auc(x_vals, y_vals, 0.0, x_max)

            row = {
                'Curve': label,
                'X Range': _fmt_xrange(x_max),
                'AUC': curve_auc,
            }

            # Compare with 32B baseline
            if 'FULL_PROOF_32B' in baseline_curves:
                bl_auc = _baseline_auc_current(baseline_curves['FULL_PROOF_32B'], x_max)
                if bl_auc is not None:
                    row['32B AUC'] = bl_auc
                    row['vs 32B'] = f"{((curve_auc / bl_auc) - 1) * 100:+.1f}%" if bl_auc > 0 else "N/A"
                else:
                    row['32B AUC'] = None
                    row['vs 32B'] = "N/A"

            # Compare with 8B baseline
            if 'FULL_PROOF_8B' in baseline_curves:
                bl_auc = _baseline_auc_current(baseline_curves['FULL_PROOF_8B'], x_max)
                if bl_auc is not None:
                    row['8B AUC'] = bl_auc
                    row['vs 8B'] = f"{((curve_auc / bl_auc) - 1) * 100:+.1f}%" if bl_auc > 0 else "N/A"
                else:
                    row['8B AUC'] = None
                    row['vs 8B'] = "N/A"

            auc_rows.append(row)

        if auc_rows:
            df_auc = pd.DataFrame(auc_rows)

            # Format AUC values
            for col in ['AUC', '32B AUC', '8B AUC']:
                if col in df_auc.columns:
                    df_auc[col] = df_auc[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

            st.dataframe(df_auc, use_container_width=True, hide_index=True)

    # Show interpretation
    if flip_axes:
        st.caption("""
        **Interpretation (flipped)**:
        - Points further **right** have higher success rate
        - Points further **up** use more compute (higher cost)
        - The **Pareto frontier** (lower-right) shows optimal trade-offs
        - AUC here = ∫ cost d(success rate); **lower** is better
        - Agentic baselines explore different combinations of full proof and lemma-level attempts
        - Full proof baselines (diamonds) show pure full-proof attempt strategies
        """)
    else:
        st.caption("""
        **Interpretation**:
        - Points further **right** use more compute (higher cost)
        - Points further **up** have higher success rate
        - The **Pareto frontier** (upper-left) shows optimal trade-offs
        - Agentic baselines explore different combinations of full proof and lemma-level attempts
        - Full proof baselines (diamonds) show pure full-proof attempt strategies
        """)

    # Data table
    st.subheader("Configuration Details")

    if filtered_comparison:
        st.markdown("### Agentic Configurations")
        df_display = pd.DataFrame(filtered_comparison)

        display_cols = ['config_name', 'n1', 'n2', 'n3', 'n4', 'n5', 'use_corrections',
                       'success_rate', 'avg_cost', 'avg_steps', 'unique_problems_solved', 'num_rollouts',
                       'solved_by_full_proof_8b', 'solved_by_full_proof_32b', 'solved_by_agentic']
        display_cols = [c for c in display_cols if c in df_display.columns]

        df_display = df_display[display_cols].copy()

        col_rename = {
            'config_name': 'Config',
            'n1': 'n1 (fp8b)',
            'n2': 'n2 (fp32b)',
            'n3': 'n3 (brkdn)',
            'n4': 'n4 (att8b)',
            'n5': 'n5 (att32b)',
            'use_corrections': 'Corr',
            'success_rate': 'Success Rate',
            'avg_cost': 'Avg Cost',
            'avg_steps': 'Avg Steps',
            'unique_problems_solved': 'Unique Solved',
            'num_rollouts': 'Total Rollouts',
            'solved_by_full_proof_8b': 'By FP8B',
            'solved_by_full_proof_32b': 'By FP32B',
            'solved_by_agentic': 'By Agentic',
        }

        # Format columns
        if 'success_rate' in df_display.columns:
            df_display['success_rate'] = df_display['success_rate'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
        if 'avg_cost' in df_display.columns:
            df_display['avg_cost'] = df_display['avg_cost'].apply(lambda x: f"{x/1e6:.2f}M" if pd.notnull(x) else "N/A")
        if 'avg_steps' in df_display.columns:
            df_display['avg_steps'] = df_display['avg_steps'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")

        df_display = df_display.rename(columns=col_rename)

        # Sort by success rate descending
        df_display = df_display.sort_values('Success Rate', ascending=False)

        st.dataframe(df_display, use_container_width=True, hide_index=True)

    if fp_summaries:
        st.markdown("### Full Proof Baselines")
        df_fp_display = pd.DataFrame(fp_summaries)

        display_cols = ['baseline', 'max_attempts', 'budget_str', 'success_rate', 'avg_cost',
                       'unique_problems_solved', 'num_rollouts']
        display_cols = [c for c in display_cols if c in df_fp_display.columns]

        if display_cols:
            df_fp_display = df_fp_display[display_cols].copy()

            if 'success_rate' in df_fp_display.columns:
                df_fp_display['success_rate'] = df_fp_display['success_rate'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
            if 'avg_cost' in df_fp_display.columns:
                df_fp_display['avg_cost'] = df_fp_display['avg_cost'].apply(lambda x: f"{x/1e6:.2f}M" if pd.notnull(x) else "N/A")

            col_rename = {
                'baseline': 'Baseline',
                'max_attempts': 'Max Attempts',
                'budget_str': 'Budget',
                'success_rate': 'Success Rate',
                'avg_cost': 'Avg Cost',
                'unique_problems_solved': 'Unique Solved',
                'num_rollouts': 'Total Rollouts',
            }
            df_fp_display = df_fp_display.rename(columns=col_rename)

            st.dataframe(df_fp_display, use_container_width=True, hide_index=True)


def render_problem_level_analysis():
    """Render the problem-level analysis section with baseline comparison."""
    st.subheader("Problem-level Analysis")
    st.markdown("Compare baseline efficiency (avg cost) per problem. The lowest cost baseline for each problem is highlighted.")

    # Path configuration in expander
    with st.expander("📁 Data Paths", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            full_proof_path = st.text_input(
                "Full Proof Directory",
                value=str(DEFAULT_FULL_PROOF_DIR),
                key="problem_level_full_proof_path"
            )
        with col2:
            agentic_path = st.text_input(
                "Agentic Directory",
                value=str(DEFAULT_AGENTIC_DIR),
                key="problem_level_agentic_path"
            )

    # Get available agentic subdirectories
    available_subdirs = get_available_agentic_subdirs(agentic_path)

    # Subdirectory selector for agentic
    if len(available_subdirs) > 1:
        selected_subdir = st.selectbox(
            "Agentic Baseline Set",
            options=available_subdirs,
            format_func=lambda x: "(root)" if x == '' else x,
            key="problem_level_agentic_subdir_selector"
        )
    else:
        selected_subdir = available_subdirs[0] if available_subdirs else ''

    # Load per-problem data from all sources
    fp_per_problem = load_full_proof_per_problem(full_proof_path)
    agentic_per_problem = load_agentic_per_problem(agentic_path, selected_subdir if selected_subdir else None)

    # Combine all baselines
    all_baselines = {}
    all_baselines.update(fp_per_problem)
    all_baselines.update(agentic_per_problem)

    if not all_baselines:
        st.warning("No per-problem data available.")
        return

    # Get list of all baselines
    baseline_names = sorted(all_baselines.keys())

    # Baseline selection
    st.markdown("**Select Baselines to Compare:**")

    # Initialize session state for baseline selections
    if 'problem_level_baselines' not in st.session_state:
        st.session_state.problem_level_baselines = set(baseline_names[:5])  # Default to first 5

    # Select/Deselect all buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select All Baselines", key="select_all_baselines"):
            st.session_state.problem_level_baselines = set(baseline_names)
            st.rerun()
    with col2:
        if st.button("Deselect All Baselines", key="deselect_all_baselines"):
            st.session_state.problem_level_baselines = set()
            st.rerun()

    # Create multiselect for baseline selection
    selected_baselines = st.multiselect(
        "Baselines",
        options=baseline_names,
        default=list(st.session_state.problem_level_baselines),
        key="baseline_multiselect",
        label_visibility="collapsed",
    )
    st.session_state.problem_level_baselines = set(selected_baselines)

    if not selected_baselines:
        st.info("Select at least one baseline to see problem-level comparison.")
        return

    # Collect all problems across selected baselines
    all_problems = set()
    for baseline in selected_baselines:
        all_problems.update(all_baselines[baseline].keys())

    all_problems = sorted(all_problems)

    if not all_problems:
        st.warning("No problems found for the selected baselines.")
        return

    # Build the dataframe
    rows = []
    for problem_id in all_problems:
        row = {'Problem': problem_id}
        costs = {}

        for baseline in selected_baselines:
            if problem_id in all_baselines.get(baseline, {}):
                metrics = all_baselines[baseline][problem_id]
                avg_cost = metrics['avg_cost']
                row[baseline] = avg_cost
                costs[baseline] = avg_cost
            else:
                row[baseline] = None

        # Find minimum cost baseline for this problem
        if costs:
            min_baseline = min(costs.keys(), key=lambda b: costs[b])
            row['_min_baseline'] = min_baseline
            row['_min_cost'] = costs[min_baseline]
        else:
            row['_min_baseline'] = None
            row['_min_cost'] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort options
    sort_col = st.selectbox(
        "Sort by",
        options=['Problem'] + selected_baselines,
        key="problem_sort_col"
    )

    sort_asc = st.checkbox("Ascending order", value=True, key="problem_sort_asc")

    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=sort_asc, na_position='last')

    # Create styled dataframe with green highlighting
    def highlight_min_cost(row):
        styles = [''] * len(row)
        min_baseline = row.get('_min_baseline')
        if min_baseline and min_baseline in row.index:
            idx = row.index.get_loc(min_baseline)
            styles[idx] = 'background-color: #90EE90; font-weight: bold'
        return styles

    # Prepare display dataframe (without internal columns)
    display_cols = ['Problem'] + selected_baselines
    df_display = df[display_cols].copy()

    # Format cost values
    for baseline in selected_baselines:
        df_display[baseline] = df_display[baseline].apply(
            lambda x: f"{x/1e6:.2f}M" if pd.notnull(x) else "-"
        )

    # Apply styling
    styled_df = df_display.style.apply(
        lambda row: highlight_min_cost_styled(row, df, selected_baselines),
        axis=1
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)

    # Summary stats
    st.markdown("---")
    st.markdown("**Summary:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Problems", len(all_problems))
    with col2:
        st.metric("Baselines Compared", len(selected_baselines))
    with col3:
        # Count how many times each baseline is the best
        best_counts = df['_min_baseline'].value_counts()
        if len(best_counts) > 0:
            top_baseline = best_counts.index[0]
            st.metric("Most Efficient Baseline", f"{top_baseline[:20]}...", delta=f"{best_counts.iloc[0]} problems")


def highlight_min_cost_styled(row, original_df, baselines):
    """Return style for each cell in the row, highlighting min cost baseline."""
    styles = [''] * len(row)

    # Get the problem from this row
    problem = row.get('Problem')
    if problem is None:
        return styles

    # Find the original row in df to get _min_baseline
    orig_row = original_df[original_df['Problem'] == problem]
    if orig_row.empty:
        return styles

    min_baseline = orig_row['_min_baseline'].iloc[0]

    if min_baseline and min_baseline in row.index:
        idx = row.index.get_loc(min_baseline)
        styles[idx] = 'background-color: #90EE90; font-weight: bold'

    return styles


# =============================================================================
# Action Cost Analysis Tab
# =============================================================================

@st.cache_data
def load_rollouts_for_action_analysis(baseline_dir: str, config_name: str) -> List[Dict[str, Any]]:
    """Load all rollouts for a specific baseline configuration.

    Loads from rollouts/{problem_id}/seed_*.json files.
    """
    config_path = Path(baseline_dir) / config_name / "rollouts"
    rollouts = []

    if not config_path.exists():
        # Try loading from rollouts_summary.json as fallback (but won't have action_cost)
        summary_file = Path(baseline_dir) / config_name / "rollouts_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return rollouts

    # Load from individual rollout files
    for problem_dir in config_path.iterdir():
        if not problem_dir.is_dir():
            continue

        for rollout_file in problem_dir.glob("seed_*.json"):
            try:
                with open(rollout_file, 'r') as f:
                    rollout = json.load(f)
                    rollouts.append(rollout)
            except Exception as e:
                pass

    return rollouts


def compute_action_cost_stats(rollouts: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute action cost statistics from rollouts.

    Returns:
        Dict mapping action_type -> {
            'count': number of times action was used,
            'total_cost': sum of all costs,
            'avg_cost': average cost,
            'min_cost': minimum cost,
            'max_cost': maximum cost,
            'success_count': number of successful actions,
            'success_rate': proportion of successful actions,
            'avg_input_tokens': average input tokens,
            'avg_output_tokens': average output tokens,
            'total_input_tokens': sum of all input tokens,
            'total_output_tokens': sum of all output tokens,
            'success_costs': list of costs for successful actions,
            'failed_costs': list of costs for failed actions,
            'avg_success_cost': average cost for successful actions,
            'avg_failed_cost': average cost for failed actions,
        }
    """
    action_stats = {}

    for rollout in rollouts:
        history = rollout.get('history', [])

        for step in history:
            action_type = step.get('action', {}).get('action_type', 'UNKNOWN')
            action_cost = step.get('action_cost', 0)
            action_success = step.get('action_success', False)

            # Extract token counts from detailed_cost
            detailed_cost = step.get('detailed_cost', {})
            input_tokens = detailed_cost.get('input_tokens', 0) if detailed_cost else 0
            output_tokens = detailed_cost.get('output_tokens', 0) if detailed_cost else 0

            if action_type not in action_stats:
                action_stats[action_type] = {
                    'costs': [],
                    'input_tokens': [],
                    'output_tokens': [],
                    'success_count': 0,
                    'success_costs': [],
                    'failed_costs': [],
                }

            action_stats[action_type]['costs'].append(action_cost)
            action_stats[action_type]['input_tokens'].append(input_tokens)
            action_stats[action_type]['output_tokens'].append(output_tokens)
            if action_success:
                action_stats[action_type]['success_count'] += 1
                action_stats[action_type]['success_costs'].append(action_cost)
            else:
                action_stats[action_type]['failed_costs'].append(action_cost)

    # Compute summary statistics
    result = {}
    for action_type, data in action_stats.items():
        costs = data['costs']
        input_tokens = data['input_tokens']
        output_tokens = data['output_tokens']
        success_costs = data['success_costs']
        failed_costs = data['failed_costs']
        if costs:
            result[action_type] = {
                'count': len(costs),
                'total_cost': sum(costs),
                'avg_cost': np.mean(costs),
                'min_cost': min(costs),
                'max_cost': max(costs),
                'std_cost': np.std(costs) if len(costs) > 1 else 0,
                'success_count': data['success_count'],
                'success_rate': data['success_count'] / len(costs) if costs else 0,
                # Token statistics
                'avg_input_tokens': np.mean(input_tokens) if input_tokens else 0,
                'avg_output_tokens': np.mean(output_tokens) if output_tokens else 0,
                'total_input_tokens': sum(input_tokens),
                'total_output_tokens': sum(output_tokens),
                'std_input_tokens': np.std(input_tokens) if len(input_tokens) > 1 else 0,
                'std_output_tokens': np.std(output_tokens) if len(output_tokens) > 1 else 0,
                # Success vs failed cost breakdown
                'success_costs': success_costs,
                'failed_costs': failed_costs,
                'avg_success_cost': np.mean(success_costs) if success_costs else 0,
                'avg_failed_cost': np.mean(failed_costs) if failed_costs else 0,
                'std_success_cost': np.std(success_costs) if len(success_costs) > 1 else 0,
                'std_failed_cost': np.std(failed_costs) if len(failed_costs) > 1 else 0,
            }

    return result


def render_action_cost_analysis():
    """Render the action cost analysis section."""
    st.subheader("💰 Action Cost Analysis")
    st.markdown("Analyze the cost distribution of different action types across rollouts.")

    # Get available baseline types
    baseline_sources = []

    # Add agentic configs (including subdirectories)
    if DEFAULT_AGENTIC_DIR.exists():
        for item in sorted(DEFAULT_AGENTIC_DIR.iterdir()):
            if not item.is_dir() or item.name.startswith('.'):
                continue
            # Check if it's a config directory (has rollouts/) or a subdirectory with configs
            if (item / "rollouts").exists():
                baseline_sources.append(('agentic', '', item.name))
            else:
                # It's a subdirectory like 8b_theorem, sft8b_theorem
                for subconfig in sorted(item.iterdir()):
                    if subconfig.is_dir() and (subconfig / "rollouts").exists():
                        baseline_sources.append(('agentic', item.name, subconfig.name))

    # Add full proof configs
    if DEFAULT_FULL_PROOF_DIR.exists():
        for baseline_dir in DEFAULT_FULL_PROOF_DIR.iterdir():
            if baseline_dir.is_dir() and not baseline_dir.name.startswith('.'):
                for max_dir in baseline_dir.iterdir():
                    if max_dir.is_dir() and max_dir.name.startswith('max_'):
                        config_name = f"{baseline_dir.name}/{max_dir.name}"
                        baseline_sources.append(('full_proof', '', config_name))

    if not baseline_sources:
        st.warning("No baseline configurations found.")
        return

    # Create display names for selection
    config_options = []
    for source, subdir, config in baseline_sources:
        if subdir:
            config_options.append(f"{source}/{subdir}: {config}")
        else:
            config_options.append(f"{source}: {config}")

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_config = st.selectbox(
            "Select Baseline Configuration",
            options=config_options,
            key="action_cost_config"
        )

    if not selected_config:
        return

    # Parse selection
    parts = selected_config.split(": ", 1)
    source_part, config_name = parts[0], parts[1]

    if "/" in source_part:
        source_type, subdir = source_part.split("/", 1)
    else:
        source_type = source_part
        subdir = ''

    # Determine base directory
    if source_type == 'agentic':
        base_dir = DEFAULT_AGENTIC_DIR / subdir if subdir else DEFAULT_AGENTIC_DIR
    else:  # full_proof
        base_dir = DEFAULT_FULL_PROOF_DIR

    # Load rollouts
    rollouts = load_rollouts_for_action_analysis(str(base_dir), config_name)

    if not rollouts:
        st.warning(f"No rollouts found for {selected_config}")
        return

    # Get list of problems
    all_problems = sorted(set(r.get('problem_id', 'unknown') for r in rollouts))

    with col2:
        problem_options = ["All Problems"] + all_problems
        selected_problem = st.selectbox(
            "Select Problem",
            options=problem_options,
            key="action_cost_problem"
        )

    # Filter rollouts by problem if needed
    if selected_problem != "All Problems":
        filtered_rollouts = [r for r in rollouts if r.get('problem_id') == selected_problem]
    else:
        filtered_rollouts = rollouts

    if not filtered_rollouts:
        st.warning("No rollouts match the selection.")
        return

    # Check if rollouts have action_cost in history
    has_action_cost = False
    for r in filtered_rollouts[:5]:  # Check first few
        history = r.get('history', [])
        if history and 'action_cost' in history[0]:
            has_action_cost = True
            break

    if not has_action_cost:
        st.warning("⚠️ These rollouts don't have per-action cost data. Re-run the baseline to generate action costs.")
        # Still show action counts from action_counts field
        st.markdown("Showing action counts from summary data instead:")

        # Aggregate action counts
        total_counts = {}
        total_success = {}
        for r in filtered_rollouts:
            action_counts = r.get('action_counts', {})
            action_success = r.get('action_success_counts', {})
            for action, count in action_counts.items():
                total_counts[action] = total_counts.get(action, 0) + count
            for action, count in action_success.items():
                total_success[action] = total_success.get(action, 0) + count

        if total_counts:
            df_counts = pd.DataFrame([
                {
                    'Action Type': action,
                    'Total Count': count,
                    'Success Count': total_success.get(action, 0),
                    'Success Rate': f"{100 * total_success.get(action, 0) / count:.1f}%" if count > 0 else "0%"
                }
                for action, count in sorted(total_counts.items())
            ])
            st.dataframe(df_counts, use_container_width=True, hide_index=True)
        return

    # Compute action cost statistics
    action_stats = compute_action_cost_stats(filtered_rollouts)

    if not action_stats:
        st.warning("No action data found in rollouts.")
        return

    # Display summary info
    st.markdown(f"**Analyzing {len(filtered_rollouts)} rollouts**")

    # Create bar chart of average costs
    action_types = list(action_stats.keys())
    avg_costs = [action_stats[a]['avg_cost'] for a in action_types]
    counts = [action_stats[a]['count'] for a in action_types]
    success_rates = [action_stats[a]['success_rate'] for a in action_types]

    # Sort by average cost descending
    sorted_indices = np.argsort(avg_costs)[::-1]
    action_types = [action_types[i] for i in sorted_indices]
    avg_costs = [avg_costs[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    success_rates = [success_rates[i] for i in sorted_indices]

    # Color mapping for action types
    color_map = {
        'FULL_PROOF_8B': '#1f77b4',
        'FULL_PROOF_32B': '#ff7f0e',
        'CREATE_BREAKDOWN': '#2ca02c',
        'ATTEMPT_8B': '#d62728',
        'ATTEMPT_32B': '#9467bd',
        'CORRECTION_32B': '#8c564b',
        'TERMINATE': '#7f7f7f',
    }
    colors = [color_map.get(a, '#17becf') for a in action_types]

    # Create figure with two subplots
    fig = go.Figure()

    # Bar chart of average costs
    fig.add_trace(go.Bar(
        x=action_types,
        y=[c / 1e6 for c in avg_costs],  # Convert to millions
        marker_color=colors,
        text=[f"{c/1e6:.3f}M" for c in avg_costs],
        textposition='auto',
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Avg Cost: %{y:.4f}M SFLOPs<br>'
            'Count: %{customdata[0]}<br>'
            'Success Rate: %{customdata[1]:.1%}<extra></extra>'
        ),
        customdata=list(zip(counts, success_rates)),
    ))

    fig.update_layout(
        title='Average Cost per Action Type',
        xaxis_title='Action Type',
        yaxis_title='Average Cost (M SFLOPs)',
        height=400,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Success vs Failed Cost Comparison
    st.markdown("### Cost per Successful vs Failed Action")

    # Prepare data for grouped bar chart
    success_avg_costs = [action_stats[a].get('avg_success_cost', 0) for a in action_types]
    failed_avg_costs = [action_stats[a].get('avg_failed_cost', 0) for a in action_types]
    success_counts = [len(action_stats[a].get('success_costs', [])) for a in action_types]
    failed_counts = [len(action_stats[a].get('failed_costs', [])) for a in action_types]

    fig_success_fail = go.Figure()

    # Add successful actions bars
    fig_success_fail.add_trace(go.Bar(
        name='Successful',
        x=action_types,
        y=[c / 1e6 for c in success_avg_costs],
        marker_color='rgba(50, 171, 96, 0.8)',
        text=[f"{c/1e6:.3f}M" if c > 0 else "N/A" for c in success_avg_costs],
        textposition='auto',
        hovertemplate=(
            '<b>%{x}</b> (Successful)<br>'
            'Avg Cost: %{y:.4f}M SFLOPs<br>'
            'Count: %{customdata}<extra></extra>'
        ),
        customdata=success_counts,
    ))

    # Add failed actions bars
    fig_success_fail.add_trace(go.Bar(
        name='Failed',
        x=action_types,
        y=[c / 1e6 for c in failed_avg_costs],
        marker_color='rgba(219, 64, 82, 0.8)',
        text=[f"{c/1e6:.3f}M" if c > 0 else "N/A" for c in failed_avg_costs],
        textposition='auto',
        hovertemplate=(
            '<b>%{x}</b> (Failed)<br>'
            'Avg Cost: %{y:.4f}M SFLOPs<br>'
            'Count: %{customdata}<extra></extra>'
        ),
        customdata=failed_counts,
    ))

    fig_success_fail.update_layout(
        title='Average Cost: Successful vs Failed Actions',
        xaxis_title='Action Type',
        yaxis_title='Average Cost (M SFLOPs)',
        barmode='group',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    st.plotly_chart(fig_success_fail, use_container_width=True)

    # Success vs Failed statistics table
    st.markdown("### Success vs Failed Cost Breakdown")

    success_fail_data = []
    for action_type in action_types:
        stats = action_stats[action_type]
        success_count = len(stats.get('success_costs', []))
        failed_count = len(stats.get('failed_costs', []))
        avg_success = stats.get('avg_success_cost', 0)
        avg_failed = stats.get('avg_failed_cost', 0)
        std_success = stats.get('std_success_cost', 0)
        std_failed = stats.get('std_failed_cost', 0)

        # Calculate cost ratio (failed/success)
        cost_ratio = avg_failed / avg_success if avg_success > 0 else float('inf') if avg_failed > 0 else 0

        success_fail_data.append({
            'Action Type': action_type,
            'Success Count': success_count,
            'Failed Count': failed_count,
            'Avg Success Cost': f"{avg_success/1e6:.4f}M" if success_count > 0 else "N/A",
            'Avg Failed Cost': f"{avg_failed/1e6:.4f}M" if failed_count > 0 else "N/A",
            'Std Success': f"{std_success/1e6:.4f}M" if success_count > 1 else "N/A",
            'Std Failed': f"{std_failed/1e6:.4f}M" if failed_count > 1 else "N/A",
            'Failed/Success Ratio': f"{cost_ratio:.2f}x" if cost_ratio != float('inf') and cost_ratio > 0 else "N/A",
        })

    df_success_fail = pd.DataFrame(success_fail_data)
    st.dataframe(df_success_fail, use_container_width=True, hide_index=True)

    # Detailed statistics table
    st.markdown("### Detailed Statistics (SFLOPs)")

    stats_data = []
    for action_type in action_types:
        stats = action_stats[action_type]
        stats_data.append({
            'Action Type': action_type,
            'Count': stats['count'],
            'Avg Cost': f"{stats['avg_cost']/1e6:.4f}M",
            'Std Dev': f"{stats['std_cost']/1e6:.4f}M",
            'Min Cost': f"{stats['min_cost']/1e6:.4f}M",
            'Max Cost': f"{stats['max_cost']/1e6:.4f}M",
            'Total Cost': f"{stats['total_cost']/1e6:.2f}M",
            'Success Count': stats['success_count'],
            'Success Rate': f"{stats['success_rate']*100:.1f}%",
        })

    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    # Token statistics table
    st.markdown("### Token Usage Statistics")

    # Check if we have token data
    has_token_data = any(stats.get('avg_output_tokens', 0) > 0 for stats in action_stats.values())

    if has_token_data:
        token_data = []
        for action_type in action_types:
            stats = action_stats[action_type]
            avg_input = stats.get('avg_input_tokens', 0)
            avg_output = stats.get('avg_output_tokens', 0)
            total_input = stats.get('total_input_tokens', 0)
            total_output = stats.get('total_output_tokens', 0)
            std_input = stats.get('std_input_tokens', 0)
            std_output = stats.get('std_output_tokens', 0)
            token_data.append({
                'Action Type': action_type,
                'Count': stats['count'],
                'Avg Input Tokens': f"{avg_input:,.0f}",
                'Avg Output Tokens': f"{avg_output:,.0f}",
                'Std Input': f"{std_input:,.0f}",
                'Std Output': f"{std_output:,.0f}",
                'Total Input': f"{total_input:,}",
                'Total Output': f"{total_output:,}",
            })

        df_tokens = pd.DataFrame(token_data)
        st.dataframe(df_tokens, use_container_width=True, hide_index=True)

        # Token bar chart
        st.markdown("### Average Tokens per Action Type")

        fig_tokens = go.Figure()

        # Add input tokens bars
        fig_tokens.add_trace(go.Bar(
            name='Input Tokens',
            x=action_types,
            y=[action_stats[a].get('avg_input_tokens', 0) for a in action_types],
            marker_color='rgba(55, 128, 191, 0.7)',
            text=[f"{action_stats[a].get('avg_input_tokens', 0):,.0f}" for a in action_types],
            textposition='auto',
        ))

        # Add output tokens bars
        fig_tokens.add_trace(go.Bar(
            name='Output Tokens',
            x=action_types,
            y=[action_stats[a].get('avg_output_tokens', 0) for a in action_types],
            marker_color='rgba(219, 64, 82, 0.7)',
            text=[f"{action_stats[a].get('avg_output_tokens', 0):,.0f}" for a in action_types],
            textposition='auto',
        ))

        fig_tokens.update_layout(
            title='Average Token Usage per Action Type',
            xaxis_title='Action Type',
            yaxis_title='Tokens',
            barmode='group',
            height=400,
        )

        st.plotly_chart(fig_tokens, use_container_width=True)
    else:
        st.info("No token data available. Run new rollouts to capture detailed_cost with token counts.")

    # Cost distribution histogram (optional, if there's variation)
    st.markdown("### Cost Distribution by Action Type")

    # Let user select which action to see distribution for
    selected_action = st.selectbox(
        "Select action type for cost distribution",
        options=action_types,
        key="cost_dist_action"
    )

    if selected_action:
        # Collect all costs for this action type
        costs_for_action = []
        for rollout in filtered_rollouts:
            history = rollout.get('history', [])
            for step in history:
                if step.get('action', {}).get('action_type') == selected_action:
                    cost = step.get('action_cost', 0)
                    costs_for_action.append(cost / 1e6)  # Convert to millions

        if costs_for_action:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=costs_for_action,
                nbinsx=50,
                marker_color=color_map.get(selected_action, '#17becf'),
            ))
            fig_hist.update_layout(
                title=f'Cost Distribution for {selected_action}',
                xaxis_title='Cost (M SFLOPs)',
                yaxis_title='Frequency',
                height=300,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Show percentiles
            percentiles = np.percentile(costs_for_action, [25, 50, 75, 90, 95, 99])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("25th percentile", f"{percentiles[0]:.4f}M")
                st.metric("50th percentile (median)", f"{percentiles[1]:.4f}M")
            with col2:
                st.metric("75th percentile", f"{percentiles[2]:.4f}M")
                st.metric("90th percentile", f"{percentiles[3]:.4f}M")
            with col3:
                st.metric("95th percentile", f"{percentiles[4]:.4f}M")
                st.metric("99th percentile", f"{percentiles[5]:.4f}M")

    # Average costs by problem difficulty
    st.markdown("### Average Costs by Problem Difficulty")

    # Load problem difficulties
    difficulties = load_problem_difficulties()

    if not difficulties:
        st.warning("Could not load problem difficulties from dataset.json")
    else:
        # Compute action costs grouped by difficulty
        difficulty_action_costs = {
            'easy': {},
            'medium': {},
            'hard': {},
        }

        for rollout in filtered_rollouts:
            problem_id = rollout.get('problem_id')
            difficulty = difficulties.get(problem_id, 'unknown')

            if difficulty not in difficulty_action_costs:
                continue

            history = rollout.get('history', [])
            for step in history:
                action_type = step.get('action', {}).get('action_type', 'UNKNOWN')
                action_cost = step.get('action_cost', 0)

                if action_type not in difficulty_action_costs[difficulty]:
                    difficulty_action_costs[difficulty][action_type] = []
                difficulty_action_costs[difficulty][action_type].append(action_cost)

        # Build data for grouped bar chart
        all_action_types = set()
        for diff_data in difficulty_action_costs.values():
            all_action_types.update(diff_data.keys())
        all_action_types = sorted(all_action_types)

        if all_action_types:
            fig_diff = go.Figure()

            difficulty_colors = {
                'easy': '#2ecc71',    # Green
                'medium': '#f39c12',  # Orange
                'hard': '#e74c3c',    # Red
            }

            for difficulty in ['easy', 'medium', 'hard']:
                avg_costs = []
                counts = []
                for action_type in all_action_types:
                    costs = difficulty_action_costs[difficulty].get(action_type, [])
                    if costs:
                        avg_costs.append(np.mean(costs) / 1e6)
                        counts.append(len(costs))
                    else:
                        avg_costs.append(0)
                        counts.append(0)

                fig_diff.add_trace(go.Bar(
                    name=difficulty.capitalize(),
                    x=all_action_types,
                    y=avg_costs,
                    marker_color=difficulty_colors[difficulty],
                    hovertemplate=(
                        f'<b>{difficulty.capitalize()}</b><br>'
                        'Action: %{x}<br>'
                        'Avg Cost: %{y:.4f}M<br>'
                        'Count: %{customdata}<extra></extra>'
                    ),
                    customdata=counts,
                ))

            fig_diff.update_layout(
                title='Average Action Cost by Problem Difficulty',
                xaxis_title='Action Type',
                yaxis_title='Average Cost (M SFLOPs)',
                barmode='group',
                height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            )

            st.plotly_chart(fig_diff, use_container_width=True)

            # Show summary table
            st.markdown("#### Summary Table")
            summary_rows = []
            for action_type in all_action_types:
                row = {'Action Type': action_type}
                for difficulty in ['easy', 'medium', 'hard']:
                    costs = difficulty_action_costs[difficulty].get(action_type, [])
                    if costs:
                        row[f'{difficulty.capitalize()} Avg'] = f"{np.mean(costs)/1e6:.4f}M"
                        row[f'{difficulty.capitalize()} Count'] = len(costs)
                    else:
                        row[f'{difficulty.capitalize()} Avg'] = "-"
                        row[f'{difficulty.capitalize()} Count'] = 0
                summary_rows.append(row)

            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

            # Token statistics by difficulty
            st.markdown("### Average Tokens by Problem Difficulty")

            # Compute token stats grouped by difficulty
            difficulty_token_stats = {
                'easy': {},
                'medium': {},
                'hard': {},
            }

            for rollout in filtered_rollouts:
                problem_id = rollout.get('problem_id')
                difficulty = difficulties.get(problem_id, 'unknown')

                if difficulty not in difficulty_token_stats:
                    continue

                history = rollout.get('history', [])
                for step in history:
                    action_type = step.get('action', {}).get('action_type', 'UNKNOWN')
                    detailed_cost = step.get('detailed_cost', {})
                    output_tokens = detailed_cost.get('output_tokens', 0) if detailed_cost else 0

                    if action_type not in difficulty_token_stats[difficulty]:
                        difficulty_token_stats[difficulty][action_type] = []
                    difficulty_token_stats[difficulty][action_type].append(output_tokens)

            # Check if we have token data
            has_diff_tokens = any(
                any(tokens for tokens in diff_data.values())
                for diff_data in difficulty_token_stats.values()
            )

            if has_diff_tokens:
                fig_diff_tokens = go.Figure()

                for difficulty in ['easy', 'medium', 'hard']:
                    avg_tokens = []
                    counts = []
                    for action_type in all_action_types:
                        tokens = difficulty_token_stats[difficulty].get(action_type, [])
                        if tokens and any(t > 0 for t in tokens):
                            avg_tokens.append(np.mean(tokens))
                            counts.append(len(tokens))
                        else:
                            avg_tokens.append(0)
                            counts.append(0)

                    fig_diff_tokens.add_trace(go.Bar(
                        name=difficulty.capitalize(),
                        x=all_action_types,
                        y=avg_tokens,
                        marker_color=difficulty_colors[difficulty],
                        hovertemplate=(
                            f'<b>{difficulty.capitalize()}</b><br>'
                            'Action: %{x}<br>'
                            'Avg Output Tokens: %{y:,.0f}<br>'
                            'Count: %{customdata}<extra></extra>'
                        ),
                        customdata=counts,
                    ))

                fig_diff_tokens.update_layout(
                    title='Average Output Tokens by Problem Difficulty',
                    xaxis_title='Action Type',
                    yaxis_title='Average Output Tokens',
                    barmode='group',
                    height=400,
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                )

                st.plotly_chart(fig_diff_tokens, use_container_width=True)

                # Token summary table
                st.markdown("#### Token Summary Table")
                token_summary_rows = []
                for action_type in all_action_types:
                    row = {'Action Type': action_type}
                    for difficulty in ['easy', 'medium', 'hard']:
                        tokens = difficulty_token_stats[difficulty].get(action_type, [])
                        if tokens and any(t > 0 for t in tokens):
                            row[f'{difficulty.capitalize()} Avg Tokens'] = f"{np.mean(tokens):,.0f}"
                            row[f'{difficulty.capitalize()} Count'] = len(tokens)
                        else:
                            row[f'{difficulty.capitalize()} Avg Tokens'] = "-"
                            row[f'{difficulty.capitalize()} Count'] = 0
                    token_summary_rows.append(row)

                df_token_summary = pd.DataFrame(token_summary_rows)
                st.dataframe(df_token_summary, use_container_width=True, hide_index=True)
            else:
                st.info("No token data available by difficulty. Run new rollouts to capture detailed_cost.")
