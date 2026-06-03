"""
Baseline model analysis tab for RL training viewer.

Plots predicted probability vs actual success probability at the problem level
to help diagnose model calibration issues.
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def render_baseline_model_analysis_tab(run_path: Path, run_metadata: Dict):
    """
    Render the baseline model analysis tab.

    Shows predicted vs actual success probability scatter plots to diagnose
    model calibration by comparing:
    - Model's predicted probability (avg per problem from rollouts)
    - Actual success rate (from benchmark result files)
    """
    st.markdown("## Model Calibration Analysis")
    st.markdown("""
    This tab analyzes the calibration of success models by comparing:
    - **X axis**: Average predicted probability per problem (from model during rollouts)
    - **Y axis**: Actual success rate per problem (from benchmark results)

    **Good calibration**: Points should lie along the diagonal (y = x).
    """)

    # Load config from file to find calibration result paths
    from .config_viewer import find_config_file, load_config
    config = {}
    config_file = find_config_file(run_path)
    if config_file:
        config = load_config(config_file) or {}

    results_8b_path = config.get('full_proof_8b_test_dir')
    results_32b_path = config.get('full_proof_32b_test_dir')

    # Allow manual path input if not in config
    with st.expander("📁 Calibration Data Paths"):
        col1, col2 = st.columns(2)
        with col1:
            results_8b_path = st.text_input(
                "8B Results Path",
                value=results_8b_path or ""
            )
        with col2:
            results_32b_path = st.text_input(
                "32B Results Path",
                value=results_32b_path or ""
            )

    # Load actual benchmark results
    actual_8b = load_actual_success_rates(results_8b_path)
    actual_32b = load_actual_success_rates(results_32b_path)

    if not actual_8b and not actual_32b:
        st.warning("Could not load benchmark results. Check the paths above.")
        return

    st.info(f"Loaded actual results: {len(actual_8b)} problems for 8B, {len(actual_32b)} problems for 32B")

    # Load rollouts and compute predicted probabilities
    rollouts = load_all_rollouts(run_path, run_metadata)

    if not rollouts:
        st.warning("No rollouts found for analysis.")
        return

    st.success(f"Loaded {len(rollouts)} rollouts")

    # Compute average predicted probability per problem
    predicted_8b, predicted_32b = compute_avg_predictions_per_problem(rollouts)

    st.info(f"Computed predictions: {len(predicted_8b)} problems for 8B, {len(predicted_32b)} problems for 32B")

    # Create plots
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 8B Model Calibration")
        plot_calibration_scatter(predicted_8b, actual_8b, '8B')

    with col2:
        st.markdown("### 32B Model Calibration")
        plot_calibration_scatter(predicted_32b, actual_32b, '32B')

    # Summary statistics
    st.markdown("### Calibration Summary")
    render_calibration_summary(predicted_8b, actual_8b, predicted_32b, actual_32b)

    # Raw data table
    with st.expander("📊 Problem-level Data"):
        render_problem_data_table(predicted_8b, actual_8b, predicted_32b, actual_32b)


def load_actual_success_rates(results_path: str) -> Dict[str, float]:
    """
    Load benchmark results and compute success rate per problem.

    Returns dict mapping problem_id -> success_rate (0 to 1)
    """
    if not results_path:
        return {}

    # Try both absolute and relative paths
    path = Path(results_path)
    if not path.is_absolute():
        # Try relative to repo root
        repo_root = Path(__file__).parent.parent.parent.parent.parent
        path = repo_root / results_path

    if not path.exists():
        return {}

    try:
        with open(path, 'r') as f:
            results = json.load(f)

        # Compute success rate per problem
        # Results format: list of dicts with 'origin_problem_id', 'pass', 'complete'
        problem_successes = {}
        problem_counts = {}

        for r in results:
            problem_id = r.get('origin_problem_id') or r.get('problem_id', 'unknown')
            success = r.get('pass', False) and r.get('complete', False)

            if problem_id not in problem_successes:
                problem_successes[problem_id] = 0
                problem_counts[problem_id] = 0

            if success:
                problem_successes[problem_id] += 1
            problem_counts[problem_id] += 1

        # Compute rates
        success_rates = {}
        for problem_id in problem_successes:
            if problem_counts[problem_id] > 0:
                success_rates[problem_id] = problem_successes[problem_id] / problem_counts[problem_id]

        return success_rates

    except Exception as e:
        st.error(f"Error loading {results_path}: {e}")
        return {}


def compute_avg_predictions_per_problem(rollouts: List[Dict]) -> tuple:
    """
    Compute average predicted probability per problem for 8B and 32B actions.

    Returns (predicted_8b, predicted_32b) where each is dict mapping problem_id -> avg_prob
    """
    # Collect all predictions per problem
    predictions_8b = {}  # problem_id -> list of probs
    predictions_32b = {}

    for rollout in rollouts:
        problem_id = rollout.get('problem_id', 'unknown')

        history = rollout.get('history', [])
        for step in history:
            action = step.get('action', {})
            action_type = action.get('action_type', '')

            # Get predictions
            predictions = step.get('predictions', {})
            all_scores = predictions.get('all_action_scores', [])

            # Extract predicted probs for both action types
            for score_info in all_scores:
                if isinstance(score_info, dict):
                    action_name = score_info.get('action', '')
                    prob = score_info.get('success_prob')

                    if prob is not None:
                        if action_name == 'FULL_PROOF_8B':
                            if problem_id not in predictions_8b:
                                predictions_8b[problem_id] = []
                            predictions_8b[problem_id].append(prob)
                        elif action_name == 'FULL_PROOF_32B':
                            if problem_id not in predictions_32b:
                                predictions_32b[problem_id] = []
                            predictions_32b[problem_id].append(prob)

    # Compute averages
    avg_8b = {pid: np.mean(probs) for pid, probs in predictions_8b.items()}
    avg_32b = {pid: np.mean(probs) for pid, probs in predictions_32b.items()}

    return avg_8b, avg_32b


def load_all_rollouts(run_path: Path, run_metadata: Dict) -> List[Dict]:
    """Load all rollouts from the run directory."""
    rollouts = []

    # Check for baseline/tree rollouts first
    iter_results = run_metadata.get('iteration_results', {})

    # Try to load from rounds - rollouts are in round{N}/rollouts/{problem_id}/{budget}/rollouts.json
    rounds = iter_results.get('rounds', [])
    for round_info in rounds:
        round_id = round_info.get('round_id', 0)
        round_dir = run_path / f"round{round_id}" / "rollouts"

        if round_dir.exists():
            # Look for rollouts.json files (each contains a list of rollouts)
            for json_file in round_dir.rglob('rollouts.json'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        # File contains a list of rollouts
                        if isinstance(data, list):
                            for rollout in data:
                                rollout['source'] = f'round{round_id}'
                                rollouts.append(rollout)
                        elif isinstance(data, dict):
                            data['source'] = f'round{round_id}'
                            rollouts.append(data)
                except Exception as e:
                    continue

    # Also try baseline rollouts if available
    baseline_dir = run_path / "baseline_rollouts"
    if baseline_dir.exists():
        for json_file in baseline_dir.rglob('rollouts.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for rollout in data:
                            rollout['source'] = 'baseline'
                            rollouts.append(rollout)
                    elif isinstance(data, dict):
                        data['source'] = 'baseline'
                        rollouts.append(data)
            except Exception:
                continue

    return rollouts


def compute_problem_level_stats(rollouts: List[Dict]) -> Dict:
    """
    Compute problem-level statistics for calibration analysis.

    Returns dict mapping problem_id -> {
        'FULL_PROOF_8B': {'predicted_probs': [...], 'actual_successes': [...], 'count': N},
        'FULL_PROOF_32B': {'predicted_probs': [...], 'actual_successes': [...], 'count': N},
    }
    """
    problem_stats = {}

    for rollout in rollouts:
        problem_id = rollout.get('problem_id', 'unknown')

        if problem_id not in problem_stats:
            problem_stats[problem_id] = {
                'FULL_PROOF_8B': {'predicted_probs': [], 'actual_successes': [], 'steps': []},
                'FULL_PROOF_32B': {'predicted_probs': [], 'actual_successes': [], 'steps': []},
            }

        history = rollout.get('history', [])
        for step in history:
            action = step.get('action', {})
            action_type = action.get('action_type', '')

            if action_type not in ['FULL_PROOF_8B', 'FULL_PROOF_32B']:
                continue

            # Get predicted probability from predictions (new format)
            # Structure: predictions.all_action_scores = [{"action": "FULL_PROOF_8B", "success_prob": 0.5, ...}, ...]
            predictions = step.get('predictions', {})
            all_scores = predictions.get('all_action_scores', [])

            # all_scores is a list of dicts
            pred_prob = None
            if isinstance(all_scores, list):
                for score_info in all_scores:
                    if score_info.get('action') == action_type:
                        pred_prob = score_info.get('success_prob')
                        break
            elif isinstance(all_scores, dict):
                # Old format: dict keyed by action type
                if action_type in all_scores:
                    pred_prob = all_scores[action_type].get('success_prob')

            # Also check result for actual success
            result = step.get('result', {})
            actual_success = 1 if result.get('success', False) else 0

            # If no predictions, skip
            if pred_prob is None:
                continue

            problem_stats[problem_id][action_type]['predicted_probs'].append(pred_prob)
            problem_stats[problem_id][action_type]['actual_successes'].append(actual_success)
            problem_stats[problem_id][action_type]['steps'].append(step)

    return problem_stats


def plot_calibration_scatter(predicted: Dict[str, float], actual: Dict[str, float], label: str):
    """Plot predicted vs actual success probability scatter plot.

    Uses Plotly for interactive hover tooltips showing problem IDs.
    """
    import plotly.graph_objects as go

    # Find common problems
    common_problems = sorted(set(predicted.keys()) & set(actual.keys()))

    if not common_problems:
        st.info(f"No common problems found between predictions and actual results for {label}.")
        return

    # Extract values for common problems
    problem_ids = list(common_problems)
    pred_values = [predicted[p] for p in common_problems]
    actual_values = [actual[p] for p in common_problems]

    # Create interactive scatter plot with Plotly
    fig = go.Figure()

    # Add scatter points with hover info
    fig.add_trace(go.Scatter(
        x=pred_values,
        y=actual_values,
        mode='markers',
        marker=dict(
            size=10,
            color='steelblue',
            opacity=0.7,
        ),
        text=problem_ids,
        customdata=np.column_stack([problem_ids, pred_values, actual_values]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Predicted: %{customdata[1]:.3f}<br>"
            "Actual: %{customdata[2]:.3f}<br>"
            "<extra></extra>"
        ),
        name='Problems'
    ))

    # Add diagonal line (perfect calibration)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect calibration',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=f'{label} Model Calibration ({len(common_problems)} problems)',
        xaxis_title='Predicted Probability (avg per problem)',
        yaxis_title='Actual Success Rate (from benchmark)',
        xaxis=dict(range=[-0.05, 1.05], scaleanchor='y', scaleratio=1),
        yaxis=dict(range=[-0.05, 1.05]),
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='closest',
    )

    st.plotly_chart(fig, use_container_width=True)

    # Compute calibration metrics
    if len(pred_values) > 1:
        correlation = np.corrcoef(pred_values, actual_values)[0, 1]
        mae = np.mean(np.abs(np.array(pred_values) - np.array(actual_values)))
        bias = np.mean(pred_values) - np.mean(actual_values)

        st.markdown(f"""
        - **Correlation**: {correlation:.3f}
        - **MAE**: {mae:.3f}
        - **Bias**: {bias:+.3f} (positive = overconfident)
        - **Mean predicted**: {np.mean(pred_values):.3f}
        - **Mean actual**: {np.mean(actual_values):.3f}
        """)


def render_calibration_summary(
    predicted_8b: Dict[str, float],
    actual_8b: Dict[str, float],
    predicted_32b: Dict[str, float],
    actual_32b: Dict[str, float]
):
    """Render summary of calibration across both models."""
    summary_data = []

    for label, predicted, actual in [
        ('FULL_PROOF_8B', predicted_8b, actual_8b),
        ('FULL_PROOF_32B', predicted_32b, actual_32b)
    ]:
        common = set(predicted.keys()) & set(actual.keys())
        if common:
            pred_vals = np.array([predicted[p] for p in common])
            actual_vals = np.array([actual[p] for p in common])
            mae = np.mean(np.abs(pred_vals - actual_vals))
            corr = np.corrcoef(pred_vals, actual_vals)[0, 1] if len(pred_vals) > 1 else 0

            summary_data.append({
                'Action': label,
                'N Problems': len(common),
                'Avg Predicted': f"{np.mean(pred_vals):.3f}",
                'Avg Actual': f"{np.mean(actual_vals):.3f}",
                'MAE': f"{mae:.3f}",
                'Bias': f"{np.mean(pred_vals) - np.mean(actual_vals):+.3f}",
                'Correlation': f"{corr:.3f}",
            })

    if summary_data:
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data available for summary.")


def render_problem_data_table(
    predicted_8b: Dict[str, float],
    actual_8b: Dict[str, float],
    predicted_32b: Dict[str, float],
    actual_32b: Dict[str, float]
):
    """Render raw problem-level data as a table."""
    rows = []

    for label, predicted, actual in [
        ('8B', predicted_8b, actual_8b),
        ('32B', predicted_32b, actual_32b)
    ]:
        common = set(predicted.keys()) & set(actual.keys())
        for problem_id in sorted(common):
            rows.append({
                'Problem': problem_id,
                'Action': label,
                'Predicted': predicted[problem_id],
                'Actual': actual[problem_id],
                'Error': predicted[problem_id] - actual[problem_id],
            })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No problem-level data available.")


# =============================================================================
# TREE ROLLOUTS CALIBRATION ANALYSIS
# =============================================================================

def render_tree_rollouts_calibration_tab(run_path: Path):
    """
    Render tree rollouts calibration analysis.

    Loads a saved TreeRolloutSuccessModel and evaluates it on tree rollout data,
    plotting predicted probability vs actual success rate per problem.
    """
    import matplotlib.pyplot as plt
    import sys
    from pathlib import Path

    st.markdown("## Tree Rollouts Calibration Analysis")
    st.markdown("""
    This analysis loads a trained TreeRolloutSuccessModel and evaluates its predictions
    against actual success rates from tree rollout data.

    - **X axis**: Predicted probability per problem (from model.predict())
    - **Y axis**: Actual success rate per problem (from tree rollout outcomes)
    """)

    # Path inputs
    repo_root = Path(__file__).parent.parent.parent.parent.parent

    # Default model path from run directory
    default_model_path = run_path / "tree_rollout_model" / "tree_rollout_success_model.pkl"

    # Try to load config to get tree rollouts path dynamically
    default_tree_rollouts_dir = ""
    try:
        from .config_viewer import find_config_file, load_config
        config_file = find_config_file(run_path)
        if config_file:
            config = load_config(config_file)
            if config:
                tree_rollouts_base = config.get('tree_rollouts_dir', '')
                tree_actions = config.get('tree_actions', ['FULL_PROOF_8B', 'FULL_PROOF_32B'])
                max_steps = config.get('tree_max_steps') or config.get('max_steps')
                budgets = config.get('budgets')

                if tree_rollouts_base:
                    # Import the function to generate config-specific subdir
                    if str(repo_root) not in sys.path:
                        sys.path.insert(0, str(repo_root))
                    from seed_prover.simulations.rl.tree_rollouts import get_tree_rollouts_config_dir

                    # Get the config-specific subdirectory
                    config_dir = get_tree_rollouts_config_dir(
                        tree_rollouts_base,
                        actions=[a.upper() for a in tree_actions],
                        budgets=budgets,
                        max_steps=max_steps,
                    )
                    default_tree_rollouts_dir = str(Path(config_dir) / "rollouts")
    except Exception as e:
        st.warning(f"Could not load config for tree rollouts path: {e}")

    with st.expander("📁 Data Paths", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            model_path = st.text_input(
                "Model Path (tree_rollout_success_model.pkl)",
                value=str(default_model_path)
            )
        with col2:
            tree_rollouts_dir = st.text_input(
                "Tree Rollouts Directory",
                value=default_tree_rollouts_dir
            )

    # Resolve paths
    model_file = repo_root / model_path if not Path(model_path).is_absolute() else Path(model_path)
    rollouts_path = repo_root / tree_rollouts_dir if not Path(tree_rollouts_dir).is_absolute() else Path(tree_rollouts_dir)

    # Check paths
    if not model_file.exists():
        st.warning(f"Model file not found: {model_file}")
        st.info("Run the RL training pipeline with `use_tree_rollout_models: true` to generate this model.")
        return

    if not rollouts_path.exists():
        st.warning(f"Tree rollouts directory not found: {rollouts_path}")
        return

    # Load model
    try:
        # Add repo root to path for imports
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from seed_prover.simulations.rl.learned_models import TreeRolloutSuccessModel

        with st.spinner("Loading model..."):
            model = TreeRolloutSuccessModel.load(str(model_file))
        st.success("Model loaded successfully")

        # Show model info
        with st.expander("🧠 Model Info"):
            st.markdown(f"""
            **Features 8B:** {model.features_8b}

            **Features 32B:** {model.features_32b}

            **Features to scale:** {model.features_to_scale}

            **Calibration 8B:** a={model.calibration_8b['a']:.4f}, b={model.calibration_8b['b']:.4f}

            **Calibration 32B:** a={model.calibration_32b['a']:.4f}, b={model.calibration_32b['b']:.4f}
            """)

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Load tree rollouts
    with st.spinner("Loading tree rollouts..."):
        rollouts = load_tree_rollouts(rollouts_path)

    if not rollouts:
        st.warning("No tree rollouts found.")
        return

    st.success(f"Loaded {len(rollouts)} tree rollouts")

    # Compute predictions for each step using the model
    with st.spinner("Computing predictions..."):
        calib_data = compute_model_predictions_on_rollouts(rollouts, model)

    if calib_data.empty:
        st.warning("No valid predictions computed.")
        return

    # Split by action type
    calib_8b = calib_data[calib_data['action_type'] == 'FULL_PROOF_8B'].copy()
    calib_32b = calib_data[calib_data['action_type'] == 'FULL_PROOF_32B'].copy()

    st.markdown(f"""
    - **8B**: {len(calib_8b)} problems
    - **32B**: {len(calib_32b)} problems
    """)

    # Plot
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 8B Model")
        if not calib_8b.empty:
            plot_tree_calibration_scatter(calib_8b, "8B")
        else:
            st.warning("No 8B data")

    with col2:
        st.markdown("### 32B Model")
        if not calib_32b.empty:
            plot_tree_calibration_scatter(calib_32b, "32B")
        else:
            st.warning("No 32B data")


def compute_model_predictions_on_rollouts(rollouts: List[Dict], model) -> pd.DataFrame:
    """
    Compute model predictions on tree rollout data and aggregate per problem.

    Computes avg LOGIT per problem first, then applies calibration once.
    This matches how calibration was fit during training.

    Returns DataFrame with columns:
        - problem_id
        - action_type
        - calibrated_prob (calibrated probability from avg logit)
        - actual_success_rate (actual success rate from the rollouts)
        - num_samples
    """
    from seed_prover.simulations.rl.actions import ActionType

    # Collect LOGITS per (problem, action_type) - not calibrated probs
    logits = {}       # (problem_id, action_type) -> list of logits
    successes = {}    # (problem_id, action_type) -> list of actual successes

    # Debug: show first rollout structure
    if rollouts:
        first_rollout = rollouts[0]
        if first_rollout.get('history'):
            first_step = first_rollout['history'][1] if len(first_rollout['history']) > 1 else first_rollout['history'][0]
            st.write(f"Sample state keys: {list(first_step.get('state', {}).keys())}")
            st.write(f"Model expects 8B features: {model.features_8b}")
            st.write(f"Model expects 32B features: {model.features_32b}")

    for rollout in rollouts:
        problem_id = rollout.get('problem_id', 'unknown')

        for step_idx, step in enumerate(rollout.get('history', [])):
            # Skip first step (all features are 0)
            if step_idx == 0:
                continue

            state = step.get('state', {})
            action = step.get('action', {})
            action_type_str = action.get('action_type', '')

            if action_type_str not in ['FULL_PROOF_8B', 'FULL_PROOF_32B']:
                continue

            # Create action object for model.predict_logit()
            try:
                action_type = ActionType[action_type_str]
            except KeyError:
                continue

            # Get RAW LOGIT from model (not calibrated)
            try:
                logit = model.predict_logit(state, action_type, None)
            except Exception as e:
                st.error(f"Error predicting logit for {problem_id}: {e}")
                continue

            # Track logit and actual outcome
            key = (problem_id, action_type_str)
            if key not in logits:
                logits[key] = []
                successes[key] = []

            logits[key].append(logit)
            successes[key].append(1 if step.get('action_success', False) else 0)

    # Debug: show how many logits were collected
    st.write(f"Collected logits for {len(logits)} (problem, action_type) pairs")

    # Aggregate to problem level: avg logit first, then calibrate once
    rows = []
    for (problem_id, action_type_str), logit_list in logits.items():
        actual = successes[(problem_id, action_type_str)]

        # Filter out invalid logits
        valid_logits = [l for l in logit_list if np.isfinite(l)]
        if not valid_logits:
            continue

        # Compute avg logit per problem
        avg_logit = np.mean(valid_logits)

        # Apply calibration ONCE on the avg logit
        calibrated_prob = model.calibrate_logit(avg_logit, action_type_str)

        rows.append({
            'problem_id': problem_id,
            'action_type': action_type_str,
            'avg_logit': avg_logit,
            'calibrated_prob': calibrated_prob,
            'actual_success_rate': np.mean(actual),
            'num_samples': len(valid_logits),
        })

    return pd.DataFrame(rows)


def load_tree_rollouts(rollouts_dir: Path) -> List[Dict]:
    """Load all tree rollout JSON files from directory structure."""
    rollouts = []
    for json_file in rollouts_dir.rglob('*.json'):
        # Skip summary files
        if 'summary' in json_file.name:
            continue
        try:
            with open(json_file, 'r') as f:
                rollout = json.load(f)
                rollout['file_path'] = str(json_file)
                rollouts.append(rollout)
        except Exception:
            continue
    return rollouts


def flatten_tree_rollouts_to_df(rollouts: List[Dict]) -> pd.DataFrame:
    """Flatten tree rollouts to DataFrame for model training."""
    rows = []

    for rollout in rollouts:
        problem_id = rollout.get('problem_id', 'unknown')
        rollout_success = rollout.get('success', False)

        for step_idx, step in enumerate(rollout.get('history', [])):
            # Skip first step (all features are 0)
            if step_idx == 0:
                continue

            # Skip steps not included in training
            if not step.get('include_in_training', True):
                continue

            state = step.get('state', {})
            action = step.get('action', {})

            row = {
                'problem_id': problem_id,
                'rollout_success': rollout_success,
                'step_idx': step_idx,
                'avg_cost_full_8b': state.get('avg_cost_full_8b', 0.0),
                'avg_cost_full_32b': state.get('avg_cost_full_32b', 0.0),
                'num_full_proof_8b_used': state.get('num_full_proof_8b_used', 0.0),
                'num_full_proof_32b_used': state.get('num_full_proof_32b_used', 0.0),
                'action_type': action.get('action_type', ''),
                'action_success': step.get('action_success', False),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def train_and_get_calibration(
    df_subset: pd.DataFrame,
    features: List[str],
    features_to_scale: List[str],
    scaler,
    name: str
) -> Optional[pd.DataFrame]:
    """Train logistic regression, fit Platt scaling, and compute calibration data."""
    from sklearn.linear_model import LogisticRegression
    from scipy.special import expit
    from scipy.optimize import minimize

    # Prepare features with partial scaling
    X = df_subset[features].copy()
    for i, feat in enumerate(features_to_scale):
        if feat in features:
            col_idx = features.index(feat)
            X.iloc[:, col_idx] = (X.iloc[:, col_idx] - scaler.mean_[i]) / scaler.scale_[i]

    y = df_subset['action_success'].astype(int)

    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)

    st.markdown(f"""
    **Model coefficients:**
    - Intercept: {model.intercept_[0]:.4f}
    - P(success) at baseline: {expit(model.intercept_[0]):.2%}
    """)

    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_[0],
    })
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

    # Compute avg logit per problem
    problem_logits = {}
    problem_successes = {}
    problem_counts = {}

    for problem_id in df_subset['problem_id'].unique():
        problem_data = df_subset[df_subset['problem_id'] == problem_id]

        # Prepare features
        X_prob = problem_data[features].copy()
        for i, feat in enumerate(features_to_scale):
            if feat in features:
                col_idx = features.index(feat)
                X_prob.iloc[:, col_idx] = (X_prob.iloc[:, col_idx] - scaler.mean_[i]) / scaler.scale_[i]

        # Compute logits
        logits = model.intercept_[0] + X_prob.values @ model.coef_[0]
        problem_logits[problem_id] = logits.mean()

        # Actual success rate from this action type's outcomes
        problem_successes[problem_id] = problem_data['action_success'].sum()
        problem_counts[problem_id] = len(problem_data)

    # Fit Platt scaling calibration: P = sigmoid(a * z_bar + b)
    z_bars = np.array([problem_logits[pid] for pid in problem_logits])
    y_true = np.array([problem_successes[pid] / problem_counts[pid] for pid in problem_logits])
    weights = np.array([problem_counts[pid] for pid in problem_logits])

    def calibration_loss(params):
        a, b = params
        p_pred = expit(a * z_bars + b)
        p_pred = np.clip(p_pred, 1e-7, 1 - 1e-7)
        loss = -y_true * np.log(p_pred) - (1 - y_true) * np.log(1 - p_pred)
        return (loss * weights).mean()

    result = minimize(calibration_loss, x0=[1.0, 0.0], method='L-BFGS-B')
    a, b = result.x

    st.markdown(f"""
    **Platt scaling calibration:** P = sigmoid({a:.3f} × z̄ + {b:.3f})
    """)

    # Create calibration DataFrame with calibrated probabilities
    rows = []
    for problem_id in problem_logits:
        avg_logit = problem_logits[problem_id]
        calibrated_prob = expit(a * avg_logit + b)
        rows.append({
            'problem_id': problem_id,
            'avg_logit': avg_logit,
            'calibrated_prob': calibrated_prob,
            'actual_success_rate': problem_successes[problem_id] / problem_counts[problem_id],
            'num_samples': problem_counts[problem_id],
        })

    return pd.DataFrame(rows)


def plot_tree_calibration_scatter(calib_df: pd.DataFrame, label: str):
    """Plot calibrated probability vs actual success rate from tree rollout data.

    Uses Plotly for interactive hover tooltips showing problem IDs.
    """
    import plotly.graph_objects as go

    if calib_df.empty:
        st.info(f"No data available for {label}")
        return

    calibrated_probs = calib_df['calibrated_prob'].values
    actual_rates = calib_df['actual_success_rate'].values
    problem_ids = calib_df['problem_id'].values
    num_samples = calib_df['num_samples'].values

    # Create interactive scatter plot with Plotly
    fig = go.Figure()

    # Add scatter points with hover info
    fig.add_trace(go.Scatter(
        x=calibrated_probs,
        y=actual_rates,
        mode='markers',
        marker=dict(
            size=10,
            color='steelblue',
            opacity=0.7,
        ),
        text=problem_ids,
        customdata=np.column_stack([problem_ids, num_samples, calibrated_probs, actual_rates]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Predicted: %{customdata[2]:.3f}<br>"
            "Actual: %{customdata[3]:.3f}<br>"
            "Samples: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
        name='Problems'
    ))

    # Add diagonal line (perfect calibration)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect calibration',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=f'{label} Tree Rollouts: Predicted vs Actual ({len(calib_df)} problems)',
        xaxis_title='Predicted Probability (calibrated)',
        yaxis_title='Actual Success Rate',
        xaxis=dict(range=[-0.05, 1.05], scaleanchor='y', scaleratio=1),
        yaxis=dict(range=[-0.05, 1.05]),
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='closest',
    )

    st.plotly_chart(fig, use_container_width=True)

    # Compute metrics
    if len(calib_df) > 1:
        correlation = np.corrcoef(calibrated_probs, actual_rates)[0, 1]
        mae = np.mean(np.abs(calibrated_probs - actual_rates))
        bias = np.mean(calibrated_probs) - np.mean(actual_rates)

        st.markdown(f"""
        **Calibration Metrics:**
        - **Correlation**: {correlation:.3f}
        - **MAE**: {mae:.3f}
        - **Bias**: {bias:+.3f} (positive = overconfident)
        - **Mean predicted prob**: {np.mean(calibrated_probs):.3f}
        - **Mean actual success rate**: {np.mean(actual_rates):.3f}
        """)

    # Data table
    with st.expander("📊 Problem-level Data"):
        display_df = calib_df[['problem_id', 'calibrated_prob', 'actual_success_rate', 'num_samples']].copy()
        display_df = display_df.sort_values('calibrated_prob', ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
