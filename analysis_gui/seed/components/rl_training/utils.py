"""
Shared utility functions for RL Training Analysis.

This module contains data loading, caching, and metric computation
functions used across the various tab components.
"""
import streamlit as st
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


def format_budget_display(budget: Optional[float], precision: int = 1) -> str:
    """Format budget for display in UI (e.g., '16.0M' or 'unlimited')."""
    if budget is None:
        return "unlimited"
    return f"{budget/1e6:.{precision}f}M"


def get_available_rl_runs(rollouts_dir: Path) -> List[Tuple[str, Path]]:
    """
    Discover available RL training runs from the rollouts directory.

    Returns list of (display_name, path) tuples, sorted by date descending.
    """
    runs = []

    if not rollouts_dir.exists():
        return runs

    # Look for timestamped directories (YYYY/MM/DD/HHMMSS pattern)
    for year_dir in sorted(rollouts_dir.iterdir(), reverse=True):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue

        for month_dir in sorted(year_dir.iterdir(), reverse=True):
            if not month_dir.is_dir():
                continue

            for day_dir in sorted(month_dir.iterdir(), reverse=True):
                if not day_dir.is_dir():
                    continue

                for time_dir in sorted(day_dir.iterdir(), reverse=True):
                    if not time_dir.is_dir():
                        continue

                    # Check if this is an RL run (has iteration_results.json or roundX folders)
                    has_iteration_results = (time_dir / "iteration_results.json").exists()
                    has_rounds = any((time_dir / f"round{i}").exists() for i in range(10))

                    if has_iteration_results or has_rounds:
                        display_name = f"{year_dir.name}/{month_dir.name}/{day_dir.name}/{time_dir.name}"
                        runs.append((display_name, time_dir))

    return runs


def get_available_baseline_datasets(rollouts_dir: Path) -> List[str]:
    """Get list of available baseline datasets (e.g., 'putnam', 'minif2f').

    The baseline structure is: rollouts/baselines/<dataset>/<strategy>/...
    """
    baselines_dir = rollouts_dir / "baselines"
    if not baselines_dir.exists():
        return []

    datasets = []
    for d in baselines_dir.iterdir():
        if d.is_dir() and not d.name.startswith('.'):
            # Check if this looks like a dataset (has subdirectories that could be strategies)
            # A dataset directory should have at least one subdirectory
            has_subdirs = any(sd.is_dir() for sd in d.iterdir() if not sd.name.startswith('.'))
            if has_subdirs:
                datasets.append(d.name)

    return sorted(datasets)


def get_available_baselines(rollouts_dir: Path, dataset: Optional[str] = None) -> List[str]:
    """Get list of available baseline strategies.

    Args:
        rollouts_dir: Root rollouts directory
        dataset: Optional dataset name (e.g., 'putnam'). If provided, looks in
                 rollouts/baselines/<dataset>/ for strategies.
    """
    if dataset:
        baselines_dir = rollouts_dir / "baselines" / dataset
    else:
        baselines_dir = rollouts_dir / "baselines"

    if not baselines_dir.exists():
        return []

    return [d.name for d in baselines_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]


def load_baseline_data(rollouts_dir: Path, strategy: str, dataset: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load baseline data for a given strategy.

    Args:
        rollouts_dir: Root rollouts directory
        strategy: Baseline strategy name (e.g., 'full_proof')
        dataset: Optional dataset name (e.g., 'putnam'). If provided, looks in
                 rollouts/baselines/<dataset>/<strategy>/
    """
    if dataset:
        base_path = rollouts_dir / "baselines" / dataset / strategy
    else:
        base_path = rollouts_dir / "baselines" / strategy

    if not base_path.exists():
        return None

    data = {}

    # Load summary
    summary_file = base_path / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data['summary'] = json.load(f)

    # Load per-problem summary
    per_problem_file = base_path / "per_problem_summary.json"
    if per_problem_file.exists():
        with open(per_problem_file, 'r') as f:
            data['per_problem'] = json.load(f)

    # Load rollouts (may be large - load lazily)
    rollouts_file = base_path / "rollouts.json"
    if rollouts_file.exists():
        data['rollouts_path'] = rollouts_file

    return data


def load_rl_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """Load RL run metadata without loading full rollouts."""
    metadata = {
        'path': run_dir,
        'rounds': [],
        'iteration_results': None,
        'budget_lambda_comparison': None,
    }

    # Load iteration results
    iter_results_file = run_dir / "iteration_results.json"
    if iter_results_file.exists():
        with open(iter_results_file, 'r') as f:
            metadata['iteration_results'] = json.load(f)

    # Load budget lambda comparison
    budget_lambda_file = run_dir / "budget_lambda_comparison.json"
    if budget_lambda_file.exists():
        with open(budget_lambda_file, 'r') as f:
            metadata['budget_lambda_comparison'] = json.load(f)

    # Find available rounds
    for i in range(20):  # Check up to 20 rounds
        round_dir = run_dir / f"round{i}"
        if round_dir.exists():
            metadata['rounds'].append(i)

    return metadata


def load_round_summary(run_dir: Path, round_id: int) -> Optional[Dict[str, Any]]:
    """Load summary for a specific round."""
    round_dir = run_dir / f"round{round_id}"
    summary_file = round_dir / "summary.json"

    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    return None


def load_round_per_problem(run_dir: Path, round_id: int) -> Optional[Dict[str, Any]]:
    """Load per-problem summary for a specific round."""
    round_dir = run_dir / f"round{round_id}"
    per_problem_file = round_dir / "per_problem_summary.json"

    if per_problem_file.exists():
        with open(per_problem_file, 'r') as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300, show_spinner=False)
def _load_rollouts_raw(rollouts_path: str) -> List[Dict]:
    """Load rollouts from file (cached, no filtering)."""
    with open(rollouts_path, 'r') as f:
        return json.load(f)


_NO_BUDGET_FILTER = object()  # Sentinel to indicate no filtering

def load_rollouts_cached(rollouts_path: str, budget_filter: Optional[float] = _NO_BUDGET_FILTER) -> List[Dict]:
    """Load rollouts with optional budget filter (exact match).

    Optimization: The raw file is cached separately, so loading multiple budgets
    from the same file only reads the file once.

    Supports both legacy rollouts.json and new rollouts_summary.json format.

    Args:
        rollouts_path: Path to rollouts JSON file
        budget_filter: Budget value to filter by. None means filter for unlimited budget.
                       Default (_NO_BUDGET_FILTER) means no filtering at all.
    """
    rollouts = _load_rollouts_raw(rollouts_path)

    if budget_filter is not _NO_BUDGET_FILTER:
        # Filter by exact budget match (including None for unlimited)
        rollouts = [r for r in rollouts if r.get('budget') == budget_filter]

    return rollouts


def _format_lambda_dirname(lambda_val: float) -> str:
    """Format lambda value as directory name (e.g., 1e-08, 5e-07)."""
    return f"{lambda_val:.0e}".replace("+", "")


def _format_budget_dirname(budget: Optional[float]) -> str:
    """Format budget as directory name (e.g., 16M, 32M, or 'unlimited')."""
    if budget is None:
        return "unlimited"
    return f"{int(budget / 1e6)}M"


def get_rollouts_file(round_dir: Path) -> Optional[Path]:
    """
    Get the rollouts file for a round directory.

    Prefers rollouts_summary.json (new format) over rollouts.json (legacy).
    """
    summary_file = round_dir / "rollouts_summary.json"
    if summary_file.exists():
        return summary_file

    legacy_file = round_dir / "rollouts.json"
    if legacy_file.exists():
        return legacy_file

    return None


@st.cache_data(ttl=300, show_spinner=False)
def load_single_rollout(
    round_dir: str,
    problem_id: str,
    lambda_val: float,
    budget: float,
    seed_idx: int
) -> Optional[Dict]:
    """
    Load a single full rollout from the file structure.

    New path: round_dir/rollouts/<problem_id>/<budget>/rollouts.json
    Legacy path: round_dir/rollouts/<problem_id>/<lambda>/<budget>/<seed>.json

    Falls back to loading from legacy rollouts.json if individual files don't exist.
    """
    round_path = Path(round_dir)

    # Try new consolidated structure first: rollouts/<problem>/<budget>/rollouts.json
    new_rollout_file = (
        round_path / "rollouts" / problem_id /
        _format_budget_dirname(budget) /
        "rollouts.json"
    )

    if new_rollout_file.exists():
        with open(new_rollout_file, 'r') as f:
            rollouts = json.load(f)
        # Find the specific rollout by lambda and seed
        for r in rollouts:
            if (abs(float(r.get('lambda', 0)) - lambda_val) < 1e-12 and
                r.get('seed_idx') == seed_idx):
                return r

    # Try legacy per-seed structure: rollouts/<problem>/<lambda>/<budget>/<seed>.json
    legacy_per_seed_file = (
        round_path / "rollouts" / problem_id /
        _format_lambda_dirname(lambda_val) /
        _format_budget_dirname(budget) /
        f"{seed_idx}.json"
    )

    if legacy_per_seed_file.exists():
        with open(legacy_per_seed_file, 'r') as f:
            return json.load(f)

    # Fallback to legacy rollouts.json
    legacy_file = round_path / "rollouts.json"
    if legacy_file.exists():
        rollouts = _load_rollouts_raw(str(legacy_file))
        for r in rollouts:
            if (r.get('problem_id') == problem_id and
                abs(float(r.get('lambda', 0)) - lambda_val) < 1e-12 and
                r.get('budget') == budget and
                r.get('seed_idx') == seed_idx):
                return r

    return None


def load_model_parameters(model_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load trained model parameters from a directory.

    Args:
        model_dir: Directory containing success_models.pkl and cost_models.pkl
                   OR tree_rollout_success_model.pkl (for initial tree rollout model)

    Returns:
        Dict with 'success' and 'cost' keys, each containing model data,
        or None if files don't exist.

    The returned structure is normalized to:
        {
            'success': {
                'models': {action_type: model, ...},
                'scaler': scaler,
                'feature_names': [...],
                'features_8b': [...],  # optional, for tree rollout models
                'features_32b': [...], # optional, for tree rollout models
            },
            'cost': {...}
        }
    """
    result = {}

    # Check for standard names first, then tree rollout names
    success_file = model_dir / "success_models.pkl"
    is_tree_rollout_format = False
    if not success_file.exists():
        success_file = model_dir / "tree_rollout_success_model.pkl"
        is_tree_rollout_format = True

    cost_file = model_dir / "cost_models.pkl"
    if not cost_file.exists():
        cost_file = model_dir / "tree_rollout_cost_model.pkl"

    if success_file.exists():
        try:
            with open(success_file, 'rb') as f:
                data = pickle.load(f)

            # Normalize tree rollout format to standard format
            if is_tree_rollout_format and 'model_8b' in data:
                result['success'] = _normalize_tree_rollout_model(data)
            else:
                result['success'] = data
        except Exception as e:
            st.warning(f"Error loading success models: {e}")

    if cost_file.exists():
        try:
            with open(cost_file, 'rb') as f:
                result['cost'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Error loading cost models: {e}")

    return result if result else None


def _normalize_tree_rollout_model(data: Dict) -> Dict:
    """
    Normalize tree rollout model format to standard model format.

    Tree rollout format:
        {'model_8b': ..., 'model_32b': ..., 'scaler': ..., 'features_8b': [...], ...}

    Standard format:
        {'models': {ActionType: model, ...}, 'scaler': ..., 'feature_names': [...]}
    """
    # Import ActionType for keying
    try:
        from seed_prover.simulations.rl.action_types import ActionType
        models = {
            ActionType.FULL_PROOF_8B: data.get('model_8b'),
            ActionType.FULL_PROOF_32B: data.get('model_32b'),
        }
    except ImportError:
        # Fallback to string keys if ActionType not available
        models = {
            'FULL_PROOF_8B': data.get('model_8b'),
            'FULL_PROOF_32B': data.get('model_32b'),
        }

    # Remove None models
    models = {k: v for k, v in models.items() if v is not None}

    # Combine feature lists for display (but keep originals too)
    features_8b = data.get('features_8b', [])
    features_32b = data.get('features_32b', [])
    # Use 32B features as the "main" feature list since it's typically a superset
    feature_names = features_32b if features_32b else features_8b

    return {
        'models': models,
        'scaler': data.get('scaler'),
        'feature_names': feature_names,
        'features_8b': features_8b,
        'features_32b': features_32b,
        'features_to_scale': data.get('features_to_scale', []),
        'calibration_8b': data.get('calibration_8b'),
        'calibration_32b': data.get('calibration_32b'),
    }


def extract_model_coefficients(model_data: Dict[str, Any], model_type: str) -> List[Dict]:
    """
    Extract coefficients from loaded model data.

    Args:
        model_data: Loaded pickle data containing 'models' and 'scaler' (shared)
        model_type: 'success' or 'cost'

    Returns:
        List of dicts with action_type, feature_name, coefficient, etc.
    """
    if not model_data:
        return []

    models = model_data.get('models', {})
    scaler = model_data.get('scaler', None)  # Single shared scaler
    feature_names = model_data.get('feature_names', None)

    # If feature_names not in model, try to infer from scaler
    if not feature_names and scaler:
        num_features = scaler.n_features_in_
        feature_names = [f"feature_{i}" for i in range(num_features)]

    rows = []

    for action_type, model in models.items():
        action_name = action_type.name if hasattr(action_type, 'name') else str(action_type)

        # Get coefficients
        if hasattr(model, 'coef_'):
            coefs = model.coef_.flatten()
            intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_

            for i, coef in enumerate(coefs):
                feat_name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
                row = {
                    'action_type': action_name,
                    'feature': feat_name,
                    'coefficient': coef,
                    'model_type': model_type,
                }
                # Add shared scaler parameters
                if scaler and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                    row['scaler_mean'] = scaler.mean_[i]
                    row['scaler_scale'] = scaler.scale_[i]
                rows.append(row)

            # Add intercept as separate row
            rows.append({
                'action_type': action_name,
                'feature': '(intercept)',
                'coefficient': intercept,
                'model_type': model_type,
            })

    return rows


def load_all_round_rollouts(
    run_dir: Path,
    rounds: List[int],
    target_budget: float,
    progress_container=None
) -> Dict[int, List[Dict]]:
    """
    Load rollouts for all rounds with progress indication.

    Returns dict mapping round_id -> rollouts list

    Supports both new rollouts_summary.json and legacy rollouts.json format.
    """
    all_rollouts = {}

    if progress_container:
        progress_bar = progress_container.progress(0, text="Loading rollouts...")

    for i, round_id in enumerate(rounds):
        if progress_container:
            progress_bar.progress(
                (i + 1) / len(rounds),
                text=f"Loading round {round_id}/{len(rounds)-1}..."
            )

        round_dir = run_dir / f"round{round_id}"
        rollouts_file = get_rollouts_file(round_dir)

        if rollouts_file is not None:
            rollouts = load_rollouts_cached(str(rollouts_file), target_budget)
            all_rollouts[round_id] = rollouts

    if progress_container:
        progress_bar.empty()

    return all_rollouts


@st.cache_data(ttl=300)
def load_baseline_rollouts_at_budget(rollouts_path: str, budget_cutoff: float) -> List[Dict]:
    """
    Load baseline rollouts and evaluate success at a budget cutoff.

    Baselines run to completion (e.g., 1B budget) but we can evaluate
    what success rate they'd have at smaller budgets by checking final_cost.

    A rollout is considered successful at budget_cutoff if:
    - success=True AND final_cost <= budget_cutoff
    """
    with open(rollouts_path, 'r') as f:
        rollouts = json.load(f)

    # Add a field to indicate if it would succeed at this budget
    for r in rollouts:
        original_success = r.get('success', False)
        final_cost = r.get('final_cost', float('inf'))
        # Success at this budget = originally successful AND cost within budget
        r['success_at_budget'] = original_success and final_cost <= budget_cutoff

    return rollouts


def compute_metrics_from_rollouts(rollouts: List[Dict], use_budget_success: bool = False) -> Dict[str, Any]:
    """
    Compute aggregate metrics from rollouts.

    Args:
        rollouts: List of rollout dictionaries (supports both summary and full format)
        use_budget_success: If True, use 'success_at_budget' field instead of 'success'
    """
    if not rollouts:
        return {}

    success_key = 'success_at_budget' if use_budget_success else 'success'
    num_successful = sum(1 for r in rollouts if r.get(success_key, False))
    total = len(rollouts)

    # Count unique problems solved
    solved_problems = set()
    for r in rollouts:
        if r.get(success_key, False):
            problem_id = r.get('problem_id', r.get('origin_problem_id', ''))
            if problem_id:
                solved_problems.add(problem_id)

    costs = [r.get('final_cost', 0) for r in rollouts]
    # Support both summary (num_steps) and full (history) format
    steps = [r.get('num_steps', len(r.get('history', []))) for r in rollouts]
    proven_lemmas = [r.get('num_proven_lemmas', 0) for r in rollouts]
    used_lemmas = [r.get('num_used_lemmas', 0) for r in rollouts]

    return {
        'success_rate': num_successful / total if total > 0 else 0,
        'num_successful': num_successful,
        'unique_solved': len(solved_problems),
        'total_rollouts': total,
        'avg_cost': np.mean(costs) if costs else 0,
        'avg_steps': np.mean(steps) if steps else 0,
        'avg_proven_lemmas': np.mean(proven_lemmas) if proven_lemmas else 0,
        'avg_used_lemmas': np.mean(used_lemmas) if used_lemmas else 0,
    }


def compute_action_distribution(rollouts: List[Dict]) -> Dict[str, int]:
    """Compute action type distribution from rollouts.

    Supports both summary format (action_counts field) and full format (history).
    """
    action_counts = defaultdict(int)

    for r in rollouts:
        # Prefer action_counts from summary format
        if 'action_counts' in r:
            for action_type, count in r['action_counts'].items():
                action_counts[action_type] += count
        else:
            # Fall back to iterating history (legacy/full format)
            history = r.get('history', [])
            for step in history:
                action = step.get('action', {})
                action_type = action.get('action_type', 'unknown')
                action_counts[action_type] += 1

    return dict(action_counts)


def compute_action_success_rates(rollouts: List[Dict]) -> Dict[str, Dict[str, int]]:
    """
    Compute success rates per action type from rollouts.

    Supports both summary format (action_counts + action_success_counts) and full format (history).

    Returns:
        Dict mapping action_type -> {'total': N, 'successful': M}
    """
    action_stats = defaultdict(lambda: {'total': 0, 'successful': 0})

    for r in rollouts:
        # Prefer summary fields if available
        if 'action_counts' in r and 'action_success_counts' in r:
            for action_type, count in r['action_counts'].items():
                action_stats[action_type]['total'] += count
            for action_type, count in r.get('action_success_counts', {}).items():
                action_stats[action_type]['successful'] += count
        else:
            # Fall back to iterating history (legacy/full format)
            history = r.get('history', [])
            for step in history:
                action = step.get('action', {})
                action_type = action.get('action_type', 'unknown')
                result = step.get('result', {})
                success = result.get('success', False)

                action_stats[action_type]['total'] += 1
                if success:
                    action_stats[action_type]['successful'] += 1

    return dict(action_stats)


def budget_selector(budgets: List[Optional[float]], key_suffix: str) -> Optional[float]:
    """Helper to create a budget selector dropdown that doesn't reload the page.

    Args:
        budgets: List of budget values (None represents unlimited/no constraint)
        key_suffix: Unique key suffix for the widget

    Returns:
        Selected budget value, or None for unlimited
    """
    budget_options = {"unlimited" if b is None else f"{b/1e6:.1f}M": b for b in budgets}
    selected = st.selectbox(
        "Select Budget",
        options=list(budget_options.keys()),
        index=len(budget_options) - 1 if budget_options else 0,
        key=f"budget_select_{key_suffix}"
    )
    return budget_options.get(selected, budgets[-1] if budgets else None)


def lambda_selector(lambda_values: List[float], key_suffix: str, include_all: bool = True) -> float:
    """Helper to create a lambda selector dropdown.

    Args:
        lambda_values: List of lambda values to choose from
        key_suffix: Unique key suffix for the widget
        include_all: If True, adds an "All" option that returns None

    Returns:
        Selected lambda value, or None if "All" is selected
    """
    if include_all:
        lambda_options = {"All": None}
    else:
        lambda_options = {}

    for lv in sorted(lambda_values):
        if lv == 0:
            lambda_options["0"] = 0.0
        else:
            lambda_options[f"{lv:.0e}"] = lv

    selected = st.selectbox(
        "Select Lambda",
        options=list(lambda_options.keys()),
        index=0,  # Default to "All" or first value
        key=f"lambda_select_{key_suffix}"
    )
    return lambda_options.get(selected)


def load_full_proof_baselines(rollouts_dir: Path, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load full_proof baselines from the nested structure.

    The structure is:
        rollouts/baselines/<dataset>/full_proof/
            combined_summary.json  <- Pre-aggregated data for all models/budgets
            FULL_PROOF_8B/
                max_4/summary.json
                max_8/summary.json
                ...
            FULL_PROOF_32B/
                max_4/summary.json
                ...

    Args:
        rollouts_dir: Root rollouts directory
        dataset: Optional dataset name (e.g., 'putnam'). If provided, looks in
                 rollouts/baselines/<dataset>/full_proof/

    Returns:
        List of dicts with keys: baseline, max_attempts, success_rate, avg_cost, etc.
    """
    if dataset:
        full_proof_dir = rollouts_dir / "baselines" / dataset / "full_proof"
    else:
        full_proof_dir = rollouts_dir / "baselines" / "full_proof"

    if not full_proof_dir.exists():
        return []

    # Prefer combined_summary.json if it exists
    combined_summary = full_proof_dir / "combined_summary.json"
    if combined_summary.exists():
        with open(combined_summary, 'r') as f:
            return json.load(f)

    # Otherwise, scan subdirectories and aggregate
    results = []
    for model_dir in full_proof_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue

        for max_dir in model_dir.iterdir():
            if not max_dir.is_dir() or not max_dir.name.startswith('max_'):
                continue

            summary_file = max_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    results.append(summary)

    return results
