"""
RL Training Analysis Components.

This package provides a modular structure for the RL Training Analysis viewer,
with separate modules for each tab/functionality.
"""

from .utils import (
    get_available_rl_runs,
    get_available_baseline_datasets,
    get_available_baselines,
    load_baseline_data,
    load_rl_run_metadata,
    load_round_summary,
    load_round_per_problem,
    load_rollouts_cached,
    load_single_rollout,
    load_model_parameters,
    get_rollouts_file,
    compute_metrics_from_rollouts,
    compute_action_distribution,
    compute_action_success_rates,
    budget_selector,
    lambda_selector,
    load_full_proof_baselines,
)

from .overview import render_overview_tab
from .learning_curves import render_learning_curves_tab
from .solve_types import render_solve_types_tab
from .per_problem import render_per_problem_tab
from .lambda_analysis import render_lambda_analysis_tab
from .rollout_detail import render_rollout_detail_tab
from .model_parameters import render_model_parameters_tab
from .cost_quality import render_cost_quality_tab
from .config_viewer import render_config_tab
from .training_data import render_training_data_tab

__all__ = [
    # Utility functions
    'get_available_rl_runs',
    'get_available_baseline_datasets',
    'get_available_baselines',
    'load_baseline_data',
    'load_rl_run_metadata',
    'load_round_summary',
    'load_round_per_problem',
    'load_rollouts_cached',
    'load_single_rollout',
    'load_model_parameters',
    'get_rollouts_file',
    'compute_metrics_from_rollouts',
    'compute_action_distribution',
    'compute_action_success_rates',
    'budget_selector',
    'lambda_selector',
    'load_full_proof_baselines',
    # Tab renderers
    'render_overview_tab',
    'render_learning_curves_tab',
    'render_solve_types_tab',
    'render_per_problem_tab',
    'render_lambda_analysis_tab',
    'render_rollout_detail_tab',
    'render_model_parameters_tab',
    'render_cost_quality_tab',
    'render_config_tab',
    'render_training_data_tab',
]
