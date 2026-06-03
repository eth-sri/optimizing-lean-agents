"""Configuration loading and policy/runner construction from YAML."""

import inspect
import itertools
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .policies.base import Policy
from .policies.fixed import FixedPolicy
from .policies.cost_models import RunningAverageCostModel
from .policies.cost_quality import CostQualityPolicy
from .policies.quality import (
    OracleProbabilityModel,
    OracleNoiseProbabilityModel,
    PretrainedLogisticModel,
)
from .problem import SimulatedProblem
from .features import (
    StateTracker, ComputedFeature, NoisyOracleFeature, OracleFeature,
    AttemptCountFeature, AverageCostFeature, PredictedProbComputed,
    CodePreviewFeature,
    NormalizedSimilarityFeature,
    ErrorDiversityFeature, UniqueProofsFeature, SubgoalRepetitionFeature,
)
from .data.loader import load_problems


FEATURE_REGISTRY = {
    "oracle": OracleFeature,
    "noisy_oracle": NoisyOracleFeature,
    "attempt_count": AttemptCountFeature,
    "avg_cost": AverageCostFeature,
    "code_preview": CodePreviewFeature,
    "normalized_similarity": NormalizedSimilarityFeature,
    "error_diversity": ErrorDiversityFeature,
    "unique_proofs": UniqueProofsFeature,
    "subgoal_repetition": SubgoalRepetitionFeature,
}

COMPUTED_REGISTRY = {
    "predicted_prob": PredictedProbComputed,
}


POLICY_REGISTRY = {
    "fixed": FixedPolicy,
}

PROB_MODEL_REGISTRY = {
    "oracle": OracleProbabilityModel,
    "noisy_oracle": OracleNoiseProbabilityModel,
    "pretrained_logistic": PretrainedLogisticModel,
}

COST_MODEL_REGISTRY = {
    "running_average": RunningAverageCostModel,
}


def build_prob_model(prob_model_type: str, params: dict, problems=None,
                     full_proof_sources=None):
    """Construct a probability model from type string and params.

    Returns the fitted model. Fitting can be expensive, so callers can cache
    the result and pass it via build_policy(prob_model=...).
    """
    if prob_model_type not in PROB_MODEL_REGISTRY:
        raise ValueError(f"Unknown prob_model: {prob_model_type}. "
                         f"Available: {list(PROB_MODEL_REGISTRY.keys())}")

    prob_model_cls = PROB_MODEL_REGISTRY[prob_model_type]
    if prob_model_type == "pretrained_logistic":
        prob_model_kwargs = {
            "model_path": str(params["model_path"]),
            "sigma": float(params.get("sigma", 0.1)),
        }
        if "feature_mapping" in params:
            prob_model_kwargs["feature_mapping"] = params["feature_mapping"]
    elif prob_model_type == "noisy_oracle":
        prob_model_kwargs = {
            "problems": problems,
            "sigma": float(params.get("sigma", 0.0)),
        }
    else:
        prob_model_kwargs = {"problems": problems}
    return prob_model_cls(**prob_model_kwargs)


def build_policy(policy_type: str, params: dict, problems=None, allowed_actions=None,
                  full_proof_sources=None, prob_model=None) -> "Policy":
    """Construct a policy from type string and params.

    Args:
        policy_type: Policy type name ("fixed", "cost_quality")
        params: Policy constructor kwargs
        problems: Required for "cost_quality" policy type
        allowed_actions: Optional list of allowed action strings for cost_quality policy
        full_proof_sources: Optional dict of model_name -> path (accepted for API
            compatibility; unused by the current prob_models)
        prob_model: Optional pre-built probability model (avoids re-fitting)

    Returns:
        Instantiated Policy
    """
    if policy_type == "cost_quality":
        if problems is None:
            raise ValueError("cost_quality policy requires problems")

        prob_model_type = params.get("prob_model", "oracle")
        cost_model_type = params.get("cost_model", "running_average")
        lambda_val = float(params.get("lambda_val", 0.0))

        # lambda=0 short-circuit: decision is fixed (always pick highest-p real
        # action over decompose, modulo hot_start), so features add nothing.
        # Swap in a FixedPolicy that decomposes to max and proves every lemma.
        if lambda_val == 0.0:
            hot_start = params.get("hot_start") or {}
            max_breakdowns = int(params.get("max_breakdowns", 8))
            max_corrections = int(params.get("max_corrections", 0))
            # Respect allowed_actions: only prove with models for which prove_{m}
            # is allowed. If allowed_actions is None, allow all hot_start models.
            allowed_models = None
            if allowed_actions is not None:
                allowed_models = {a.split("prove_", 1)[1] for a in allowed_actions
                                  if a.startswith("prove_")}
            candidate_models = list(hot_start.keys()) or ["8b"]
            if allowed_models is not None:
                candidate_models = [m for m in candidate_models if m in allowed_models]
            if not candidate_models:
                candidate_models = ["8b"]
            # Budget per model: 64 attempts per lemma (matches the max of the
            # fixed-baseline sweep).
            breakdown_proof_budget = {m: 64 for m in candidate_models}
            full_proof_budget = {m: 0 for m in candidate_models}
            policy = FixedPolicy(
                full_proof_budget=full_proof_budget,
                max_breakdowns=max_breakdowns,
                breakdown_proof_budget=breakdown_proof_budget,
                max_corrections=max_corrections,
            )
            return policy

        if cost_model_type not in COST_MODEL_REGISTRY:
            raise ValueError(f"Unknown cost_model: {cost_model_type}. "
                             f"Available: {list(COST_MODEL_REGISTRY.keys())}")

        if prob_model is None:
            prob_model = build_prob_model(prob_model_type, params, problems=problems,
                                          full_proof_sources=full_proof_sources)
        # Update sigma on cached prob_model for sweep reuse across configs
        if hasattr(prob_model, 'sigma') and 'sigma' in params:
            prob_model.sigma = float(params['sigma'])
        cost_model_cls = COST_MODEL_REGISTRY[cost_model_type]
        if cost_model_type == "running_average":
            cost_model_kwargs = {}
            if "default_cost" in params:
                cost_model_kwargs["default_cost"] = float(params["default_cost"])
            cost_model = cost_model_cls(**cost_model_kwargs)
        else:
            cost_model = cost_model_cls(problems=problems)

        hot_start = params.get("hot_start")  # e.g. {"8b": 1} or None

        policy = CostQualityPolicy(
            prob_model=prob_model,
            cost_model=cost_model,
            lambda_val=lambda_val,
            allowed_actions=allowed_actions,
            hot_start=hot_start,
        )
        policy.set_problems(problems)
        return policy

    elif policy_type in POLICY_REGISTRY:
        return POLICY_REGISTRY[policy_type](**params)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. "
                         f"Available: {list(POLICY_REGISTRY.keys()) + ['cost_quality']}")


def build_state_tracker(config: dict, state_tracker_params: Optional[dict] = None) -> Optional[StateTracker]:
    """Build a StateTracker from config, if state_tracker section exists.

    Args:
        config: Full config dict (must have state_tracker section with features list)
        state_tracker_params: Optional override params (e.g. {"sigma": 0.1} from sweep)

    Returns:
        StateTracker instance, or None if no state_tracker config
    """
    st_cfg = config.get("state_tracker")
    if st_cfg is None:
        return None

    features_cfg = st_cfg.get("features", [])
    computed_cfg = st_cfg.get("computed", [])

    if not features_cfg and not computed_cfg:
        return None

    features = []
    for feat_cfg in features_cfg:
        feat_type = feat_cfg["type"]
        if feat_type not in FEATURE_REGISTRY:
            raise ValueError(f"Unknown feature type: {feat_type}. "
                             f"Available: {list(FEATURE_REGISTRY.keys())}")
        feat_cls = FEATURE_REGISTRY[feat_type]

        # Collect kwargs: start from config, override with state_tracker_params
        kwargs = {k: v for k, v in feat_cfg.items() if k != "type"}
        if state_tracker_params is not None:
            kwargs.update(state_tracker_params)

        # Filter to only kwargs the constructor accepts
        sig = inspect.signature(feat_cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        features.append(feat_cls(**kwargs))

    computed = []
    for comp_cfg in computed_cfg:
        comp_type = comp_cfg["type"]
        if comp_type not in COMPUTED_REGISTRY:
            raise ValueError(f"Unknown computed type: {comp_type}. "
                             f"Available: {list(COMPUTED_REGISTRY.keys())}")
        comp_cls = COMPUTED_REGISTRY[comp_type]

        kwargs = {k: v for k, v in comp_cfg.items() if k != "type"}
        if state_tracker_params is not None:
            kwargs.update(state_tracker_params)

        sig = inspect.signature(comp_cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        computed.append(comp_cls(**kwargs))

    return StateTracker(features, computed=computed)


def load_config(config_path: str) -> dict:
    """Load and validate a simulation config YAML.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Parsed config dict
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required top-level keys
    for key in ("data", "policy", "simulation", "output"):
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate policy
    policy_cfg = config["policy"]
    if "type" not in policy_cfg:
        raise ValueError("policy.type is required")

    # Validate output
    output_cfg = config["output"]
    if "name" not in output_cfg:
        raise ValueError("output.name is required")

    return config


def extract_prove_models(allowed_actions: Optional[List[str]]) -> Set[str]:
    """Extract model names from prove_* entries in allowed_actions."""
    if not allowed_actions:
        return set()
    return {a.removeprefix("prove_") for a in allowed_actions if a.startswith("prove_")}


def validate_breakdown_model_filter(
    agent_config: Optional[Dict],
    allowed_actions: Optional[List[str]],
) -> Optional[Set[str]]:
    """Validate config and derive breakdown model filter from allowed_actions.

    With shared_breakdowns=false and multiple agent sources, exactly one
    prove model must be specified in allowed_actions. Returns the set of
    model names to filter breakdown templates by, or None if no filtering needed.

    Raises ValueError if the configuration is invalid.
    """
    if not agent_config:
        return None

    sources = agent_config.get("sources", {})
    shared_breakdowns = agent_config.get("shared_breakdowns", False)

    if shared_breakdowns or len(sources) <= 1:
        return None

    # Multiple agent sources, not shared — must have exactly one prove model
    prove_models = extract_prove_models(allowed_actions)

    if len(prove_models) != 1:
        raise ValueError(
            f"With shared_breakdowns=false and multiple agent sources "
            f"({list(sources.keys())}), allowed_actions must include exactly "
            f"one prove_<model> action. "
            f"Got allowed_actions={allowed_actions}"
        )

    return prove_models


def build_from_config(config: dict) -> Tuple[List[SimulatedProblem], Policy, dict]:
    """Build problems, policy, and runner settings from config dict.

    Args:
        config: Parsed config dict (from load_config)

    Returns:
        (problems, policy, runner_kwargs) tuple where runner_kwargs
        can be passed to SimulationRunner.__init__
    """
    # 1. Load data
    data_cfg = config["data"]
    full_proof_sources = data_cfg.get("full_proof")
    agent_sources = data_cfg.get("agent")

    seed = config["simulation"].get("seeds", [42])[0]

    load_code = data_cfg.get("load_code", False)

    problems = load_problems(
        full_proof_sources=full_proof_sources,
        agent_config=agent_sources,
        seed=seed,
        load_code=load_code,
    )

    # Apply max_problems limit
    max_problems = config["simulation"].get("max_problems")
    if max_problems is not None:
        problems = problems[:max_problems]

    # 2. Validate breakdown model filter
    policy_cfg = config["policy"]
    allowed_actions = policy_cfg.get("allowed_actions")
    breakdown_model_filter = validate_breakdown_model_filter(agent_sources, allowed_actions)

    # 3. Build policy
    policy_type = policy_cfg["type"]
    policy_params = policy_cfg.get("params", {})
    policy = build_policy(
        policy_type, policy_params, problems=problems,
        allowed_actions=allowed_actions,
        full_proof_sources=full_proof_sources,
    )

    # 4. Build state tracker
    state_tracker = build_state_tracker(config)

    # 5. Build runner kwargs
    sim_cfg = config["simulation"]
    runner_kwargs = {
        "max_steps": sim_cfg.get("max_steps", 1000),
    }
    if sim_cfg.get("max_breakdowns") is not None:
        runner_kwargs["max_breakdowns"] = sim_cfg["max_breakdowns"]
    if sim_cfg.get("max_corrections") is not None:
        runner_kwargs["max_corrections"] = sim_cfg["max_corrections"]
    if breakdown_model_filter is not None:
        runner_kwargs["breakdown_model_filter"] = breakdown_model_filter
    if state_tracker is not None:
        runner_kwargs["state_tracker"] = state_tracker

    return problems, policy, runner_kwargs


def resolve_seeds(sim_cfg: dict) -> List[int]:
    """Resolve seeds from simulation config, supporting both 'seeds' list and 'n_seeds'.

    Args:
        sim_cfg: The simulation section of config

    Returns:
        List of integer seeds
    """
    if "n_seeds" in sim_cfg:
        return list(range(1, sim_cfg["n_seeds"] + 1))
    return sim_cfg.get("seeds", [42])


def _collect_leaf_lists(d: dict, prefix: Tuple[str, ...] = ()) -> List[Tuple[Tuple[str, ...], list]]:
    """Walk a nested dict and collect (key_path, values_list) for each leaf list.

    Scalar values are wrapped in a single-element list automatically.
    """
    leaves = []
    for key, value in d.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            leaves.extend(_collect_leaf_lists(value, path))
        elif isinstance(value, list):
            leaves.append((path, value))
        else:
            # Scalar → treat as single-element list
            leaves.append((path, [value]))
    return leaves


def _build_nested(key_path: Tuple[str, ...], value) -> dict:
    """Build a nested dict from a key path tuple and a scalar value."""
    result = {}
    d = result
    for part in key_path[:-1]:
        d[part] = {}
        d = d[part]
    d[key_path[-1]] = value
    return result


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, modifying base in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def expand_sweep_params(sweep_params: dict) -> List[dict]:
    """Expand nested sweep params into cartesian product of policy param dicts.

    Input:  {"full_proof_budget": {"8b": [0, 16]}, "max_breakdowns": [0, 3]}
    Output: [
        {"full_proof_budget": {"8b": 0}, "max_breakdowns": 0},
        {"full_proof_budget": {"8b": 0}, "max_breakdowns": 3},
        {"full_proof_budget": {"8b": 16}, "max_breakdowns": 0},
        {"full_proof_budget": {"8b": 16}, "max_breakdowns": 3},
    ]
    """
    leaves = _collect_leaf_lists(sweep_params)
    paths = [path for path, _ in leaves]
    value_lists = [values for _, values in leaves]

    combos = []
    for values in itertools.product(*value_lists):
        params = {}
        for path, val in zip(paths, values):
            _deep_merge(params, _build_nested(path, val))
        combos.append(params)

    return combos
