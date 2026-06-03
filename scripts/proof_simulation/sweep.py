#!/usr/bin/env python3
"""Hyperparameter sweep runner for proof simulation.

Usage: python scripts/proof_simulation/sweep.py --config configs/proof_simulation/sweep_cost_quality.yaml

Expands sweep_params (cartesian product of lists), runs each config
across multiple seeds with deterministic seeding, and saves results.
"""

import argparse
import copy
import itertools
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add repo root to path so `proof_simulation` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from proof_simulation.actions import DetailedCost
from proof_simulation.analysis import build_seed_summary
from proof_simulation.config import (
    _deep_merge,
    build_policy,
    build_prob_model,
    build_state_tracker,
    expand_sweep_params,
    extract_prove_models,
    load_config,
    resolve_seeds,
    validate_breakdown_model_filter,
)
from proof_simulation.data.loader import load_problems
from proof_simulation.simulation import SimulationRunner


_RUNNER_KEYS = {"max_breakdowns", "max_corrections"}
"""Sweep param keys that are runner-level, not policy-level."""

_NON_POLICY_KEYS = {"_state_tracker"}
"""Sweep param keys that should be stripped before passing to build_policy."""


def _canonical_params_key(params: dict) -> str:
    """Return a canonical JSON string for a param combo (order-independent)."""
    return json.dumps(params, sort_keys=True)


def _load_existing_configs(output_dir: Path) -> tuple:
    """Scan existing config_NNN/ dirs and return completed results.

    A config is "complete" only if both params.json and summary.json exist
    and parse correctly.

    Returns:
        (existing, key_to_index, max_idx) where existing is a dict mapping
        canonical_params_key -> summary dict, key_to_index maps
        canonical_params_key -> config index, and max_idx is the highest
        config index found (-1 if none).
    """
    existing: Dict[str, dict] = {}
    key_to_index: Dict[str, int] = {}
    max_idx = -1

    if not output_dir.exists():
        return existing, key_to_index, max_idx

    for child in sorted(output_dir.iterdir()):
        m = re.match(r"config_(\d+)$", child.name)
        if not m or not child.is_dir():
            continue
        idx = int(m.group(1))
        max_idx = max(max_idx, idx)

        params_file = child / "params.json"
        summary_file = child / "summary.json"
        if not params_file.exists() or not summary_file.exists():
            continue

        try:
            with open(params_file) as f:
                params = json.load(f)
            with open(summary_file) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        key = _canonical_params_key(params)
        existing[key] = summary
        key_to_index[key] = idx

    return existing, key_to_index, max_idx


def _split_runner_params(params: dict, base_runner_kwargs: dict) -> tuple:
    """Split sweep params into (policy_params, runner_kwargs).

    Keys in _RUNNER_KEYS are merged into runner_kwargs.
    Keys in _NON_POLICY_KEYS are removed from the returned policy params.
    Returns (policy_params, runner_kwargs).
    """
    runner_kwargs = dict(base_runner_kwargs)
    for key in _RUNNER_KEYS:
        if key in params:
            runner_kwargs[key] = params[key]
    policy_params = {k: v for k, v in params.items() if k not in _NON_POLICY_KEYS}
    return policy_params, runner_kwargs


def _get_breakdown_model_filter(
    params: dict,
    agent_sources: dict,
    shared_breakdowns: bool,
    allowed_actions: Optional[List[str]] = None,
) -> Optional[Set[str]]:
    """Extract active breakdown models from params/allowed_actions and validate.

    Returns set of model names to filter breakdown templates by,
    or None if no filtering needed.
    """
    # First try explicit breakdown_proof_budget
    budget = params.get("breakdown_proof_budget", {})
    active_models = {m for m, b in budget.items() if b > 0}

    # Fall back to allowed_actions
    if not active_models and allowed_actions:
        active_models = extract_prove_models(allowed_actions)

    if not active_models:
        return None

    # With shared_breakdowns=false and multiple agent sources,
    # can't have multiple models active (would pick wrong templates)
    if not shared_breakdowns and len(agent_sources) > 1 and len(active_models) > 1:
        raise ValueError(
            f"With shared_breakdowns=false and multiple agent sources, "
            f"only one prove model can be active. "
            f"Got active models: {sorted(active_models)}"
        )

    return active_models


# ── Per-process state for parallel workers ──────────────────────────
_sweep_worker_data: Dict = {}


def _parse_problem_split(split_file: str, split: str) -> set:
    """Parse a train/test split file and return the set of problem IDs for the given split."""
    problems = set()
    in_section = False
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith(f"{split} (") or line.lower().startswith(f"{split}("):
                in_section = True
                continue
            if in_section:
                if not line or line.lower().startswith(("train ", "test ", "train(", "test(")):
                    break
                problems.add(line)
    return problems


def _filter_problems_by_split(problems, split_cfg):
    """Filter problems by train/test split config."""
    if split_cfg is None:
        return problems
    split_file = split_cfg.get("file")
    split = split_cfg.get("split")  # "train" or "test"
    if not split_file or not split:
        return problems
    allowed = _parse_problem_split(split_file, split)
    filtered = [p for p in problems if p.problem_id in allowed]
    print(f"  Split filter ({split}): {len(problems)} -> {len(filtered)} problems")
    return filtered


def _init_sweep_worker(data_cfg: dict, max_problems, policy_base_params: dict = None, problem_split=None):
    """Process initializer: load problem data once per worker."""
    full_proof_sources = data_cfg.get("full_proof")
    problems = load_problems(
        full_proof_sources=full_proof_sources,
        agent_config=data_cfg.get("agent"),
        seed=1,
        load_code=data_cfg.get("load_code", False),
    )
    if max_problems is not None:
        problems = problems[:max_problems]
    problems = _filter_problems_by_split(problems, problem_split)
    _sweep_worker_data["problems"] = problems
    _sweep_worker_data["full_proof_sources"] = full_proof_sources

    # Pre-fit prob model once per worker to avoid re-fitting per seed
    if policy_base_params is not None:
        prob_model_type = policy_base_params.get("prob_model", "oracle")
        _sweep_worker_data["prob_model"] = build_prob_model(
            prob_model_type, policy_base_params, problems=problems,
            full_proof_sources=full_proof_sources,
        )


def _run_config_worker(args: tuple) -> dict:
    """Worker function: run one config across all seeds."""
    (config_idx, params, seeds, output_dir, save_trajectories,
     runner_kwargs, policy_type, n_configs, breakdown_model_filter,
     allowed_actions, config, st_params, save_params) = args

    problems = _sweep_worker_data["problems"]
    full_proof_sources = _sweep_worker_data.get("full_proof_sources")
    cached_prob_model = _sweep_worker_data.get("prob_model")
    output_dir = Path(output_dir)

    # Build policy (reuses pre-fitted prob_model if available)
    policy = build_policy(policy_type, params, problems=problems,
                          allowed_actions=allowed_actions,
                          full_proof_sources=full_proof_sources,
                          prob_model=cached_prob_model)

    # Build state tracker (forward sigma from policy params).
    # Skip when lambda=0: policy is fixed, features add nothing.
    if float(params.get("lambda_val", -1)) == 0.0:
        state_tracker = None
    else:
        effective_st_params = dict(st_params or {})
        if "sigma" in params:
            effective_st_params["sigma"] = params["sigma"]
        state_tracker = build_state_tracker(config, state_tracker_params=effective_st_params)

    runner = SimulationRunner(
        problems, policy, show_progress=False,
        breakdown_model_filter=breakdown_model_filter,
        state_tracker=state_tracker, **runner_kwargs,
    )

    return run_config(
        runner=runner,
        seeds=seeds,
        config_idx=config_idx,
        n_configs=n_configs,
        params=save_params,
        output_dir=output_dir,
        save_trajectories=save_trajectories,
    )


def _run_seed_worker_sweep(args: tuple) -> dict:
    """Worker function: run one seed for one config, return per-seed summary."""
    (config_idx, seed, params, output_dir_str, save_trajectories,
     runner_kwargs, policy_type, breakdown_model_filter,
     allowed_actions, config, st_params) = args

    problems = _sweep_worker_data["problems"]
    full_proof_sources = _sweep_worker_data.get("full_proof_sources")
    cached_prob_model = _sweep_worker_data.get("prob_model")
    config_dir = Path(output_dir_str) / f"config_{config_idx:03d}"

    # Build fresh policy per seed (reuses pre-fitted prob_model if available)
    policy = build_policy(policy_type, params, problems=problems,
                          allowed_actions=allowed_actions,
                          full_proof_sources=full_proof_sources,
                          prob_model=cached_prob_model)

    # Build state tracker (forward sigma from policy params).
    # Skip when lambda=0: policy is fixed, features add nothing.
    if float(params.get("lambda_val", -1)) == 0.0:
        state_tracker = None
    else:
        effective_st_params = dict(st_params or {})
        if "sigma" in params:
            effective_st_params["sigma"] = params["sigma"]
        state_tracker = build_state_tracker(config, state_tracker_params=effective_st_params)

    runner = SimulationRunner(
        problems, policy, show_progress=False,
        breakdown_model_filter=breakdown_model_filter,
        state_tracker=state_tracker, **runner_kwargs,
    )
    results = runner.run(seed)

    if save_trajectories:
        seed_dir = config_dir / "trajectories" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for pid, traj in results.items():
            with open(seed_dir / f"{pid}.json", "w") as f:
                json.dump(traj.to_dict(), f, indent=2)

    return build_seed_summary(results, seed=seed)


def _build_config_summary(
    config_idx: int,
    params: dict,
    per_seed: List[dict],
    config_dir: Path,
) -> dict:
    """Aggregate per-seed results into a config summary and save to disk."""
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    solve_rates = [s["solve_rate"] for s in per_seed]
    avg_solve_rate = sum(solve_rates) / len(solve_rates) if solve_rates else 0.0
    std_solve_rate = (
        (sum((r - avg_solve_rate) ** 2 for r in solve_rates) / len(solve_rates)) ** 0.5
        if len(solve_rates) > 1 else 0.0
    )

    avg_total_cost = DetailedCost()
    for s in per_seed:
        avg_total_cost += DetailedCost.from_dict(s["total_cost"])
    n = len(per_seed) or 1
    avg_total_cost = DetailedCost(
        input_sflops=avg_total_cost.input_sflops // n,
        output_sflops=avg_total_cost.output_sflops // n,
        input_tokens=avg_total_cost.input_tokens // n,
        output_tokens=avg_total_cost.output_tokens // n,
    )

    config_summary = {
        "config_id": config_idx,
        "params": params,
        "avg_solve_rate": avg_solve_rate,
        "std_solve_rate": std_solve_rate,
        "avg_total_cost": avg_total_cost.to_dict(),
        "per_seed": per_seed,
    }

    with open(config_dir / "summary.json", "w") as f:
        json.dump(config_summary, f, indent=2)

    return config_summary


def _save_prob_model_params(policy, config_dir: Path):
    """Save probability model parameters if the model supports it."""
    prob_model = getattr(policy, "prob_model", None)
    if prob_model is not None and hasattr(prob_model, "get_model_params"):
        config_dir.mkdir(parents=True, exist_ok=True)
        prob_model.save_model_params(str(config_dir / "model_params.json"))


def run_config(
    runner: SimulationRunner,
    seeds: List[int],
    config_idx: int,
    n_configs: int,
    params: dict,
    output_dir: Path,
    save_trajectories: bool = False,
) -> dict:
    """Run a single config across all seeds and save per-config results.

    Returns:
        Config summary dict for inclusion in sweep_summary.json
    """
    config_dir = output_dir / f"config_{config_idx:03d}"

    _save_prob_model_params(runner.policy, config_dir)

    per_seed = []

    for seed in seeds:
        results = runner.run(seed)
        per_seed.append(build_seed_summary(results, seed=seed))

        if save_trajectories:
            seed_dir = config_dir / "trajectories" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            for pid, traj in results.items():
                with open(seed_dir / f"{pid}.json", "w") as f:
                    json.dump(traj.to_dict(), f, indent=2)

    return _build_config_summary(config_idx, params, per_seed, config_dir)


def main():
    parser = argparse.ArgumentParser(description="Run a proof simulation hyperparameter sweep.")
    parser.add_argument("--config", required=True, help="Path to YAML config file with sweep_params")
    parser.add_argument(
        "--model-path", default=None,
        help="Override policy.params.model_path (for feature-subset variants reusing a base config).",
    )
    parser.add_argument(
        "--output-name", default=None,
        help="Override output.name (the results subdirectory).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Force output.overwrite=true regardless of the config value.",
    )
    parser.add_argument(
        "--state-features", nargs="+", default=None,
        help="Override state_tracker.features with this list of feature TYPES "
             "(e.g. avg_cost attempt_count). Only these are computed, so expensive "
             "unused features (e.g. normalized_similarity) are skipped. Leaves "
             "state_tracker.computed untouched.",
    )
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    # CLI overrides (used by feature-subset pipelines to reuse one base config)
    if args.model_path is not None:
        config["policy"].setdefault("params", {})["model_path"] = args.model_path
    if args.output_name is not None:
        config["output"]["name"] = args.output_name
    if args.overwrite:
        config["output"]["overwrite"] = True
    if args.state_features is not None:
        config.setdefault("state_tracker", {})["features"] = [
            {"type": t} for t in args.state_features
        ]

    # Expand sweep params
    policy_cfg = config["policy"]
    st_cfg = config.get("state_tracker", {})

    if "sweep_params" not in policy_cfg and "sweep_params" not in st_cfg:
        print("Error: policy.sweep_params or state_tracker.sweep_params is required for sweep mode")
        sys.exit(1)

    base_params = policy_cfg.get("params", {})

    # Expand policy sweep params
    if "sweep_params" in policy_cfg:
        policy_combos = expand_sweep_params(policy_cfg["sweep_params"])
    else:
        policy_combos = [{}]
    if base_params:
        policy_combos = [_deep_merge(copy.deepcopy(base_params), combo) for combo in policy_combos]

    # Expand state_tracker sweep params
    if "sweep_params" in st_cfg:
        st_combos = expand_sweep_params(st_cfg["sweep_params"])
    else:
        st_combos = [{}]

    # Cartesian product of policy x state_tracker params
    combined_combos = list(itertools.product(policy_combos, st_combos))
    # For canonical key and params.json, merge both into one dict
    param_combos = []
    st_params_list = []
    for p, st in combined_combos:
        merged = copy.deepcopy(p)
        if st:
            merged["_state_tracker"] = st
        param_combos.append(merged)
        st_params_list.append(st if st else None)
    n_configs = len(param_combos)

    # Load data ONCE
    data_cfg = config["data"]
    sim_cfg = config["simulation"]

    problems = load_problems(
        full_proof_sources=data_cfg.get("full_proof"),
        agent_config=data_cfg.get("agent"),
        seed=1,  # initial seed for data loading
        load_code=data_cfg.get("load_code", False),
    )

    max_problems = sim_cfg.get("max_problems")
    if max_problems is not None:
        problems = problems[:max_problems]

    problem_split = sim_cfg.get("problem_split")
    problems = _filter_problems_by_split(problems, problem_split)

    # Seeds
    n_seeds = sim_cfg.get("n_seeds", len(resolve_seeds(sim_cfg)))
    seed_offset = sim_cfg.get("seed_offset", 0)
    # When shared_seeds is true, every config (e.g. every lambda value) uses the
    # SAME seed set, so all configs see identical data shuffling. This is a paired
    # / common-random-numbers comparison: differences across configs reflect the
    # swept parameter rather than seed noise, keeping the solve-rate-vs-lambda
    # curve smooth (concave). When false, each config gets its own decorrelated
    # seed block (the original behavior).
    shared_seeds = sim_cfg.get("shared_seeds", True)

    def _config_seeds(config_idx: int) -> List[int]:
        base_seed = seed_offset if shared_seeds else config_idx * 1000 + seed_offset
        return [base_seed + s for s in range(1, n_seeds + 1)]

    # Output setup
    output_cfg = config["output"]
    base_dir = output_cfg.get("dir", "results/simulations")
    name = output_cfg["name"]
    output_dir = Path(base_dir) / name
    output_dir = output_dir.resolve()
    overwrite = output_cfg.get("overwrite", False)
    if overwrite and output_dir.exists():
        # Targeted cleanup: only remove known sweep outputs
        for child in output_dir.iterdir():
            if child.is_dir() and re.match(r"config_\d+$", child.name):
                shutil.rmtree(child, ignore_errors=True)
            elif child.name in ("sweep_config.json", "sweep_summary.json"):
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_trajectories = output_cfg.get("save_trajectories", False)

    # ── Incremental resume: match param combos to existing results ────
    if overwrite:
        existing_configs: Dict[str, dict] = {}
        key_to_index: Dict[str, int] = {}
        next_idx = 0
    else:
        existing_configs, key_to_index, max_existing_idx = _load_existing_configs(output_dir)
        next_idx = max_existing_idx + 1

    cached_summaries: List[dict] = []
    pending_combos: List[tuple] = []  # (config_idx, params, st_params)
    config_map: List[dict] = []  # maps param combos to config_NNN indices

    for params, st_params in zip(param_combos, st_params_list):
        key = _canonical_params_key(params)
        if key in existing_configs:
            cached_summaries.append(existing_configs[key])
            config_map.append({"index": key_to_index[key], "params": params})
        else:
            pending_combos.append((next_idx, params, st_params))
            config_map.append({"index": next_idx, "params": params})
            next_idx += 1

    n_cached = len(cached_summaries)
    n_pending = len(pending_combos)
    n_total = n_cached + n_pending

    if n_cached > 0:
        tqdm.write(f"Resume: {n_cached} cached, {n_pending} new to run")

    # Save full sweep config
    sweep_meta = {
        "config_path": config_path,
        "n_configs": n_total,
        "n_seeds": n_seeds,
        "n_problems": len(problems),
        "param_combos": param_combos,
        "config_map": config_map,
    }
    with open(output_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_meta, f, indent=2)

    # Runner kwargs (max_breakdowns can be overridden per-config via sweep_params)
    base_runner_kwargs = {
        "max_steps": sim_cfg.get("max_steps", 1000),
    }
    if sim_cfg.get("max_breakdowns") is not None:
        base_runner_kwargs["max_breakdowns"] = sim_cfg["max_breakdowns"]

    policy_type = policy_cfg["type"]
    allowed_actions = policy_cfg.get("allowed_actions")
    num_workers = sim_cfg.get("num_workers", 1)
    parallel_over = sim_cfg.get("parallel_over", "configs")

    # Determine agent sources for validation
    agent_sources = data_cfg.get("agent", {}).get("sources", {})
    shared_breakdowns = data_cfg.get("agent", {}).get("shared_breakdowns", False)

    # Early validation: crash if shared_breakdowns=false with multiple sources
    # and allowed_actions doesn't specify exactly one prove model
    agent_cfg = data_cfg.get("agent")
    validate_breakdown_model_filter(agent_cfg, allowed_actions)

    # Run pending configs (skip if all cached)
    all_config_summaries = list(cached_summaries)

    if n_pending > 0:
        if num_workers > 1 and parallel_over == "configs":
            # Parallel over configs — each worker runs one config (all seeds)
            tasks = []
            for config_idx, params, st_params in pending_combos:
                seeds = _config_seeds(config_idx)
                policy_params, runner_kwargs = _split_runner_params(params, base_runner_kwargs)
                breakdown_model_filter = _get_breakdown_model_filter(
                    policy_params, agent_sources, shared_breakdowns, allowed_actions,
                )
                tasks.append((
                    config_idx, policy_params, seeds, str(output_dir), save_trajectories,
                    runner_kwargs, policy_type, n_total, breakdown_model_filter,
                    allowed_actions, config, st_params, params,
                ))

            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_sweep_worker,
                initargs=(data_cfg, max_problems, base_params, problem_split),
            ) as executor:
                futures = {
                    executor.submit(_run_config_worker, t): t[0]
                    for t in tasks
                }
                config_bar = tqdm(
                    as_completed(futures), total=n_pending,
                    desc=f"{name} configs", unit="cfg",
                )
                for future in config_bar:
                    config_summary = future.result()
                    all_config_summaries.append(config_summary)
                    config_bar.set_postfix(
                        rate=f"{config_summary['avg_solve_rate']:.1%}",
                        std=f"{config_summary['std_solve_rate']:.3f}",
                    )

        elif num_workers > 1 and parallel_over == "seeds":
            # Parallel over seeds — configs run sequentially, seeds in parallel
            full_proof_sources = data_cfg.get("full_proof")
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_sweep_worker,
                initargs=(data_cfg, max_problems, base_params, problem_split),
            ) as executor:
                config_bar = tqdm(
                    pending_combos, total=n_pending,
                    desc=f"{name} configs", unit="cfg",
                )
                for config_idx, params, st_params in config_bar:
                    seeds = _config_seeds(config_idx)
                    policy_params, runner_kwargs = _split_runner_params(params, base_runner_kwargs)
                    breakdown_model_filter = _get_breakdown_model_filter(
                        policy_params, agent_sources, shared_breakdowns,
                    )

                    # Save prob model params (built in main process, uses module cache)
                    prob_model_type = policy_params.get("prob_model", "oracle")
                    prob_model_for_save = build_prob_model(
                        prob_model_type, policy_params, problems=problems,
                        full_proof_sources=full_proof_sources,
                    )
                    config_dir_seeds = output_dir / f"config_{config_idx:03d}"
                    config_dir_seeds.mkdir(parents=True, exist_ok=True)
                    if hasattr(prob_model_for_save, "save_model_params"):
                        prob_model_for_save.save_model_params(
                            str(config_dir_seeds / "model_params.json")
                        )

                    seed_tasks = [
                        (config_idx, seed, policy_params, str(output_dir),
                         save_trajectories, runner_kwargs, policy_type,
                         breakdown_model_filter, allowed_actions,
                         config, st_params)
                        for seed in seeds
                    ]
                    futures = [executor.submit(_run_seed_worker_sweep, t) for t in seed_tasks]
                    per_seed = []
                    seed_bar = tqdm(
                        futures, total=len(futures),
                        desc=f"  config_{config_idx:03d} seeds", unit="seed",
                        leave=False,
                    )
                    for f in seed_bar:
                        per_seed.append(f.result())
                    per_seed.sort(key=lambda x: x["seed"])

                    config_dir = output_dir / f"config_{config_idx:03d}"
                    config_summary = _build_config_summary(
                        config_idx, params, per_seed, config_dir,
                    )
                    all_config_summaries.append(config_summary)
                    config_bar.set_postfix(
                        rate=f"{config_summary['avg_solve_rate']:.1%}",
                        std=f"{config_summary['std_solve_rate']:.3f}",
                    )

        else:
            # Sequential execution — pre-fit prob model once
            full_proof_sources = data_cfg.get("full_proof")
            cached_prob_model = build_prob_model(
                base_params.get("prob_model", "oracle"), base_params,
                problems=problems, full_proof_sources=full_proof_sources,
            )
            config_bar = tqdm(
                pending_combos, total=n_pending,
                desc=f"{name} configs", unit="cfg",
            )
            for config_idx, params, st_params in config_bar:
                seeds = _config_seeds(config_idx)
                policy_params, runner_kwargs = _split_runner_params(params, base_runner_kwargs)
                breakdown_model_filter = _get_breakdown_model_filter(
                    policy_params, agent_sources, shared_breakdowns, allowed_actions,
                )

                # Build policy (reuses pre-fitted prob_model)
                policy = build_policy(policy_type, policy_params, problems=problems,
                                      allowed_actions=allowed_actions,
                                      full_proof_sources=full_proof_sources,
                                      prob_model=cached_prob_model)

                # Build state tracker (forward sigma from policy params).
                # Skip when lambda=0: policy is fixed, features add nothing.
                if float(policy_params.get("lambda_val", -1)) == 0.0:
                    state_tracker = None
                else:
                    effective_st_params = dict(st_params or {})
                    if "sigma" in policy_params:
                        effective_st_params["sigma"] = policy_params["sigma"]
                    state_tracker = build_state_tracker(config, state_tracker_params=effective_st_params)

                runner = SimulationRunner(
                    problems, policy, show_progress=False,
                    breakdown_model_filter=breakdown_model_filter,
                    state_tracker=state_tracker, **runner_kwargs,
                )

                config_summary = run_config(
                    runner=runner,
                    seeds=seeds,
                    config_idx=config_idx,
                    n_configs=n_total,
                    params=params,
                    output_dir=output_dir,
                    save_trajectories=save_trajectories,
                )
                all_config_summaries.append(config_summary)

                config_bar.set_postfix(
                    rate=f"{config_summary['avg_solve_rate']:.1%}",
                    std=f"{config_summary['std_solve_rate']:.3f}",
                )

    # Sort by config_id and save sweep summary
    all_config_summaries.sort(key=lambda x: x["config_id"])

    sweep_summary = {
        "n_configs": n_total,
        "n_seeds": n_seeds,
        "n_problems": len(problems),
        "config_map": config_map,
        "configs": all_config_summaries,
    }
    with open(output_dir / "sweep_summary.json", "w") as f:
        json.dump(sweep_summary, f, indent=2)

    tqdm.write(
        f"Done. {n_total} configs ({n_cached} cached, {n_pending} new) "
        f"x {n_seeds} seeds -> {output_dir}"
    )


if __name__ == "__main__":
    main()
