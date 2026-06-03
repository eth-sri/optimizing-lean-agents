#!/usr/bin/env python3
"""Run a single proof simulation config across multiple seeds.

Usage: python scripts/proof_simulation/run.py --config configs/proof_simulation/fixed_putnam.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add repo root to path so `proof_simulation` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from proof_simulation.analysis import build_seed_summary
from proof_simulation.config import load_config, build_from_config, build_policy, build_state_tracker, resolve_seeds
from proof_simulation.simulation import SimulationRunner
from proof_simulation.trajectory import Trajectory


# ── Per-process state for parallel workers ──────────────────────────
_run_worker_data: Dict = {}


def _init_run_worker(config_path: str):
    """Process initializer: load config and problems once per worker."""
    config = load_config(config_path)
    problems, _, runner_kwargs = build_from_config(config)
    # Remove state_tracker from shared kwargs — rebuilt per seed in worker
    runner_kwargs.pop("state_tracker", None)
    _run_worker_data["config"] = config
    _run_worker_data["problems"] = problems
    _run_worker_data["runner_kwargs"] = runner_kwargs


def _run_seed_worker(args: tuple) -> dict:
    """Worker function: run one seed, save results to disk, return summary."""
    seed, output_dir, save_trajectories, num_seeds = args

    config = _run_worker_data["config"]
    problems = _run_worker_data["problems"]
    runner_kwargs = _run_worker_data["runner_kwargs"]

    # Build fresh policy per seed (policies have mutable state)
    policy_cfg = config["policy"]
    policy_type = policy_cfg["type"]
    policy_params = policy_cfg.get("params", {})
    policy = build_policy(
        policy_type, policy_params, problems=problems,
        allowed_actions=policy_cfg.get("allowed_actions"),
    )

    # Build state tracker per seed (has mutable RNG state)
    state_tracker = build_state_tracker(config)
    rk = dict(runner_kwargs)
    if state_tracker is not None:
        rk["state_tracker"] = state_tracker

    runner = SimulationRunner(problems, policy, show_progress=False, **rk)
    results = runner.run(seed)

    if save_trajectories:
        seed_dir = output_dir if num_seeds == 1 else f"{output_dir}/seed_{seed}"
        runner.save_results(results, seed_dir, config=config, seed=seed)

    # Return lightweight summary (no Trajectory objects across process boundary)
    return build_seed_summary(results, seed=seed)


def save_multi_seed_summary(
    all_results: List[Dict[str, Trajectory]],
    seeds: List[int],
    output_dir: str,
):
    """Save aggregate summary across all seeds."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total_problems = len(all_results[0]) if all_results else 0
    per_seed = []
    for seed, results in zip(seeds, all_results):
        per_seed.append(build_seed_summary(results, seed=seed))

    avg_solve_rate = (
        sum(s["solve_rate"] for s in per_seed) / len(per_seed)
        if per_seed else 0.0
    )

    summary = {
        "total_problems": total_problems,
        "num_seeds": len(seeds),
        "avg_solve_rate": avg_solve_rate,
        "per_seed": per_seed,
    }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run a single proof simulation config.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    seeds = resolve_seeds(config["simulation"])
    sim_cfg = config["simulation"]
    num_workers = sim_cfg.get("num_workers", 1)

    output_cfg = config["output"]
    base_dir = output_cfg.get("dir", "results/simulations")
    name = output_cfg["name"]
    output_dir = str(Path(base_dir) / name)

    save_trajectories = output_cfg.get("save_trajectories", True)
    save_summary = output_cfg.get("save_summary", True)

    if num_workers > 1 and len(seeds) > 1:
        # Parallel across seeds — each worker loads data independently
        tasks = [
            (seed, output_dir, save_trajectories, len(seeds))
            for seed in seeds
        ]

        all_seed_summaries = []
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_run_worker,
            initargs=(config_path,),
        ) as executor:
            futures = {
                executor.submit(_run_seed_worker, t): t[0]
                for t in tasks
            }
            seed_bar = tqdm(
                as_completed(futures), total=len(seeds),
                desc=f"{name}", unit="seed",
            )
            for future in seed_bar:
                summary = future.result()
                all_seed_summaries.append(summary)
                seed_bar.set_postfix(
                    solved=f"{summary['solved']}/{summary['total_problems']}",
                    rate=f"{100*summary['solve_rate']:.1f}%",
                )

        # Save multi-seed summary from worker results
        if save_summary and len(seeds) > 1:
            all_seed_summaries.sort(key=lambda x: x["seed"])
            total_problems = all_seed_summaries[0]["total_problems"] if all_seed_summaries else 0
            avg_solve_rate = (
                sum(s["solve_rate"] for s in all_seed_summaries) / len(all_seed_summaries)
                if all_seed_summaries else 0.0
            )
            per_seed = [
                {k: v for k, v in s.items() if k != "total_problems"}
                for s in all_seed_summaries
            ]
            summary = {
                "total_problems": total_problems,
                "num_seeds": len(seeds),
                "avg_solve_rate": avg_solve_rate,
                "per_seed": per_seed,
            }
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
    else:
        # Sequential execution
        problems, policy, runner_kwargs = build_from_config(config)
        runner = SimulationRunner(problems, policy, **runner_kwargs)

        all_results = []
        seed_bar = tqdm(seeds, desc=f"{name}", unit="seed")
        for seed in seed_bar:
            results = runner.run(seed)

            solved = sum(1 for t in results.values() if t.solved)
            seed_bar.set_postfix(solved=f"{solved}/{len(results)}", rate=f"{100*solved/len(results):.1f}%")

            if save_trajectories:
                seed_dir = output_dir if len(seeds) == 1 else f"{output_dir}/seed_{seed}"
                runner.save_results(results, seed_dir, config=config, seed=seed)

            all_results.append(results)

        # Save multi-seed summary
        if save_summary and len(seeds) > 1:
            save_multi_seed_summary(all_results, seeds, output_dir)

    tqdm.write(f"Done. Output at {output_dir}")


if __name__ == "__main__":
    main()
