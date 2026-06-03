"""Aggregate statistics and analysis for simulation results."""

from typing import Dict, List, Optional

from .trajectory import Trajectory
from .actions import ActionType, DetailedCost


def compute_solve_rate(trajectories: Dict[str, Trajectory]) -> float:
    """Fraction of problems solved."""
    if not trajectories:
        return 0.0
    solved = sum(1 for t in trajectories.values() if t.solved)
    return solved / len(trajectories)


def compute_cost_distribution(trajectories: Dict[str, Trajectory]) -> Dict[str, float]:
    """Compute aggregate cost statistics."""
    costs = [t.total_cost.total_sflops for t in trajectories.values()]
    if not costs:
        return {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'total': 0}

    sorted_costs = sorted(costs)
    n = len(sorted_costs)
    return {
        'mean': sum(costs) / n,
        'median': sorted_costs[n // 2],
        'min': sorted_costs[0],
        'max': sorted_costs[-1],
        'total': sum(costs),
    }


def compute_cumulative_solve_rate(
    trajectories: Dict[str, Trajectory],
    budgets: List[int],
) -> List[float]:
    """Solve rate at each budget threshold (SFLOPs).

    Args:
        trajectories: Results from a simulation run
        budgets: List of budget thresholds (in total SFLOPs)

    Returns:
        List of solve rates, one per budget
    """
    rates = []
    for budget in budgets:
        solved = 0
        total = 0
        for traj in trajectories.values():
            total += 1
            # Check if problem was solved within budget
            cumulative = 0
            was_solved = False
            for step in traj.steps:
                cumulative += step.result.cost.total_sflops
                if cumulative > budget:
                    break
                if (step.result.success
                        and step.action.type in (ActionType.PROVE, ActionType.CORRECT)
                        and traj.solved):
                    was_solved = True
                    break
            if was_solved:
                solved += 1
        rates.append(solved / max(total, 1))
    return rates


def build_seed_summary(results: Dict[str, Trajectory], seed: Optional[int] = None) -> dict:
    """Build per-seed summary statistics from trajectories.

    Args:
        results: Dict mapping problem_id to Trajectory
        seed: Optional seed identifier

    Returns:
        Dict with seed, solved, total_problems, solve_rate, total_steps, total_cost
    """
    total_problems = len(results)
    solved = sum(1 for t in results.values() if t.solved)
    total_cost = DetailedCost()
    for t in results.values():
        total_cost += t.total_cost

    return {
        "seed": seed,
        "solved": solved,
        "total_problems": total_problems,
        "solve_rate": solved / total_problems if total_problems > 0 else 0.0,
        "total_steps": sum(len(t.steps) for t in results.values()),
        "total_cost": total_cost.to_dict(),
    }


def compare_policies(
    policy_results: Dict[str, List[Dict[str, Trajectory]]],
) -> Dict[str, Dict[str, float]]:
    """Compare multiple policies across multiple seeds.

    Args:
        policy_results: {policy_name: [seed_results, ...]}
            where each seed_results is {problem_id: Trajectory}

    Returns:
        {policy_name: {metric: value}}
    """
    comparison = {}
    for policy_name, seed_runs in policy_results.items():
        all_solve_rates = []
        all_costs = []

        for run in seed_runs:
            rate = compute_solve_rate(run)
            all_solve_rates.append(rate)

            for traj in run.values():
                all_costs.append(traj.total_cost.total_sflops)

        n = len(all_solve_rates) or 1
        avg_rate = sum(all_solve_rates) / n
        std_rate = (sum((r - avg_rate) ** 2 for r in all_solve_rates) / max(n - 1, 1)) ** 0.5

        avg_cost = sum(all_costs) / max(len(all_costs), 1)

        comparison[policy_name] = {
            'solve_rate_mean': avg_rate,
            'solve_rate_std': std_rate,
            'avg_cost_sflops': avg_cost,
            'num_seeds': len(seed_runs),
            'num_problems': len(seed_runs[0]) if seed_runs else 0,
        }

    return comparison
