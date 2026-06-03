"""SimulationRunner: orchestrates running policies across problems."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from .actions import ActionType
from .problem import SimulatedProblem
from .policies.base import Policy
from .features import StateTracker
from .trajectory import Trajectory


class SimulationRunner:
    """Run proof simulations with a given policy across problems."""

    def __init__(
        self,
        problems: List[SimulatedProblem],
        policy: Policy,
        max_steps: int = 1000,
        show_progress: bool = True,
        breakdown_model_filter: Optional[set] = None,
        max_breakdowns: Optional[int] = None,
        max_corrections: Optional[int] = None,
        state_tracker: Optional[StateTracker] = None,
    ):
        """Initialize the simulation runner.

        Args:
            problems: List of SimulatedProblem instances
            policy: Policy to use for action selection
            max_steps: Maximum steps per problem (safety limit)
            show_progress: Show per-problem tqdm bar inside run()
            breakdown_model_filter: If set, only use breakdown templates from these models
            max_breakdowns: If set, limit number of breakdown templates per target
            max_corrections: If set, limit number of correction attempts per prove
            state_tracker: Optional StateTracker for recording per-step features
        """
        self.problems = problems
        self.policy = policy
        self.max_steps = max_steps
        self.show_progress = show_progress
        self.breakdown_model_filter = breakdown_model_filter
        self.max_breakdowns = max_breakdowns
        self.max_corrections = max_corrections
        self.state_tracker = state_tracker

    def run(self, seed: int = 42) -> Dict[str, Trajectory]:
        """Run all problems with given seed.

        Args:
            seed: Random seed (each problem gets this seed for reset)

        Returns:
            Dict mapping problem_id to Trajectory
        """
        results = {}
        solved = 0
        if self.state_tracker is not None:
            self.state_tracker.seed(seed)
        problems_iter = tqdm(
            self.problems,
            desc=f"seed {seed}",
            unit="prob",
            leave=False,
            disable=not self.show_progress,
        )
        for problem in problems_iter:
            problem.reset(seed, breakdown_model_filter=self.breakdown_model_filter, max_breakdowns=self.max_breakdowns, max_corrections=self.max_corrections)
            if hasattr(self.policy, 'reset'):
                self.policy.reset()
            cost_model = getattr(self.policy, 'cost_model', None)
            if cost_model is not None and hasattr(cost_model, 'reset'):
                cost_model.reset()
            prob_model = getattr(self.policy, 'prob_model', None)
            if prob_model is not None and hasattr(prob_model, 'seed'):
                prob_model.seed(seed)
            if hasattr(self.policy, 'seed'):
                self.policy.seed(seed)
            trajectory = self._run_problem(problem, seed)
            results[problem.problem_id] = trajectory
            if trajectory.solved:
                solved += 1
            problems_iter.set_postfix(solved=f"{solved}/{len(results)}")
        return results

    def run_multi_seed(self, seeds: List[int]) -> List[Dict[str, Trajectory]]:
        """Run across N seeds for statistical analysis.

        Args:
            seeds: List of random seeds

        Returns:
            List of trajectory dicts, one per seed
        """
        all_results = []
        for seed in seeds:
            # Reset fixed sequence policies if they support it
            if hasattr(self.policy, 'reset'):
                self.policy.reset()
            results = self.run(seed)
            all_results.append(results)
        return all_results

    def _run_problem(self, problem: SimulatedProblem, seed: int = 42) -> Trajectory:
        """Core simulation loop for a single problem."""
        trajectory = Trajectory(problem_id=problem.problem_id, seed=seed)
        if self.state_tracker is not None:
            self.state_tracker.reset()

        for step_num in range(self.max_steps):
            if problem.is_done():
                break

            # Get valid actions
            valid_actions = problem.get_valid_actions()
            if not valid_actions:
                break

            # Get state snapshot
            state = problem.get_state()

            # Track state before action
            tracked_state = None
            if self.state_tracker is not None:
                tracked_state = self.state_tracker.get_tracked_state(state, problem)

            # Choose action
            action = self.policy.choose_action(state, valid_actions, tracked_state=tracked_state)
            decision_metadata = self.policy.get_decision_metadata()

            # Execute
            result = problem.simulate_action(action)

            # Feed observed cost to cost model
            cost_model = getattr(self.policy, 'cost_model', None)
            if cost_model is not None and hasattr(cost_model, 'observe'):
                cost_model.observe(action, result.cost, target_id=state.target_id)

            # Observe action result in state tracker
            if self.state_tracker is not None:
                self.state_tracker.observe(state, action, result, problem)

            # Record
            trajectory.add_step(state, action, result, decision_metadata=decision_metadata, tracked_state=tracked_state)

        trajectory.solved = problem.is_solved()
        trajectory.total_cost = problem.total_cost
        return trajectory

    def save_results(
        self,
        results: Dict[str, Trajectory],
        output_dir: str,
        config: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        """Save trajectories and summary to disk.

        Args:
            results: Dict mapping problem_id to Trajectory
            output_dir: Directory to save to (e.g., results/simulations/my_run)
            config: Optional config dict to save alongside
            seed: Optional seed used for this run
        """
        out = Path(output_dir)
        traj_dir = out / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        if config is not None:
            with open(out / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        # Save each trajectory
        for pid, traj in results.items():
            with open(traj_dir / f"{pid}.json", "w") as f:
                json.dump(traj.to_dict(), f, indent=2)

        # Save summary
        summary = self._build_summary(results, seed=seed)
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    @staticmethod
    def _build_summary(results: Dict[str, Trajectory], seed: Optional[int] = None) -> dict:
        """Build aggregate stats from trajectories."""
        from .analysis import build_seed_summary
        summary = build_seed_summary(results, seed=seed)
        total_problems = summary["total_problems"]
        summary["avg_steps"] = summary["total_steps"] / total_problems if total_problems > 0 else 0.0
        return summary

    @staticmethod
    def load_trajectory(path: str) -> Trajectory:
        """Load a single trajectory from JSON.

        Args:
            path: Path to a trajectory JSON file

        Returns:
            Trajectory instance
        """
        with open(path, "r") as f:
            d = json.load(f)
        return Trajectory.from_dict(d)

    @staticmethod
    def load_results(output_dir: str) -> Dict[str, Trajectory]:
        """Load all trajectories from a results directory.

        Args:
            output_dir: Directory containing trajectories/ subdirectory

        Returns:
            Dict mapping problem_id to Trajectory
        """
        traj_dir = Path(output_dir) / "trajectories"
        results = {}
        for f in sorted(traj_dir.glob("*.json")):
            traj = SimulationRunner.load_trajectory(str(f))
            results[traj.problem_id] = traj
        return results
