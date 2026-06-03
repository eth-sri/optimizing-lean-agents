#!/usr/bin/env python3
"""Train a cost-based logistic regression model on full proof data and save to JSON.

Fits P(success | output_sflops) per model_name from minified full proof data.
The saved model can be used as a PredictedProbFeature in the simulation.

Usage:
    python scripts/proof_simulation/train_cost_logistic.py --config configs/proof_simulation/train_cost_logistic.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from proof_simulation.data.full_proof import load_full_proof_data
from proof_simulation.data.agent import load_agent_data_flat


def parse_feature_transform(feat: str) -> Tuple[str, str]:
    """Parse a feature spec that may include a transform prefix.

    Supported transforms:
        "avg_cost"           -> ("avg_cost", None)
        "1/x:num_attempts"   -> ("num_attempts", "1/x")
        "log:avg_cost"       -> ("avg_cost", "log")

    Returns:
        (feature_name, transform) tuple. transform is None if no prefix.
    """
    if ":" in feat:
        transform, name = feat.split(":", 1)
        return name, transform
    return feat, None


def apply_feature_transform(val: float, transform: str) -> float:
    """Apply a transform to a feature value."""
    if transform is None:
        return val
    if transform == "1/x":
        return 1.0 / val if val != 0 else 0.0
    if transform == "log":
        return float(np.log(val + 1e-10))
    if transform == "logit":
        val = max(1e-6, min(val, 1 - 1e-6))
        return float(np.log(val / (1 - val)))
    raise ValueError(f"Unknown feature transform: {transform}")


def load_simulation_trajectory_data(
    trajectory_dir: str,
    model_name: str,
    features: List[str],
    feature_mapping: Dict[str, str],
    skip_first: int = 2,
    problem_ids: set = None,
) -> Tuple[List[List[float]], List[int]]:
    """Load training data from simulation trajectory outputs.

    Args:
        trajectory_dir: Path to sweep dir (with config_NNN/ subdirs) or single config dir.
        model_name: Model name to filter prove actions for (e.g. "8b").
        features: List of feature names to extract. Supports transform prefixes
            like "1/x:num_attempts" or "log:avg_cost".
        feature_mapping: Maps feature names to tracked_state keys.
        skip_first: Skip steps where num_attempts[model_name] < skip_first.

    Returns:
        (X_rows, y_labels) tuple.
    """
    traj_path = Path(trajectory_dir)
    X_rows: List[List[float]] = []
    y_labels: List[int] = []

    # Parse feature transforms
    parsed_features = [parse_feature_transform(feat) for feat in features]

    # Detect sweep mode vs single config
    config_dirs = sorted(traj_path.glob("config_*"))
    if config_dirs:
        base_dirs = config_dirs
    else:
        base_dirs = [traj_path]

    for config_dir in base_dirs:
        traj_root = config_dir / "trajectories"
        if not traj_root.exists():
            continue
        for seed_dir in sorted(traj_root.glob("seed_*")):
            for traj_file in sorted(seed_dir.glob("*.json")):
                if problem_ids is not None:
                    # Filename is problem_id.json
                    pid = traj_file.stem
                    if pid not in problem_ids:
                        continue
                with open(traj_file) as f:
                    traj = json.load(f)
                for step in traj.get("steps", []):
                    action = step.get("action", {})
                    if action.get("type") != "prove" or action.get("model") != model_name:
                        continue
                    tracked_state = step.get("tracked_state", {})
                    num_attempts = tracked_state.get("num_attempts", {})
                    if num_attempts.get(model_name, 0) < skip_first:
                        continue
                    # Extract feature vector (mirrors PredictedProbComputed._get_feature_vector)
                    fv = []
                    skip = False
                    for feat_name, transform in parsed_features:
                        tracker_key = feature_mapping.get(feat_name, feat_name)
                        feat_data = tracked_state.get(tracker_key)
                        if feat_data is None:
                            skip = True
                            break
                        if isinstance(feat_data, dict):
                            val = feat_data.get(model_name)
                        else:
                            val = feat_data
                        if val is None:
                            skip = True
                            break
                        v = apply_feature_transform(float(val), transform)
                        if not np.isfinite(v):
                            skip = True
                            break
                        fv.append(v)
                    if skip:
                        continue
                    result = step.get("result", {})
                    X_rows.append(fv)
                    y_labels.append(int(result.get("success", False)))

    return X_rows, y_labels


def main():
    parser = argparse.ArgumentParser(description="Train cost-based logistic regression on full proof data")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument(
        "--features", nargs="+", default=None,
        help="Override the config's feature list (e.g. for LOO / single-feature variants). "
             "Supports transform prefixes like '1/x:num_attempts'.",
    )
    parser.add_argument(
        "--output-path", default=None,
        help="Override the config's output_path for the saved model JSON.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides (used by feature-subset pipelines to reuse one base config)
    if args.features is not None:
        cfg["features"] = args.features
    if args.output_path is not None:
        cfg["output_path"] = args.output_path

    sources: Dict[str, dict] = cfg["sources"]
    features: List[str] = cfg.get("features", ["output_sflops"])
    feature_mapping: Dict[str, str] = cfg.get("feature_mapping", {})
    C: float = float(cfg.get("C", 1.0))
    output_path: str = cfg["output_path"]

    # Parse problem split for filtering training data
    split_cfg = cfg.get("problem_split")
    train_problem_ids = None
    if split_cfg:
        split_file = split_cfg["file"]
        split_name = split_cfg["split"]
        train_problem_ids = set()
        in_section = False
        with open(split_file) as sf:
            for line in sf:
                line = line.strip()
                if line.lower().startswith(f"{split_name} (") or line.lower().startswith(f"{split_name}("):
                    in_section = True
                    continue
                if in_section:
                    if not line or line.lower().startswith(("train ", "test ", "train(", "test(")):
                        break
                    train_problem_ids.add(line)
        print(f"  Problem split: {split_name} ({len(train_problem_ids)} problems)")

    print(f"Training cost-based logistic regression...")
    print(f"  features: {features}")
    print(f"  feature_mapping: {feature_mapping}")
    print(f"  C: {C}")

    models_dict = {}
    all_X_rows: Dict[str, List[List[float]]] = {}  # per-model for default_costs

    for model_name, src in sources.items():
        src_type = src["type"]
        print(f"\n--- {model_name} ({src_type}) ---")
        if src_type == "simulation":
            X_rows, y_labels = load_simulation_trajectory_data(
                src["trajectory_dir"], model_name, features, feature_mapping,
                src.get("skip_first", 2),
                problem_ids=train_problem_ids,
            )
        else:
            path = src["path"]
            print(f"  path: {path}")
            if src_type == "full_proof":
                pairs_by_problem = load_full_proof_data(path, model_name)
            elif src_type == "agent":
                pairs_by_problem = load_agent_data_flat(path, model_name)
            else:
                raise ValueError(f"Unknown source type: {src_type}")

            X_rows: List[List[float]] = []
            y_labels: List[int] = []

            for pairs in pairs_by_problem.values():
                for pair in pairs:
                    # Initial attempt
                    fv = [float(getattr(pair.initial.cost, f)) for f in features]
                    X_rows.append(fv)
                    y_labels.append(int(pair.initial.success))
                    # Corrections
                    for corr in pair.corrections:
                        fv = [float(getattr(corr.cost, f)) for f in features]
                        X_rows.append(fv)
                        y_labels.append(int(corr.success))

        all_X_rows[model_name] = X_rows

        n = len(y_labels)
        pos = sum(y_labels)
        print(f"  {n} samples (pos={pos}, neg={n - pos}, balance={pos/n:.2%})" if n > 0 else "  0 samples")

        if not X_rows:
            models_dict[model_name] = {"type": "constant", "value": 0.0}
            continue

        X = np.array(X_rows)
        y = np.array(y_labels)

        # Degenerate: all same class
        if len(set(y_labels)) < 2:
            models_dict[model_name] = {"type": "constant", "value": float(y[0])}
            print(f"  Degenerate (constant): p = {y[0]}")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"  scaler mean: {dict(zip(features, scaler.mean_.tolist()))}")
        print(f"  scaler scale: {dict(zip(features, scaler.scale_.tolist()))}")

        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_scaled, y)

        coefs = model.coef_.flatten()
        intercept = model.intercept_.flatten()
        print(f"  Intercept: {intercept[0]:.4f}")
        for feat, coef in zip(features, coefs):
            print(f"  {feat:>25s}: {coef:+.6f}")

        models_dict[model_name] = {
            "type": "logistic_regression",
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "classes": model.classes_.tolist(),
            "features": features,
            "scaler": {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
        }

    # Compute default_costs: mean feature values per model
    default_costs = {}
    for model_name, rows in all_X_rows.items():
        if rows:
            means = np.mean(rows, axis=0)
            default_costs[model_name] = {feat: float(m) for feat, m in zip(features, means)}
            print(f"\n  default_costs[{model_name}]: {default_costs[model_name]}")

    # Save
    result = {
        "features": features,
        "feature_mapping": feature_mapping,
        "default_costs": default_costs,
        "C": C,
        "models": models_dict,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()
