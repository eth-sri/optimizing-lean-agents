#!/usr/bin/env python3
"""Train a logistic regression model on trajectory data and save to JSON.

Usage:
    python scripts/proof_simulation/train_logistic.py --config configs/proof_simulation/train_logistic.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from proof_simulation.policies.quality.trajectory_logistic import TrajectoryLogisticRegressionModel


def print_training_summary(model: TrajectoryLogisticRegressionModel):
    """Print a summary of the trained model to console."""
    params = model.get_model_params()

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    print(f"\nFeatures: {params['features']}")
    print(f"Regularization C: {params['C']}")
    print(f"Sub-models: {len(params['models'])}")

    for key, mdict in sorted(params["models"].items()):
        print(f"\n--- {key} ---")
        if mdict["type"] == "constant":
            print(f"  Degenerate (constant): p = {mdict['value']}")
        else:
            coefs = np.array(mdict["coefficients"]).flatten()
            intercept = np.array(mdict["intercept"]).flatten()
            features = mdict["features"]
            print(f"  Intercept: {intercept[0]:.4f}")
            for feat, coef in zip(features, coefs):
                print(f"  {feat:>25s}: {coef:+.4f}")
            if "scaler" in mdict:
                print(f"  Scaler mean:  {mdict['scaler']['mean']}")
                print(f"  Scaler scale: {mdict['scaler']['scale']}")

    # Print per-key sample counts from raw training data
    print(f"\n--- Training data ---")
    for (sigma, model_name), (X_rows, y_labels) in sorted(model._fit_data.items()):
        n = len(y_labels)
        pos = sum(y_labels)
        neg = n - pos
        print(f"  sigma={sigma}, model={model_name}: "
              f"{n} samples (pos={pos}, neg={neg}, "
              f"balance={pos/n:.2%})" if n > 0 else f"  sigma={sigma}, model={model_name}: 0 samples")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train logistic regression on trajectory data")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trajectory_base_dir = cfg["trajectory_base_dir"]
    features = cfg.get("features", ["noisy_p", "num_attempts"])
    C = float(cfg.get("C", 1.0))
    model_type = cfg.get("model_type", "logistic")
    alpha = float(cfg.get("alpha", 1.0))
    output_path = cfg["output_path"]

    print(f"Training {model_type} regression model...")
    print(f"  trajectory_base_dir: {trajectory_base_dir}")
    print(f"  features: {features}")
    print(f"  C: {C}, alpha: {alpha}")

    model = TrajectoryLogisticRegressionModel(
        trajectory_base_dir=trajectory_base_dir,
        features=features,
        C=C,
        model_type=model_type,
        alpha=alpha,
    )

    # Store raw training data reference for summary
    # (the model already computed this during __init__, we need to re-load for counts)
    raw = model._load_training_data(Path(trajectory_base_dir))
    model._fit_data = raw

    print_training_summary(model)

    # Save model with trajectory_base_dir for GUI calibration
    params = model.get_model_params()
    params["trajectory_base_dir"] = trajectory_base_dir

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()
