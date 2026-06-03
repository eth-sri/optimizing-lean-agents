"""End-to-end feature extraction pipeline.

Usage:
    uv run python scripts/feature_engineering/run_features.py --run_dir <path>
    uv run python scripts/feature_engineering/run_features.py --run_dir <path> --correction_rounds 1 --max_success_rate 0.5
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_engineering.core.data_loader import (
    compute_success_rates,
    filter_by_success_rate,
    filter_failed_only,
    load_run_data,
)
from feature_engineering.features.error_diversity import ErrorDiversity
from feature_engineering.features.have_repetition import HaveRepetition
from feature_engineering.features.error_persistence import ErrorPersistence
from feature_engineering.features.lean_similarity import LeanSimilarity
from feature_engineering.features.lean_similarity_normalized import NormalizedLeanSimilarity
from feature_engineering.features.lean_similarity_structural import StructuralLeanSimilarity
from feature_engineering.features.output_tokens import OutputSflops
from feature_engineering.features.uncertainty_words import UncertaintyWords
from feature_engineering.models.gradient_boosting import GradientBoosting, gb_to_json_dict
from feature_engineering.models.linear import LinearRegression
from feature_engineering.models.logistic import LogisticRegression
from feature_engineering.models.random_forest import RandomForest

# Register all features here
FEATURES = [
    OutputSflops(),
    # LeanSimilarity(),
    NormalizedLeanSimilarity(),
    StructuralLeanSimilarity(),
    ErrorDiversity(),
    # ErrorPersistence(),
    UncertaintyWords(),
    HaveRepetition(),
]


def _plot_predicted_vs_actual(result: dict, df: pd.DataFrame, title: str, save_path: Path, stat_text: str):
    """Plot predicted vs actual success rate scatter."""
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pred = result.get("y_prob", result.get("y_pred"))
    y_actual = df.loc[result["problem_ids"], "success_rate"].values
    ax.scatter(y_actual, y_pred, alpha=0.7, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="ideal")
    if len(y_actual) <= 100:
        for pid, actual, pred in zip(result["problem_ids"], y_actual, y_pred):
            ax.annotate(str(pid).replace("putnam_", ""), (actual, pred),
                        textcoords="offset points", xytext=(5, 5), fontsize=5, alpha=0.8)
    if stat_text:
        ax.text(0.02, 0.98, stat_text,
                transform=ax.transAxes, fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Actual Success Rate", fontsize=12)
    ax.set_ylabel("Predicted Success Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Feature extraction pipeline")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--correction_rounds", type=int, default=0)
    parser.add_argument("--max_success_rate", type=float, default=None)
    parser.add_argument("--min_success_rate", type=float, default=None,
                        help="Minimum success rate threshold. Use 0 to exclude unsolved problems.")
    parser.add_argument("--no_failed_only", action="store_true",
                        help="Use all attempts instead of only failed ones (default: failed only).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path. Defaults to <run_dir>/features.csv")
    parser.add_argument("--features", nargs="+", default=None,
                        help="Only compute these features (e.g. --features output_sflops)")
    parser.add_argument("--no_std", action="store_true",
                        help="Exclude _std columns from the feature DataFrame")
    parser.add_argument("--no_cv", action="store_true",
                        help="Exclude _cv columns from the feature DataFrame")
    parser.add_argument("--problem_split_file", type=Path, default=None,
                        help="Path to train/test split file")
    parser.add_argument("--problem_split", type=str, default=None, choices=["train", "test"],
                        help="Which split to use (train or test)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()

    # Filter features if specified
    features_to_compute = FEATURES
    if args.features:
        features_to_compute = [f for f in FEATURES if f.name in args.features]
        if not features_to_compute:
            print(f"Error: no matching features found for {args.features}")
            print(f"Available: {[f.name for f in FEATURES]}")
            sys.exit(1)
        print(f"Using features: {[f.name for f in features_to_compute]}")

    # Build output directory based on settings
    attempts_tag = "all_attempts" if args.no_failed_only else "failed_only"
    out_dir = run_dir / f"corr{args.correction_rounds}_{attempts_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {run_dir} (correction_rounds={args.correction_rounds})...")
    data = load_run_data(run_dir, correction_rounds=args.correction_rounds)
    print(f"Loaded {len(data)} problems, {sum(len(v) for v in data.values())} total attempts")

    # Filter by problem split (matches base problem id, e.g. putnam_1962_a5)
    if args.problem_split_file and args.problem_split:
        import re as _re
        allowed = set()
        in_section = False
        with open(args.problem_split_file) as sf:
            for line in sf:
                line = line.strip()
                if line.lower().startswith(f"{args.problem_split} ("):
                    in_section = True
                    continue
                if in_section:
                    if not line or line.lower().startswith(("train ", "test ")):
                        break
                    allowed.add(line)
        data = {pid: chains for pid, chains in data.items()
                if _re.match(r'(putnam_\d+_[a-z]\d+)', pid) and
                _re.match(r'(putnam_\d+_[a-z]\d+)', pid).group(1) in allowed}
        print(f"Filtered to {args.problem_split} split: {len(data)} problems ({len(allowed)} base problems)")

    # Compute success rates before any attempt filtering
    success_rates = compute_success_rates(data)

    # Filter by success rate (removes entire problems)
    if args.min_success_rate is not None:
        data = {pid: chains for pid, chains in data.items() if success_rates[pid] > args.min_success_rate}
        success_rates = {pid: success_rates[pid] for pid in data}
        print(f"Filtered to {len(data)} problems with success rate > {args.min_success_rate}")
    if args.max_success_rate is not None:
        data = filter_by_success_rate(data, max_rate=args.max_success_rate)
        success_rates = {pid: success_rates[pid] for pid in data}
        print(f"Filtered to {len(data)} problems with success rate <= {args.max_success_rate}")

    # Filter to failed attempts only (default, removes successful attempts within each problem)
    if not args.no_failed_only:
        data = filter_failed_only(data)
        print(f"Filtered to failed attempts: {sum(len(v) for v in data.values())} attempts across {len(data)} problems")

    # Build DataFrame with pre-computed success rates, only for problems still in data
    success_rates = {pid: success_rates[pid] for pid in data}
    df = pd.DataFrame.from_dict(success_rates, orient="index", columns=["success_rate"])
    df.index.name = "problem_id"

    # Compute and join all features
    for feature in features_to_compute:
        print(f"Computing {feature.name}...")
        feature_df = feature.compute(data)
        df = df.join(feature_df)

    # Drop _std and/or _cv columns if requested
    drop_cols = []
    if args.no_std:
        drop_cols += [c for c in df.columns if c.endswith("_std")]
    if args.no_cv:
        drop_cols += [c for c in df.columns if c.endswith("_cv")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped columns: {drop_cols}")

    # Sort by success rate
    if "output_sflops" in df.columns:
        df["output_sflops"] = df["output_sflops"].round(2)
    df = df.sort_values("success_rate", ascending=False)

    # Sanitize: replace inf with NaN, report
    feature_cols_all = [c for c in df.columns if c != "success_rate"]
    n_inf = np.isinf(df[feature_cols_all].values).sum()
    n_nan = df[feature_cols_all].isna().sum().sum()
    if n_inf > 0 or n_nan > 0:
        print(f"\nWarning: {n_inf} inf and {n_nan} NaN values in features, replacing inf with NaN")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f"\nShape: {df.shape}")
    print(f"\nCorrelations with success_rate:")
    feature_cols = [c for c in df.columns if c != "success_rate"]
    for col in feature_cols:
        corr = df["success_rate"].corr(df[col])
        print(f"  {col}: {corr:.4f}")

    # Correlation heatmap (all features + success_rate)
    corr_matrix = df[["success_rate"] + feature_cols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(feature_cols) * 1.5), max(6, len(feature_cols) * 1.2)))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = list(corr_matrix.columns)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_matrix.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    fig.tight_layout()
    corr_path = out_dir / "correlation_matrix.pdf"
    fig.savefig(corr_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Correlation matrix saved to {corr_path}")

    # Logistic regression (soft labels for fractional success rates)
    log_model = LogisticRegression(soft_labels=True)

    # Per-feature: scatter plot + logistic curve
    print(f"\n{'='*60}")
    print("Per-feature logistic regression (50/50 train/test split):")
    print(f"{'='*60}")
    for col in feature_cols:
        result = log_model.fit(df, feature_cols=[col])
        print(f"  {col}: acc_train={result['acc_train']:.4f}, acc_test={result['accuracy']:.4f}, "
              f"mae={result['mae']:.4f}, rmse={result['rmse']:.4f}, "
              f"coef={result['coefficients'][col]:.6f}, intercept={result['intercept']:.4f} "
              f"(train={result['n_train']}, test={result['n_test']})")

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(df[col], df["success_rate"], alpha=0.7, edgecolors="none", label="data")

        # Plot fitted logistic curve
        x_range = np.linspace(df[col].min(), df[col].max(), 200).reshape(-1, 1)
        x_scaled = result["scaler"].transform(x_range)
        y_prob = result["model"].predict_proba(x_scaled)[:, 1]
        ax.plot(x_range, y_prob, color="red", linewidth=2, label="logistic fit")

        if len(df) <= 100:
            for pid, row in df.iterrows():
                ax.annotate(str(pid).replace("putnam_", ""), (row[col], row["success_rate"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=5, alpha=0.8)

        if len(df) >= 2:
            r = np.corrcoef(df[col], df["success_rate"])[0, 1]
            ax.text(0.02, 0.98, f"r = {r:.3f}, acc = {result['accuracy']:.3f} (n={len(df)})",
                    transform=ax.transAxes, fontsize=11, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        ax.set_title(f"{col} vs Success Rate", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if "similarity" in col:
            ax.set_xlim(0, 1)

        plot_path = out_dir / f"{col}_vs_success.pdf"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved to {plot_path}")

    # Combined model
    if len(feature_cols) > 1:
        print(f"\n{'='*60}")
        print("Combined logistic regression (all features, 50/50 split):")
        print(f"{'='*60}")
        result = log_model.fit(df, feature_cols=feature_cols)
        print(f"  acc_train={result['acc_train']:.4f}, acc_test={result['accuracy']:.4f}, "
              f"mae={result['mae']:.4f}, rmse={result['rmse']:.4f} "
              f"(train={result['n_train']}, test={result['n_test']})")
        for col, coef in result['coefficients'].items():
            print(f"    {col}: {coef:.6f}")
        print(f"    intercept: {result['intercept']:.4f}")

    # Logistic: predicted vs actual plot
    _plot_predicted_vs_actual(
        result, df,
        title="Logistic: Predicted vs Actual Success Rate",
        save_path=out_dir / "logistic_predicted_vs_actual.pdf",
        stat_text="",
    )

    # Linear regression (continuous success_rate)
    lin_model = LinearRegression()

    print(f"\n{'='*60}")
    print("Per-feature linear regression (50/50 train/test split):")
    print(f"{'='*60}")
    for col in feature_cols:
        lin_result = lin_model.fit(df, feature_cols=[col])
        print(f"  {col}: R²_train={lin_result['r2_train']:.4f}, R²_test={lin_result['r2']:.4f}, "
              f"MAE={lin_result['mae']:.4f}, RMSE={lin_result['rmse']:.4f}, "
              f"coef={lin_result['coefficients'][col]:.6f}, intercept={lin_result['intercept']:.4f} "
              f"(train={lin_result['n_train']}, test={lin_result['n_test']})")

        _plot_predicted_vs_actual(
            lin_result, df,
            title=f"Linear ({col}): Predicted vs Actual Success Rate",
            save_path=out_dir / f"linear_{col}_predicted_vs_actual.pdf",
            stat_text="",
        )

    if len(feature_cols) > 1:
        print(f"\n{'='*60}")
        print("Combined linear regression (all features, 50/50 split):")
        print(f"{'='*60}")
        lin_result = lin_model.fit(df, feature_cols=feature_cols)
        print(f"  R²_train={lin_result['r2_train']:.4f}, R²_test={lin_result['r2']:.4f}, "
              f"MAE={lin_result['mae']:.4f}, RMSE={lin_result['rmse']:.4f} "
              f"(train={lin_result['n_train']}, test={lin_result['n_test']})")
        for col, coef in lin_result['coefficients'].items():
            print(f"    {col}: {coef:.6f}")
        print(f"    intercept: {lin_result['intercept']:.4f}")

    # Linear: predicted vs actual plot
    _plot_predicted_vs_actual(
        lin_result, df,
        title="Linear: Predicted vs Actual Success Rate",
        save_path=out_dir / "linear_predicted_vs_actual.pdf",
        stat_text="",
    )

    # Random forest regression
    rf_model = RandomForest()

    print(f"\n{'='*60}")
    print("Per-feature random forest (80/20 train/test split):")
    print(f"{'='*60}")
    for col in feature_cols:
        rf_result = rf_model.fit(df, feature_cols=[col])
        print(f"  {col}: R²_train={rf_result['r2_train']:.4f}, R²_test={rf_result['r2']:.4f}, "
              f"MAE={rf_result['mae']:.4f}, RMSE={rf_result['rmse']:.4f} "
              f"(train={rf_result['n_train']}, test={rf_result['n_test']})")

    if len(feature_cols) > 1:
        print(f"\n{'='*60}")
        print("Combined random forest (all features, 80/20 split):")
        print(f"{'='*60}")
        rf_result = rf_model.fit(df, feature_cols=feature_cols)
        print(f"  R²_train={rf_result['r2_train']:.4f}, R²_test={rf_result['r2']:.4f}, "
              f"MAE={rf_result['mae']:.4f}, RMSE={rf_result['rmse']:.4f} "
              f"(train={rf_result['n_train']}, test={rf_result['n_test']})")
        print("  Feature importance:")
        for col, imp in sorted(rf_result['feature_importance'].items(), key=lambda x: -x[1]):
            print(f"    {col}: {imp:.4f}")

    # Random forest: predicted vs actual plot (test set only)
    _plot_predicted_vs_actual(
        rf_result, df,
        title="Random Forest: Predicted vs Actual Success Rate (test set)",
        save_path=out_dir / "rf_predicted_vs_actual.pdf",
        stat_text="",
    )

    # Map feature column names to state tracker feature names
    _FEATURE_TO_TRACKER = {
        "output_sflops": "avg_cost",
        "lean_similarity": "similarity",
        "normalized_similarity": "normalized_similarity",
        "shallow_similarity_d3": "shallow_similarity",
        "skeleton_similarity": "skeleton_similarity",
        "structural_similarity": "structural_similarity",
        "similarity_high_ratio_tactics": "similarity_high_ratio",
        "error_diversity": "error_diversity",
        "avg_code_length": "avg_code_length",
        "reasoning_uncertainty": "reasoning_uncertainty",
        "error_persistence": "error_persistence",
        "subgoal_repetition": "subgoal_repetition",
    }
    feature_mapping = {col: _FEATURE_TO_TRACKER[col] for col in feature_cols if col in _FEATURE_TO_TRACKER}

    # Save logistic model
    final_result = result  # last fitted model (combined if >1 feature, else single)
    model_params = {
        "features": feature_cols,
        "feature_mapping": feature_mapping,
        "C": final_result["model"].C,
        "default_costs": {
            "8b": {col: float(df[col].mean()) for col in feature_cols},
        },
        "models": {
            "8b": {
                "type": "logistic_regression",
                "coefficients": final_result["model"].coef_.tolist(),
                "intercept": final_result["model"].intercept_.tolist(),
                "classes": final_result["model"].classes_.tolist(),
                "features": feature_cols,
                "scaler": {
                    "mean": final_result["scaler"].mean_.tolist(),
                    "scale": final_result["scaler"].scale_.tolist(),
                },
            },
        },
    }
    model_path = out_dir / "model.json"
    with open(model_path, "w") as f:
        json.dump(model_params, f, indent=2)
    print(f"\nLogistic model saved to {model_path}")

    # Save linear model
    linear_model_params = {
        "features": feature_cols,
        "feature_mapping": feature_mapping,
        "default_costs": {
            "8b": {col: float(df[col].mean()) for col in feature_cols},
        },
        "models": {
            "8b": {
                "type": "linear_regression",
                "coefficients": lin_result["model"].coef_.tolist(),
                "intercept": [float(lin_result["intercept"])],
                "features": feature_cols,
                "scaler": {
                    "mean": lin_result["scaler"].mean_.tolist(),
                    "scale": lin_result["scaler"].scale_.tolist(),
                },
            },
        },
    }
    linear_model_path = out_dir / "linear_model.json"
    with open(linear_model_path, "w") as f:
        json.dump(linear_model_params, f, indent=2)
    print(f"Linear model saved to {linear_model_path}")

    # Gradient boosting regression (clamped to [0, 1])
    gb_model = GradientBoosting()
    print(f"\n{'='*60}")
    print("Per-feature gradient boosting (50/50 train/test split):")
    print(f"{'='*60}")
    for col in feature_cols:
        gb_result = gb_model.fit(df, feature_cols=[col])
        print(f"  {col}: R²_train={gb_result['r2_train']:.4f}, R²_test={gb_result['r2']:.4f}, "
              f"MAE={gb_result['mae']:.4f}, RMSE={gb_result['rmse']:.4f} "
              f"(train={gb_result['n_train']}, test={gb_result['n_test']})")

    if len(feature_cols) > 1:
        print(f"\n{'='*60}")
        print("Combined gradient boosting (all features, 50/50 split):")
        print(f"{'='*60}")
        gb_result = gb_model.fit(df, feature_cols=feature_cols)
        print(f"  R²_train={gb_result['r2_train']:.4f}, R²_test={gb_result['r2']:.4f}, "
              f"MAE={gb_result['mae']:.4f}, RMSE={gb_result['rmse']:.4f} "
              f"(train={gb_result['n_train']}, test={gb_result['n_test']})")
        print("  Feature importance:")
        for col, imp in sorted(gb_result['feature_importance'].items(), key=lambda x: -x[1]):
            print(f"    {col}: {imp:.4f}")

    _plot_predicted_vs_actual(
        gb_result, df,
        title="Gradient Boosting: Predicted vs Actual Success Rate (test set)",
        save_path=out_dir / "gb_predicted_vs_actual.pdf",
        stat_text="",
    )

    gb_model_params = {
        "features": feature_cols,
        "feature_mapping": feature_mapping,
        "default_costs": {
            "8b": {col: float(df[col].mean()) for col in feature_cols},
        },
        "models": {
            "8b": gb_to_json_dict(gb_result["model"], feature_cols),
        },
    }
    gb_model_path = out_dir / "gradient_boosting_model.json"
    with open(gb_model_path, "w") as f:
        json.dump(gb_model_params, f, indent=2)
    print(f"Gradient boosting model saved to {gb_model_path}")

    # Save features CSV
    output_path = args.output or out_dir / "features.csv"
    df.to_csv(output_path, sep="\t")
    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    main()
