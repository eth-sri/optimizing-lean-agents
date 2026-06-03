"""
Model Analysis viewer.

Compares predicted probabilities from logistic models against empirical
success rates and oracle probabilities. Provides calibration analysis,
deviation plots, and temperature scaling diagnostics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import math

from .trajectory_analysis_viewer import discover_simulation_runs, DEFAULT_SIM_DIR
from .individual_trajectory_viewer import list_configs, list_seeds, list_problems

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_model_json(model_path: str) -> dict:
    """Load a model JSON file."""
    with open(model_path) as f:
        return json.load(f)


@st.cache_data
def load_trajectory_features(config_path: str) -> pd.DataFrame:
    """Scan trajectories and extract per-step features for prove actions.

    Returns DataFrame with columns:
        problem_id, seed, step, prover_model, avg_cost, similarity,
        oracle_p, num_attempts, predicted_prob, success
    """
    traj_dir = Path(config_path) / "trajectories"
    if not traj_dir.exists():
        return pd.DataFrame()

    rows = []
    for seed_dir in sorted(traj_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        seed_name = seed_dir.name
        for traj_file in sorted(seed_dir.glob("*.json")):
            try:
                with open(traj_file) as f:
                    traj = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            problem_id = traj.get("problem_id", traj_file.stem)
            for step in traj.get("steps", []):
                action = step.get("action", {})
                if action.get("type") != "prove":
                    continue

                prover_model = action.get("model", "unknown")
                ts = step.get("tracked_state", {})
                result = step.get("result", {})

                # Extract features
                oracle_p_dict = ts.get("oracle_p", {})
                num_attempts_dict = ts.get("num_attempts", {})
                avg_cost_dict = ts.get("avg_cost", {})
                similarity = ts.get("similarity")

                # Extract all predicted_prob* keys
                pp_values = {}
                for key, val in ts.items():
                    if key.startswith("predicted_prob"):
                        if isinstance(val, dict) and prover_model in val:
                            pp_values[key] = val[prover_model]

                row = {
                    "problem_id": problem_id,
                    "seed": seed_name,
                    "step": step.get("step", 0),
                    "prover_model": prover_model,
                    "avg_cost": avg_cost_dict.get(prover_model),
                    "similarity": similarity,
                    "oracle_p": oracle_p_dict.get(prover_model),
                    "num_attempts": num_attempts_dict.get(prover_model, 0),
                    "success": result.get("success", False),
                }
                row.update(pp_values)
                rows.append(row)

    return pd.DataFrame(rows)


def run_inference(features_df: pd.DataFrame, model_dict: dict, prover_model: str) -> pd.Series:
    """Run model inference on feature rows, return predicted P(success)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    model_info = model_dict.get("models", {}).get(prover_model)
    if model_info is None:
        return pd.Series(dtype=float)

    if model_info.get("type") == "constant":
        return pd.Series(float(model_info["value"]), index=features_df.index)

    feature_names = model_dict["features"]
    feature_mapping = model_dict.get("feature_mapping", {})
    default_costs = model_dict.get("default_costs", {}).get(prover_model, {})

    # Reconstruct model
    lr = LogisticRegression()
    lr.coef_ = np.array(model_info["coefficients"])
    lr.intercept_ = np.array(model_info["intercept"])
    lr.classes_ = np.array(model_info.get("classes", [0, 1]))

    scaler = None
    if "scaler" in model_info:
        scaler = StandardScaler()
        scaler.mean_ = np.array(model_info["scaler"]["mean"])
        scaler.scale_ = np.array(model_info["scaler"]["scale"])
        scaler.n_features_in_ = len(scaler.mean_)

    # Build feature matrix
    X = []
    valid_mask = []
    for _, row in features_df.iterrows():
        fv = []
        valid = True
        for feat in feature_names:
            col = feature_mapping.get(feat, feat)
            val = row.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                default_val = default_costs.get(feat)
                if default_val is not None:
                    val = default_val
                else:
                    valid = False
                    break
            fv.append(float(val))
        X.append(fv)
        valid_mask.append(valid)

    results = pd.Series(np.nan, index=features_df.index)
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    if not valid_indices:
        return results

    X_valid = np.array([X[i] for i in valid_indices])
    if scaler is not None:
        X_valid = scaler.transform(X_valid)

    probs = lr.predict_proba(X_valid)[:, 1]
    for idx, prob in zip(valid_indices, probs):
        results.iloc[idx] = prob

    return results


@st.cache_data
def compute_empirical_success_rates(full_proof_path: str) -> pd.DataFrame:
    """Load minified JSON and compute per-problem empirical success rates."""
    with open(full_proof_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # success = pass AND complete
    df["success"] = df["pass"] & df["complete"]

    grouped = df.groupby("origin_problem_id").agg(
        n_attempts=("success", "count"),
        n_successes=("success", "sum"),
    ).reset_index()
    grouped["empirical_rate"] = grouped["n_successes"] / grouped["n_attempts"]
    grouped.rename(columns={"origin_problem_id": "problem_id"}, inplace=True)
    return grouped


def build_per_problem_summary(
    features_df: pd.DataFrame,
    empirical_df: pd.DataFrame,
    prover_model: str,
    predicted_col: str = "predicted_prob",
) -> pd.DataFrame:
    """Build per-problem summary with predicted prob, oracle, and empirical rates."""
    pdf = features_df[features_df["prover_model"] == prover_model].copy()
    if pdf.empty:
        return pd.DataFrame()

    agg = pdf.groupby("problem_id").agg(
        avg_predicted_prob=(predicted_col, "mean"),
        avg_oracle_p=("oracle_p", "mean"),
        total_attempts=("success", "count"),
        total_successes=("success", "sum"),
    ).reset_index()
    agg["sim_success_rate"] = agg["total_successes"] / agg["total_attempts"]

    # Merge with empirical
    merged = agg.merge(empirical_df, on="problem_id", how="left")
    merged["deviation_pred_emp"] = merged["avg_predicted_prob"] - merged["empirical_rate"]
    merged["deviation_pred_oracle"] = merged["avg_predicted_prob"] - merged["avg_oracle_p"]
    return merged


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def compute_calibration_metrics(predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10):
    """Compute ECE, Brier score, and calibration curve data."""
    brier = float(np.mean((predicted - actual) ** 2))

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_means_pred = []
    bin_means_actual = []
    bin_counts = []
    ece = 0.0
    n = len(predicted)

    for i in range(n_bins):
        mask = (predicted >= bins[i]) & (predicted < bins[i + 1])
        if i == n_bins - 1:  # include right edge
            mask = mask | (predicted == bins[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        mean_pred = predicted[mask].mean()
        mean_actual = actual[mask].mean()
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_means_pred.append(mean_pred)
        bin_means_actual.append(mean_actual)
        bin_counts.append(count)
        ece += abs(mean_pred - mean_actual) * count / n

    return {
        "ece": ece,
        "brier": brier,
        "bin_centers": bin_centers,
        "bin_means_pred": bin_means_pred,
        "bin_means_actual": bin_means_actual,
        "bin_counts": bin_counts,
    }


def fit_temperature(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Find temperature T that minimizes ECE via grid search."""
    best_t = 1.0
    best_ece = float("inf")
    for t in np.arange(0.1, 5.01, 0.05):
        logits = np.clip(np.log(predicted / (1 - predicted + 1e-10) + 1e-10), -20, 20)
        calibrated = 1 / (1 + np.exp(-logits / t))
        cal = compute_calibration_metrics(calibrated, actual)
        if cal["ece"] < best_ece:
            best_ece = cal["ece"]
            best_t = t
    return best_t


def apply_temperature(predicted: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling: calibrated_p = sigmoid(logit(p) / T)."""
    eps = 1e-10
    logits = np.clip(np.log(predicted / (1 - predicted + eps) + eps), -20, 20)
    return 1 / (1 + np.exp(-logits / T))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_model_params(model_dict: dict, model_name: str):
    """Display model parameters in a formatted way."""
    st.markdown(f"**Features:** {model_dict.get('features', [])}")
    st.markdown(f"**C (regularization):** {model_dict.get('C', 'N/A')}")

    feature_mapping = model_dict.get("feature_mapping", {})
    if feature_mapping:
        st.markdown(f"**Feature mapping:** {feature_mapping}")

    for prover_model, minfo in model_dict.get("models", {}).items():
        st.markdown(f"#### Prover model: {prover_model}")
        if minfo.get("type") == "constant":
            st.markdown(f"Type: constant, value = {minfo['value']}")
            continue

        st.markdown(f"**Type:** {minfo.get('type', 'unknown')}")

        features = model_dict.get("features", [])
        coeffs = minfo.get("coefficients", [[]])[0]
        scaler_info = minfo.get("scaler", {})
        means = scaler_info.get("mean", [])
        scales = scaler_info.get("scale", [])
        defaults = model_dict.get("default_costs", {}).get(prover_model, {})

        rows = []
        for i, feat in enumerate(features):
            rows.append({
                "Feature": feat,
                "Coefficient": coeffs[i] if i < len(coeffs) else "N/A",
                "Scaler Mean": means[i] if i < len(means) else "N/A",
                "Scaler Scale": scales[i] if i < len(scales) else "N/A",
                "Default": defaults.get(feat, "N/A"),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown(f"**Intercept:** {minfo.get('intercept', 'N/A')}")


def render_plots(summary_df: pd.DataFrame, prover_model: str, model_name: str):
    """Render calibration/analysis plots for one model x prover combination."""
    if summary_df.empty:
        st.warning("No data for this combination.")
        return

    # Filter to rows with valid data
    df = summary_df.dropna(subset=["avg_predicted_prob", "empirical_rate"]).copy()
    if df.empty:
        st.warning("No rows with both predicted prob and empirical rate.")
        return

    predicted = df["avg_predicted_prob"].values.astype(float)
    empirical = df["empirical_rate"].values.astype(float)

    st.caption(f"Predicted range: [{np.nanmin(predicted):.4f}, {np.nanmax(predicted):.4f}], "
               f"Empirical range: [{np.nanmin(empirical):.4f}, {np.nanmax(empirical):.4f}]")

    pearson_r = np.corrcoef(predicted, empirical)[0, 1] if len(df) > 1 else float("nan")

    mc1, mc2 = st.columns(2)
    mc1.metric("Pearson r (pred vs emp)", f"{pearson_r:.4f}")
    mc2.metric("N problems", len(df))

    # --- Scatter: avg predicted prob vs empirical success rate ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), name="y=x",
    ))
    fig.add_trace(go.Scatter(
        x=empirical.tolist(), y=predicted.tolist(), mode="markers",
        name="Problems",
        marker=dict(size=8, color="#1f77b4"),
        text=df["problem_id"].tolist(),
        hovertemplate="%{text}<br>Empirical: %{x:.3f}<br>Avg Predicted: %{y:.3f}<extra></extra>",
    ))
    # Best fit line
    slope, intercept = np.polyfit(empirical, predicted, 1)
    fit_x = np.array([0, 1])
    fit_y = slope * fit_x + intercept
    fig.add_trace(go.Scatter(
        x=fit_x.tolist(), y=fit_y.tolist(), mode="lines",
        line=dict(color="red", width=2), name=f"Fit: y={slope:.3f}x+{intercept:.3f}",
    ))
    fig.update_layout(
        height=600,
        title=f"Predicted Prob vs Empirical Success Rate (r={pearson_r:.3f}, y={slope:.3f}x+{intercept:.3f})",
        xaxis_title="Empirical Success Rate",
        yaxis_title="Avg Predicted Probability",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_model_analysis_viewer():
    col1, col2 = st.columns([6, 1])
    with col1:
        st.header("Model Analysis")
    with col2:
        if st.button("Refresh", key="refresh_model_analysis", help="Clear cached data and reload"):
            st.cache_data.clear()
            st.rerun()

    # --- Input controls in main area ---
    sim_dir = str(DEFAULT_SIM_DIR)
    runs = discover_simulation_runs(sim_dir)

    if not runs:
        st.warning(f"No simulation runs found in `{sim_dir}`.")
        return

    with st.expander("Data Sources", expanded=True):
        # Row 1: Run selector + config selector
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            run_labels = [f"{r['name']}  ({r['type']})" for r in runs]
            selected_run_label = st.selectbox(
                "Simulation run",
                options=run_labels,
                key="model_analysis_run",
            )
            selected_run = runs[run_labels.index(selected_run_label)]

        with r1c2:
            # Config selector (for sweeps with multiple configs)
            if selected_run["type"] == "sweep":
                configs = list_configs(selected_run["path"])
                config_labels = [c["label"] for c in configs]
                selected_config_label = st.selectbox(
                    "Config",
                    options=config_labels,
                    key="model_analysis_config",
                )
                config_path = configs[config_labels.index(selected_config_label)]["path"]
            else:
                st.text_input("Config", value="(single run)", disabled=True)
                config_path = selected_run["path"]

        # Row 2: Full proof path + model JSONs
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            full_proof_path = st.text_input(
                "Full proof minified JSON path",
                value=str(PROJECT_ROOT / "outputs" / "putnam" / "full_proof_8b" / "minified_8b.json"),
                key="model_analysis_full_proof",
            )
        with r2c2:
            pass  # reserved for future controls

    # --- Load data ---
    with st.spinner("Loading trajectory features..."):
        features_df = load_trajectory_features(config_path)

    if features_df.empty:
        st.error(f"No trajectory data found in {config_path}")
        return

    st.caption(f"Loaded {len(features_df)} prove steps from {features_df['problem_id'].nunique()} problems, "
               f"{features_df['seed'].nunique()} seeds")

    # Load empirical rates
    empirical_df = None
    if Path(full_proof_path).exists():
        with st.spinner("Loading empirical success rates..."):
            empirical_df = compute_empirical_success_rates(full_proof_path)
        st.caption(f"Empirical rates: {len(empirical_df)} problems from {full_proof_path}")
    else:
        st.warning(f"Full proof file not found: {full_proof_path}")

    # --- Predicted prob column selector ---
    pp_cols = [c for c in features_df.columns if c.startswith("predicted_prob")]
    if not pp_cols:
        st.warning("No predicted_prob columns found in trajectory data.")
        return

    predicted_col = st.selectbox(
        "Predicted probability field",
        options=pp_cols,
        key="model_analysis_pp_col",
    )

    # --- Per prover model plots ---
    prover_models = features_df["prover_model"].unique().tolist()

    if empirical_df is None:
        st.warning("No empirical data available for comparison.")
        return

    if len(prover_models) == 1:
        pm = prover_models[0]
        st.markdown(f"**Prover model: {pm}**")
        summary = build_per_problem_summary(features_df, empirical_df, pm, predicted_col)
        render_plots(summary, pm, predicted_col)
    else:
        prover_tabs = st.tabs(prover_models)
        for prover_tab, pm in zip(prover_tabs, prover_models):
            with prover_tab:
                summary = build_per_problem_summary(features_df, empirical_df, pm, predicted_col)
                render_plots(summary, pm, predicted_col)
