"""
Simulation Runs viewer (formerly "Individual Trajectory viewer").

Provides tabs:
1. Analyze Trajectories – step-by-step trajectory table and per-target charts
2. Model Parameters   – compare logistic-regression model_params.json across configs
3. Performance Analysis – calibration plots, per-problem bars, prediction scatter
4. Noise – prediction noise analysis (predicted_prob vs oracle_p)
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
import json
import math

from .trajectory_analysis_viewer import discover_simulation_runs, DEFAULT_SIM_DIR


# ---------------------------------------------------------------------------
# Data discovery helpers
# ---------------------------------------------------------------------------

@st.cache_data
def list_configs(sweep_path: str) -> List[Dict[str, Any]]:
    """List config_NNN directories that have params.json."""
    p = Path(sweep_path)
    configs = []
    for entry in sorted(p.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("config_"):
            continue
        params = {}
        params_file = entry / "params.json"
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
        # Build compact label from key params (flatten _state_tracker)
        flat_params = {k: v for k, v in params.items() if k != "_state_tracker"}
        if "_state_tracker" in params and isinstance(params["_state_tracker"], dict):
            for sk, sv in params["_state_tracker"].items():
                flat_params[f"st_{sk}"] = sv
        label_parts = []
        for k in ("lambda_val", "sigma", "st_sigma", "max_n", "max_breakdowns"):
            if k in flat_params:
                label_parts.append(f"{k}={flat_params[k]}")
        label = f"{entry.name} ({', '.join(label_parts)})" if label_parts else entry.name
        configs.append({
            "name": entry.name,
            "path": str(entry),
            "params": params,
            "label": label,
        })
    return configs


@st.cache_data
def list_seeds(config_path: str) -> List[str]:
    """List seed_N directories inside a config's trajectories/ dir."""
    traj_dir = Path(config_path) / "trajectories"
    if not traj_dir.exists():
        return []
    seeds = []
    for entry in sorted(traj_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("seed_"):
            seeds.append(entry.name)
    return seeds


@st.cache_data
def get_trajectory_stats(seed_path: str, problem: str) -> Dict[str, Any]:
    """Get solved status and step count for a single trajectory."""
    f = Path(seed_path) / f"{problem}.json"
    if not f.exists():
        return {}
    try:
        with open(f) as fh:
            traj = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}

    # Extract oracle_p from decision metadata of steps
    oracle_p = None
    for step in traj.get("steps", []):
        dm = step.get("decision_metadata") or {}
        if "oracle_p" in dm:
            oracle_p = dm["oracle_p"]
            break
        # Also check action_scores entries
        for scored in dm.get("action_scores", []):
            if "oracle_p" in scored:
                oracle_p = scored["oracle_p"]
                break
        if oracle_p is not None:
            break

    return {
        "solved": traj.get("solved", False),
        "steps": traj.get("num_steps", len(traj.get("steps", []))),
        "oracle_p": oracle_p,
    }


@st.cache_data
def list_problems(seed_path: str) -> List[str]:
    """List .json problem files in a seed directory."""
    p = Path(seed_path)
    if not p.exists():
        return []
    return sorted(f.stem for f in p.iterdir() if f.suffix == ".json")


def load_trajectory(file_path: str) -> dict:
    """Load a single trajectory JSON. Not cached for minimal memory."""
    with open(file_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Color map for action types
# ---------------------------------------------------------------------------

ACTION_COLORS = {
    "prove_8b": "#1f77b4",       # blue
    "prove_32b": "#ff7f0e",      # orange
    "correct": "#2ca02c",        # green
    "decompose": "#9467bd",      # purple
    "create_breakdown": "#8c564b",  # brown
    "terminate": "#d62728",      # red
}

# Colors for overlaid scalar fields on combined charts
OVERLAY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf",
]


def _action_key(action: dict) -> str:
    """Get a display key for an action dict."""
    atype = action.get("type", "unknown")
    model = action.get("model")
    if model:
        return f"{atype}_{model}"
    return atype


def _action_display(action: dict) -> str:
    """Get a human-readable display string for an action."""
    atype = action.get("type", "unknown")
    model = action.get("model")
    if model:
        return f"{atype} ({model})"
    return atype


# ---------------------------------------------------------------------------
# Decision metadata helpers
# ---------------------------------------------------------------------------

def _flatten_decision_metadata(dm: dict, chosen_action: dict) -> dict:
    """Extract all scalar values from decision metadata for display.

    For action_scores, includes fields from ALL candidate actions with
    action-prefixed keys (e.g. ``prove_8b_score``, ``terminate_p``).
    Skips nested dicts/lists (except action_scores which is handled specially).
    """
    flat = {}
    if not dm:
        return flat

    for k, v in dm.items():
        if k == "action_scores":
            for scored in v:
                sa = scored.get("action", {})
                prefix = _action_key(sa)
                for sk, sv in scored.items():
                    if sk == "action":
                        continue
                    if isinstance(sv, (int, float, bool, str)) or sv is None:
                        flat[f"{prefix}_{sk}"] = sv
        elif k == "action":
            continue
        elif isinstance(v, (int, float, bool, str)) or v is None:
            flat[k] = v

    return flat


def _flatten_tracked_state(step: dict) -> dict:
    """Flatten tracked_state from a step into display-friendly keys.

    tracked_state looks like: {noisy_p: {8b: 0.01}, num_attempts: {8b: 3}}
    Output: {noisy_p_8b: 0.01, num_attempts_8b: 3}
    """
    tracked = step.get("tracked_state") or {}
    flat = {}
    for k, v in tracked.items():
        if isinstance(v, dict):
            for mk, mv in v.items():
                if isinstance(mv, (int, float, bool, str)) or mv is None:
                    flat[f"{k}_{mk}"] = mv
        elif isinstance(v, (int, float, bool, str)) or v is None:
            flat[k] = v
    return flat


def _format_dm_value(v) -> str:
    """Format a decision metadata value for table display."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        abs_v = abs(v)
        if abs_v == 0:
            return "0"
        if abs_v < 0.001 or abs_v >= 1e6:
            return f"{v:.2e}"
        if abs_v < 1:
            return f"{v:.4f}"
        return f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 1000 else str(v)
    return str(v)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_individual_trajectory_viewer():
    col1, col2 = st.columns([6, 1])
    with col1:
        st.header("Analyze Simulation Runs")
    with col2:
        if st.button("Refresh", key="refresh_indiv_traj", help="Clear cached data and reload"):
            st.cache_data.clear()
            st.rerun()

    # --- Sim directory ---
    sim_dir = st.text_input(
        "Simulations directory",
        value=str(DEFAULT_SIM_DIR),
        key="indiv_traj_sim_dir",
    )

    runs = discover_simulation_runs(sim_dir)
    # Only show sweep runs (they have config subdirs with trajectories)
    sweep_runs = [r for r in runs if r["type"] == "sweep"]
    if not sweep_runs:
        st.warning(f"No sweep runs found in `{sim_dir}`.")
        return

    # --- Selection controls ---
    # Row 1: run selector
    run_names = [r["name"] for r in sweep_runs]
    selected_run_name = st.selectbox("Sweep run", run_names, key="indiv_traj_run")
    selected_run = next(r for r in sweep_runs if r["name"] == selected_run_name)

    # Row 2: per-sweep-param dropdowns
    configs = list_configs(selected_run["path"])
    if not configs:
        st.warning("No config directories found.")
        return

    # Detect which scalar params vary across configs
    all_param_keys = set()
    for c in configs:
        for k, v in c["params"].items():
            if isinstance(v, (int, float, str, bool)):
                all_param_keys.add(k)

    sweep_param_options = {}  # param_name -> sorted unique values
    for key in sorted(all_param_keys):
        vals = list({c["params"][key] for c in configs
                     if key in c["params"] and isinstance(c["params"][key], (int, float, str, bool))})
        # Sort numerically when possible (handles string-encoded numbers like "1e-9")
        try:
            vals.sort(key=lambda v: float(v))
        except (TypeError, ValueError):
            try:
                vals.sort()
            except TypeError:
                vals.sort(key=str)
        if len(vals) > 1:
            sweep_param_options[key] = vals

    selected_param_values = {}
    if sweep_param_options:
        cols_sw = st.columns(len(sweep_param_options))
        for col, (pname, pvals) in zip(cols_sw, sweep_param_options.items()):
            with col:
                selected = st.selectbox(pname, pvals, key=f"indiv_traj_sweep_{pname}")
                selected_param_values[pname] = selected

    # Filter configs to match all selected param values
    matching = configs
    for pname, pval in selected_param_values.items():
        matching = [c for c in matching if c["params"].get(pname) == pval]

    if not matching:
        st.warning("No config matches the selected parameters.")
        return
    selected_config = matching[0]

    # --- Tabs ---
    st.markdown("---")
    tab_traj, tab_model, tab_noise = st.tabs(
        ["Analyze Trajectories", "Model & Calibration", "Noise"]
    )

    with tab_traj:
        _render_trajectory_tab(selected_config)

    with tab_model:
        _render_model_and_calibration_tab(selected_run["path"], selected_config, sweep_param_options)

    with tab_noise:
        _render_noise_tab(selected_run["path"])


# ---------------------------------------------------------------------------
# Tab 1: Analyze Trajectories (original trajectory viewer)
# ---------------------------------------------------------------------------

def _render_trajectory_tab(selected_config: dict):
    """Render the trajectory analysis tab (problem/seed selection + step table + per-target charts)."""
    seeds = list_seeds(selected_config["path"])
    if not seeds:
        st.warning("No seed directories found.")
        return
    first_seed_path = str(Path(selected_config["path"]) / "trajectories" / seeds[0])

    # Row 3: problem + seed side by side
    sel_cols = st.columns(2)

    with sel_cols[0]:
        problems = list_problems(first_seed_path)
        if not problems:
            st.warning("No trajectory files found.")
            return
        # Build labels with oracle_p as n/256
        problem_labels = []
        for prob in problems:
            ts = get_trajectory_stats(first_seed_path, prob)
            op = ts.get("oracle_p") if ts else None
            if op is not None:
                n_256 = round(op * 256)
                problem_labels.append(f"{prob} ({n_256}/256)")
            else:
                problem_labels.append(prob)
        selected_label = st.selectbox("Problem", problem_labels, key="indiv_traj_problem")
        selected_problem = problems[problem_labels.index(selected_label)]

    with sel_cols[1]:
        seed_labels = []
        for s in seeds:
            sp = str(Path(selected_config["path"]) / "trajectories" / s)
            ts = get_trajectory_stats(sp, selected_problem)
            if ts:
                mark = "\u2705" if ts["solved"] else "\u274c"
                seed_labels.append(f"{s} {mark} {ts['steps']} steps")
            else:
                seed_labels.append(s)
        selected_seed_label = st.selectbox("Seed", seed_labels, key="indiv_traj_seed")
        selected_seed = seeds[seed_labels.index(selected_seed_label)]

    seed_path = str(Path(selected_config["path"]) / "trajectories" / selected_seed)

    # --- Load trajectory ---
    traj_file = str(Path(seed_path) / f"{selected_problem}.json")
    traj = load_trajectory(traj_file)

    # --- Header ---
    st.markdown("---")
    h_cols = st.columns(4)
    with h_cols[0]:
        st.metric("Problem", traj["problem_id"])
    with h_cols[1]:
        solved = traj.get("solved", False)
        st.metric("Status", "Solved" if solved else "Not Solved")
    with h_cols[2]:
        total_cost = traj.get("total_cost", {})
        total_sflops = total_cost.get("input_sflops", 0) + total_cost.get("output_sflops", 0)
        st.metric("Total Cost", f"{total_sflops / 1e6:.2f}M SFLOPs")
    with h_cols[3]:
        st.metric("Total Steps", traj.get("num_steps", len(traj.get("steps", []))))

    # --- Build step data ---
    steps = traj.get("steps", [])
    if not steps:
        st.info("No steps in this trajectory.")
        return

    # --- Full trajectory summary table ---
    st.subheader("Full Trajectory Summary")

    # First pass: discover tracked_state + decision metadata keys across steps
    all_tracked_keys = []
    seen_tracked_keys = set()
    step_tracked_flat = []

    all_dm_keys = []
    seen_dm_keys = set()
    step_dm_flat = []

    for step in steps:
        action = step.get("action", {})
        dm = step.get("decision_metadata") or {}

        tracked_flat = _flatten_tracked_state(step)
        step_tracked_flat.append(tracked_flat)
        for k in tracked_flat:
            if k not in seen_tracked_keys:
                all_tracked_keys.append(k)
                seen_tracked_keys.add(k)

        dm_flat = _flatten_decision_metadata(dm, action)
        step_dm_flat.append(dm_flat)
        for k in dm_flat:
            if k not in seen_dm_keys:
                all_dm_keys.append(k)
                seen_dm_keys.add(k)

    # Build rows: fixed cols + tracked state + decision metadata
    table_rows = []
    for step, tracked_flat, dm_flat in zip(steps, step_tracked_flat, step_dm_flat):
        state = step.get("state", {})
        action = step.get("action", {})
        result = step.get("result", {})

        cost = result.get("cost", {})
        cost_sflops = cost.get("input_sflops", 0) + cost.get("output_sflops", 0)
        used_lemmas = result.get("used_lemma_ids")

        row = {
            "Step": step.get("step", 0),
            "Target": state.get("target_id", ""),
            "Action": _action_display(action),
            "Success": result.get("success", False),
            "Cost": f"{cost_sflops:,}",
        }
        if used_lemmas:
            row["Lemmas"] = ",".join(str(l) for l in used_lemmas)

        for k in all_tracked_keys:
            row[f"s| {k}"] = _format_dm_value(tracked_flat.get(k))

        for k in all_dm_keys:
            row[f"d| {k}"] = _format_dm_value(dm_flat.get(k))

        table_rows.append(row)

    df = pd.DataFrame(table_rows)

    # Color-code rows
    def _row_color(row):
        if row["Success"]:
            return ["background-color: rgba(0, 180, 0, 0.15)"] * len(row)
        else:
            return ["background-color: rgba(220, 50, 50, 0.10)"] * len(row)

    styled = df.style.apply(_row_color, axis=1)
    st.dataframe(styled, use_container_width=True, height=min(400, 35 * len(df) + 38), hide_index=True)

    # --- Group steps by target ---
    target_order = OrderedDict()  # target_id -> list of (global_step_idx, step_dict)
    for i, step in enumerate(steps):
        tid = step.get("state", {}).get("target_id", "unknown")
        if tid not in target_order:
            target_order[tid] = []
        target_order[tid].append((i, step))

    # --- Per-target tabs ---
    st.subheader("Per-Target Analysis")
    target_ids = list(target_order.keys())
    tabs = st.tabs(target_ids)

    for tab, tid in zip(tabs, target_ids):
        with tab:
            target_steps = target_order[tid]
            _render_target_tab(tid, target_steps)


# ---------------------------------------------------------------------------
# Tab 2: Model & Calibration (on training data)
# ---------------------------------------------------------------------------

@st.cache_data
def _load_model_params(sweep_path: str) -> Optional[dict]:
    """Load model_params.json from the first config that has one (identical across configs)."""
    p = Path(sweep_path)
    for entry in sorted(p.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("config_"):
            continue
        f = entry / "model_params.json"
        if f.exists():
            try:
                with open(f) as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _parse_feature_spec(name: str):
    """Parse 'log:num_attempts' into (transform, base_name)."""
    if ":" in name:
        transform, base = name.split(":", 1)
        return transform, base
    return None, name


@st.cache_data
def _load_training_data_with_predictions(
    training_base_dir: str, sigma: float,
    model_coeffs: list, model_intercept: float,
    feature_specs: list, model_key_model: str,
) -> Optional[pd.DataFrame]:
    """Load training trajectories for a given sigma, apply the logistic model, return per-step data.

    Returns DataFrame: problem_id, step, noisy_p, num_attempts, predicted_p, success, oracle_p
    """
    base = Path(training_base_dir)
    # Find the config for this sigma via sweep_config.json or params.json
    sigma_config_dir = None
    sweep_cfg_file = base / "sweep_config.json"
    if sweep_cfg_file.exists():
        try:
            with open(sweep_cfg_file) as fh:
                sweep_cfg = json.load(fh)
            for i, combo in enumerate(sweep_cfg.get("param_combos", [])):
                cfg_sigma = combo.get("_state_tracker", {}).get("sigma")
                if cfg_sigma is not None and abs(cfg_sigma - sigma) < 1e-9:
                    sigma_config_dir = base / f"config_{i:03d}"
                    break
        except (json.JSONDecodeError, OSError):
            pass

    if sigma_config_dir is None:
        # Fallback: scan params.json in each config
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("config_"):
                continue
            pf = entry / "params.json"
            if pf.exists():
                try:
                    params = json.load(open(pf))
                    cfg_sigma = params.get("_state_tracker", {}).get("sigma", params.get("sigma"))
                    if cfg_sigma is not None and abs(float(cfg_sigma) - sigma) < 1e-9:
                        sigma_config_dir = entry
                        break
                except (json.JSONDecodeError, OSError, TypeError):
                    continue

    if sigma_config_dir is None or not (sigma_config_dir / "trajectories").exists():
        return None

    parsed_specs = [_parse_feature_spec(f) for f in feature_specs]

    records = []
    traj_dir = sigma_config_dir / "trajectories"
    for seed_dir in sorted(traj_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        for traj_file in sorted(seed_dir.glob("*.json")):
            try:
                with open(traj_file) as fh:
                    traj = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            problem_id = traj.get("problem_id", traj_file.stem)
            for step in traj.get("steps", []):
                tracked = step.get("tracked_state") or {}
                action = step.get("action", {})
                result = step.get("result", {})
                action_model = action.get("model", "")

                if action_model != model_key_model:
                    continue

                # Extract features
                feat_vals = []
                raw_vals = {}
                valid = True
                for transform, base_name in parsed_specs:
                    feat_data = tracked.get(base_name, {})
                    if isinstance(feat_data, dict):
                        val = feat_data.get(action_model)
                    else:
                        val = feat_data
                    if val is None:
                        valid = False
                        break
                    raw_vals[base_name] = float(val)
                    fval = float(val)
                    
                    if transform == "log":
                        fval = math.log1p(fval)
                    elif transform == "sqrt":
                        fval = math.sqrt(max(fval, 0))
                    elif transform == "1/x":
                        fval = 1 / fval if fval != 0 else 0.0

                    feat_vals.append(fval)

                if not valid:
                    continue

                # Apply logistic model
                logit = model_intercept
                for c, fv in zip(model_coeffs, feat_vals):
                    logit += c * fv
                predicted_p = _sigmoid(logit)

                oracle_data = tracked.get("oracle_p", {})
                oracle_p = oracle_data.get(action_model) if isinstance(oracle_data, dict) else oracle_data

                records.append({
                    "problem_id": problem_id,
                    "seed": seed_dir.name,
                    "step": step.get("step", 0),
                    "noisy_p": raw_vals.get("noisy_p"),
                    "num_attempts": raw_vals.get("num_attempts"),
                    "predicted_p": predicted_p,
                    "success": bool(result.get("success", False)),
                    "oracle_p": float(oracle_p) if oracle_p is not None else None,
                })

    if not records:
        return None
    df = pd.DataFrame(records)
    df["predicted_p"] = pd.to_numeric(df["predicted_p"], errors="coerce")
    df["oracle_p"] = pd.to_numeric(df["oracle_p"], errors="coerce")
    return df


def _render_model_and_calibration_tab(sweep_path: str, selected_config: dict, sweep_param_options: dict):
    """Show model parameter heatmap + calibration on training data."""
    # --- Model parameter heatmap ---
    st.subheader("Model Parameters")
    mp = _load_model_params(sweep_path)
    if mp is None:
        st.warning("No model_params.json found.")
        return

    # Sigma selector (independent of lambda)
    sigma_vals = sweep_param_options.get("sigma", [])
    config_sigma = selected_config["params"].get("sigma")
    if sigma_vals:
        default_idx = sigma_vals.index(config_sigma) if config_sigma in sigma_vals else 0
        selected_sigma = st.selectbox(
            "sigma (model noise level)",
            sigma_vals,
            index=default_idx,
            key="model_cal_sigma",
        )
    elif config_sigma is not None:
        selected_sigma = config_sigma
    else:
        st.warning("No sigma parameter found.")
        return

    _render_model_heatmap(mp, float(selected_sigma))

    st.markdown("---")

    # --- Calibration on training data ---
    st.subheader("Calibration (on training data)")

    # Find the training base dir from config params or model_params.json
    training_base_dir = None
    for cfg_dir in sorted(Path(sweep_path).iterdir()):
        if not cfg_dir.is_dir() or not cfg_dir.name.startswith("config_"):
            continue
        pf = cfg_dir / "params.json"
        if pf.exists():
            params = json.load(open(pf))
            training_base_dir = params.get("trajectory_base_dir")
            break

    # Fall back to trajectory_base_dir stored in model_params.json (pretrained models)
    if not training_base_dir and mp:
        training_base_dir = mp.get("trajectory_base_dir")

    if not training_base_dir:
        st.warning("No `trajectory_base_dir` found in config params or model_params.json.")
        return

    # Resolve relative path
    training_base = Path(training_base_dir)
    if not training_base.is_absolute():
        # Try relative to repo root
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        training_base = repo_root / training_base_dir
    if not training_base.exists():
        st.warning(f"Training data directory not found: {training_base}")
        return

    # Get model for selected sigma
    sigma_f = float(selected_sigma)
    model_key = None
    model_data = None
    for mk, md in mp.get("models", {}).items():
        if f"sigma={selected_sigma}" in mk or f"sigma={sigma_f}" in mk:
            model_key = mk
            model_data = md
            break

    if model_data is None:
        st.warning(f"No model found for sigma={selected_sigma}")
        return

    coeffs = model_data["coefficients"][0]
    intercept = model_data["intercept"][0]
    features = model_data.get("features", mp.get("features", []))
    # Extract model name (e.g. "8b") from model_key like "sigma=0.01_model=8b"
    model_name = model_key.split("model=")[-1] if "model=" in model_key else "8b"

    st.caption(f"Model: {model_key} | Training data: {training_base.name}")

    df = _load_training_data_with_predictions(
        str(training_base), sigma_f, coeffs, intercept, features, model_name,
    )
    if df is None or df.empty:
        st.warning("No training data found for this sigma.")
        return

    st.caption(f"{len(df)} training steps across {df['problem_id'].nunique()} problems")

    # Aggregate per-problem
    per_problem = df.groupby("problem_id").agg(
        avg_predicted=("predicted_p", "mean"),
        avg_success=("success", "mean"),
        oracle_p=("oracle_p", "first"),
        n_steps=("success", "count"),
    ).reset_index()

    _render_calibration_plot(per_problem)
    _render_prediction_scatter(per_problem)


def _render_model_heatmap(mp: dict, selected_sigma: float):
    """Heatmap of model parameters: rows=models, cols=parameters."""
    models = mp.get("models", {})
    if not models:
        st.info("No models found.")
        return

    model_keys = list(models.keys())
    first_model = models[model_keys[0]]
    features = first_model.get("features", mp.get("features", []))
    param_names = ["intercept"] + [f"coeff:{f}" for f in features]

    z = []
    active_row = None
    for i, mk in enumerate(model_keys):
        m = models[mk]
        intercept = m.get("intercept", [0])[0]
        coeffs = m.get("coefficients", [[]])[0]
        z.append([intercept] + list(coeffs))
        if f"sigma={selected_sigma}" in mk:
            active_row = i

    fig = go.Figure(go.Heatmap(
        z=z,
        x=param_names,
        y=model_keys,
        colorscale="RdBu_r",
        zmid=0,
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=13),
        hovertemplate="Model: %{y}<br>Param: %{x}<br>Value: %{z:.4f}<extra></extra>",
    ))

    if active_row is not None:
        fig.add_shape(
            type="rect",
            x0=-0.5, x1=len(param_names) - 0.5,
            y0=active_row - 0.5, y1=active_row + 0.5,
            line=dict(color="lime", width=3),
        )

    fig.update_layout(
        height=max(250, 60 * len(model_keys) + 80),
        xaxis_title="Parameter",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    c_val = mp.get("C", "N/A")
    st.caption(f"Regularization C = {c_val} | Features: {', '.join(features)}")


# ---------------------------------------------------------------------------
# Shared: trajectory data loader
# ---------------------------------------------------------------------------

@st.cache_data
def _load_all_trajectories_for_config(config_path: str) -> Optional[pd.DataFrame]:
    """Load all trajectory steps from a config, extracting predictions and outcomes.

    Returns DataFrame with columns:
        problem_id, seed, step, predicted_p, oracle_p, success, action_type, action_model, hot_start
    """
    traj_dir = Path(config_path) / "trajectories"
    if not traj_dir.exists():
        return None

    records = []
    for seed_dir in sorted(traj_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        seed_name = seed_dir.name
        for traj_file in sorted(seed_dir.glob("*.json")):
            try:
                with open(traj_file) as fh:
                    traj = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            problem_id = traj.get("problem_id", traj_file.stem)
            for step in traj.get("steps", []):
                action = step.get("action", {})
                result = step.get("result", {})
                dm = step.get("decision_metadata") or {}

                action_type = action.get("type", "")
                action_model = action.get("model", "")
                is_hot_start = dm.get("hot_start", False)
                success = result.get("success", False)

                # Extract predicted_p and oracle_p from the chosen action's score entry
                predicted_p = None
                oracle_p = None
                for scored in dm.get("action_scores", []):
                    sa = scored.get("action", {})
                    if sa.get("type") == action_type and sa.get("model", "") == action_model:
                        predicted_p = scored.get("p")
                        oracle_p = scored.get("oracle_p")
                        break
                # Fallback: oracle_p at top level
                if oracle_p is None:
                    oracle_p = dm.get("oracle_p")

                records.append({
                    "problem_id": problem_id,
                    "seed": seed_name,
                    "step": step.get("step", 0),
                    "predicted_p": predicted_p,
                    "oracle_p": oracle_p,
                    "success": success,
                    "action_type": action_type,
                    "action_model": action_model,
                    "hot_start": is_hot_start,
                })

    if not records:
        return None
    df = pd.DataFrame(records)
    # Ensure correct dtypes (avoid object columns from None mixing)
    df["predicted_p"] = pd.to_numeric(df["predicted_p"], errors="coerce")
    df["oracle_p"] = pd.to_numeric(df["oracle_p"], errors="coerce")
    df["success"] = df["success"].astype(bool)
    df["hot_start"] = df["hot_start"].astype(bool)
    return df


def _filter_prediction_steps(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to non-hot-start, non-terminate steps with valid predictions."""
    mask = (~df["hot_start"]) & (df["action_type"] != "terminate") & (df["predicted_p"].notna())
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Calibration plot (per-problem: avg predicted_p vs avg success rate)
# ---------------------------------------------------------------------------

def _render_calibration_plot(per_problem: pd.DataFrame):
    """Per-problem calibration: sorted bars of predicted_p vs actual success rate.

    Expects pre-aggregated DataFrame with columns:
        problem_id, avg_predicted, avg_success, n_steps
    """
    if per_problem.empty:
        st.info("Not enough data for calibration plot.")
        return

    corr = per_problem["avg_predicted"].corr(per_problem["avg_success"])
    sorted_df = per_problem.sort_values("avg_predicted", ascending=False)
    pids = sorted_df["problem_id"].values.tolist()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pids,
        y=sorted_df["avg_predicted"].values.tolist(),
        name="Avg Predicted P",
        marker_color="#1f77b4",
        opacity=0.7,
    ))

    fig.add_trace(go.Bar(
        x=pids,
        y=sorted_df["avg_success"].values.tolist(),
        name="Avg Success Rate",
        marker_color="#ff7f0e",
        opacity=0.7,
    ))

    fig.update_layout(
        title=f"Per-Problem: Predicted P vs Actual Success Rate (r = {corr:.3f})",
        xaxis_title="Problem (sorted by predicted P)",
        yaxis_title="Value",
        barmode="group",
        height=500,
        xaxis_tickangle=-45,
        xaxis_tickfont=dict(size=8),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show data table
    with st.expander("Per-problem data"):
        st.dataframe(
            sorted_df,
            use_container_width=True,
            hide_index=True,
        )


def _render_prediction_scatter(per_problem: pd.DataFrame):
    """Scatter plot: avg predicted_p vs oracle_p per problem (if oracle_p available).

    Expects pre-aggregated DataFrame with columns:
        problem_id, avg_predicted, oracle_p
    """
    if "oracle_p" not in per_problem.columns or per_problem["oracle_p"].isna().all():
        return

    valid = per_problem.dropna(subset=["oracle_p", "avg_predicted"])
    if valid.empty:
        return

    corr = valid["avg_predicted"].corr(valid["oracle_p"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=valid["oracle_p"].values.tolist(),
        y=valid["avg_predicted"].values.tolist(),
        mode="markers+text",
        text=valid["problem_id"].values.tolist(),
        textposition="top center",
        textfont=dict(size=7),
        hovertemplate="<b>%{text}</b><br>Oracle P: %{x:.4f}<br>Avg Predicted P: %{y:.4f}<extra></extra>",
        marker=dict(size=10, color="#ff7f0e", opacity=0.8),
        name="Problems",
    ))

    max_val = max(valid["oracle_p"].max(), valid["avg_predicted"].max(), 0.01)
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color="red", dash="dash", width=1),
        name="y = x",
    ))

    fig.update_layout(
        title=f"Oracle P vs Avg Predicted P (r = {corr:.3f})",
        xaxis_title="Oracle P",
        yaxis_title="Avg Predicted P",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Per-target chart (used by Tab 1)
# ---------------------------------------------------------------------------

def _render_target_tab(target_id: str, target_steps: List[Tuple[int, dict]]):
    """Render the per-target tab with dynamic charts from decision metadata."""
    st.caption(f"{len(target_steps)} steps on this target")

    # 1. Scan all steps to discover numeric fields
    #    - action_score_fields: fields inside action_scores entries (per-action)
    #    - scalar_fields: top-level numeric fields in decision_metadata
    action_score_fields = []
    scalar_fields = []
    seen_as = set()
    seen_sc = set()

    chart_data = []
    local_idx = 0

    for global_idx, step in target_steps:
        dm = step.get("decision_metadata") or {}
        chosen_action = step.get("action", {})
        result = step.get("result", {})
        is_hot_start = dm.get("hot_start", False)

        action_scores = dm.get("action_scores", [])
        for scored in action_scores:
            for k, v in scored.items():
                if k != "action" and isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
                    if k not in seen_as:
                        action_score_fields.append(k)
                        seen_as.add(k)

        scalars = {}
        for k, v in dm.items():
            if k in ("action_scores", "action"):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
                scalars[k] = v
                if k not in seen_sc:
                    scalar_fields.append(k)
                    seen_sc.add(k)

        chart_data.append({
            "local_idx": local_idx,
            "global_idx": global_idx,
            "action_scores": action_scores,
            "scalars": scalars,
            "chosen_action": chosen_action,
            "is_hot_start": is_hot_start,
            "success": result.get("success", False),
        })
        local_idx += 1

    n_as = len(action_score_fields)
    n_sc = len(scalar_fields)

    if n_as + n_sc == 0:
        st.info("No numeric decision metadata for this target.")
        return

    # --- Overlay selector: let users combine scalar fields onto one chart ---
    overlay_fields: List[str] = []
    individual_scalar_fields = list(scalar_fields)
    if n_sc >= 2:
        # Suggest tau_stop + best_tau_continue as default if both present
        default_overlay = [f for f in scalar_fields if f in ("tau_stop", "best_tau_continue")]
        overlay_fields = st.multiselect(
            "Overlay fields on same chart",
            options=scalar_fields,
            default=default_overlay,
            key=f"overlay_{target_id}",
            help="Select 2+ scalar fields to plot on a single shared chart for comparison.",
        )
        if len(overlay_fields) >= 2:
            individual_scalar_fields = [f for f in scalar_fields if f not in overlay_fields]
        else:
            overlay_fields = []  # need at least 2 to combine

    has_overlay = len(overlay_fields) >= 2
    n_individual = len(individual_scalar_fields)
    n_total = n_as + n_individual + (1 if has_overlay else 0)

    # Build subplot titles
    titles = list(action_score_fields)
    if has_overlay:
        titles.append(" + ".join(overlay_fields))
    titles.extend(individual_scalar_fields)

    fig = make_subplots(
        rows=n_total, cols=1,
        shared_xaxes=True,
        vertical_spacing=max(0.02, 0.3 / max(n_total, 1)),
        subplot_titles=titles,
    )

    # 2. Plot action_score fields (per-action traces with chosen/alternative markers)
    if n_as > 0:
        legend_added = set()

        for cd in chart_data:
            x = cd["local_idx"]
            if not cd["action_scores"]:
                continue

            for scored in cd["action_scores"]:
                sa = scored.get("action", {})
                ak = _action_key(sa)
                color = ACTION_COLORS.get(ak, "#7f7f7f")
                is_chosen = (sa.get("type") == cd["chosen_action"].get("type") and
                             sa.get("model") == cd["chosen_action"].get("model"))

                if cd["success"] and is_chosen:
                    symbol, size = "star", 12
                elif cd["is_hot_start"]:
                    symbol = "square" if is_chosen else "square-open"
                    size = 10
                elif is_chosen:
                    symbol, size = "circle", 9
                else:
                    symbol, size = "circle-open", 8

                show_legend = ak not in legend_added
                legend_added.add(ak)

                # Build hover text from all fields
                hover_parts = [
                    f"Step {cd['global_idx']}",
                    f"Action: {_action_display(sa)}",
                    "CHOSEN" if is_chosen else "alternative",
                ]
                for field in action_score_fields:
                    v = scored.get(field)
                    if v is not None:
                        hover_parts.append(f"{field}={_format_dm_value(v)}")
                hover_text = "<br>".join(hover_parts)

                common = dict(
                    marker=dict(color=color, symbol=symbol, size=size, line=dict(width=1, color=color)),
                    name=_action_display(sa),
                    legendgroup=ak,
                    showlegend=show_legend,
                    hovertext=hover_text,
                    hoverinfo="text",
                )

                for i, field in enumerate(action_score_fields):
                    v = scored.get(field)
                    if v is not None:
                        fig.add_trace(
                            go.Scatter(x=[x], y=[v], mode="markers",
                                       **common | {"showlegend": show_legend if i == 0 else False}),
                            row=i + 1, col=1,
                        )

        # Connect chosen actions with dotted lines
        chosen_by_key: Dict[str, List[dict]] = {}
        for cd in chart_data:
            if not cd["action_scores"]:
                continue
            for scored in cd["action_scores"]:
                sa = scored.get("action", {})
                if (sa.get("type") == cd["chosen_action"].get("type") and
                        sa.get("model") == cd["chosen_action"].get("model")):
                    entry = {"x": cd["local_idx"]}
                    for field in action_score_fields:
                        entry[field] = scored.get(field)
                    ak = _action_key(sa)
                    chosen_by_key.setdefault(ak, []).append(entry)
                    break

        for ak, points in chosen_by_key.items():
            if len(points) < 2:
                continue
            color = ACTION_COLORS.get(ak, "#7f7f7f")
            xs = [pt["x"] for pt in points]
            for i, field in enumerate(action_score_fields):
                ys = [pt.get(field) for pt in points]
                valid = [(vx, vy) for vx, vy in zip(xs, ys) if vy is not None]
                if len(valid) >= 2:
                    fig.add_trace(go.Scatter(
                        x=[v[0] for v in valid], y=[v[1] for v in valid],
                        mode="lines", line=dict(color=color, width=1, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ), row=i + 1, col=1)

    # 3a. Plot overlay (combined) scalar fields on one shared subplot
    overlay_row_offset = n_as  # row offset for individual scalars (updated below if overlay exists)
    if has_overlay:
        overlay_row = n_as + 1
        for fi, field in enumerate(overlay_fields):
            color = OVERLAY_COLORS[fi % len(OVERLAY_COLORS)]
            xs, ys, hover_texts = [], [], []
            for cd in chart_data:
                v = cd["scalars"].get(field)
                if v is not None:
                    xs.append(cd["local_idx"])
                    ys.append(v)
                    hover_parts = [
                        f"Step {cd['global_idx']}",
                        f"Action: {_action_display(cd['chosen_action'])}",
                    ]
                    for of in overlay_fields:
                        ov = cd["scalars"].get(of)
                        if ov is not None:
                            hover_parts.append(f"{of}={_format_dm_value(ov)}")
                    hover_texts.append("<br>".join(hover_parts))
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers+lines",
                    marker=dict(color=color, size=7),
                    line=dict(color=color, width=2),
                    name=field,
                    legendgroup=f"overlay_{field}",
                    showlegend=True,
                    hovertext=hover_texts, hoverinfo="text",
                ), row=overlay_row, col=1)
        overlay_row_offset = n_as + 1

    # 3b. Plot individual scalar fields (one subplot each)
    for i, field in enumerate(individual_scalar_fields):
        row = overlay_row_offset + i + 1
        xs = []
        ys = []
        colors = []
        symbols = []
        sizes = []
        hover_texts = []

        for cd in chart_data:
            v = cd["scalars"].get(field)
            if v is not None:
                xs.append(cd["local_idx"])
                ys.append(v)

                ak = _action_key(cd["chosen_action"])
                colors.append(ACTION_COLORS.get(ak, "#7f7f7f"))

                if cd["success"]:
                    symbols.append("star"); sizes.append(12)
                elif cd["is_hot_start"]:
                    symbols.append("square"); sizes.append(10)
                else:
                    symbols.append("circle"); sizes.append(9)

                hover_parts = [
                    f"Step {cd['global_idx']}",
                    f"Action: {_action_display(cd['chosen_action'])}",
                ]
                for sf in scalar_fields:
                    sv = cd["scalars"].get(sf)
                    if sv is not None:
                        hover_parts.append(f"{sf}={_format_dm_value(sv)}")
                hover_texts.append("<br>".join(hover_parts))

        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers+lines",
                marker=dict(color=colors, symbol=symbols, size=sizes),
                line=dict(color="#7f7f7f", width=1),
                hovertext=hover_texts, hoverinfo="text",
                showlegend=False,
            ), row=row, col=1)

    # Update axes
    for i, field in enumerate(titles):
        fig.update_yaxes(title_text=field, row=i + 1, col=1)
    fig.update_xaxes(title_text="Step (within target)", row=n_total, col=1)

    fig.update_layout(
        height=max(400, 220 * n_total),
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Noise Analysis
# ---------------------------------------------------------------------------

def _load_all_noise_data(sweep_path: str) -> pd.DataFrame:
    """Load prediction vs oracle_p data from all trajectories in the sweep.

    Builds one row per (step, model) from two sources:
    - action_scores entries (fields: p, oracle_p)
    - tracked_state dict-valued entries keyed by model name (e.g. predicted_prob, noisy_p, oracle_p)

    When action_scores exist, rows are created per scored action.
    Otherwise, rows are created per model found in tracked_state dicts.
    """
    rows = []
    sweep = Path(sweep_path)
    for config_dir in sorted(sweep.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
            continue
        traj_dir = config_dir / "trajectories"
        if not traj_dir.exists():
            continue
        # Load config params for labeling
        config_label = config_dir.name
        params_file = config_dir / "params.json"
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
            # Flatten _state_tracker params to top level for labeling
            flat_params = {k: v for k, v in params.items() if k != "_state_tracker"}
            if "_state_tracker" in params and isinstance(params["_state_tracker"], dict):
                for sk, sv in params["_state_tracker"].items():
                    flat_params[f"st_{sk}"] = sv
            label_parts = []
            for k in ("lambda_val", "sigma", "st_sigma", "max_n", "max_breakdowns", "full_proof_budget"):
                if k in flat_params:
                    label_parts.append(f"{k}={flat_params[k]}")
            if label_parts:
                config_label = ", ".join(label_parts)
        for seed_dir in sorted(traj_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            for traj_file in sorted(seed_dir.glob("*.json")):
                with open(traj_file) as f:
                    traj = json.load(f)
                problem_id = traj.get("problem_id", traj_file.stem)
                for step in traj.get("steps", []):
                    target_id = step.get("state", {}).get("target_id", "")
                    dm = step.get("decision_metadata") or {}
                    tracked = step.get("tracked_state") or {}

                    # Skip hot start steps
                    if dm.get("hot_start"):
                        continue

                    action_scores = dm.get("action_scores", [])
                    scored_actions = [
                        s for s in action_scores
                        if "p" in s and "oracle_p" in s
                    ]

                    if scored_actions:
                        # One row per scored action
                        for scored in scored_actions:
                            model = scored.get("action", {}).get("model", "")
                            row = {
                                "config": config_label,
                                "problem_id": problem_id,
                                "target_id": target_id,
                                "as_p": scored["p"],
                                "as_oracle_p": scored["oracle_p"],
                                "model": model,
                            }
                            for ts_key, ts_val in tracked.items():
                                if isinstance(ts_val, dict) and model in ts_val:
                                    v = ts_val[model]
                                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                                        row[f"ts_{ts_key}"] = v
                            rows.append(row)
                    else:
                        # No action_scores — create rows from tracked_state models
                        # Find all models present in any dict-valued tracked_state field
                        models = set()
                        for ts_val in tracked.values():
                            if isinstance(ts_val, dict):
                                for k, v in ts_val.items():
                                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                                        models.add(k)
                        for model in models:
                            row = {
                                "config": config_label,
                                "problem_id": problem_id,
                                "target_id": target_id,
                                "model": model,
                            }
                            for ts_key, ts_val in tracked.items():
                                if isinstance(ts_val, dict) and model in ts_val:
                                    v = ts_val[model]
                                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                                        row[f"ts_{ts_key}"] = v
                            rows.append(row)
    return pd.DataFrame(rows)


def _render_noise_tab(sweep_path: str):
    """Render the noise analysis tab: predicted_prob vs oracle_p noise."""
    st.subheader("Prediction Noise Analysis")
    st.caption(
        "Analyzes how noisy a chosen prediction field is relative to `oracle_p` "
        "by pooling data across all configs, seeds, problems, and targets."
    )

    if st.button("Load Noise Data", key="load_noise_data"):
        with st.spinner("Loading all trajectory data..."):
            st.session_state["noise_data"] = _load_all_noise_data(sweep_path)

    if "noise_data" not in st.session_state:
        st.info("Click **Load Noise Data** to begin analysis.")
        return

    df = st.session_state["noise_data"]
    if df.empty:
        st.warning("No (p, oracle_p) pairs found in any trajectory.")
        return

    # Build list of available numeric fields
    numeric_cols = []
    for col in sorted(df.columns):
        if col in ("problem_id", "target_id", "model"):
            continue
        if df[col].dtype in ("float64", "int64", "float32", "int32"):
            numeric_cols.append(col)

    def _fmt_field(x: str) -> str:
        if x.startswith("ts_"):
            return f"tracked_state.{x[3:]}"
        if x.startswith("as_"):
            return f"action_scores.{x[3:]}"
        return x

    col1, col2 = st.columns(2)
    with col1:
        # Default to ts_oracle_p if available, else as_oracle_p, else first
        oracle_defaults = [c for c in numeric_cols if "oracle_p" in c]
        oracle_default_idx = numeric_cols.index(oracle_defaults[0]) if oracle_defaults else 0
        oracle_field = st.selectbox(
            "Oracle (ground truth) field",
            numeric_cols,
            index=oracle_default_idx,
            format_func=_fmt_field,
            key="noise_oracle_field",
        )
    with col2:
        # Default to first non-oracle field
        pred_defaults = [c for c in numeric_cols if c != oracle_field]
        selected_field = st.selectbox(
            "Prediction field to compare",
            pred_defaults,
            format_func=_fmt_field,
            key="noise_pred_field",
        )

    # Filter to rows where both fields are present
    working = df.dropna(subset=[selected_field, oracle_field]).copy()
    if working.empty:
        st.warning(f"No data with both `{_fmt_field(selected_field)}` and `{_fmt_field(oracle_field)}`.")
        return

    working["error"] = working[selected_field] - working[oracle_field]

    # --- 1. Per-config summary ---
    st.markdown("### Per-Config Noise Summary")

    configs = sorted(working["config"].unique()) if "config" in working.columns else ["all"]
    per_config_rows = []
    for cfg in configs:
        cfg_df = working[working["config"] == cfg] if "config" in working.columns else working
        per_config_rows.append({
            "config": cfg,
            "std": cfg_df["error"].std(),
            "bias": cfg_df["error"].mean(),
            "mae": cfg_df["error"].abs().mean(),
            "n": len(cfg_df),
        })
    per_config = pd.DataFrame(per_config_rows)
    st.dataframe(per_config, use_container_width=True, hide_index=True)

    # --- 2. Per-problem noise (grouped by config) ---
    st.markdown("### Per-Problem Noise")

    per_problem = (
        working.groupby(["config", "problem_id"])
        .agg(
            oracle_p=(oracle_field, "mean"),
            avg_predicted=(selected_field, "mean"),
            std=("error", "std"),
            mean_error=("error", "mean"),
            mae=("error", lambda x: x.abs().mean()),
            count=("error", "count"),
        )
        .reset_index()
        .sort_values("std", ascending=False)
    )

    fig = go.Figure()
    for cfg in configs:
        cfg_df = per_problem[per_problem["config"] == cfg]
        fig.add_trace(go.Bar(
            x=cfg_df["problem_id"].tolist(),
            y=cfg_df["std"].tolist(),
            name=f"Std — {cfg}",
        ))
    fig.update_layout(
        title="Per-Problem Prediction Noise",
        xaxis_title="Problem",
        yaxis_title="Std(error)",
        barmode="group",
        height=500,
        xaxis_tickangle=-45,
        xaxis_tickfont=dict(size=8),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Per-problem data table"):
        st.dataframe(per_problem, use_container_width=True, hide_index=True)

    # --- 3. Per-problem/target noise ---
    st.markdown("### Per-Problem/Target Noise")

    per_target = (
        working.groupby(["config", "problem_id", "target_id"])
        .agg(
            oracle_p=(oracle_field, "mean"),
            avg_predicted=(selected_field, "mean"),
            std=("error", "std"),
            mean_error=("error", "mean"),
            mae=("error", lambda x: x.abs().mean()),
            count=("error", "count"),
        )
        .reset_index()
        .sort_values("std", ascending=False)
    )
    st.dataframe(per_target, use_container_width=True, hide_index=True)
