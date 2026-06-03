"""
Trajectory Analysis viewer.

Loads simulation results (single-run and sweep) from results/simulations/,
normalizes them to a common data model, and provides:
- Pareto cost-vs-solve-rate scatter with optional upper envelopes
- AUC comparison across curves
- Filterable configuration details table
- Paper-quality PDF export with LaTeX fonts
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import colorsys
import io
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

from .baseline_rollouts_viewer import (
    compute_upper_envelope,
    interpolate_baseline_auc,
)

# Project root (lean-breakdown directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_SIM_DIR = PROJECT_ROOT / "results" / "simulations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flatten_params(params: dict, prefix: str = "") -> dict:
    """Flatten nested params dict to dot-notation keys.

    {"full_proof_budget": {"8b": 32}} -> {"full_proof_budget.8b": 32}
    """
    flat = {}
    for k, v in params.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(flatten_params(v, key))
        else:
            flat[key] = v
    return flat


def _generate_color_shades(base_hex: str, n: int) -> List[str]:
    """Generate *n* distinct shades from a base hex colour using HLS variation."""
    if n <= 1:
        return [base_hex]

    base_hex = base_hex.lstrip("#")
    r, g, b = (
        int(base_hex[:2], 16) / 255,
        int(base_hex[2:4], 16) / 255,
        int(base_hex[4:6], 16) / 255,
    )
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    shades = []
    for i in range(n):
        # Vary lightness between 0.3 and 0.7
        new_l = 0.3 + (0.4 * i / (n - 1)) if n > 1 else l
        if n > 6:
            # Add slight hue rotation for extra distinguishability
            new_h = (h + 0.05 * (i / (n - 1) - 0.5)) % 1.0
        else:
            new_h = h
        nr, ng, nb = colorsys.hls_to_rgb(new_h, new_l, min(s, 1.0))
        shades.append(f"#{int(nr*255):02x}{int(ng*255):02x}{int(nb*255):02x}")

    return shades


# ---------------------------------------------------------------------------
# Data discovery & loading  (all cached)
# ---------------------------------------------------------------------------

@st.cache_data
def discover_simulation_runs(sim_dir: str) -> List[Dict[str, Any]]:
    """Scan *sim_dir* for subdirectories that look like simulation results.

    Returns a list of dicts:
        {"name": str, "path": str, "type": "sweep"|"single"|"partial_sweep"}
    """
    sim_path = Path(sim_dir)
    if not sim_path.exists():
        return []

    runs = []
    for entry in sorted(sim_path.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        if (entry / "sweep_summary.json").exists():
            runs.append({"name": entry.name, "path": str(entry), "type": "sweep"})
        elif (entry / "summary.json").exists() and (entry / "config.json").exists():
            runs.append({"name": entry.name, "path": str(entry), "type": "single"})
        else:
            # Check for partial sweep: config_NNN dirs with summary.json
            config_dirs = sorted(entry.glob("config_*/summary.json"))
            if config_dirs:
                n_done = len(config_dirs)
                runs.append({"name": f"{entry.name}  ({n_done} configs)", "path": str(entry), "type": "partial_sweep"})

    return runs


@st.cache_data
def load_sweep_run(path: str) -> List[Dict[str, Any]]:
    """Load a sweep run, returning one data point per config entry."""
    p = Path(path)
    with open(p / "sweep_summary.json") as f:
        sweep = json.load(f)

    n_problems = sweep.get("n_problems", 1)
    n_seeds = sweep.get("n_seeds", 1)
    run_name = p.name

    points = []
    for cfg in sweep.get("configs", []):
        params = cfg.get("params", {})
        avg_cost_obj = cfg.get("avg_total_cost", {})
        avg_cost = (
            (avg_cost_obj.get("input_sflops", 0) + avg_cost_obj.get("output_sflops", 0))
            / max(n_problems, 1)
        )

        points.append({
            "run_name": run_name,
            "config_label": f"config_{cfg.get('config_id', 0):03d}",
            "params": params,
            "flat_params": flatten_params(params),
            "solve_rate": cfg.get("avg_solve_rate", 0.0),
            "std_solve_rate": cfg.get("std_solve_rate", 0.0),
            "avg_cost_per_problem": avg_cost,
            "total_problems": n_problems,
            "num_seeds": n_seeds,
        })

    return points


@st.cache_data
def load_partial_sweep(path: str) -> List[Dict[str, Any]]:
    """Load a partial sweep by reading individual config_NNN/summary.json files."""
    p = Path(path)
    run_name = p.name

    points = []
    for config_dir in sorted(p.glob("config_*")):
        summary_path = config_dir / "summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path) as f:
            cfg = json.load(f)

        params = cfg.get("params", {})
        n_problems = cfg.get("per_seed", [{}])[0].get("total_problems", 1) if cfg.get("per_seed") else 1
        n_seeds = len(cfg.get("per_seed", []))
        avg_cost_obj = cfg.get("avg_total_cost", {})
        avg_cost = (
            (avg_cost_obj.get("input_sflops", 0) + avg_cost_obj.get("output_sflops", 0))
            / max(n_problems, 1)
        )

        points.append({
            "run_name": run_name,
            "config_label": config_dir.name,
            "params": params,
            "flat_params": flatten_params(params),
            "solve_rate": cfg.get("avg_solve_rate", 0.0),
            "std_solve_rate": cfg.get("std_solve_rate", 0.0),
            "avg_cost_per_problem": avg_cost,
            "total_problems": n_problems,
            "num_seeds": n_seeds,
        })

    return points


@st.cache_data
def load_single_run(path: str) -> List[Dict[str, Any]]:
    """Load a single-run result (1 or multi seed) as one data point."""
    p = Path(path)
    with open(p / "config.json") as f:
        config = json.load(f)
    with open(p / "summary.json") as f:
        summary = json.load(f)

    run_name = p.name
    params = config.get("policy", {}).get("params", {})

    # Detect single-seed vs multi-seed
    if "per_seed" in summary:
        # Multi-seed format
        total_problems = summary.get("total_problems", 1)
        num_seeds = summary.get("num_seeds", len(summary["per_seed"]))
        solve_rate = summary.get("avg_solve_rate", 0.0)

        # Average cost across seeds
        total_input = 0
        total_output = 0
        for s in summary["per_seed"]:
            tc = s.get("total_cost", {})
            total_input += tc.get("input_sflops", 0)
            total_output += tc.get("output_sflops", 0)
        n = len(summary["per_seed"]) or 1
        avg_cost = ((total_input + total_output) / n) / max(total_problems, 1)

        # Std of solve rates
        rates = [s.get("solve_rate", 0.0) for s in summary["per_seed"]]
        std_rate = float(np.std(rates)) if len(rates) > 1 else 0.0
    else:
        # Single-seed format
        total_problems = summary.get("total_problems", 1)
        num_seeds = 1
        solve_rate = summary.get("solve_rate", 0.0)
        std_rate = 0.0
        tc = summary.get("total_cost", {})
        avg_cost = (
            (tc.get("input_sflops", 0) + tc.get("output_sflops", 0))
            / max(total_problems, 1)
        )

    return [{
        "run_name": run_name,
        "config_label": run_name,
        "params": params,
        "flat_params": flatten_params(params),
        "solve_rate": solve_rate,
        "std_solve_rate": std_rate,
        "avg_cost_per_problem": avg_cost,
        "total_problems": total_problems,
        "num_seeds": num_seeds,
    }]


# ---------------------------------------------------------------------------
# Paper-quality export
# ---------------------------------------------------------------------------

def _render_paper_export(group_curves: Dict[str, Dict], show_envelope: bool,
                         envelope_scope: str, visible_points: List[Dict],
                         run_base_colors: Dict[str, str], color_palette: List[str],
                         flip_axes: bool = False):
    """Render paper-quality PDF export controls and generate matplotlib figure.

    When flip_axes is True the exported figure matches the flipped interactive
    plot: solve rate on the x-axis, cost on the y-axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.subheader("Paper Export")

    # Legend renaming, ordering, and visibility
    st.markdown("**Legend names & order**")
    legend_names: Dict[str, str] = {}
    legend_order: Dict[str, int] = {}
    legend_visible: Dict[str, bool] = {}
    n_curves = len(group_curves)
    curve_labels = list(group_curves.keys())
    cols = st.columns(min(n_curves, 3))
    for i, label in enumerate(curve_labels):
        with cols[i % len(cols)]:
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                legend_names[label] = st.text_input(
                    label, value=label, key=f"paper_legend_{i}",
                    label_visibility="collapsed",
                )
            with c2:
                legend_order[label] = st.number_input(
                    "order", value=i, min_value=0, max_value=n_curves - 1,
                    step=1, key=f"paper_order_{i}",
                    label_visibility="collapsed",
                )
            with c3:
                legend_visible[label] = st.checkbox(
                    "show", value=True, key=f"paper_vis_{i}",
                )

    # Sort and filter curves by user-specified order and visibility
    sorted_labels = [l for l in sorted(curve_labels, key=lambda l: legend_order[l]) if legend_visible.get(l, True)]

    # Plot settings
    set_col1, set_col2, set_col3, set_col4 = st.columns(4)
    with set_col1:
        fig_width = st.number_input("Width (inches)", value=5.1, min_value=2.0, max_value=12.0, step=0.1, key="paper_w")
    with set_col2:
        fig_height = st.number_input("Height (inches)", value=3.2, min_value=2.0, max_value=10.0, step=0.1, key="paper_h")
    with set_col3:
        font_size = st.number_input("Font size", value=9, min_value=6, max_value=18, step=1, key="paper_fs")
    with set_col4:
        font_choice = st.selectbox(
            "Font",
            options=["Sans-serif (CM)", "Times (ICML)"],
            index=0,
            key="paper_font",
        )

    # ICML 2026 column-layout presets (single column = 3.25in, full text = 6.75in).
    # When chosen, override the manual width above.
    icml_layout = None
    if font_choice == "Times (ICML)":
        icml_layout = st.radio(
            "ICML layout (overrides width)",
            options=["Single column (3.25 in)", "Full text width (6.75 in)", "Custom (use width above)"],
            index=0,
            horizontal=True,
            key="paper_icml_layout",
        )
        if icml_layout == "Single column (3.25 in)":
            fig_width = 3.25
        elif icml_layout == "Full text width (6.75 in)":
            fig_width = 6.75

    # Labels are semantic (per data series), not positional, so they follow
    # their data automatically when the axes are flipped.
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    with adv_col1:
        cost_label = st.text_input("Cost-axis label", value="Avg Cost per Problem (M SFLOPs)", key="paper_costlabel")
    with adv_col2:
        rate_label = st.text_input("Solve-rate-axis label", value="Solve Rate (\%)", key="paper_ratelabel")
    with adv_col3:
        legend_loc = st.selectbox("Legend position", options=[
            "below", "best", "upper left", "upper right", "lower right", "lower left",
        ], index=0, key="paper_legend_loc")

    marker_col1, marker_col2, marker_col3, marker_col4, marker_col5, marker_col6 = st.columns(6)
    with marker_col1:
        marker_size = st.number_input("Marker size", value=5, min_value=2, max_value=15, step=1, key="paper_ms")
    with marker_col2:
        line_width = st.number_input("Line width", value=1.5, min_value=0.5, max_value=4.0, step=0.25, key="paper_lw")
    with marker_col3:
        show_paper_envelope = st.checkbox("Show envelope", value=show_envelope, key="paper_env")
    with marker_col4:
        rate_min = st.number_input("Rate min (%)", value=0, min_value=0, max_value=100, step=5, key="paper_ymin")
    with marker_col5:
        rate_max = st.number_input("Rate max (%)", value=100, min_value=0, max_value=100, step=5, key="paper_ymax")
    with marker_col6:
        legend_cols = st.number_input("Legend cols", value=1, min_value=1, max_value=5, step=1, key="paper_lcols")

    export_filename = st.text_input("Export filename", value="cost_quality_curve.pdf", key="paper_filename")

    if not st.button("Render", key="paper_render", type="primary"):
        return

    # Configure matplotlib — font preamble depends on user font choice
    if font_choice == "Times (ICML)":
        # Matches icml2026.sty: \RequirePackage{times}
        latex_preamble = r"\usepackage{times}"
        font_family = "serif"
    else:
        # CM Sans Serif (thesis default)
        latex_preamble = r"\usepackage[OT1]{fontenc}\renewcommand{\familydefault}{\sfdefault}\usepackage{sfmath}"
        font_family = "sans-serif"

    # Only use a real LaTeX installation if one is present on the system;
    # otherwise fall back to matplotlib's built-in mathtext renderer.
    use_tex = shutil.which("latex") is not None
    if not use_tex:
        st.warning(
            "LaTeX not found on this machine — rendering with matplotlib's "
            "built-in mathtext instead. Install a TeX distribution (e.g. "
            "`texlive`) for exact paper-quality fonts."
        )

    rc = {
        "text.usetex": use_tex,
        "font.family": font_family,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size - 1,
    }
    if use_tex:
        rc["text.latex.preamble"] = latex_preamble

    sns.set_theme(style="ticks", font_scale=1.0, rc=rc)

    fig_mpl, ax = plt.subplots(figsize=(fig_width, fig_height))

    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<"]

    for i, label in enumerate(sorted_labels):
        data = group_curves[label]
        display_name = legend_names.get(label, label)
        color = data["color"]
        cost_vals = data["x"]
        rate_vals = [v * 100 for v in data["y"]]  # solve rate as percentage
        marker = markers[i % len(markers)]

        plot_x, plot_y = (rate_vals, cost_vals) if flip_axes else (cost_vals, rate_vals)
        ax.plot(plot_x, plot_y, marker=marker, markersize=marker_size,
                linewidth=line_width, label=display_name, color=color,
                markeredgecolor="white", markeredgewidth=0.3)

    # Envelope
    if show_paper_envelope and group_curves:
        from .baseline_rollouts_viewer import compute_upper_envelope as _compute_env
        all_curves = [{"x": d["x"], "y": d["y"]} for d in group_curves.values() if len(d["x"]) >= 2]
        if all_curves:
            x_max = max(max(c["x"]) for c in all_curves)
            env_x, env_y = _compute_env(all_curves, 0.0, x_max)
            if env_x and env_y:
                env_cost = env_x
                env_rate = [v * 100 for v in env_y]
                env_plot_x, env_plot_y = (env_rate, env_cost) if flip_axes else (env_cost, env_rate)
                env_label = "Cost-Efficient Frontier" if flip_axes else "Upper Envelope"
                ax.plot(env_plot_x, env_plot_y, "--",
                        color="gray", linewidth=1.0, alpha=0.7, label=env_label)

    # Cost axis starts at 0; solve-rate axis honours the rate_min/rate_max controls.
    if flip_axes:
        ax.set_xlabel(rate_label)
        ax.set_ylabel(cost_label)
        ax.set_xlim(left=rate_min, right=rate_max)
        ax.set_ylim(bottom=0)
    else:
        ax.set_xlabel(cost_label)
        ax.set_ylabel(rate_label)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=rate_min, top=rate_max)
    if legend_loc == "below":
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.22),
            ncol=legend_cols, frameon=False, fontsize=font_size - 1,
        )
    else:
        ax.legend(loc=legend_loc, ncol=legend_cols, frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    sns.despine(ax=ax)
    fig_mpl.tight_layout()

    # Always show the preview
    st.pyplot(fig_mpl)

    # PDF download
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format="pdf", bbox_inches="tight", dpi=300)
    buf.seek(0)
    plt.close(fig_mpl)

    st.download_button(
        label="Download PDF",
        data=buf,
        file_name=export_filename,
        mime="application/pdf",
        key="paper_download",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_trajectory_analysis_viewer():
    col1, col2 = st.columns([6, 1])
    with col1:
        st.header("Analyze Trajectories")
    with col2:
        if st.button("Refresh", key="refresh_traj_cache", help="Clear cached data and reload"):
            st.cache_data.clear()
            st.rerun()

    # --- Sim directory ---
    sim_dir = st.text_input(
        "Simulations directory",
        value=str(DEFAULT_SIM_DIR),
        key="traj_sim_dir",
    )

    runs = discover_simulation_runs(sim_dir)
    if not runs:
        st.warning(f"No simulation runs found in `{sim_dir}`.")
        return

    # --- 1. Run selector ---
    run_options = [f"{r['name']}  ({r['type']})" for r in runs]
    selected_labels = st.multiselect(
        "Select runs",
        options=run_options,
        default=[],
        key="traj_run_select",
    )
    # Preserve user's selection order
    _label_to_run = {label: r for r, label in zip(runs, run_options)}
    selected_runs = [_label_to_run[label] for label in selected_labels if label in _label_to_run]

    if not selected_runs:
        st.info("Select at least one run above.")
        return

    # --- 2. Load & merge ---
    all_points: List[Dict[str, Any]] = []
    for r in selected_runs:
        if r["type"] == "sweep":
            all_points.extend(load_sweep_run(r["path"]))
        elif r["type"] == "partial_sweep":
            all_points.extend(load_partial_sweep(r["path"]))
        else:
            all_points.extend(load_single_run(r["path"]))

    if not all_points:
        st.warning("No data points loaded.")
        return

    # Collect all flat-param keys with >1 unique value
    all_param_keys: Dict[str, set] = {}
    for pt in all_points:
        for k, v in pt["flat_params"].items():
            all_param_keys.setdefault(k, set()).add(_hashable(v))

    varying_keys = sorted(k for k, vals in all_param_keys.items() if len(vals) > 1)

    # --- 3. Filter controls ---
    filtered_points = list(all_points)
    if varying_keys:
        with st.expander("Filters", expanded=False):
            cols = st.columns(min(len(varying_keys), 3))
            for i, key in enumerate(varying_keys):
                with cols[i % len(cols)]:
                    unique_vals = sorted(all_param_keys[key], key=lambda x: str(x))
                    st.markdown(f"**{key}**")
                    selected_vals = set()
                    for v in unique_vals:
                        if st.checkbox(str(v), value=True, key=f"traj_filt_{key}_{v}"):
                            selected_vals.add(v)
                    # Apply filter
                    filtered_points = [
                        pt for pt in filtered_points
                        if _hashable(pt["flat_params"].get(key)) in selected_vals
                        or pt["flat_params"].get(key) is None
                    ]

    if not filtered_points:
        st.warning("All data points filtered out.")
        return

    st.caption(f"{len(filtered_points)} data points after filtering")

    # --- 4. Per-run coloring ---
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
    ]

    # Group filtered points by run_name
    per_run_points: Dict[str, List[Dict]] = {}
    for pt in filtered_points:
        per_run_points.setdefault(pt["run_name"], []).append(pt)

    # Compute which params vary *within* each run
    per_run_varying_keys: Dict[str, List[str]] = {}
    for rn, rpts in per_run_points.items():
        run_param_vals: Dict[str, set] = {}
        for pt in rpts:
            for k, v in pt["flat_params"].items():
                run_param_vals.setdefault(k, set()).add(_hashable(v))
        per_run_varying_keys[rn] = sorted(
            k for k, vals in run_param_vals.items() if len(vals) > 1
        )

    # Assign each run a stable base colour
    run_names = list(per_run_points.keys())  # preserve selection order
    run_base_colors: Dict[str, str] = {}
    for i, rn in enumerate(run_names):
        run_base_colors[rn] = color_palette[i % len(color_palette)]

    # Per-run visibility and optional sub-colouring (multi-param grouping)
    run_visible: Dict[str, bool] = {}
    run_sub_color: Dict[str, Optional[List[str]]] = {}

    with st.expander("Per-run coloring", expanded=True):
        for rn in run_names:
            n_pts = len(per_run_points[rn])
            col_left, col_right = st.columns([1, 1])
            with col_left:
                run_visible[rn] = st.checkbox(
                    f"{rn}  ({n_pts} pts)",
                    value=True,
                    key=f"traj_run_vis_{rn}",
                )
            with col_right:
                vk = per_run_varying_keys.get(rn, [])
                if n_pts > 1 and vk and run_visible[rn]:
                    selected_keys = st.multiselect(
                        f"Group by ({rn})",
                        options=vk,
                        default=[],
                        key=f"traj_run_grpby_{rn}",
                        label_visibility="collapsed",
                    )
                    run_sub_color[rn] = selected_keys if selected_keys else None
                else:
                    run_sub_color[rn] = None

    # --- 5. Build groups and visible_points ---
    groups: List[Dict[str, Any]] = []  # {label, color, points}
    visible_points: List[Dict[str, Any]] = []

    for rn in run_names:
        if not run_visible.get(rn, True):
            continue

        pts = per_run_points[rn]
        sub_keys = run_sub_color.get(rn)
        base_color = run_base_colors[rn]

        if sub_keys is None:
            groups.append({"label": rn, "color": base_color, "points": pts})
            visible_points.extend(pts)
        else:
            sub_groups: Dict[str, List[Dict]] = {}
            for pt in pts:
                combo_parts = []
                for sk in sub_keys:
                    val = str(pt["flat_params"].get(sk, "N/A"))
                    combo_parts.append(f"{sk}={val}")
                combo_key = ", ".join(combo_parts)
                sub_groups.setdefault(combo_key, []).append(pt)

            sub_vals = sorted(sub_groups.keys())
            shades = _generate_color_shades(base_color, len(sub_vals))
            for sv, shade in zip(sub_vals, shades):
                label = f"{rn}: {sv}"
                groups.append({"label": label, "color": shade, "points": sub_groups[sv]})
                visible_points.extend(sub_groups[sv])

    if not visible_points:
        st.warning("No visible data points.")
        return

    # --- 6. Pareto plot ---
    st.subheader("Cost vs Solve Rate")

    flip_axes = st.checkbox(
        "Flip axes (solve rate on x, cost on y)",
        value=False,
        key="traj_flip_axes",
        help="Swap the X and Y axes. AUC is then computed as ∫ cost d(solve rate); lower is better.",
    )

    def _xy(x_orig, y_orig):
        return (y_orig, x_orig) if flip_axes else (x_orig, y_orig)

    # Plot options
    env_col1, env_col2, env_col3 = st.columns([1, 2, 1])
    with env_col1:
        show_envelope = st.checkbox("Show upper envelope", value=True, key="traj_show_env")
    with env_col3:
        include_origin = st.checkbox("Include (0,0) origin", value=True, key="traj_include_origin")
    with env_col2:
        envelope_scope = st.radio(
            "Envelope scope",
            options=["Global", "Per run"],
            horizontal=True,
            key="traj_env_scope",
        ) if show_envelope else "Global"

    fig = go.Figure()

    # Build per-group traces
    group_curves: Dict[str, Dict] = {}  # label -> {x, y, color, points}
    for grp in groups:
        gv = grp["label"]
        color = grp["color"]
        pts = list(grp["points"])

        if not pts:
            continue

        # Sort by cost
        pts.sort(key=lambda p: p["avg_cost_per_problem"])
        if include_origin:
            x_data = [0.0] + [p["avg_cost_per_problem"] / 1e6 for p in pts]
            y_data = [0.0] + [p["solve_rate"] for p in pts]
        else:
            x_data = [p["avg_cost_per_problem"] / 1e6 for p in pts]
            y_data = [p["solve_rate"] for p in pts]

        group_curves[gv] = {"x": x_data, "y": y_data, "color": color, "points": pts}

        # Build hover customdata
        customdata = [["", "", "", 0, 0, ""]] if include_origin else []  # origin row
        for p in pts:
            param_str = ", ".join(f"{k}={v}" for k, v in p["flat_params"].items())
            customdata.append([
                p["run_name"],
                p["config_label"],
                f"{p['solve_rate']:.3f}",
                p["num_seeds"],
                p["total_problems"],
                param_str,
            ])

        plot_x, plot_y = _xy(x_data, y_data)
        fig.add_trace(go.Scatter(
            x=plot_x,
            y=plot_y,
            mode="markers+lines",
            name=str(gv),
            marker=dict(size=9, color=color),
            line=dict(color=color, width=2),
            hovertemplate=(
                "<b>%{customdata[0]}</b> / %{customdata[1]}<br>"
                "Solve rate: %{customdata[2]}<br>"
                f"Cost: %{{{'y' if flip_axes else 'x'}:.2f}}M SFLOPs/problem<br>"
                "Seeds: %{customdata[3]}, Problems: %{customdata[4]}<br>"
                "%{customdata[5]}"
                "<extra></extra>"
            ),
            customdata=customdata,
        ))

    # Envelope(s)
    if show_envelope and group_curves:
        if envelope_scope == "Global":
            all_curves = [{"x": d["x"], "y": d["y"]} for d in group_curves.values() if len(d["x"]) >= 2]
            if all_curves:
                x_max = max(max(c["x"]) for c in all_curves)
                env_x, env_y = compute_upper_envelope(all_curves, 0.0, x_max)
                if env_x and env_y:
                    env_plot_x, env_plot_y = _xy(env_x, env_y)
                    env_name = "Cost-Efficient Frontier" if flip_axes else "Upper Envelope"
                    fig.add_trace(go.Scatter(
                        x=env_plot_x, y=env_plot_y, mode="lines",
                        name=env_name,
                        line=dict(color="#00FFFF", width=3, dash="dot"),
                        hovertemplate=(
                            f"<b>{env_name}</b><br>"
                            f"Solve rate: %{{{'x' if flip_axes else 'y'}:.3f}}<br>"
                            f"Cost: %{{{'y' if flip_axes else 'x'}:.2f}}M<extra></extra>"
                        ),
                    ))
        else:
            # Per run — use run_base_colors for envelope colour
            run_groups: Dict[str, List[Dict]] = {}
            for pt in visible_points:
                rn = pt["run_name"]
                run_groups.setdefault(rn, []).append(pt)

            dash_styles = ["dot", "dash", "dashdot", "longdash", "longdashdot"]
            for idx, (rn, rpts) in enumerate(sorted(run_groups.items())):
                rpts.sort(key=lambda p: p["avg_cost_per_problem"])
                cx = [0.0] + [p["avg_cost_per_problem"] / 1e6 for p in rpts]
                cy = [0.0] + [p["solve_rate"] for p in rpts]
                if len(cx) < 2:
                    continue
                x_max = max(cx)
                env_x, env_y = compute_upper_envelope([{"x": cx, "y": cy}], 0.0, x_max)
                if env_x and env_y:
                    ec = run_base_colors.get(rn, color_palette[idx % len(color_palette)])
                    env_plot_x, env_plot_y = _xy(env_x, env_y)
                    fig.add_trace(go.Scatter(
                        x=env_plot_x, y=env_plot_y, mode="lines",
                        name=f"Env: {rn}",
                        line=dict(color=ec, width=2, dash=dash_styles[idx % len(dash_styles)]),
                        hovertemplate=(
                            f"<b>Envelope: {rn}</b><br>"
                            f"Solve rate: %{{{'x' if flip_axes else 'y'}:.3f}}<br>"
                            f"Cost: %{{{'y' if flip_axes else 'x'}:.2f}}M<extra></extra>"
                        ),
                    ))

    if flip_axes:
        xaxis_title, yaxis_title = "Solve Rate", "Avg Cost per Problem (M SFLOPs)"
        xaxis_tickformat, yaxis_tickformat = ".0%", None
    else:
        xaxis_title, yaxis_title = "Avg Cost per Problem (M SFLOPs)", "Solve Rate"
        xaxis_tickformat, yaxis_tickformat = None, ".0%"
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_tickformat=xaxis_tickformat,
        yaxis_tickformat=yaxis_tickformat,
        hovermode="closest",
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 6b. Paper-quality export ---
    with st.expander("Paper Export (PDF)", expanded=False):
        _render_paper_export(
            group_curves, show_envelope, envelope_scope,
            visible_points, run_base_colors, color_palette,
            flip_axes=flip_axes,
        )

    # --- 7. Envelope AUC comparison ---
    if len(visible_points) >= 2:
        st.subheader("AUC Comparison")

        # Per-run "Envelope by" selectors
        visible_run_names = [rn for rn in run_names if run_visible.get(rn, True)]
        run_envelope_key: Dict[str, Optional[str]] = {}

        if visible_run_names:
            with st.expander("Per-run envelope grouping", expanded=True):
                for rn in visible_run_names:
                    rpts = per_run_points.get(rn, [])
                    vk = per_run_varying_keys.get(rn, [])
                    if len(rpts) > 1 and vk:
                        run_envelope_key[rn] = st.selectbox(
                            f"Envelope by ({rn})",
                            options=["(all points)"] + vk,
                            index=0,
                            key=f"traj_env_by_{rn}",
                        )
                        if run_envelope_key[rn] == "(all points)":
                            run_envelope_key[rn] = None
                    else:
                        run_envelope_key[rn] = None

        # Build envelope groups from per-run selections
        envelope_groups: Dict[str, List[Dict]] = {}
        for rn in visible_run_names:
            rpts = per_run_points.get(rn, [])
            env_key = run_envelope_key.get(rn)
            if env_key is None:
                envelope_groups[rn] = list(rpts)
            else:
                for pt in rpts:
                    val = str(pt["flat_params"].get(env_key, "N/A"))
                    group_label = f"{rn}: {env_key}={val}"
                    envelope_groups.setdefault(group_label, []).append(pt)

        # Compute raw curves per group (sorted by integration-axis x, no convex hull).
        # When flip_axes is True, x=solve_rate and y=cost (integrate cost over solve rate).
        raw_curves: Dict[str, Dict] = {}
        for gname, gpts in envelope_groups.items():
            if flip_axes:
                gpts.sort(key=lambda p: p["solve_rate"])
                if include_origin:
                    cx = [0.0] + [p["solve_rate"] for p in gpts]
                    cy = [0.0] + [p["avg_cost_per_problem"] / 1e6 for p in gpts]
                else:
                    cx = [p["solve_rate"] for p in gpts]
                    cy = [p["avg_cost_per_problem"] / 1e6 for p in gpts]
            else:
                gpts.sort(key=lambda p: p["avg_cost_per_problem"])
                if include_origin:
                    cx = [0.0] + [p["avg_cost_per_problem"] / 1e6 for p in gpts]
                    cy = [0.0] + [p["solve_rate"] for p in gpts]
                else:
                    cx = [p["avg_cost_per_problem"] / 1e6 for p in gpts]
                    cy = [p["solve_rate"] for p in gpts]
            if len(cx) < 2:
                continue
            raw_curves[gname] = {"x": cx, "y": cy}

        common_range_unit = "solve rate" if flip_axes else "M SFLOPs"
        common_range_fmt = (lambda v: f"{v*100:.1f}%") if flip_axes else (lambda v: f"{v:.2f}M")
        # When flipped, AUC = ∫ cost d(solve rate); lower is better, so reverse sort.
        higher_is_better = not flip_axes

        if len(raw_curves) >= 2:
            common_x_min = max(min(d["x"]) for d in raw_curves.values())
            common_x_max = min(max(d["x"]) for d in raw_curves.values())
            if common_x_max > common_x_min:
                st.caption(
                    f"Common x-range: {common_range_fmt(common_x_min)} - "
                    f"{common_range_fmt(common_x_max)} ({common_range_unit})"
                )
                auc_rows = []
                # Common grid for fair comparison
                x_grid = np.linspace(common_x_min, common_x_max, 500)
                for gname, data in raw_curves.items():
                    xs, ys = np.array(data["x"]), np.array(data["y"])
                    # Linear interpolation onto common grid
                    y_interp = np.interp(x_grid, xs, ys)
                    auc = float(np.trapezoid(y_interp, x_grid))
                    auc_rows.append({"Group": gname, "AUC": auc})

                if auc_rows:
                    auc_rows.sort(key=lambda r: r["AUC"], reverse=higher_is_better)
                    worst = auc_rows[-1]["AUC"]
                    for r in auc_rows:
                        r["vs Worst %"] = f"{((r['AUC'] / worst) - 1) * 100:+.1f}%" if worst > 0 else "N/A"
                        r["AUC"] = f"{r['AUC']:.4f}"
                    st.dataframe(pd.DataFrame(auc_rows), use_container_width=True, hide_index=True)
        elif len(raw_curves) == 1:
            gname, data = next(iter(raw_curves.items()))
            xs, ys = data["x"], data["y"]
            if len(xs) >= 2:
                auc = sum((xs[i+1] - xs[i]) * (ys[i] + ys[i+1]) / 2 for i in range(len(xs) - 1))
                st.dataframe(
                    pd.DataFrame([{"Group": gname, "AUC": f"{auc:.4f}"}]),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.info("Not enough data points to compute AUC.")

    # --- 8. Configuration details table ---
    st.subheader("Configuration Details")

    table_rows = []
    for pt in visible_points:
        row = {
            "Run": pt["run_name"],
            "Config": pt["config_label"],
            "Solve Rate": pt["solve_rate"],
            "Cost (M)": pt["avg_cost_per_problem"] / 1e6,
            "Seeds": pt["num_seeds"],
            "Problems": pt["total_problems"],
        }
        if pt["std_solve_rate"] > 0:
            row["Std"] = pt["std_solve_rate"]
        for k, v in pt["flat_params"].items():
            row[k] = v
        table_rows.append(row)

    df = pd.DataFrame(table_rows)
    # Format numeric columns
    if "Solve Rate" in df.columns:
        df["Solve Rate"] = df["Solve Rate"].apply(lambda x: f"{x:.3f}")
    if "Cost (M)" in df.columns:
        df["Cost (M)"] = df["Cost (M)"].apply(lambda x: f"{x:.2f}")
    if "Std" in df.columns:
        df["Std"] = df["Std"].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

    st.dataframe(df, use_container_width=True, hide_index=True, height=400)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _hashable(v):
    """Make a value hashable for use in sets."""
    if isinstance(v, list):
        return tuple(v)
    if isinstance(v, dict):
        return tuple(sorted(v.items()))
    return v
