"""
Cost/Quality Analysis tab for RL Training Analysis.

Shows scatter plot of average cost vs success rate for baselines and RL runs.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import colorsys
import numpy as np

from .utils import load_full_proof_baselines, load_rollouts_cached, get_rollouts_file


def _compute_upper_envelope(
    baselines_df: pd.DataFrame,
    y_column: str = 'success_rate'
) -> List[Tuple[float, float]]:
    """
    Compute the upper envelope (convex hull upper bound) of baseline curves.

    For each x value across both baseline models, take the max y value.
    Returns a list of (x, y) points sorted by x.

    Args:
        baselines_df: DataFrame with baseline data
        y_column: Column name for y-axis values (default: 'success_rate')
    """
    if baselines_df.empty:
        return []

    # Check if y_column exists
    if y_column not in baselines_df.columns:
        return []

    # Collect all points from all baselines
    all_points = []
    for _, row in baselines_df.iterrows():
        x = row['avg_cost_millions']
        y = row[y_column]
        all_points.append((x, y))

    if not all_points:
        return []

    # Sort by x
    all_points.sort(key=lambda p: p[0])

    # Build upper envelope: for overlapping x regions, keep max y
    # We interpolate between points and take the max
    envelope = [(0, 0)]  # Start at origin

    # Get unique x values and interpolate both curves
    all_x = sorted(set(p[0] for p in all_points))

    # Group points by baseline model for interpolation
    points_8b = [(row['avg_cost_millions'], row[y_column])
                 for _, row in baselines_df.iterrows() if row['baseline'] == 'FULL_PROOF_8B']
    points_32b = [(row['avg_cost_millions'], row[y_column])
                  for _, row in baselines_df.iterrows() if row['baseline'] == 'FULL_PROOF_32B']

    points_8b.sort(key=lambda p: p[0])
    points_32b.sort(key=lambda p: p[0])

    # Add origin to both curves
    points_8b = [(0, 0)] + points_8b
    points_32b = [(0, 0)] + points_32b

    def interpolate_curve(points: List[Tuple[float, float]], x: float) -> float:
        """Interpolate y value at x using linear interpolation."""
        if not points:
            return 0
        if x <= points[0][0]:
            return points[0][1]
        if x >= points[-1][0]:
            return points[-1][1]

        # Find the two points to interpolate between
        for i in range(len(points) - 1):
            if points[i][0] <= x <= points[i + 1][0]:
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                if x1 == x0:
                    return y0
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return points[-1][1]

    # Get all x values from both curves plus some intermediate points
    all_curve_x = set([0])
    for p in points_8b + points_32b:
        all_curve_x.add(p[0])
    all_curve_x = sorted(all_curve_x)

    # Compute upper envelope
    for x in all_curve_x:
        y_8b = interpolate_curve(points_8b, x)
        y_32b = interpolate_curve(points_32b, x)
        envelope.append((x, max(y_8b, y_32b)))

    # Remove duplicates and sort
    envelope = sorted(set(envelope), key=lambda p: p[0])

    return envelope


def _compute_auc_between_curves(
    curve1: List[Tuple[float, float]],
    curve2: List[Tuple[float, float]],
    x_min: Optional[float] = None,
    x_max: Optional[float] = None
) -> Tuple[float, float, float]:
    """
    Compute AUC between two curves using trapezoidal integration.

    Args:
        curve1: First curve as list of (x, y) points
        curve2: Second curve as list of (x, y) points
        x_min: Minimum x value for integration (defaults to min of both curves)
        x_max: Maximum x value for integration (defaults to max of both curves)

    Returns:
        (auc_diff, auc_curve1, auc_curve2) where auc_diff = auc_curve1 - auc_curve2
        Positive auc_diff means curve1 is better (higher).
    """
    if not curve1 or not curve2:
        return 0.0, 0.0, 0.0

    # Determine x range
    if x_min is None:
        x_min = min(
            min(p[0] for p in curve1) if curve1 else 0,
            min(p[0] for p in curve2) if curve2 else 0
        )
    if x_max is None:
        x_max = max(
            max(p[0] for p in curve1) if curve1 else 0,
            max(p[0] for p in curve2) if curve2 else 0
        )

    if x_max <= x_min:
        return 0.0, 0.0, 0.0

    def interpolate_curve(points: List[Tuple[float, float]], x: float) -> float:
        """Interpolate y value at x using linear interpolation."""
        if not points:
            return 0
        points = sorted(points, key=lambda p: p[0])
        if x <= points[0][0]:
            return points[0][1]
        if x >= points[-1][0]:
            return points[-1][1]

        for i in range(len(points) - 1):
            if points[i][0] <= x <= points[i + 1][0]:
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                if x1 == x0:
                    return y0
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return points[-1][1]

    # Get all x values from both curves
    all_x = set([x_min])
    for p in curve1 + curve2:
        if x_min <= p[0] <= x_max:
            all_x.add(p[0])
    all_x.add(x_max)
    all_x = sorted(all_x)

    # Compute AUC using trapezoidal rule
    auc1 = 0.0
    auc2 = 0.0

    for i in range(len(all_x) - 1):
        x0, x1 = all_x[i], all_x[i + 1]
        dx = x1 - x0

        y1_0 = interpolate_curve(curve1, x0)
        y1_1 = interpolate_curve(curve1, x1)
        y2_0 = interpolate_curve(curve2, x0)
        y2_1 = interpolate_curve(curve2, x1)

        # Trapezoidal area
        auc1 += dx * (y1_0 + y1_1) / 2
        auc2 += dx * (y2_0 + y2_1) / 2

    return auc1 - auc2, auc1, auc2


def _compute_avg_unique_problems_per_seed(rollouts: List[Dict]) -> float:
    """
    Compute average unique problems solved per seed from rollouts.

    Groups rollouts by seed_idx, counts unique problems solved per seed,
    and returns the average.
    """
    if not rollouts:
        return 0.0

    # Group by seed_idx
    seed_indices = set(r.get('seed_idx', 0) for r in rollouts)
    if not seed_indices:
        return 0.0

    problems_solved_per_seed = []
    for seed_idx in seed_indices:
        seed_rollouts = [r for r in rollouts if r.get('seed_idx', 0) == seed_idx]
        unique_solved = len(set(
            r.get('problem_id', r.get('origin_problem_id', ''))
            for r in seed_rollouts if r.get('success', False)
        ))
        problems_solved_per_seed.append(unique_solved)

    return sum(problems_solved_per_seed) / len(problems_solved_per_seed) if problems_solved_per_seed else 0.0


def _get_round_colors(num_rounds: int) -> List[str]:
    """Generate distinct colors for each round."""
    # Use a color palette that's visually distinct
    base_colors = [
        (0.0, 0.7, 0.5),   # Red
        (0.3, 0.7, 0.5),   # Green
        (0.6, 0.7, 0.5),   # Blue
        (0.1, 0.7, 0.5),   # Orange
        (0.8, 0.7, 0.5),   # Purple
        (0.5, 0.7, 0.5),   # Cyan
    ]

    colors = []
    for i in range(num_rounds):
        h, s, l = base_colors[i % len(base_colors)]
        colors.append((h, s, l))

    return colors


def _adjust_color_for_lambda(base_hsl: tuple, lambda_idx: int, num_lambdas: int) -> str:
    """
    Adjust color lightness based on lambda index.

    Lower lambda (idx=0) -> lighter color
    Higher lambda (idx=num_lambdas-1) -> darker color
    """
    h, s, l = base_hsl

    # Scale lightness: low lambda = light (0.7), high lambda = dark (0.3)
    if num_lambdas > 1:
        # Linear interpolation from light to dark
        lightness = 0.7 - (lambda_idx / (num_lambdas - 1)) * 0.4
    else:
        lightness = 0.5

    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(h, lightness, s)
    return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'


def render_cost_quality_tab(
    rollouts_dir: Path,
    run_path: Optional[Path] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    dataset: Optional[str] = None
):
    """
    Render cost/quality analysis scatter plot for baselines and RL runs.

    Shows average cost per problem (x-axis) vs success rate (y-axis).
    Each point represents a baseline configuration or an RL round/lambda combination.

    Args:
        rollouts_dir: Root rollouts directory
        run_path: Optional path to the RL run directory
        run_metadata: Optional metadata for the RL run
        dataset: Optional dataset name (e.g., 'putnam') for baseline data location
    """
    st.subheader("Cost/Quality Analysis")

    # Baseline selection checkboxes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_8b = st.checkbox("Show 8B Baseline", value=True, key="show_8b_baseline")
    with col2:
        show_32b = st.checkbox("Show 32B Baseline", value=True, key="show_32b_baseline")
    with col3:
        show_training = st.checkbox("Show Training Results", value=True, key="show_training_results")
    with col4:
        show_shaded = st.checkbox("Show Shaded Region", value=False, key="show_shaded_region")

    # Load full_proof baselines
    all_baselines = load_full_proof_baselines(rollouts_dir, dataset)

    # Filter baselines based on checkbox selection
    baselines = []
    if all_baselines:
        for b in all_baselines:
            if b.get('baseline') == 'FULL_PROOF_8B' and show_8b:
                baselines.append(b)
            elif b.get('baseline') == 'FULL_PROOF_32B' and show_32b:
                baselines.append(b)

    # Create figure
    fig = go.Figure()

    has_baseline_data = False
    has_rl_data = False

    # Add baseline data if available
    if baselines:
        has_baseline_data = True
        df_baselines = pd.DataFrame(baselines)

        # Ensure required columns exist
        required_cols = ['baseline', 'max_attempts', 'success_rate', 'avg_cost']
        missing_cols = [c for c in required_cols if c not in df_baselines.columns]

        if not missing_cols:
            # Create display label combining model and max_attempts
            df_baselines['label'] = df_baselines['baseline'] + ' (max=' + df_baselines['max_attempts'].astype(str) + ')'
            df_baselines['avg_cost_millions'] = df_baselines['avg_cost'] / 1e6

            # Add baseline points grouped by model
            baseline_colors = {'FULL_PROOF_8B': 'lightgray', 'FULL_PROOF_32B': 'darkgray'}
            for model in df_baselines['baseline'].unique():
                model_df = df_baselines[df_baselines['baseline'] == model].sort_values('max_attempts')
                color = baseline_colors.get(model, 'gray')

                # Add points
                fig.add_trace(go.Scatter(
                    x=model_df['avg_cost_millions'],
                    y=model_df['success_rate'],
                    mode='markers+lines',
                    name=f'Baseline: {model}',
                    marker=dict(size=8, color=color, symbol='diamond'),
                    line=dict(dash='dot', width=1, color=color),
                    hovertemplate=(
                        f'<b>{model}</b><br>'
                        'Max attempts: %{customdata[0]}<br>'
                        'Success rate: %{y:.3f}<br>'
                        'Avg cost: %{x:.2f}M<extra></extra>'
                    ),
                    customdata=model_df[['max_attempts']].values,
                    legendgroup='baselines',
                ))

    # Collect lambda points and baseline envelope for AUC computation
    lambda_curve_points: List[Tuple[float, float]] = []
    lambda_curves_by_round: Dict[int, List[Tuple[float, float]]] = {}
    baseline_envelope: List[Tuple[float, float]] = []

    # Compute baseline upper envelope if we have baseline data
    if has_baseline_data and baselines:
        df_baselines_for_envelope = pd.DataFrame(baselines)
        if 'avg_cost' in df_baselines_for_envelope.columns:
            df_baselines_for_envelope['avg_cost_millions'] = df_baselines_for_envelope['avg_cost'] / 1e6
            baseline_envelope = _compute_upper_envelope(df_baselines_for_envelope)

            # Add upper envelope trace
            if baseline_envelope:
                envelope_x = [p[0] for p in baseline_envelope]
                envelope_y = [p[1] for p in baseline_envelope]
                fig.add_trace(go.Scatter(
                    x=envelope_x,
                    y=envelope_y,
                    mode='lines',
                    name='Baseline Upper Envelope',
                    line=dict(color='rgba(100, 100, 100, 0.5)', width=2, dash='dash'),
                    fill='tozeroy' if show_shaded else None,
                    fillcolor='rgba(200, 200, 200, 0.2)' if show_shaded else None,
                    hoverinfo='skip',
                    legendgroup='envelope',
                ))

    # Add RL run data if available
    if show_training and run_metadata and run_metadata.get('iteration_results'):
        iter_results = run_metadata['iteration_results']
        rounds_data = iter_results.get('rounds', [])
        lambda_values = iter_results.get('lambda_values', [])

        if rounds_data and lambda_values:
            has_rl_data = True
            num_rounds = len(rounds_data)
            num_lambdas = len(lambda_values)
            sorted_lambdas = sorted(lambda_values)

            # Get colors for each round
            round_colors = _get_round_colors(num_rounds)

            # Collect all lambda points first
            all_lambda_points = []

            # Extract data points from by_budget_lambda
            for round_info in rounds_data:
                round_id = round_info.get('round_id', 0)
                by_budget_lambda = round_info.get('by_budget_lambda', {})

                if not by_budget_lambda:
                    continue

                round_base_color = round_colors[round_id % len(round_colors)]

                # For each budget (we'll use 'unlimited' or the first available)
                for budget_key, lambda_data in by_budget_lambda.items():
                    for lambda_str, stats in lambda_data.items():
                        try:
                            lambda_val = float(lambda_str)
                        except ValueError:
                            continue

                        lambda_idx = sorted_lambdas.index(lambda_val) if lambda_val in sorted_lambdas else 0
                        point_color = _adjust_color_for_lambda(round_base_color, lambda_idx, num_lambdas)

                        avg_cost = stats.get('avg_cost', 0) / 1e6
                        success_rate = stats.get('success_rate', 0)

                        all_lambda_points.append({
                            'cost': avg_cost,
                            'success_rate': success_rate,
                            'lambda_val': lambda_val,
                            'lambda_idx': lambda_idx,
                            'round_id': round_id,
                            'budget_key': budget_key,
                            'color': point_color,
                        })

            # Group points by round and sort each round's points by cost
            points_by_round: Dict[int, List[Dict]] = {}
            for p in all_lambda_points:
                rid = p['round_id']
                if rid not in points_by_round:
                    points_by_round[rid] = []
                points_by_round[rid].append(p)

            # Sort each round's points by cost
            for rid in points_by_round:
                points_by_round[rid].sort(key=lambda p: p['cost'])

            # Build lambda curves per round (for AUC calculation)
            lambda_curves_by_round: Dict[int, List[Tuple[float, float]]] = {}
            for rid, points in points_by_round.items():
                curve = [(0, 0)]  # Start at origin
                for p in points:
                    curve.append((p['cost'], p['success_rate']))
                lambda_curves_by_round[rid] = curve

            # Also build overall lambda curve for backwards compatibility
            all_lambda_points.sort(key=lambda p: p['cost'])
            lambda_curve_points = [(0, 0)]
            for p in all_lambda_points:
                lambda_curve_points.append((p['cost'], p['success_rate']))

            # Add connected line for each round (with round-specific color)
            for rid in sorted(points_by_round.keys()):
                points = points_by_round[rid]
                if not points:
                    continue

                # Get the base color for this round (convert HSL to RGB for the line)
                round_base_hsl = round_colors[rid % len(round_colors)]
                h, s, l = round_base_hsl
                r, g, b = colorsys.hls_to_rgb(h, 0.5, s)
                line_color = f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)'

                # Build curve points for this round
                curve_x = [0] + [p['cost'] for p in points]
                curve_y = [0] + [p['success_rate'] for p in points]

                fig.add_trace(go.Scatter(
                    x=curve_x,
                    y=curve_y,
                    mode='lines',
                    name=f'Round {rid} Curve',
                    line=dict(color=line_color, width=2),
                    hoverinfo='skip',
                    legendgroup=f'round_{rid}',
                    showlegend=False,
                ))

            # Add individual lambda points as markers
            for p in all_lambda_points:
                # Only show legend for first point of each round
                show_legend = (p == next((pt for pt in all_lambda_points if pt['round_id'] == p['round_id']), None))

                fig.add_trace(go.Scatter(
                    x=[p['cost']],
                    y=[p['success_rate']],
                    mode='markers',
                    name=f'Round {p["round_id"]}' if show_legend else None,
                    showlegend=show_legend,
                    marker=dict(size=12, color=p['color'], symbol='circle'),
                    hovertemplate=(
                        f'<b>Round {p["round_id"]}</b><br>'
                        f'Lambda: {p["lambda_val"]:.0e}<br>'
                        f'Budget: {p["budget_key"]}<br>'
                        'Success rate: %{y:.3f}<br>'
                        'Avg cost: %{x:.2f}M<extra></extra>'
                    ),
                    legendgroup=f'round_{p["round_id"]}',
                ))

    if not has_baseline_data and not has_rl_data:
        if all_baselines and not (show_8b or show_32b):
            st.warning("No baselines selected. Check at least one baseline checkbox above.")
        else:
            st.warning("No data available. No baselines found and no RL run data loaded.")
        return

    # Update layout
    title = 'Cost vs Quality Trade-off'
    if has_baseline_data and has_rl_data:
        title += ' (Baselines + RL Rounds)'
    elif has_baseline_data:
        title += ' (Baselines)'
    elif has_rl_data:
        title += ' (RL Rounds)'

    fig.update_layout(
        title=title,
        xaxis_title='Average Cost per Problem (M SFLOPs)',
        yaxis_title='Success Rate',
        yaxis_tickformat='.0%',
        hovermode='closest',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Compute and display AUC score per round
    if has_rl_data and baseline_envelope and len(baseline_envelope) > 1 and has_baseline_data:
        # Use the actual baseline data points' x-range for integration
        df_baselines_for_auc = pd.DataFrame(baselines)
        df_baselines_for_auc['avg_cost_millions'] = df_baselines_for_auc['avg_cost'] / 1e6
        baseline_costs = df_baselines_for_auc['avg_cost_millions'].tolist()

        min_x = min(baseline_costs)
        max_x = max(baseline_costs)

        # Compute baseline AUC once
        _, _, auc_baseline = _compute_auc_between_curves(
            baseline_envelope, baseline_envelope, x_min=min_x, x_max=max_x
        )

        # Compute AUC for each round
        round_aucs: Dict[int, Tuple[float, float, float]] = {}  # round_id -> (auc_diff, auc_round, auc_baseline)
        for rid, curve in lambda_curves_by_round.items():
            if len(curve) > 1:
                auc_diff, auc_round, auc_bl = _compute_auc_between_curves(
                    curve, baseline_envelope, x_min=min_x, x_max=max_x
                )
                round_aucs[rid] = (auc_diff, auc_round, auc_bl)

        # Display AUC metrics per round
        st.markdown("### AUC Analysis by Round")

        # Show baseline AUC first
        st.metric(
            "Baseline Envelope AUC",
            f"{auc_baseline:.2f}",
            help="Area under the baseline upper envelope (reference)"
        )

        # Create columns for round metrics (up to 4 per row)
        sorted_rounds = sorted(round_aucs.keys())
        if sorted_rounds:
            # Display in rows of 4 columns
            for row_start in range(0, len(sorted_rounds), 4):
                row_rounds = sorted_rounds[row_start:row_start + 4]
                cols = st.columns(len(row_rounds))

                for col, rid in zip(cols, row_rounds):
                    auc_diff, auc_round, _ = round_aucs[rid]
                    delta_pct = (auc_diff / auc_baseline * 100) if auc_baseline > 0 else 0

                    with col:
                        st.metric(
                            f"Round {rid}",
                            f"{auc_round:.2f}",
                            delta=f"{delta_pct:+.1f}% vs baseline",
                            delta_color="normal",
                            help=f"Round {rid} AUC: {auc_round:.2f}, Improvement: {auc_diff:+.2f}"
                        )

        # Summary table
        if len(round_aucs) > 1:
            st.markdown("#### Summary Table")
            summary_data = []
            for rid in sorted_rounds:
                auc_diff, auc_round, _ = round_aucs[rid]
                delta_pct = (auc_diff / auc_baseline * 100) if auc_baseline > 0 else 0
                summary_data.append({
                    'Round': rid,
                    'AUC': f"{auc_round:.2f}",
                    'vs Baseline': f"{auc_diff:+.2f}",
                    'Improvement %': f"{delta_pct:+.1f}%"
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        st.caption(f"**AUC Score**: Area under cost-quality curve. Higher is better. "
                   f"Integration range: [{min_x:.1f}, {max_x:.1f}]M SFLOPs")

    # Show legend explanation for RL data
    if has_rl_data:
        st.caption("**RL rounds**: Each round has a different color. Within each round, "
                   "lighter shades = lower λ, darker shades = higher λ.")

    # === Second Plot: Unique Problems Solved per Seed vs Cost ===
    st.markdown("---")
    fig2 = go.Figure()

    has_baseline_data2 = False
    has_rl_data2 = False

    # Add baseline data if available
    if baselines:
        df_baselines2 = pd.DataFrame(baselines)

        # Check if we have the required columns
        required_cols2 = ['baseline', 'max_attempts', 'avg_cost']

        if all(c in df_baselines2.columns for c in required_cols2):
            # Check if we have pre-computed avg_unique_problems_per_seed
            has_unique_per_seed = 'avg_unique_problems_per_seed' in df_baselines2.columns
            has_unique_total = 'unique_problems_solved' in df_baselines2.columns
            has_num_seeds = 'num_seeds' in df_baselines2.columns

            if has_unique_per_seed:
                df_baselines2['unique_per_seed'] = df_baselines2['avg_unique_problems_per_seed']
                has_baseline_data2 = True
            elif has_unique_total and has_num_seeds:
                df_baselines2['unique_per_seed'] = df_baselines2['unique_problems_solved'] / df_baselines2['num_seeds']
                has_baseline_data2 = True
            else:
                # Try to compute from rollouts
                if dataset:
                    full_proof_dir = rollouts_dir / "baselines" / dataset / "full_proof"
                else:
                    full_proof_dir = rollouts_dir / "baselines" / "full_proof"
                unique_per_seed_values = []

                for _, row in df_baselines2.iterrows():
                    baseline_name = row['baseline']
                    max_attempts = row['max_attempts']
                    rollouts_file = full_proof_dir / baseline_name / f"max_{max_attempts}" / "rollouts_summary.json"

                    if rollouts_file.exists():
                        rollouts = load_rollouts_cached(str(rollouts_file))
                        unique_per_seed = _compute_avg_unique_problems_per_seed(rollouts)
                    else:
                        unique_per_seed = 0.0
                    unique_per_seed_values.append(unique_per_seed)

                df_baselines2['unique_per_seed'] = unique_per_seed_values
                if any(v > 0 for v in unique_per_seed_values):
                    has_baseline_data2 = True

            if has_baseline_data2:
                df_baselines2['avg_cost_millions'] = df_baselines2['avg_cost'] / 1e6

                # Add baseline points grouped by model
                baseline_colors = {'FULL_PROOF_8B': 'lightgray', 'FULL_PROOF_32B': 'darkgray'}
                for model in df_baselines2['baseline'].unique():
                    model_df = df_baselines2[df_baselines2['baseline'] == model].sort_values('max_attempts')
                    color = baseline_colors.get(model, 'gray')

                    fig2.add_trace(go.Scatter(
                        x=model_df['avg_cost_millions'],
                        y=model_df['unique_per_seed'],
                        mode='markers+lines',
                        name=f'Baseline: {model}',
                        marker=dict(size=8, color=color, symbol='diamond'),
                        line=dict(dash='dot', width=1, color=color),
                        hovertemplate=(
                            f'<b>{model}</b><br>'
                            'Max attempts: %{customdata[0]}<br>'
                            'Unique problems/seed: %{y:.1f}<br>'
                            'Avg cost: %{x:.2f}M<extra></extra>'
                        ),
                        customdata=model_df[['max_attempts']].values,
                        legendgroup='baselines2',
                    ))

    # Collect data for AUC calculation
    unique_curves_by_round: Dict[int, List[Tuple[float, float]]] = {}
    baseline_envelope2: List[Tuple[float, float]] = []

    # Compute baseline upper envelope for unique problems plot (convex hull of 8B and 32B)
    if has_baseline_data2:
        baseline_envelope2 = _compute_upper_envelope(df_baselines2, y_column='unique_per_seed')

    # Add RL run data if available
    if show_training and run_metadata and run_metadata.get('iteration_results'):
        iter_results = run_metadata['iteration_results']
        rounds_data = iter_results.get('rounds', [])
        lambda_values = iter_results.get('lambda_values', [])
        run_path = run_metadata.get('path')

        if rounds_data and lambda_values:
            num_rounds = len(rounds_data)
            num_lambdas = len(lambda_values)
            sorted_lambdas = sorted(lambda_values)
            round_colors = _get_round_colors(num_rounds)

            # Cache for rollouts loaded per round
            round_rollouts_cache: Dict[int, List[Dict]] = {}

            # Collect all points first
            all_unique_points = []

            for round_info in rounds_data:
                round_id = round_info.get('round_id', 0)
                by_budget_lambda = round_info.get('by_budget_lambda', {})

                if not by_budget_lambda:
                    continue

                round_base_color = round_colors[round_id % len(round_colors)]

                for budget_key, lambda_data in by_budget_lambda.items():
                    for lambda_str, stats in lambda_data.items():
                        try:
                            lambda_val = float(lambda_str)
                        except ValueError:
                            continue

                        # Check if we have unique problems per seed data
                        unique_per_seed = stats.get('avg_unique_problems_per_seed')
                        if unique_per_seed is None:
                            # Try to compute from unique_solved and num_seeds
                            unique_solved = stats.get('unique_solved', stats.get('unique_problems_solved', 0))
                            num_seeds = stats.get('num_seeds')
                            if num_seeds and num_seeds > 0:
                                unique_per_seed = unique_solved / num_seeds
                            elif run_path:
                                # Try to compute from rollouts
                                if round_id not in round_rollouts_cache:
                                    round_dir = Path(run_path) / f"round{round_id}"
                                    rollouts_file = get_rollouts_file(round_dir)
                                    if rollouts_file:
                                        round_rollouts_cache[round_id] = load_rollouts_cached(str(rollouts_file))
                                    else:
                                        round_rollouts_cache[round_id] = []

                                # Filter rollouts for this budget and lambda
                                budget_val = None if budget_key == "unlimited" else float(budget_key.replace("M", "")) * 1e6
                                filtered_rollouts = [
                                    r for r in round_rollouts_cache[round_id]
                                    if r.get('budget') == budget_val and abs(float(r.get('lambda', 0)) - lambda_val) < 1e-12
                                ]
                                if filtered_rollouts:
                                    unique_per_seed = _compute_avg_unique_problems_per_seed(filtered_rollouts)
                                else:
                                    continue
                            else:
                                continue

                        has_rl_data2 = True
                        lambda_idx = sorted_lambdas.index(lambda_val) if lambda_val in sorted_lambdas else 0
                        point_color = _adjust_color_for_lambda(round_base_color, lambda_idx, num_lambdas)

                        avg_cost = stats.get('avg_cost', 0) / 1e6

                        all_unique_points.append({
                            'cost': avg_cost,
                            'unique_per_seed': unique_per_seed,
                            'lambda_val': lambda_val,
                            'lambda_idx': lambda_idx,
                            'round_id': round_id,
                            'budget_key': budget_key,
                            'color': point_color,
                        })

            # Group points by round
            points_by_round2: Dict[int, List[Dict]] = {}
            for p in all_unique_points:
                rid = p['round_id']
                if rid not in points_by_round2:
                    points_by_round2[rid] = []
                points_by_round2[rid].append(p)

            # Sort each round's points by cost
            for rid in points_by_round2:
                points_by_round2[rid].sort(key=lambda p: p['cost'])

            # Build curves per round for AUC
            for rid, points in points_by_round2.items():
                curve = [(0, 0)]
                for p in points:
                    curve.append((p['cost'], p['unique_per_seed']))
                unique_curves_by_round[rid] = curve

            # Add per-round lines
            for rid in sorted(points_by_round2.keys()):
                points = points_by_round2[rid]
                if not points:
                    continue

                round_base_hsl = round_colors[rid % len(round_colors)]
                h, s, l = round_base_hsl
                r, g, b = colorsys.hls_to_rgb(h, 0.5, s)
                line_color = f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)'

                curve_x = [0] + [p['cost'] for p in points]
                curve_y = [0] + [p['unique_per_seed'] for p in points]

                fig2.add_trace(go.Scatter(
                    x=curve_x,
                    y=curve_y,
                    mode='lines',
                    name=f'Round {rid} Curve',
                    line=dict(color=line_color, width=2),
                    hoverinfo='skip',
                    legendgroup=f'round2_{rid}',
                    showlegend=False,
                ))

            # Add individual points as markers
            for p in all_unique_points:
                show_legend = (p == next((pt for pt in all_unique_points if pt['round_id'] == p['round_id']), None))

                fig2.add_trace(go.Scatter(
                    x=[p['cost']],
                    y=[p['unique_per_seed']],
                    mode='markers',
                    name=f'Round {p["round_id"]}' if show_legend else None,
                    showlegend=show_legend,
                    marker=dict(size=12, color=p['color'], symbol='circle'),
                    hovertemplate=(
                        f'<b>Round {p["round_id"]}</b><br>'
                        f'Lambda: {p["lambda_val"]:.0e}<br>'
                        f'Budget: {p["budget_key"]}<br>'
                        'Unique problems/seed: %{y:.1f}<br>'
                        'Avg cost: %{x:.2f}M<extra></extra>'
                    ),
                    legendgroup=f'round2_{p["round_id"]}',
                ))

    if has_baseline_data2 or has_rl_data2:
        title2 = 'Cost vs Unique Problems Solved per Seed'
        if has_baseline_data2 and has_rl_data2:
            title2 += ' (Baselines + RL Rounds)'
        elif has_baseline_data2:
            title2 += ' (Baselines)'
        elif has_rl_data2:
            title2 += ' (RL Rounds)'

        fig2.update_layout(
            title=title2,
            xaxis_title='Average Cost per Problem (M SFLOPs)',
            yaxis_title='Avg Unique Problems Solved per Seed',
            hovermode='closest',
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            ),
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Compute and display AUC for unique problems plot
        if has_rl_data2 and unique_curves_by_round and has_baseline_data2 and baseline_envelope2:
            # Use baseline data range for integration
            baseline_costs2 = [p[0] for p in baseline_envelope2 if p[0] > 0]
            if baseline_costs2:
                min_x2 = min(baseline_costs2)
                max_x2 = max(baseline_costs2)

                # Compute baseline AUC
                _, _, auc_baseline2 = _compute_auc_between_curves(
                    baseline_envelope2, baseline_envelope2, x_min=min_x2, x_max=max_x2
                )

                # Compute AUC per round
                round_aucs2: Dict[int, Tuple[float, float, float]] = {}
                for rid, curve in unique_curves_by_round.items():
                    if len(curve) > 1:
                        auc_diff, auc_round, auc_bl = _compute_auc_between_curves(
                            curve, baseline_envelope2, x_min=min_x2, x_max=max_x2
                        )
                        round_aucs2[rid] = (auc_diff, auc_round, auc_bl)

                # Display AUC metrics
                st.markdown("### AUC Analysis (Unique Problems) by Round")

                st.metric(
                    "Baseline Envelope AUC",
                    f"{auc_baseline2:.2f}",
                    help="Area under the baseline envelope (reference)"
                )

                sorted_rounds2 = sorted(round_aucs2.keys())
                if sorted_rounds2:
                    for row_start in range(0, len(sorted_rounds2), 4):
                        row_rounds = sorted_rounds2[row_start:row_start + 4]
                        cols = st.columns(len(row_rounds))

                        for col, rid in zip(cols, row_rounds):
                            auc_diff, auc_round, _ = round_aucs2[rid]
                            delta_pct = (auc_diff / auc_baseline2 * 100) if auc_baseline2 > 0 else 0

                            with col:
                                st.metric(
                                    f"Round {rid}",
                                    f"{auc_round:.2f}",
                                    delta=f"{delta_pct:+.1f}% vs baseline",
                                    delta_color="normal",
                                )

                # Summary table
                if len(round_aucs2) > 1:
                    st.markdown("#### Summary Table")
                    summary_data2 = []
                    for rid in sorted_rounds2:
                        auc_diff, auc_round, _ = round_aucs2[rid]
                        delta_pct = (auc_diff / auc_baseline2 * 100) if auc_baseline2 > 0 else 0
                        summary_data2.append({
                            'Round': rid,
                            'AUC': f"{auc_round:.2f}",
                            'vs Baseline': f"{auc_diff:+.2f}",
                            'Improvement %': f"{delta_pct:+.1f}%"
                        })
                    st.dataframe(pd.DataFrame(summary_data2), use_container_width=True, hide_index=True)

                st.caption(f"**AUC Score**: Area under cost vs unique-problems curve. "
                           f"Integration range: [{min_x2:.1f}, {max_x2:.1f}]M SFLOPs")

        if has_rl_data2:
            st.caption("**RL rounds**: Each round has a different color. Within each round, "
                       "lighter shades = lower λ, darker shades = higher λ.")
    else:
        st.info("No unique problems per seed data available for this plot. "
                "Requires 'avg_unique_problems_per_seed' or 'unique_problems_solved' + 'num_seeds' fields.")

    # Show data table below for baselines
    if has_baseline_data and baselines:
        st.markdown("### Baseline Details")
        df_baselines = pd.DataFrame(baselines)

        required_cols = ['baseline', 'max_attempts', 'success_rate', 'avg_cost']
        if all(c in df_baselines.columns for c in required_cols):
            display_cols = ['baseline', 'max_attempts', 'success_rate', 'avg_cost']
            if 'num_successful' in df_baselines.columns:
                display_cols.append('num_successful')
            if 'unique_problems_solved' in df_baselines.columns:
                display_cols.append('unique_problems_solved')
            if 'num_rollouts' in df_baselines.columns:
                display_cols.append('num_rollouts')

            display_df = df_baselines[display_cols].copy()
            display_df.columns = ['Model', 'Max Attempts', 'Success Rate', 'Avg Cost'] + \
                                 (['Solved'] if 'num_successful' in display_cols else []) + \
                                 (['Unique Solved'] if 'unique_problems_solved' in display_cols else []) + \
                                 (['Total Rollouts'] if 'num_rollouts' in display_cols else [])
            display_df['Success Rate'] = display_df['Success Rate'].apply(lambda x: f"{x:.3f}")
            display_df['Avg Cost'] = display_df['Avg Cost'].apply(lambda x: f"{x/1e6:.2f}M")
            display_df = display_df.sort_values(['Model', 'Max Attempts'])

            st.dataframe(display_df, use_container_width=True, hide_index=True)
