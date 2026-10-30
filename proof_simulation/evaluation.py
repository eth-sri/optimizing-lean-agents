"""Paper AUCs: output cost, linear segments, fixed-1/fixed-64 routing anchors.

Points are (mean output cost per problem in M SFLOPs, solve rate in [0, 1]).
All settings are retained in cost order, including non-monotone solve rates.
"""

import json
import math
from pathlib import Path


def curve(points):
    points = sorted(((float(c), float(r)) for c, r in points), key=lambda p: p[0])
    if len(points) < 2 or any(
        not math.isfinite(c) or not math.isfinite(r) or c < 0 or not 0 <= r <= 1
        for c, r in points
    ):
        raise ValueError("A curve needs at least two finite (cost >= 0, rate in [0,1]) points")
    return points


def integrate(points, low, high, *, inverse=False):
    """Integrate clipped original segments, preserving their direction.

    Inverse integration computes cost over solve rate. A solve-rate reversal
    contributes signed area; a repeated rate contributes zero. Sorting by solve
    rate or discarding repeated rates would change the recorded polyline.
    """
    points = curve(points)
    if inverse:
        points = [(r, c) for c, r in points]
    if not min(p[0] for p in points) <= low < high <= max(p[0] for p in points):
        raise ValueError("Integration bounds must lie within the measured curve")
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 == x1:
            continue
        left, right = max(low, min(x0, x1)), min(high, max(x0, x1))
        if left >= right:
            continue
        yl = y0 + (y1 - y0) * (left - x0) / (x1 - x0)
        yr = y0 + (y1 - y0) * (right - x0) / (x1 - x0)
        area += (right - left) * (yl + yr) / 2 * (1 if x1 > x0 else -1)
    return area


def routing_metrics(fixed, adaptive):
    """Connect adaptive settings to fixed k=1 and k=64, never to the origin.

    The fixed curve must contain the chosen budgets from 1 through 64. Its
    lowest/highest cost points supply the shared evaluation anchors; they are
    not estimates of an adaptive policy with lambda=infinity.
    """
    fixed = curve(fixed)
    adaptive = curve(adaptive)
    adaptive = curve(adaptive + [p for p in (fixed[0], fixed[-1]) if p not in adaptive])
    low, high = fixed[0][0], fixed[-1][0]
    rlow = max(min(r for _, r in fixed), min(r for _, r in adaptive))
    rhigh = min(max(r for _, r in fixed), max(r for _, r in adaptive))
    cost_ref = integrate(fixed, rlow, rhigh, inverse=True)
    solve_ref = integrate(fixed, low, high)
    if cost_ref <= 0 or solve_ref <= 0:
        raise ValueError("Reference AUC must be positive")
    return {
        "cost_auc_reduction_pct": 100 * (1 - integrate(adaptive, rlow, rhigh, inverse=True) / cost_ref),
        "solve_auc_gain_pct": 100 * (integrate(adaptive, low, high) / solve_ref - 1),
        "solve_range": [rlow, rhigh],
        "cost_range_m": [low, high],
    }


def data_plane_metrics(fixed, whole_proof):
    """Compare all curves over the data plane's fixed-1 to fixed-64 cost range."""
    fixed = curve(fixed)
    low, high = fixed[0][0], fixed[-1][0]
    areas = {name: integrate(points, low, high) for name, points in whole_proof.items()}
    areas["fixed"] = integrate(fixed, low, high)
    if any(a <= 0 for a in areas.values()):
        raise ValueError("Reference AUC must be positive")
    return {
        "gain_over_8b_pct": 100 * (areas["fixed"] / areas["8b"] - 1),
        "deficit_to_32b_pct": 100 * (1 - areas["fixed"] / areas["32b"]),
        "cost_range_m": [low, high],
        "solve_auc": areas,
    }


def sweep_curve(directory, *, sigma=None, fixed_routing=False):
    """Read a single sweep curve without rounded avg_total_cost summaries.

    Select a sigma when reading a noisy-oracle sweep. Seeds and problem counts
    must match across settings. Callers must use the same problem split for
    sweeps they compare.
    """
    directory = Path(directory)
    files = sorted(directory.glob("config_*/summary.json"))
    if not files:
        raise ValueError(f"No config summaries in {directory}")
    points, signatures = [], set()
    sigmas, budgets, breakdown_limits = set(), set(), set()
    for file in files:
        summary = json.loads(file.read_text())
        params = summary["params"]
        sigmas.add(params.get("sigma"))
        if sigma is not None and float(params.get("sigma", 0)) != sigma:
            continue
        if fixed_routing:
            active = [v for v in params["breakdown_proof_budget"].values() if v > 0]
            if len(active) != 1 or any(params.get("full_proof_budget", {}).values()):
                raise ValueError("Use a fixed data-plane sweep with one active prover")
            budgets.add(active[0])
            breakdown_limits.add(params["max_breakdowns"])
        rows = summary["per_seed"]
        if not rows or any(s["total_problems"] <= 0 for s in rows):
            raise ValueError(f"Empty seed/problem data in {file}")
        signature = tuple(sorted((s["seed"], s["total_problems"]) for s in rows))
        if len({s["seed"] for s in rows}) != len(rows):
            raise ValueError(f"Repeated seed IDs in {file}")
        signatures.add(signature)
        cost = sum(s["total_cost"]["output_sflops"] / s["total_problems"] for s in rows) / len(rows)
        rate = sum(s["solved"] / s["total_problems"] for s in rows) / len(rows)
        points.append([cost / 1e6, rate])
    if sigma is None and len(sigmas) > 1:
        raise ValueError("Select --sigma; this sweep contains several noisy-oracle curves")
    if len(signatures) != 1:
        raise ValueError("Sweep settings must share seed IDs and problem counts")
    if fixed_routing and (min(budgets) != 1 or max(budgets) != 64 or len(breakdown_limits) != 1):
        raise ValueError("Fixed routing sweep needs k=1 and k=64 and one breakdown limit")
    return curve(points), signatures.pop()
