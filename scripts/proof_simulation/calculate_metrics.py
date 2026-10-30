#!/usr/bin/env python3
"""Compare sweep summaries using the paper's AUC rule."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from proof_simulation.evaluation import routing_metrics, sweep_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-sweep", type=Path, required=True, help="Fixed routing sweep with k=1..64 endpoints")
    parser.add_argument("--method-sweep", type=Path, required=True, help="Adaptive sweep for the same split and seeds")
    parser.add_argument("--sigma", type=float, help="Select one noisy-oracle curve")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fixed, fixed_seeds = sweep_curve(args.fixed_sweep, fixed_routing=True)
    method, method_seeds = sweep_curve(args.method_sweep, sigma=args.sigma)
    if fixed_seeds != method_seeds:
        parser.error("Compared sweeps must share seed IDs and problem counts")
    result = routing_metrics(fixed, method)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
