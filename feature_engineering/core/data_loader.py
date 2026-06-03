"""Load run data into a standardized structure for feature extraction.

Output structure:
    dict[str, list[AttemptChain]]

Where each AttemptChain is:
    {
        "origin_problem_id": str,
        "attempt_id": str,
        "rounds": [
            {
                "correction_round": 0,  # 0 = base, 1 = corr1, ...
                "full_code": str,
                "output": str,
                "detailed_cost": {"cost": ..., "input_tokens": ..., "output_tokens": ..., "input_sflops": ..., "output_sflops": ...},
                "compilation_result": {"pass": bool, "complete": bool, "errors": [...], "warnings": [...]},
            },
            ...
        ],
        success: bool,  # True if any round in the chain has pass=True and complete=True
    }
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from .id_utils import extract_attempt_base, extract_origin_problem


def _iter_chunk_results(run_dir: Path) -> Iterator[Path]:
    """Yield chunk_*/results/ directories."""
    for chunk_dir in sorted(run_dir.iterdir()):
        if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk_"):
            results_dir = chunk_dir / "results"
            if results_dir.exists():
                yield results_dir


def _load_json_across_chunks(run_dir: Path, filename: str) -> list[dict]:
    """Load and concatenate JSON arrays from filename across all chunks."""
    records = []
    for results_dir in _iter_chunk_results(run_dir):
        path = results_dir / filename
        if path.exists():
            with open(path) as f:
                records.extend(json.load(f))
    return records


def load_run_data(
    run_dir: Path,
    correction_rounds: int = 0,
) -> dict[str, list[dict]]:
    """Load run data into standardized AttemptChain structure.

    Args:
        run_dir: Path to the run directory containing chunk_* subdirectories.
        correction_rounds: Number of correction rounds to include (0 = base only).

    Returns:
        dict mapping origin_problem_id -> list of AttemptChain dicts.
    """
    inference_files = ["to_inference_codes.json"]
    compilation_files = ["code_compilation_repl.json"]
    for cr in range(1, correction_rounds + 1):
        inference_files.append(f"to_inference_codes_corr{cr}.json")
        compilation_files.append(f"code_compilation_repl_corr{cr}.json")

    # Load all inference records
    inference_records = []
    for fname in inference_files:
        inference_records.extend(_load_json_across_chunks(run_dir, fname))

    # Load all compilation records, keyed by problem_id
    comp_by_id: dict[str, dict] = {}
    for fname in compilation_files:
        for r in _load_json_across_chunks(run_dir, fname):
            pid = r.get("problem_id") or r.get("name", "")
            if pid:
                comp_by_id[pid] = r.get("compilation_result", {})

    # Group inference records by (origin_problem, base_attempt_id)
    attempts: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for record in inference_records:
        origin = record.get("origin_problem_id", "")
        if not origin:
            continue

        attempt_id = record.get("problem_id", "")
        base_id = extract_attempt_base(attempt_id)

        # Determine correction round from filename convention
        # Base records have no _corr suffix, corr records have _corrN_pM in problem_id
        corr_round = 0
        stripped = attempt_id[len(base_id):]
        if stripped:
            import re
            m = re.match(r"_corr(\d+)_", stripped)
            if m:
                corr_round = int(m.group(1))

        round_data = {
            "correction_round": corr_round,
            "full_code": record.get("full_code", ""),
            "output": record.get("model_output", ""),
            "detailed_cost": record.get("detailed_cost", {}),
            "compilation_result": comp_by_id.get(attempt_id, {}),
        }

        attempts[origin][base_id].append(round_data)

    # Build final structure
    result: dict[str, list[dict]] = defaultdict(list)
    for origin, base_attempts in attempts.items():
        for base_id, rounds in base_attempts.items():
            rounds.sort(key=lambda r: r["correction_round"])
            # Success is determined by the last round in the chain
            last_comp = rounds[-1]["compilation_result"]
            success = bool(
                last_comp.get("pass", False) and last_comp.get("complete", False)
            )

            result[origin].append({
                "origin_problem_id": origin,
                "attempt_id": base_id,
                "rounds": rounds,
                "success": success,
            })

    return dict(result)


def filter_failed_only(data: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Keep only failed attempts within each problem.

    Removes problems that have no failed attempts.

    Returns:
        Filtered data dict with the same structure.
    """
    result = {}
    for pid, chains in data.items():
        failed = [c for c in chains if not c["success"]]
        if failed:
            result[pid] = failed
    return result


def filter_by_success_rate(
    data: dict[str, list[dict]],
    max_rate: float,
) -> dict[str, list[dict]]:
    """Filter to only problems with success rate <= max_rate.

    Args:
        data: Output of load_run_data().
        max_rate: Maximum success rate threshold (0-1).

    Returns:
        Filtered data dict with the same structure.
    """
    rates = compute_success_rates(data)
    return {pid: chains for pid, chains in data.items() if rates[pid] <= max_rate}


def compute_success_rates(data: dict[str, list[dict]]) -> dict[str, float]:
    """Compute per-problem success rate from loaded run data.

    An attempt succeeds if pass=True and complete=True in ANY round of its chain.

    Returns:
        {origin_problem_id: success_rate}
    """
    rates = {}
    for origin, chains in data.items():
        n_success = sum(1 for chain in chains if chain["success"])
        rates[origin] = n_success / len(chains) if chains else 0.0
    return rates
