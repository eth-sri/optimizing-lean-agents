"""Load minified full proof JSON data."""

import json
from collections import defaultdict
from typing import Dict, List

from .types import AttemptData, AttemptPair
from ..actions import DetailedCost


def load_full_proof_data(path: str, model_name: str) -> Dict[str, List[AttemptPair]]:
    """Load minified JSON -> {problem_id: [AttemptPair]}.

    Groups by proof_id to pair initials with corrections (sorted by correction_round_id).
    Currently full proof data has no corrections (all correction_round_id=0),
    so each AttemptPair has an empty corrections list.

    Args:
        path: Path to minified JSON file (e.g., minified_8b.json)
        model_name: Model name for identification (e.g., "8b")

    Returns:
        Dict mapping problem_id to list of AttemptPairs
    """
    with open(path) as f:
        records = json.load(f)

    # Group by (problem_id, proof_id)
    by_problem_proof = defaultdict(list)
    for record in records:
        pid = record["origin_problem_id"]
        proof_id = record.get("proof_id", record.get("attempt_id"))
        by_problem_proof[(pid, proof_id)].append(record)

    # Build AttemptPairs per problem
    result: Dict[str, List[AttemptPair]] = defaultdict(list)

    for (pid, proof_id), attempts in by_problem_proof.items():
        # Sort by correction_round_id
        attempts.sort(key=lambda r: r.get("correction_round_id", 0))

        initial_record = None
        correction_records = []

        for record in attempts:
            crid = record.get("correction_round_id", 0)
            if crid == 0:
                if initial_record is not None:
                    # Multiple initials for same proof_id - treat each as separate pair
                    pair = AttemptPair(
                        initial=_record_to_attempt_data(initial_record),
                        corrections=[_record_to_attempt_data(c) for c in correction_records],
                    )
                    result[pid].append(pair)
                    correction_records = []
                initial_record = record
            else:
                correction_records.append(record)

        # Add the last pair
        if initial_record is not None:
            pair = AttemptPair(
                initial=_record_to_attempt_data(initial_record),
                corrections=[_record_to_attempt_data(c) for c in correction_records],
            )
            result[pid].append(pair)

    return dict(result)


def _record_to_attempt_data(record: dict) -> AttemptData:
    """Convert a minified JSON record to AttemptData."""
    dc = record.get("detailed_cost", {})
    return AttemptData(
        success=record.get("pass", False) and record.get("complete", False),
        cost=DetailedCost(
            input_sflops=dc.get("input_sflops", 0),
            output_sflops=dc.get("output_sflops", 0),
            input_tokens=dc.get("input_tokens", 0),
            output_tokens=dc.get("output_tokens", 0),
        ),
        proof_length=record.get("proof_length"),
        num_errors=record.get("num_errors"),
        correction_round_id=record.get("correction_round_id", 0),
        attempt_id=record.get("proof_id", record.get("attempt_id")),
        code=record.get("full_code"),
    )
