"""Error diversity feature: measures how varied compilation errors are across attempts.

Uses exact string matching on error messages (no normalization) to compute
the ratio of unique errors to total errors.

Low diversity (close to 0) = model keeps hitting the same error.
High diversity (close to 1) = model fails in many different ways.
"""

import pandas as pd

from .base import BaseFeature


def _extract_error_messages(compilation_result: dict) -> list[str]:
    """Extract error message strings from a compilation result."""
    errors = compilation_result.get("errors", [])
    messages = []
    for err in errors:
        if isinstance(err, dict):
            data = err.get("data", "")
            if data:
                messages.append(data.strip())
        elif isinstance(err, str) and err.strip():
            messages.append(err.strip())
    return messages


class ErrorDiversity(BaseFeature):
    """Ratio of unique error messages to total errors per problem (exact match)."""

    name = "error_diversity"

    def compute(self, data: dict) -> pd.DataFrame:
        rows = []
        for pid in sorted(data.keys()):
            all_errors = []
            for chain in data[pid]:
                for round_data in chain["rounds"]:
                    comp = round_data.get("compilation_result", {})
                    all_errors.extend(_extract_error_messages(comp))

            if all_errors:
                diversity = len(set(all_errors)) / len(all_errors)
            else:
                diversity = float("nan")

            rows.append({"problem_id": pid, self.name: diversity})

        return pd.DataFrame(rows).set_index("problem_id")
