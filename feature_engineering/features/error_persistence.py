"""Error persistence feature: measures the fraction of errors that survive correction rounds.

For each attempt chain with multiple rounds, compares normalized errors from
the first round to the last round. Errors present in both are "persistent".

High persistence = corrections are failing to resolve errors.
Low persistence = corrections are effective at fixing errors.

Requires correction_rounds > 0 to be meaningful; returns NaN for single-round chains.
"""

import pandas as pd

from .base import BaseFeature
from .error_diversity import _extract_error_messages


def _get_error_set(round_data: dict) -> set[str]:
    """Get set of error messages from a round."""
    comp = round_data.get("compilation_result", {})
    return set(_extract_error_messages(comp))


class ErrorPersistence(BaseFeature):
    """Average fraction of initial errors that persist after correction rounds."""

    name = "error_persistence"

    def compute(self, data: dict) -> pd.DataFrame:
        rows = []
        for pid in sorted(data.keys()):
            persistence_ratios = []
            for chain in data[pid]:
                rounds = chain["rounds"]
                if len(rounds) < 2:
                    continue

                initial_errors = _get_error_set(rounds[0])
                if not initial_errors:
                    continue

                final_errors = _get_error_set(rounds[-1])
                persistent = initial_errors & final_errors
                persistence_ratios.append(len(persistent) / len(initial_errors))

            if persistence_ratios:
                avg_persistence = sum(persistence_ratios) / len(persistence_ratios)
            else:
                avg_persistence = float("nan")

            rows.append({"problem_id": pid, self.name: avg_persistence})

        return pd.DataFrame(rows).set_index("problem_id")
