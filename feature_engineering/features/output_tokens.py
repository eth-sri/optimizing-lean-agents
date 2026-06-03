import numpy as np
import pandas as pd

from .base import BaseFeature


class OutputSflops(BaseFeature):
    """Average output sflops per attempt for each problem."""

    name = "output_sflops"

    def compute(self, data: dict) -> pd.DataFrame:
        rows = []
        for pid in sorted(data.keys()):
            chains = data[pid]
            attempt_totals = [
                sum(r["detailed_cost"].get("output_sflops", 0) for r in chain["rounds"])
                for chain in chains
            ]
            mean = np.mean(attempt_totals) if attempt_totals else 0.0
            std = np.std(attempt_totals) if len(attempt_totals) > 1 else 0.0
            cv = std / mean if mean > 0 else 0.0
            rows.append({
                "problem_id": pid,
                self.name: mean,
                f"{self.name}_std": std,
                f"{self.name}_cv": cv,
            })

        return pd.DataFrame(rows).set_index("problem_id")
