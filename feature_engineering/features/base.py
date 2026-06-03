from abc import ABC, abstractmethod

import pandas as pd


class BaseFeature(ABC):
    """Base class for all features."""

    name: str

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def compute(self, data: dict) -> pd.DataFrame:
        """Compute feature values from loaded data.

        Args:
            data: standardized data dict from data_loader.load_run_data()

        Returns:
            DataFrame with problem_id as index and feature column(s).
        """
        ...
