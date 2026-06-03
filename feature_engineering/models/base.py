from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """Base class for fitting models on feature DataFrames."""

    name: str

    @abstractmethod
    def fit(self, df: pd.DataFrame, feature_cols: list[str], target: str = "success_rate") -> dict:
        """Fit model on feature DataFrame.

        Args:
            df: DataFrame with feature columns + target column.
            feature_cols: Which columns to use as features.
            target: Column name to predict.

        Returns:
            dict with metrics (r2, coefficients, etc.)
        """
        ...
