import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base import BaseModel


class LinearRegression(BaseModel):
    """Ridge regression with train/test split."""

    name = "linear_regression"

    def __init__(self, test_size: float = 0.5, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, df: pd.DataFrame, feature_cols: list[str], target: str = "success_rate") -> dict:
        df_clean = df[feature_cols + [target]].dropna()
        X = df_clean[feature_cols].values
        y = df_clean[target].values
        problem_ids = np.array(df_clean.index)

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, problem_ids, test_size=self.test_size, random_state=self.random_state,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        # Train metrics
        y_pred_train = model.predict(X_train_scaled)
        ss_res_train = np.sum((y_train - y_pred_train) ** 2)
        ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
        r2_train = 1 - ss_res_train / ss_tot_train if ss_tot_train > 0 else 0.0

        # Test metrics
        y_pred_test = model.predict(X_test_scaled)
        ss_res = np.sum((y_test - y_pred_test) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mae = np.mean(np.abs(y_pred_test - y_test))
        rmse = np.sqrt(np.mean((y_pred_test - y_test) ** 2))

        return {
            "model": model,
            "scaler": scaler,
            "r2_train": r2_train,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "y_pred": y_pred_test,
            "problem_ids": list(ids_test),
            "coefficients": dict(zip(feature_cols, model.coef_)),
            "intercept": model.intercept_,
            "n_samples": len(df_clean),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
