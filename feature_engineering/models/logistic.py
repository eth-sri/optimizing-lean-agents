import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base import BaseModel


class LogisticRegression(BaseModel):
    """Logistic regression with train/test split.

    Supports two modes:
    - soft_labels=False (default): Converts success_rate to binary (> threshold),
      then fits standard logistic regression.
    - soft_labels=True: Uses fractional labels directly via sample weight duplication.
      Each sample is duplicated: once as positive (weight=y) and once as negative (weight=1-y).
    """

    name = "logistic_regression"

    def __init__(self, threshold: float = 0.5, soft_labels: bool = True,
                 test_size: float = 0.5, random_state: int = 42):
        self.threshold = threshold
        self.soft_labels = soft_labels
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, df: pd.DataFrame, feature_cols: list[str], target: str = "success_rate") -> dict:
        df_clean = df[feature_cols + [target]].dropna()
        X = df_clean[feature_cols].values
        y_actual = df_clean[target].values
        problem_ids = np.array(df_clean.index)

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y_actual, problem_ids, test_size=self.test_size, random_state=self.random_state,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SkLogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000)

        if self.soft_labels:
            X_expanded = np.vstack([X_train_scaled, X_train_scaled])
            y_expanded = np.concatenate([np.ones(len(y_train)), np.zeros(len(y_train))])
            weights = np.concatenate([y_train, 1 - y_train])
            model.fit(X_expanded, y_expanded, sample_weight=weights)
        else:
            y_binary_train = (y_train > self.threshold).astype(int)
            model.fit(X_train_scaled, y_binary_train)

        # Train metrics
        y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
        y_pred_train = model.predict(X_train_scaled)
        y_binary_train = (y_train > self.threshold).astype(int)
        acc_train = np.mean(y_pred_train == y_binary_train)

        # Test metrics
        y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
        y_pred_test = model.predict(X_test_scaled)
        y_binary_test = (y_test > self.threshold).astype(int)
        accuracy = np.mean(y_pred_test == y_binary_test)

        mae = np.mean(np.abs(y_prob_test - y_test))
        rmse = np.sqrt(np.mean((y_prob_test - y_test) ** 2))

        return {
            "model": model,
            "scaler": scaler,
            "acc_train": acc_train,
            "accuracy": accuracy,
            "mae": mae,
            "rmse": rmse,
            "y_prob": y_prob_test,
            "problem_ids": list(ids_test),
            "coefficients": dict(zip(feature_cols, model.coef_[0])),
            "intercept": model.intercept_[0],
            "n_samples": len(df_clean),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_positive": int(y_binary_test.sum()),
            "n_negative": int(len(y_binary_test) - y_binary_test.sum()),
        }
