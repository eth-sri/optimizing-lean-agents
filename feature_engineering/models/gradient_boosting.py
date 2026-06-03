"""Gradient boosting regressor for predicting per-problem success rate.

Wraps sklearn's GradientBoostingRegressor and provides a JSON serialization of
the fitted tree ensemble that the proof_simulation consumer can load and
evaluate without sklearn (predictions clamped to [0, 1]).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from .base import BaseModel


def _tree_to_dict(tree) -> dict:
    """Serialize a sklearn DecisionTreeRegressor's internal tree to plain lists.

    Leaves are nodes whose children indices are -1 (TREE_LEAF). Regression
    node values have shape (node_count, 1, 1); we flatten to a list of floats.
    """
    t = tree.tree_
    return {
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "value": t.value.reshape(-1).tolist(),
    }


def gb_to_json_dict(model: GradientBoostingRegressor, feature_cols: list[str]) -> dict:
    """Convert a fitted GradientBoostingRegressor into a framework-free dict.

    Consumer-side prediction is:
        raw = init + learning_rate * sum(tree.predict(x) for tree in trees)
        y   = clip(raw, clamp[0], clamp[1])
    """
    trees = [_tree_to_dict(est) for est in model.estimators_.ravel()]
    init_const = float(np.array(model.init_.constant_).ravel()[0])
    return {
        "type": "gradient_boosting",
        "learning_rate": float(model.learning_rate),
        "init": init_const,
        "features": list(feature_cols),
        "trees": trees,
        "clamp": [0.0, 1.0],
    }


class GradientBoosting(BaseModel):
    """Gradient boosting regression with train/test split."""

    name = "gradient_boosting"

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 3,
        subsample: float = 0.8,
        test_size: float = 0.5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, df: pd.DataFrame, feature_cols: list[str], target: str = "success_rate") -> dict[str, Any]:
        df_clean = df[feature_cols + [target]].dropna()
        X = df_clean[feature_cols].values
        y = df_clean[target].values
        problem_ids = np.array(df_clean.index)

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, problem_ids, test_size=self.test_size, random_state=self.random_state,
        )

        model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)

        # Clamped predictions match inference behavior
        y_pred_train = np.clip(model.predict(X_train), 0.0, 1.0)
        ss_res_train = np.sum((y_train - y_pred_train) ** 2)
        ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
        r2_train = 1 - ss_res_train / ss_tot_train if ss_tot_train > 0 else 0.0

        y_pred_test = np.clip(model.predict(X_test), 0.0, 1.0)
        ss_res = np.sum((y_test - y_pred_test) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mae = float(np.mean(np.abs(y_pred_test - y_test)))
        rmse = float(np.sqrt(np.mean((y_pred_test - y_test) ** 2)))

        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))

        return {
            "model": model,
            "r2_train": float(r2_train),
            "r2": float(r2),
            "mae": mae,
            "rmse": rmse,
            "y_pred": y_pred_test,
            "problem_ids": list(ids_test),
            "n_samples": int(len(df_clean)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "feature_importance": importance,
        }
