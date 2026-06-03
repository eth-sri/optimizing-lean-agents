"""Predicted probability computed feature using pretrained logistic model.

Reads feature values from the tracked_state dict (phase 1) and applies
a pretrained logistic model to produce predicted probabilities per model.
No internal cost tracking — relies on AverageCostFeature being in the
feature list.
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..problem import SimulatedProblem
from ..state import SimulationState
from .base import ComputedFeature


def _reconstruct_linear(model_dict: dict):
    """Reconstruct a linear regression model (and optional scaler) from saved coefficients."""
    from sklearn.preprocessing import StandardScaler

    coef = np.array(model_dict["coefficients"]).flatten()
    intercept = float(np.array(model_dict["intercept"]).flatten()[0])

    scaler = None
    if "scaler" in model_dict:
        scaler = StandardScaler()
        scaler.mean_ = np.array(model_dict["scaler"]["mean"])
        scaler.scale_ = np.array(model_dict["scaler"]["scale"])
        scaler.n_features_in_ = len(scaler.mean_)

    return (coef, intercept), scaler


class _GradientBoostingModel:
    """Tiny inference-only gradient boosting model.

    Stores the per-tree arrays (feature, threshold, children_left,
    children_right, value) and evaluates them with a vectorized walk.
    Prediction:
        raw = init + lr * sum(tree(x) for tree in trees)
        y   = clip(raw, clamp[0], clamp[1])
    """

    __slots__ = ("learning_rate", "init", "trees", "clamp", "n_features")

    def __init__(self, model_dict: dict):
        self.learning_rate = float(model_dict["learning_rate"])
        self.init = float(model_dict["init"])
        self.clamp = tuple(model_dict.get("clamp", [0.0, 1.0]))
        self.n_features = len(model_dict.get("features", []))
        # Pre-convert each tree's arrays to ndarrays once.
        self.trees = []
        for t in model_dict["trees"]:
            self.trees.append({
                "feature": np.asarray(t["feature"], dtype=np.int64),
                "threshold": np.asarray(t["threshold"], dtype=np.float64),
                "children_left": np.asarray(t["children_left"], dtype=np.int64),
                "children_right": np.asarray(t["children_right"], dtype=np.int64),
                "value": np.asarray(t["value"], dtype=np.float64),
            })

    @staticmethod
    def _walk(tree: dict, x: np.ndarray) -> float:
        feature = tree["feature"]
        threshold = tree["threshold"]
        left = tree["children_left"]
        right = tree["children_right"]
        value = tree["value"]
        node = 0
        # Leaves have children_left == -1 (TREE_LEAF in sklearn).
        while left[node] != -1:
            if x[feature[node]] <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        return float(value[node])

    def predict_one(self, x: list[float]) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        raw = self.init
        for tree in self.trees:
            raw += self.learning_rate * self._walk(tree, x_arr)
        lo, hi = self.clamp
        return float(min(max(raw, lo), hi))


def _reconstruct_gradient_boosting(model_dict: dict):
    """Reconstruct a gradient boosting model (and optional scaler) from JSON."""
    from sklearn.preprocessing import StandardScaler

    gb = _GradientBoostingModel(model_dict)

    scaler = None
    if "scaler" in model_dict:
        scaler = StandardScaler()
        scaler.mean_ = np.array(model_dict["scaler"]["mean"])
        scaler.scale_ = np.array(model_dict["scaler"]["scale"])
        scaler.n_features_in_ = len(scaler.mean_)

    return gb, scaler


def _reconstruct_logistic(model_dict: dict):
    """Reconstruct a fitted LogisticRegression (and optional scaler) from saved coefficients."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    lr = LogisticRegression()
    lr.coef_ = np.array(model_dict["coefficients"])
    lr.intercept_ = np.array(model_dict["intercept"])
    lr.classes_ = np.array(model_dict["classes"])

    scaler = None
    if "scaler" in model_dict:
        scaler = StandardScaler()
        scaler.mean_ = np.array(model_dict["scaler"]["mean"])
        scaler.scale_ = np.array(model_dict["scaler"]["scale"])
        scaler.n_features_in_ = len(scaler.mean_)

    return lr, scaler


class PredictedProbComputed(ComputedFeature):
    """Predicted success probability from a cost-based logistic model.

    Loads a pretrained logistic model and predicts P(success) per model
    using feature values from the tracked_state dict. The feature_mapping
    in the model JSON maps model feature names to tracked_state keys.
    """

    def __init__(self, model_path: str):
        self._model_path = model_path

        with open(model_path) as f:
            self._saved = json.load(f)

        self._features: List[str] = self._saved["features"]
        self._default_costs: Dict[str, Dict[str, float]] = self._saved.get("default_costs", {})
        self._feature_mapping: Dict[str, str] = self._saved.get("feature_mapping", {})

        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}
        self._model_types: Dict[str, str] = {}
        for model_name, model_dict in self._saved.get("models", {}).items():
            mtype = model_dict["type"]
            self._model_types[model_name] = mtype
            if mtype == "constant":
                self._models[model_name] = float(model_dict["value"])
            elif mtype == "logistic_regression":
                model, scaler = _reconstruct_logistic(model_dict)
                self._models[model_name] = model
                if scaler is not None:
                    self._scalers[model_name] = scaler
            elif mtype == "linear_regression":
                model, scaler = _reconstruct_linear(model_dict)
                self._models[model_name] = model
                if scaler is not None:
                    self._scalers[model_name] = scaler
            elif mtype == "gradient_boosting":
                model, scaler = _reconstruct_gradient_boosting(model_dict)
                self._models[model_name] = model
                if scaler is not None:
                    self._scalers[model_name] = scaler

    def name(self) -> str:
        return "predicted_prob"

    def compute(self, tracked_state: dict, state: SimulationState, problem: SimulatedProblem) -> Dict[str, float]:
        result = {}
        for model_name, model in self._models.items():
            fv = self._get_feature_vector(tracked_state, model_name)
            if fv is None:
                continue
            if isinstance(model, float):
                result[model_name] = model
            elif self._model_types.get(model_name) == "linear_regression":
                coef, intercept = model
                if model_name in self._scalers:
                    fv = self._scalers[model_name].transform([fv])[0].tolist()
                result[model_name] = float(np.clip(np.dot(coef, fv) + intercept, 0.0, 1.0))
            elif self._model_types.get(model_name) == "gradient_boosting":
                if model_name in self._scalers:
                    fv = self._scalers[model_name].transform([fv])[0].tolist()
                result[model_name] = model.predict_one(fv)
            else:
                if model_name in self._scalers:
                    fv = self._scalers[model_name].transform([fv])[0].tolist()
                result[model_name] = float(model.predict_proba([fv])[0, 1])
        return result

    @staticmethod
    def _parse_feature_transform(feat: str):
        """Parse 'transform:name' -> (name, transform) or (feat, None)."""
        if ":" in feat:
            transform, name = feat.split(":", 1)
            return name, transform
        return feat, None

    @staticmethod
    def _apply_transform(val: float, transform) -> float:
        if transform is None:
            return val
        if transform == "1/x":
            return 1.0 / val if val != 0 else 0.0
        if transform == "log":
            return float(np.log(val + 1e-10))
        if transform == "logit":
            val = max(1e-6, min(val, 1 - 1e-6))
            return float(np.log(val / (1 - val)))
        raise ValueError(f"Unknown feature transform: {transform}")

    def _get_feature_vector(self, tracked_state: dict, model_name: str) -> Optional[List[float]]:
        """Get feature vector from tracked_state, falling back to default costs."""
        values = []
        for feat in self._features:
            base_name, transform = self._parse_feature_transform(feat)
            tracker_key = self._feature_mapping.get(feat, self._feature_mapping.get(base_name, base_name))
            feat_data = tracked_state.get(tracker_key, {})
            if isinstance(feat_data, dict):
                val = feat_data.get(model_name)
            else:
                val = feat_data
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                values.append(float(val))
            else:
                # Cold start: use default costs
                defaults = self._default_costs.get(model_name)
                if defaults is None:
                    return None
                default_val = defaults.get(feat)
                if default_val is None:
                    return None
                values.append(float(default_val))
        return values
