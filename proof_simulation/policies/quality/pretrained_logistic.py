"""Pretrained logistic regression probability model.

Loads a previously trained model from a JSON file (produced by
TrajectoryLogisticRegressionModel.save_model_params) and uses it for
prediction. Supports feature name mapping so the state tracker can use
different feature names than the ones the model was trained on.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import ProbabilityModel
from .trajectory_logistic import TRANSFORMS, _parse_feature_spec
from ...actions import Action, ActionType
from ...state import SimulationState
from ...problem import SimulatedProblem
from ...features.predicted_prob import (
    _GradientBoostingModel,
    _reconstruct_gradient_boosting,
)


def _reconstruct_logistic(model_dict: dict) -> Tuple[LogisticRegression, Optional[StandardScaler]]:
    """Reconstruct a fitted LogisticRegression (and optional scaler) from saved coefficients."""
    lr = LogisticRegression()
    lr.coef_ = np.array(model_dict["coefficients"])
    lr.intercept_ = np.array(model_dict["intercept"])
    lr.classes_ = np.array(model_dict.get("classes", [0, 1]))

    scaler = None
    if "scaler" in model_dict:
        scaler = StandardScaler()
        scaler.mean_ = np.array(model_dict["scaler"]["mean"])
        scaler.scale_ = np.array(model_dict["scaler"]["scale"])
        scaler.n_features_in_ = len(scaler.mean_)

    return lr, scaler


class _LinearRegressionModel:
    """Simple linear regression model reconstructed from saved coefficients."""

    def __init__(self, coefficients, intercept):
        self.coef_ = np.array(coefficients).flatten()
        self.intercept_ = float(np.array(intercept).flatten()[0])

    def predict(self, X):
        return np.clip(X @ self.coef_ + self.intercept_, 0.0, 1.0)


def _reconstruct_linear(model_dict: dict) -> Tuple[_LinearRegressionModel, Optional[StandardScaler]]:
    """Reconstruct a fitted linear regression (and optional scaler) from saved coefficients."""
    lr = _LinearRegressionModel(model_dict["coefficients"], model_dict["intercept"])

    scaler = None
    if "scaler" in model_dict:
        scaler = StandardScaler()
        scaler.mean_ = np.array(model_dict["scaler"]["mean"])
        scaler.scale_ = np.array(model_dict["scaler"]["scale"])
        scaler.n_features_in_ = len(scaler.mean_)

    return lr, scaler


class PretrainedLogisticModel(ProbabilityModel):
    """Logistic regression loaded from a pretrained model JSON file.

    Args:
        model_path: Path to the saved model_params.json.
        feature_mapping: Optional dict mapping model feature names to state
            tracker feature names (e.g. {"noisy_p": "predicted_p"}).
            If None, features are read by their original names.
        sigma: Which (sigma, model_name) sub-model to use. Mutable for sweeps.
    """

    def __init__(
        self,
        model_path: str,
        feature_mapping: Optional[Dict[str, str]] = None,
        sigma: float = 0.1,
    ):
        self.sigma = sigma
        self._model_path = model_path

        # Load saved params
        with open(model_path) as f:
            self._saved_params = json.load(f)

        # Parse feature specs from saved model
        raw_specs = self._saved_params["features"]
        self._feature_specs: List[Tuple[str, Optional[str]]] = [
            _parse_feature_spec(s) for s in raw_specs
        ]
        self._feature_names = [name for name, _ in self._feature_specs]
        self._feature_raw_specs = raw_specs

        # Feature mapping: explicit arg > JSON > empty
        self._feature_mapping: Dict[str, str] = feature_mapping or self._saved_params.get("feature_mapping", {})

        # Reconstruct models: store (model_or_float, scaler_or_None)
        self._models: Dict[Tuple[float, str], Union[Tuple[LogisticRegression, Optional[StandardScaler]], float]] = {}
        for key_str, model_dict in self._saved_params.get("models", {}).items():
            sigma_val, model_name = self._parse_model_key(key_str)
            if model_dict["type"] == "constant":
                self._models[(sigma_val, model_name)] = float(model_dict["value"])
            elif model_dict["type"] == "logistic_regression":
                self._models[(sigma_val, model_name)] = _reconstruct_logistic(model_dict)
            elif model_dict["type"] == "linear_regression":
                self._models[(sigma_val, model_name)] = _reconstruct_linear(model_dict)
            elif model_dict["type"] == "gradient_boosting":
                self._models[(sigma_val, model_name)] = _reconstruct_gradient_boosting(model_dict)

    @staticmethod
    def _parse_model_key(key_str: str) -> Tuple[float, str]:
        """Parse 'sigma=0.1_model=8b' or plain '8b' into (sigma, model_name)."""
        if "_model=" in key_str:
            parts = key_str.split("_model=")
            sigma_val = float(parts[0].replace("sigma=", ""))
            model_name = parts[1]
            return sigma_val, model_name
        # Plain key like "8b" — use sigma=0.0 as sentinel
        return 0.0, key_str

    def _resolve_feature_name(self, model_feature_name: str) -> str:
        """Map a model feature name to the state tracker feature name."""
        return self._feature_mapping.get(model_feature_name, model_feature_name)

    def predict(
        self,
        state: SimulationState,
        action: Action,
        problem: SimulatedProblem,
        tracked_state: dict = None,
    ) -> float:
        if action.type != ActionType.PROVE or action.model is None:
            return 0.0

        model_name = action.model
        tracked = tracked_state or {}

        raw_features = {}
        for feat_name in self._feature_names:
            tracker_name = self._resolve_feature_name(feat_name)
            feat_data = tracked.get(tracker_name, {})
            if isinstance(feat_data, dict):
                val = feat_data.get(model_name)
            else:
                val = feat_data
            if val is None or (isinstance(val, float) and val != val):
                val = 0.0
            raw_features[feat_name] = float(val)

        # Build transformed feature vector
        fv = self._build_feature_vector(raw_features)

        # Look up fitted model with sigma fallback
        key = (self.sigma, model_name)
        entry = self._models.get(key)
        if entry is None:
            # Fallback to sigma=0.0 (plain key format)
            entry = self._models.get((0.0, model_name))
        if entry is None:
            return 0.0

        if isinstance(entry, float):
            return entry

        model, scaler = entry
        if scaler is not None:
            fv = scaler.transform([fv])[0].tolist()

        if isinstance(model, _LinearRegressionModel):
            return float(model.predict([fv])[0])
        if isinstance(model, _GradientBoostingModel):
            return model.predict_one(fv)
        return float(model.predict_proba([fv])[0, 1])

    def _build_feature_vector(self, raw_features: Dict[str, float]) -> List[float]:
        """Apply transforms and build the feature vector."""
        fv = []
        for feat_name, transform_key in self._feature_specs:
            val = raw_features[feat_name]
            transform_fn = TRANSFORMS[transform_key]
            fv.append(float(transform_fn(np.array(val))))
        return fv

    def get_model_params(self) -> Dict[str, Any]:
        """Return the loaded model parameters."""
        return self._saved_params

    def save_model_params(self, path: str):
        """Save model parameters to a JSON file."""
        with open(path, "w") as f:
            json.dump(self._saved_params, f, indent=2)
