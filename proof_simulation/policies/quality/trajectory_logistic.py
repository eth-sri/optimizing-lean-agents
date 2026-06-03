"""Logistic regression probability model trained on trajectory data.

Learns P(success | noisy_p, num_attempts) from recorded simulation
trajectories, fitting one model per (sigma, model_name) pair.

At prediction time, reads features from the state tracker (which the user
configures with NoisyOracleFeature + AttemptCountFeature). No internal noise
sampling — the state tracker owns all feature computation.
"""

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .base import ProbabilityModel
from ...actions import Action, ActionType
from ...state import SimulationState
from ...problem import SimulatedProblem


TRANSFORMS: Dict[Optional[str], Callable[[np.ndarray], np.ndarray]] = {
    None: lambda x: x,
    "log": lambda x: np.log1p(x),
    "sqrt": lambda x: np.sqrt(x),
    "1/x": lambda x: np.where(x != 0, 1 / x, 0.0),
    "logit": lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6))),
}


def _parse_feature_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Parse a feature spec like 'log:num_attempts' into (name, transform_key).

    Returns (feature_name, transform_key) where transform_key is None for identity.
    """
    if ":" in spec:
        transform_key, feature_name = spec.split(":", 1)
        return feature_name, transform_key
    return spec, None


class TrajectoryLogisticRegressionModel(ProbabilityModel):
    """Logistic regression trained on per-step trajectory features.

    Scans a trajectory output directory (from a fixed_feature_tracker sweep),
    extracts (noisy_p, num_attempts) -> success labels for each prove action,
    and fits one logistic regression per (sigma, model_name) pair.

    At prediction time, reads feature values from the state tracker rather
    than computing them internally. The user must configure the state tracker
    with the matching features (e.g. NoisyOracleFeature, AttemptCountFeature).

    Args:
        problems: List of simulated problems (unused, kept for API compatibility).
        trajectory_base_dir: Path to sweep output (contains config_NNN/ dirs).
        trajectory_dir: Path to a single config dir (alternative to base_dir).
        features: Feature specs with optional transforms, e.g. ["noisy_p", "log:num_attempts"].
        sigma: Which sigma sub-model to use at prediction time. Mutable.
        C: Logistic regression regularization parameter.
    """

    def __init__(
        self,
        problems: Optional[List[SimulatedProblem]] = None,
        trajectory_base_dir: Optional[str] = None,
        trajectory_dir: Optional[str] = None,
        features: Optional[List[str]] = None,
        sigma: float = 0.1,
        C: float = 1.0,
        model_type: str = "logistic",  # "logistic" or "linear"
        alpha: float = 1.0,  # Ridge regularization for linear
    ):
        self.sigma = sigma
        self.C = C
        self.model_type = model_type
        self.alpha = alpha

        # Parse feature specs
        features = features or ["noisy_p", "num_attempts"]
        self._feature_specs: List[Tuple[str, Optional[str]]] = [
            _parse_feature_spec(f) for f in features
        ]
        self._feature_names = [name for name, _ in self._feature_specs]
        self._feature_raw_specs = features

        # Load training data and fit models
        if trajectory_base_dir is not None:
            traj_path = Path(trajectory_base_dir)
            raw = self._load_training_data(traj_path)
        elif trajectory_dir is not None:
            traj_path = Path(trajectory_dir)
            raw = self._load_training_data_single(traj_path)
        else:
            raise ValueError("Either trajectory_base_dir or trajectory_dir is required")

        if not raw:
            raise ValueError(
                f"No training data found in {traj_path}. "
                f"Ensure the directory contains trajectory JSON files with "
                f"prove actions that have the required features: {features}"
            )

        self._models = self._fit_models(raw)

    # ── Training data loading ──────────────────────────────────────────

    def _build_sigma_map(self, base_dir: Path) -> Dict[int, float]:
        """Build config_index -> sigma mapping from sweep_config.json.

        The sweep runner strips _state_tracker from per-config params.json,
        but the full param combos (including _state_tracker.sigma) are stored
        in sweep_config.json.
        """
        sigma_map: Dict[int, float] = {}

        sweep_cfg_file = base_dir / "sweep_config.json"
        if sweep_cfg_file.exists():
            with open(sweep_cfg_file) as f:
                sweep_cfg = json.load(f)
            for idx, combo in enumerate(sweep_cfg.get("param_combos", [])):
                st = combo.get("_state_tracker", {})
                sigma_map[idx] = float(st.get("sigma", 0.1))

        return sigma_map

    def _load_training_data(
        self, base_dir: Path,
    ) -> Dict[Tuple[float, str], Tuple[List[List[float]], List[int]]]:
        """Load training samples from all config dirs in a sweep output.

        Returns dict mapping (sigma, model_name) -> (X_rows, y_labels).
        """
        samples: Dict[Tuple[float, str], Tuple[List[List[float]], List[int]]] = {}
        sigma_map = self._build_sigma_map(base_dir)

        for config_dir in sorted(base_dir.iterdir()):
            m = re.match(r"config_(\d+)$", config_dir.name)
            if not config_dir.is_dir() or not m:
                continue

            config_idx = int(m.group(1))

            # Get sigma: prefer params.json (ground truth), fall back to sweep_config
            params_file = config_dir / "params.json"
            if params_file.exists():
                with open(params_file) as f:
                    params = json.load(f)
                st_params = params.get("_state_tracker", {})
                config_sigma = float(st_params.get("sigma", sigma_map.get(config_idx, 0.1)))
            elif config_idx in sigma_map:
                config_sigma = sigma_map[config_idx]
            else:
                continue

            # Scan trajectory files
            traj_dir = config_dir / "trajectories"
            if not traj_dir.exists():
                continue

            for seed_dir in sorted(traj_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                for traj_file in sorted(seed_dir.iterdir()):
                    if not traj_file.suffix == ".json":
                        continue
                    self._extract_samples_from_trajectory(
                        traj_file, config_sigma, samples,
                    )

        return samples

    def _load_training_data_single(
        self, config_dir: Path,
    ) -> Dict[Tuple[float, str], Tuple[List[List[float]], List[int]]]:
        """Load training samples from a single config directory."""
        samples: Dict[Tuple[float, str], Tuple[List[List[float]], List[int]]] = {}

        params_file = config_dir / "params.json"
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
            st_params = params.get("_state_tracker", {})
            config_sigma = float(st_params.get("sigma", 0.1))
        else:
            config_sigma = 0.1

        traj_dir = config_dir / "trajectories"
        if not traj_dir.exists():
            return samples

        for seed_dir in sorted(traj_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            for traj_file in sorted(seed_dir.iterdir()):
                if not traj_file.suffix == ".json":
                    continue
                self._extract_samples_from_trajectory(
                    traj_file, config_sigma, samples,
                )

        return samples

    def _extract_samples_from_trajectory(
        self,
        traj_file: Path,
        config_sigma: float,
        samples: Dict[Tuple[float, str], Tuple[List[List[float]], List[int]]],
    ):
        """Extract (feature_vector, label) pairs from one trajectory file."""
        with open(traj_file) as f:
            traj = json.load(f)

        for step in traj.get("steps", []):
            action = step.get("action", {})
            if action.get("type") != "prove":
                continue

            model_name = action.get("model")
            if model_name is None:
                continue

            tracked = step.get("tracked_state", {})
            result = step.get("result", {})

            # Skip steps with no prior attempts (hot-start territory)
            num_att = tracked.get("num_attempts", {})
            n = num_att.get(model_name) if isinstance(num_att, dict) else num_att
            if n is not None and float(n) == 0:
                continue

            # Extract raw feature values
            raw_features = {}
            for feat_name in self._feature_names:
                feat_data = tracked.get(feat_name, {})
                if isinstance(feat_data, dict):
                    val = feat_data.get(model_name)
                else:
                    val = feat_data
                if val is None:
                    break
                raw_features[feat_name] = float(val)
            else:
                # All features found — build feature vector with transforms
                fv = self._build_feature_vector(raw_features)
                label = int(result.get("success", False))

                key = (config_sigma, model_name)
                if key not in samples:
                    samples[key] = ([], [])
                samples[key][0].append(fv)
                samples[key][1].append(label)

    def _build_feature_vector(self, raw_features: Dict[str, float]) -> List[float]:
        """Apply transforms and build the feature vector."""
        fv = []
        for feat_name, transform_key in self._feature_specs:
            val = raw_features[feat_name]
            transform_fn = TRANSFORMS[transform_key]
            fv.append(float(transform_fn(np.array(val))))
        return fv

    # ── Model fitting ──────────────────────────────────────────────────

    def _fit_models(
        self,
        samples: Dict[Tuple[float, str], Tuple[List[List[float]], List[int]]],
    ) -> Dict[Tuple[float, str], Union[Tuple, float]]:
        """Fit one model (with scaler) per (sigma, model_name) group."""
        models: Dict[Tuple[float, str], Union[Tuple, float]] = {}

        for key, (X_rows, y_labels) in samples.items():
            if not X_rows:
                models[key] = 0.0
                continue

            X = np.array(X_rows)
            y = np.array(y_labels)

            unique_classes = set(y_labels)
            if len(unique_classes) < 2:
                # Degenerate: all same class
                models[key] = float(y[0])
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if self.model_type == "linear":
                model = Ridge(alpha=self.alpha)
                model.fit(X_scaled, y)
            else:
                model = LogisticRegression(C=self.C, max_iter=1000)
                model.fit(X_scaled, y)
            models[key] = (model, scaler)

        return models

    # ── Prediction ─────────────────────────────────────────────────────

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
            feat_data = tracked.get(feat_name, {})
            if isinstance(feat_data, dict):
                val = feat_data.get(model_name)
            else:
                val = feat_data
            if val is None:
                return 0.0
            raw_features[feat_name] = float(val)

        # Build transformed feature vector
        fv = self._build_feature_vector(raw_features)

        # Look up fitted model
        key = (self.sigma, model_name)
        entry = self._models.get(key)
        if entry is None:
            return 0.0

        if isinstance(entry, float):
            return entry

        model, scaler = entry
        fv = scaler.transform([fv])[0].tolist()
        if isinstance(model, Ridge):
            return float(np.clip(model.predict([fv])[0], 0.0, 1.0))
        return float(model.predict_proba([fv])[0, 1])

    # ── Serialization ──────────────────────────────────────────────────

    def get_model_params(self) -> Dict[str, Any]:
        """Return model parameters as a serializable dict."""
        params: Dict[str, Any] = {
            "features": self._feature_raw_specs,
            "C": self.C,
            "models": {},
        }

        for (sigma, model_name), entry in self._models.items():
            key = f"sigma={sigma}_model={model_name}"
            if isinstance(entry, float):
                params["models"][key] = {
                    "type": "constant",
                    "value": entry,
                }
            else:
                model, scaler = entry
                if isinstance(model, Ridge):
                    model_dict: Dict[str, Any] = {
                        "type": "linear_regression",
                        "coefficients": model.coef_.tolist(),
                        "intercept": [float(model.intercept_)],
                        "features": self._feature_raw_specs,
                        "scaler": {
                            "mean": scaler.mean_.tolist(),
                            "scale": scaler.scale_.tolist(),
                        },
                    }
                else:
                    model_dict: Dict[str, Any] = {
                        "type": "logistic_regression",
                        "coefficients": model.coef_.tolist(),
                        "intercept": model.intercept_.tolist(),
                        "classes": model.classes_.tolist(),
                        "features": self._feature_raw_specs,
                        "C": self.C,
                        "scaler": {
                            "mean": scaler.mean_.tolist(),
                            "scale": scaler.scale_.tolist(),
                        },
                    }
                params["models"][key] = model_dict

        return params

    def save_model_params(self, path: str):
        """Save model parameters to a JSON file."""
        params = self.get_model_params()
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
