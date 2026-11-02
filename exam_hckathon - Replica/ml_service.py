"""Isolation Forest inference for aggregated exam behavior.

Artifacts are loaded once when the Flask app is created. joblib uses pickle
under the hood: only load artifacts produced by this project's trusted
training pipeline, and never accept an artifact path from an API request.
"""

import math
import numbers
import os
from pathlib import Path

import joblib
import pandas as pd


FEATURE_COLUMNS = (
    "avg_key_interval",
    "std_key_interval",
    "typing_speed",
    "backspace_rate",
    "paste_count",
    "tab_switch",
    "focus_loss",
    "avg_idle",
    "max_idle",
    "edit_count",
    "time_to_first_key",
    "answer_duration",
)
FEATURE_SCHEMA_VERSION = 1


class ModelUnavailableError(RuntimeError):
    """Raised when a trusted trained artifact is not available."""


class InvalidBehaviorFeatures(ValueError):
    """Raised when feature data does not match the training input schema."""


class BehaviorModelService:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path or os.getenv(
            "BEHAVIOR_MODEL_PATH", Path(__file__).resolve().parent / "behavior_model.pkl"
        ))
        self.model = None
        self.scaler = None
        self.decision_threshold = None
        self.metadata = None
        self.load_error = None
        self._load_model()

    @property
    def available(self):
        return self.model is not None

    def _load_model(self):
        try:
            artifact = joblib.load(self.model_path)
            if not isinstance(artifact, dict):
                raise ValueError("Artifact must be a dictionary")
            model = artifact.get("model")
            scaler = artifact.get("scaler")
            columns = artifact.get("feature_columns")
            if model is None or scaler is None:
                raise ValueError("Artifact must contain model and scaler")
            if columns is None or tuple(columns) != FEATURE_COLUMNS:
                raise ValueError("Artifact feature columns do not match the training schema")
            if not callable(getattr(model, "predict", None)) or not callable(
                getattr(model, "decision_function", None)
            ):
                raise ValueError("Artifact model does not support prediction and scoring")
            if not callable(getattr(scaler, "transform", None)):
                raise ValueError("Artifact scaler does not support transform")
            expected = getattr(model, "n_features_in_", len(FEATURE_COLUMNS))
            if int(expected) != len(FEATURE_COLUMNS):
                raise ValueError("Artifact model feature count does not match the schema")
            scaler_features = getattr(scaler, "n_features_in_", len(FEATURE_COLUMNS))
            if int(scaler_features) != len(FEATURE_COLUMNS):
                raise ValueError("Artifact scaler feature count does not match the schema")
            scaler_columns = getattr(scaler, "feature_names_in_", None)
            if scaler_columns is not None and tuple(scaler_columns) != FEATURE_COLUMNS:
                raise ValueError("Artifact scaler feature order does not match the schema")
            metadata = artifact.get("metadata")
            if metadata is not None:
                if not isinstance(metadata, dict):
                    raise ValueError("Artifact metadata must be an object")
                metadata_columns = metadata.get("feature_columns")
                if metadata_columns is not None and tuple(metadata_columns) != FEATURE_COLUMNS:
                    raise ValueError("Artifact metadata feature columns do not match the schema")
                schema_version = metadata.get("feature_schema_version")
                if schema_version is not None and schema_version != FEATURE_SCHEMA_VERSION:
                    raise ValueError("Artifact feature schema version is unsupported")
            threshold = artifact.get("decision_threshold")
            if threshold is not None:
                if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real) or not math.isfinite(float(threshold)):
                    raise ValueError("Artifact decision threshold must be finite numeric")
                self.decision_threshold = float(threshold)
            self.metadata = metadata
            self.model, self.scaler = model, scaler
        except Exception as exc:  # Keep serving the rest of the Flask app.
            self.load_error = exc

    @staticmethod
    def validate_features(features):
        if not isinstance(features, dict):
            raise InvalidBehaviorFeatures("features must be a JSON object")
        missing = [name for name in FEATURE_COLUMNS if name not in features]
        extra = [name for name in features if name not in FEATURE_COLUMNS]
        if missing or extra:
            parts = []
            if missing:
                parts.append("missing features: " + ", ".join(missing))
            if extra:
                parts.append("unexpected features: " + ", ".join(map(str, extra)))
            raise InvalidBehaviorFeatures("; ".join(parts))

        values = {}
        for name in FEATURE_COLUMNS:
            value = features[name]
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise InvalidBehaviorFeatures(f"{name} must be a finite number")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise InvalidBehaviorFeatures(f"{name} must be a finite number")
            if numeric < 0:
                raise InvalidBehaviorFeatures(f"{name} must not be negative")
            if name == "backspace_rate" and numeric > 1:
                raise InvalidBehaviorFeatures("backspace_rate must be between 0 and 1")
            values[name] = numeric
        return values

    def predict(self, features):
        values = self.validate_features(features)
        if not self.available:
            raise ModelUnavailableError("Behavior model is not available") from self.load_error
        # Match train_model.py: StandardScaler.transform on the original ordered columns.
        frame = pd.DataFrame([[values[name] for name in FEATURE_COLUMNS],], columns=FEATURE_COLUMNS)
        transformed = self.scaler.transform(frame)
        score = float(self.model.decision_function(transformed)[0])
        if self.decision_threshold is not None:
            label = -1 if score < self.decision_threshold else 1
        else:
            # Artifacts created before validation threshold calibration remain loadable.
            label = int(self.model.predict(transformed)[0])
        if label not in (-1, 1) or not math.isfinite(score):
            raise RuntimeError("Model returned an invalid prediction")
        anomalous = label == -1
        return {
            "prediction": label,
            "is_anomaly": anomalous,
            "decision_score": score,
            "alert_category": "review" if anomalous else "none",
            "alert_category_definition": (
                "review means the model flagged this summary for human review; none means "
                "the model did not flag it. Neither category is proof of cheating."
            ),
        }
