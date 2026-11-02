import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

from ml_service import (
    BehaviorModelService,
    FEATURE_COLUMNS,
    InvalidBehaviorFeatures,
    ModelUnavailableError,
)
from server import create_app


def valid_features():
    return {
        "avg_key_interval": 0.2,
        "std_key_interval": 0.05,
        "typing_speed": 300,
        "backspace_rate": 0.08,
        "paste_count": 0,
        "tab_switch": 1,
        "focus_loss": 0,
        "avg_idle": 3.0,
        "max_idle": 8.0,
        "edit_count": 2,
        "time_to_first_key": 5.0,
        "answer_duration": 120.0,
    }


class FakeScaler:
    def __init__(self):
        self.columns = None

    def transform(self, frame):
        self.columns = tuple(frame.columns)
        return frame.to_numpy(dtype=float)


class FakeModel:
    n_features_in_ = len(FEATURE_COLUMNS)

    def __init__(self, label=1, score=0.25):
        self.label = label
        self.score = score
        self.seen = None

    def predict(self, data):
        self.seen = data
        return np.array([self.label])

    def decision_function(self, data):
        return np.array([self.score])


class BehaviorModelServiceTests(unittest.TestCase):
    def test_loads_artifact_once_and_uses_training_scaler_and_feature_order(self):
        model, scaler = FakeModel(label=-1, score=-0.42), FakeScaler()
        artifact = {"model": model, "scaler": scaler, "feature_columns": list(FEATURE_COLUMNS)}
        with patch("ml_service.joblib.load", return_value=artifact) as loader:
            service = BehaviorModelService("trusted-model.pkl")
            result = service.predict(valid_features())
            service.predict(valid_features())
        loader.assert_called_once_with(Path("trusted-model.pkl"))
        self.assertEqual(scaler.columns, FEATURE_COLUMNS)
        self.assertEqual(model.seen.shape, (1, len(FEATURE_COLUMNS)))
        self.assertEqual(result["prediction"], -1)
        self.assertTrue(result["is_anomaly"])
        self.assertEqual(result["decision_score"], -0.42)
        self.assertEqual(result["alert_category"], "review")
        self.assertIn("proof of cheating", result["alert_category_definition"])

    def test_valid_inlier_uses_none_category(self):
        artifact = {
            "model": FakeModel(label=1),
            "scaler": FakeScaler(),
            "feature_columns": list(FEATURE_COLUMNS),
        }
        with patch("ml_service.joblib.load", return_value=artifact):
            result = BehaviorModelService("model.pkl").predict(valid_features())
        self.assertFalse(result["is_anomaly"])
        self.assertEqual(result["alert_category"], "none")

    def test_validation_calibrated_threshold_overrides_legacy_model_predict(self):
        artifact = {
            "model": FakeModel(label=1, score=-0.4),
            "scaler": FakeScaler(),
            "feature_columns": list(FEATURE_COLUMNS),
            "decision_threshold": -0.2,
            "metadata": {"feature_columns": list(FEATURE_COLUMNS), "feature_schema_version": 1},
        }
        with patch("ml_service.joblib.load", return_value=artifact):
            result = BehaviorModelService("calibrated.pkl").predict(valid_features())
        self.assertEqual(result["prediction"], -1)
        self.assertEqual(result["alert_category"], "review")

    def test_rejects_missing_extra_malformed_nonfinite_and_invalid_numbers(self):
        invalid_inputs = []
        missing = valid_features()
        del missing["avg_idle"]
        invalid_inputs.append(missing)
        extra = valid_features()
        extra["answers"] = "not sent"
        invalid_inputs.append(extra)
        for value in ("1.2", True, float("nan"), float("inf"), -1):
            malformed = valid_features()
            malformed["avg_idle"] = value
            invalid_inputs.append(malformed)
        malformed = valid_features()
        malformed["paste_count"] = -1
        invalid_inputs.append(malformed)
        malformed = valid_features()
        malformed["backspace_rate"] = 1.01
        invalid_inputs.append(malformed)

        artifact = {"model": FakeModel(), "scaler": FakeScaler(), "feature_columns": list(FEATURE_COLUMNS)}
        with patch("ml_service.joblib.load", return_value=artifact):
            service = BehaviorModelService("model.pkl")
            for features in invalid_inputs:
                with self.subTest(features=features):
                    with self.assertRaises(InvalidBehaviorFeatures):
                        service.predict(features)

    def test_model_load_failure_is_retained_and_reported(self):
        with patch("ml_service.joblib.load", side_effect=FileNotFoundError("missing")) as loader:
            service = BehaviorModelService("absent.pkl")
            self.assertFalse(service.available)
            self.assertIsInstance(service.load_error, FileNotFoundError)
            with self.assertRaises(ModelUnavailableError):
                service.predict(valid_features())
        loader.assert_called_once()

    def test_rejects_artifact_with_incompatible_feature_schema(self):
        artifact = {"model": FakeModel(), "scaler": FakeScaler(), "feature_columns": list(reversed(FEATURE_COLUMNS))}
        with patch("ml_service.joblib.load", return_value=artifact):
            service = BehaviorModelService("wrong-schema.pkl")
        self.assertFalse(service.available)

    def test_existing_training_artifact_format_runs_with_real_scaler_and_model(self):
        training_rows = pd.DataFrame(
            [[float(row + column + 1) for column in range(len(FEATURE_COLUMNS))] for row in range(20)],
            columns=FEATURE_COLUMNS,
        )
        scaler = StandardScaler().fit(training_rows)
        model = IsolationForest(n_estimators=10, random_state=7).fit(scaler.transform(training_rows))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behavior_model.pkl"
            joblib.dump({"model": model, "scaler": scaler, "feature_columns": list(FEATURE_COLUMNS)}, path)
            service = BehaviorModelService(path)
            result = service.predict(valid_features())
        self.assertIn(result["prediction"], (-1, 1))
        self.assertTrue(np.isfinite(result["decision_score"]))


class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        self.artifact = {"model": FakeModel(label=-1, score=-0.31), "scaler": FakeScaler(), "feature_columns": list(FEATURE_COLUMNS)}
        with patch("ml_service.joblib.load", return_value=self.artifact):
            self.app = create_app(model_path="model.pkl")
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_post_predict_valid_payload(self):
        response = self.client.post("/api/predict", json={"features": valid_features()})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["prediction"], -1)
        self.assertEqual(body["alert_category"], "review")
        self.assertEqual(body["decision_score"], -0.31)

    def test_post_predict_invalid_payloads_are_400(self):
        missing = valid_features()
        del missing["typing_speed"]
        for payload in ({"features": missing}, {"features": []}, {"features": valid_features(), "student_id": "x"}):
            with self.subTest(payload=payload):
                response = self.client.post("/api/predict", json=payload)
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.post("/api/predict", data="{}", content_type="text/plain").status_code, 400)

    def test_unavailable_model_returns_503_without_blocking_app(self):
        missing = BehaviorModelService("missing.pkl")
        app = create_app(model_service=missing)
        app.config["TESTING"] = True
        client = app.test_client()
        response = client.post("/api/predict", json={"features": valid_features()})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"], "model_unavailable")
        self.assertEqual(client.get("/api/questions").status_code, 200)

    def test_prediction_exception_returns_generic_500(self):
        class BrokenService:
            def predict(self, _features):
                raise RuntimeError("private model details")
        app = create_app(model_service=BrokenService())
        app.config["TESTING"] = True
        response = app.test_client().post("/api/predict", json={"features": valid_features()})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private model details", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
