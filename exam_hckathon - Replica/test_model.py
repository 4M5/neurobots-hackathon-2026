"""Evaluate the saved model on the independently held-out test split.

Optional interactive scoring is a demonstration only and is not part of the
evaluation metrics.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_service import BehaviorModelService, FEATURE_COLUMNS
from train_model import LABEL_COLUMN, _metrics_for_threshold, validate_dataset


def load_model(model_path="behavior_model.pkl"):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Run train_model.py first.")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be an object")
    required = {"model", "scaler", "feature_columns"}
    if not required.issubset(artifact):
        raise ValueError("Model artifact is missing required training fields")
    if tuple(artifact["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Model artifact feature columns do not match the current schema")
    return artifact


def load_test_split(test_path, metadata):
    test_path = Path(test_path)
    if not test_path.is_file():
        raise FileNotFoundError(f"Held-out test split not found: {test_path}. Run train_model.py first.")
    frame = pd.read_csv(test_path)
    if "source_index" not in frame.columns:
        raise ValueError("Test split must include source_index values")
    source_ids = set(pd.to_numeric(frame.pop("source_index"), errors="raise").astype(int))
    split_metadata = metadata.get("splits")
    if not isinstance(split_metadata, dict) or not {"train", "validation", "test"}.issubset(split_metadata):
        raise ValueError("Artifact metadata does not contain audited split information")
    train_ids = set(split_metadata["train"].get("source_indices", []))
    validation_ids = set(split_metadata["validation"].get("source_indices", []))
    expected_test_ids = set(split_metadata["test"].get("source_indices", []))
    if not source_ids or len(source_ids) != len(frame) or source_ids != expected_test_ids:
        raise ValueError("Test split rows do not match the model metadata")
    if source_ids & train_ids or source_ids & validation_ids or train_ids & validation_ids:
        raise ValueError("Recorded train, validation, and test splits are not independent")
    return validate_dataset(frame)


def predict_behavior(model, scaler, samples, feature_columns, decision_threshold=None):
    if tuple(feature_columns) != FEATURE_COLUMNS:
        raise ValueError("Feature columns do not match the model schema")
    values = samples.loc[:, feature_columns]
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Prediction features must be finite numeric values")
    transformed = scaler.transform(values)
    scores = np.asarray(model.decision_function(transformed), dtype=float)
    if decision_threshold is None:
        predictions = np.asarray(model.predict(transformed), dtype=int)
    else:
        predictions = np.where(scores < float(decision_threshold), -1, 1)
    return predictions, scores


def evaluate_test_split(artifact, test_frame):
    model, scaler = artifact["model"], artifact["scaler"]
    threshold = artifact.get("decision_threshold")
    if threshold is None:
        raise ValueError("This artifact has no validation-calibrated threshold; retrain it with train_model.py")
    predictions, scores = predict_behavior(model, scaler, test_frame, FEATURE_COLUMNS, threshold)
    y_pred = (predictions == -1).astype(int)
    metrics = _metrics_for_threshold(test_frame[LABEL_COLUMN].to_numpy(dtype=int), scores, threshold)
    if not np.array_equal(y_pred, (scores < threshold).astype(int)):
        raise RuntimeError("Prediction threshold and evaluation metrics disagree")
    expected = (artifact.get("metadata") or {}).get("test_metrics")
    if expected:
        for key in ("precision", "recall", "f1", "false_positive_rate"):
            if not np.isclose(metrics[key], expected[key], rtol=1e-12, atol=1e-12):
                raise ValueError("Stored test metrics do not match recalculated held-out metrics")
        for key in ("confusion_matrix", "tn", "fp", "fn", "tp", "support"):
            if metrics[key] != expected[key]:
                raise ValueError("Stored test metrics do not match recalculated held-out metrics")
    return metrics


def interactive_score(artifact):
    defaults = {
        "avg_key_interval": 0.2,
        "std_key_interval": 0.05,
        "typing_speed": 200,
        "backspace_rate": 0.05,
        "paste_count": 0,
        "tab_switch": 1,
        "focus_loss": 0,
        "avg_idle": 5,
        "max_idle": 20,
        "edit_count": 4,
        "time_to_first_key": 10,
        "answer_duration": 150,
    }
    sample = {}
    for feature, default in defaults.items():
        entered = input(f"{feature} [{default}]: ").strip()
        sample[feature] = float(entered) if entered else default
    features = BehaviorModelService.validate_features(sample)
    frame = pd.DataFrame([[features[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    predictions, scores = predict_behavior(
        artifact["model"], artifact["scaler"], frame, FEATURE_COLUMNS, artifact.get("decision_threshold")
    )
    print(f"Illustrative score only: {'review' if predictions[0] == -1 else 'no model flag'}")
    print(f"Decision score: {scores[0]:.6f}")
    print("This single example is not an evaluation result or evidence of cheating.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="behavior_model.pkl")
    parser.add_argument("--test-split", default="behavior_test_split.csv")
    parser.add_argument("--interactive", action="store_true", help="Optionally score a hand-entered example after evaluation")
    args = parser.parse_args()

    artifact = load_model(args.model)
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Model artifact has no evaluation metadata")
    test_frame = load_test_split(args.test_split, metadata)
    metrics = evaluate_test_split(artifact, test_frame)
    print("Held-out test split diagnostics (dataset limitations still apply):")
    print(f"Samples: {metrics['support']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(json.dumps(metrics["confusion_matrix"]))
    print(f"Validation-selected threshold: {metrics['threshold']:.8f}")
    print("These metrics describe only this held-out split; synthetic metrics do not estimate school performance.")
    if args.interactive:
        interactive_score(artifact)


if __name__ == "__main__":
    main()
