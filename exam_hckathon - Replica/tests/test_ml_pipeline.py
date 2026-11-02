import json

import numpy as np
import pandas as pd
import pytest

from generate_dataset import generate_dataset
from test_model import evaluate_test_split, load_model, load_test_split
from train_model import calibrate_threshold, split_dataset, train_pipeline, validate_dataset
from ml_service import FEATURE_COLUMNS


def test_synthetic_dataset_generation_is_seeded():
    left = generate_dataset(normal_samples=50, anomaly_samples=20, seed=91)
    right = generate_dataset(normal_samples=50, anomaly_samples=20, seed=91)
    pd.testing.assert_frame_equal(left, right)


def test_stratified_splits_are_reproducible_disjoint_and_cover_rows():
    frame = generate_dataset(normal_samples=100, anomaly_samples=40, seed=10)
    train, validation, test = split_dataset(frame, seed=73)
    train2, validation2, test2 = split_dataset(frame, seed=73)
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)
    assert set(train.index) | set(validation.index) | set(test.index) == set(frame.index)
    assert train.index.tolist() == train2.index.tolist()
    assert validation.index.tolist() == validation2.index.tolist()
    assert test.index.tolist() == test2.index.tolist()
    assert set(FEATURE_COLUMNS).issubset(train.columns)


def test_dataset_schema_rejects_missing_extra_nonfinite_and_invalid_labels():
    frame = generate_dataset(normal_samples=20, anomaly_samples=10, seed=4)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_dataset(frame.drop(columns=[FEATURE_COLUMNS[0]]))
    extra = frame.assign(unexpected=1)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_dataset(extra)
    nonfinite = frame.copy()
    nonfinite.loc[nonfinite.index[0], FEATURE_COLUMNS[0]] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_dataset(nonfinite)
    bad_label = frame.copy()
    bad_label.loc[bad_label.index[0], "label"] = 9
    with pytest.raises(ValueError, match="label"):
        validate_dataset(bad_label)


def test_threshold_calibration_uses_validation_labels_and_computes_confusion_matrix():
    labels = np.array([1, 0, 0, 0])
    scores = np.array([-0.4, -0.2, 0.1, 0.3])
    threshold, analysis = calibrate_threshold(labels, scores)
    assert threshold == pytest.approx(-0.3)
    selected = next(row for row in analysis if row["threshold"] == threshold)
    assert selected["confusion_matrix"] == [[3, 0], [0, 1]]
    assert selected["precision"] == pytest.approx(1.0)
    assert selected["recall"] == pytest.approx(1.0)
    assert selected["f1"] == pytest.approx(1.0)


def test_pipeline_saves_metadata_and_evaluates_only_disjoint_test_rows(tmp_path):
    frame = generate_dataset(normal_samples=120, anomaly_samples=40, seed=44)
    result = train_pipeline(frame, output_directory=tmp_path, seed=18, n_estimators=20)
    metadata = result["metadata"]
    train_ids = set(metadata["splits"]["train"]["source_indices"])
    validation_ids = set(metadata["splits"]["validation"]["source_indices"])
    test_ids = set(metadata["splits"]["test"]["source_indices"])
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert metadata["threshold_calibration"]["calibration_partition"] == "validation"
    assert metadata["test_metrics"]["support"] == len(test_ids)
    assert (tmp_path / "behavior_model.pkl").is_file()
    assert (tmp_path / "model_metadata.json").is_file()
    assert (tmp_path / "evaluation_results.json").is_file()
    assert (tmp_path / "validation_threshold_analysis.csv").is_file()

    artifact = load_model(tmp_path / "behavior_model.pkl")
    test_frame = load_test_split(tmp_path / "behavior_test_split.csv", artifact["metadata"])
    metrics = evaluate_test_split(artifact, test_frame)
    assert metrics == metadata["test_metrics"]
    persisted = json.loads((tmp_path / "model_metadata.json").read_text(encoding="utf-8"))
    assert persisted["feature_columns"] == list(FEATURE_COLUMNS)
    assert persisted["decision_threshold"] == result["threshold"]


def test_threshold_and_split_metadata_are_reproducible(tmp_path):
    frame = generate_dataset(normal_samples=120, anomaly_samples=40, seed=57)
    first = train_pipeline(frame, output_directory=tmp_path / "one", seed=92, n_estimators=15)
    second = train_pipeline(frame, output_directory=tmp_path / "two", seed=92, n_estimators=15)
    assert first["threshold"] == second["threshold"]
    assert first["metadata"]["splits"] == second["metadata"]["splits"]
    assert first["metadata"]["validation_metrics"] == second["metadata"]["validation_metrics"]
    assert first["metadata"]["test_metrics"] == second["metadata"]["test_metrics"]
