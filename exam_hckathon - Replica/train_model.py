"""Train an Isolation Forest with seeded train/validation/test evaluation.

Training is unsupervised on only the normal-labeled training partition.
The validation partition selects a score threshold; test data is held out
until that threshold is fixed. Synthetic metrics are development diagnostics,
not estimates of performance on real students.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_service import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION


LABEL_COLUMN = "label"
DEFAULT_SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


def validate_dataset(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Dataset must be a non-empty pandas DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names")
    expected = set(FEATURE_COLUMNS) | {LABEL_COLUMN}
    missing = expected - set(frame.columns)
    extra = set(frame.columns) - expected
    if missing or extra:
        raise ValueError(f"Dataset schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    if not frame.index.is_unique:
        raise ValueError("Dataset row index must be unique to audit the splits")

    clean = frame.loc[:, [*FEATURE_COLUMNS, LABEL_COLUMN]].copy()
    for column in [*FEATURE_COLUMNS, LABEL_COLUMN]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    values = clean.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Feature values must all be finite numeric values")
    if (values < 0).any():
        raise ValueError("Behavior feature values must not be negative")
    if (clean["backspace_rate"] > 1).any():
        raise ValueError("backspace_rate must be between 0 and 1")
    labels = clean[LABEL_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("label must contain only 0 (normal) and 1 (synthetic anomaly) values")
    clean[LABEL_COLUMN] = clean[LABEL_COLUMN].astype(int)
    counts = clean[LABEL_COLUMN].value_counts()
    if len(counts) != 2 or counts.min() < 3:
        raise ValueError("At least three samples from each label are required for stratified splits")
    return clean


def split_dataset(frame, seed=DEFAULT_SEED):
    """Return disjoint stratified partitions while retaining source row IDs."""
    clean = validate_dataset(frame)
    train, remainder = train_test_split(
        clean,
        test_size=VALIDATION_FRACTION + TEST_FRACTION,
        random_state=seed,
        stratify=clean[LABEL_COLUMN],
    )
    relative_test_fraction = TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION)
    validation, test = train_test_split(
        remainder,
        test_size=relative_test_fraction,
        random_state=seed,
        stratify=remainder[LABEL_COLUMN],
    )

    train_ids, validation_ids, test_ids = map(set, (train.index, validation.index, test.index))
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("Train, validation, and test splits overlap")
    if train_ids | validation_ids | test_ids != set(clean.index):
        raise RuntimeError("Split rows do not cover the input dataset exactly")
    return train.sort_index(), validation.sort_index(), test.sort_index()


def fit_isolation_forest(training_frame, seed=DEFAULT_SEED, n_estimators=100):
    normal_rows = training_frame.loc[training_frame[LABEL_COLUMN] == 0, FEATURE_COLUMNS]
    if len(normal_rows) < 2:
        raise ValueError("Training split must contain at least two normal samples")
    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(normal_rows)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        max_samples="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(training_scaled)
    return model, scaler


def _metrics_for_threshold(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores) or len(y_true) == 0:
        raise ValueError("y_true and scores must be non-empty one-dimensional arrays of equal length")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite")
    y_pred = (scores < threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    false_positive_rate = float(fp / (fp + tn)) if fp + tn else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support": int(len(y_true)),
    }


def calibrate_threshold(validation_labels, validation_scores):
    """Select the validation threshold maximizing F1; ties prefer fewer false positives."""
    scores = np.asarray(validation_scores, dtype=float)
    labels = np.asarray(validation_labels, dtype=int)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels) or len(scores) == 0:
        raise ValueError("Validation labels and scores must be non-empty arrays of equal length")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Validation data contains invalid labels or scores")
    unique = np.unique(scores)
    midpoints = (unique[:-1] + unique[1:]) / 2 if len(unique) > 1 else np.array([], dtype=float)
    candidates = np.unique(np.concatenate((
        [np.nextafter(unique[0], -np.inf)],
        midpoints,
        [np.nextafter(unique[-1], np.inf)],
    )))
    analysis = [_metrics_for_threshold(labels, scores, candidate) for candidate in candidates]
    selected = max(
        analysis,
        key=lambda row: (
            row["f1"],
            row["precision"],
            -row["false_positive_rate"],
            -row["threshold"],
        ),
    )
    return selected["threshold"], analysis


def evaluate_partition(model, scaler, frame, threshold):
    features = frame.loc[:, FEATURE_COLUMNS]
    transformed = scaler.transform(features)
    scores = np.asarray(model.decision_function(transformed), dtype=float)
    labels = frame[LABEL_COLUMN].to_numpy(dtype=int)
    return _metrics_for_threshold(labels, scores, threshold), scores


def _row_id_hash(indices):
    canonical = ",".join(str(value) for value in sorted(int(index) for index in indices))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _dataset_hash(dataset_path, frame):
    if dataset_path is not None and Path(dataset_path).is_file():
        payload = Path(dataset_path).read_bytes()
    else:
        payload = frame.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _split_summary(frame):
    return {
        "sample_count": int(len(frame)),
        "class_counts": {str(int(label)): int(count) for label, count in frame[LABEL_COLUMN].value_counts().sort_index().items()},
        "source_indices": sorted(int(index) for index in frame.index),
        "source_index_sha256": _row_id_hash(frame.index),
    }


def train_pipeline(dataset, output_directory=".", seed=DEFAULT_SEED, dataset_kind="synthetic", n_estimators=100):
    """Train, calibrate on validation, and evaluate once on disjoint test rows."""
    dataset_path = dataset if isinstance(dataset, (str, Path)) else None
    frame = pd.read_csv(dataset_path) if dataset_path is not None else dataset.copy()
    frame = validate_dataset(frame)
    train, validation, test = split_dataset(frame, seed=seed)
    model, scaler = fit_isolation_forest(train, seed=seed, n_estimators=n_estimators)

    validation_scores = np.asarray(
        model.decision_function(scaler.transform(validation.loc[:, FEATURE_COLUMNS])), dtype=float
    )
    threshold, threshold_analysis = calibrate_threshold(
        validation[LABEL_COLUMN].to_numpy(dtype=int), validation_scores
    )
    validation_metrics, _ = evaluate_partition(model, scaler, validation, threshold)
    test_metrics, test_scores = evaluate_partition(model, scaler, test, threshold)

    metadata = {
        "metadata_version": 1,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_mapping": {"0": "synthetic_normal_profile", "1": "synthetic_anomaly_profile"},
        "dataset_kind": dataset_kind,
        "dataset_sha256": _dataset_hash(dataset_path, frame),
        "random_seed": int(seed),
        "split_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "splits": {
            "train": _split_summary(train),
            "validation": _split_summary(validation),
            "test": _split_summary(test),
        },
        "preprocessing": "StandardScaler fitted only on normal rows in the training partition",
        "model": {
            "algorithm": "IsolationForest",
            "n_estimators": int(n_estimators),
            "contamination": "auto",
            "random_state": int(seed),
        },
        "threshold_calibration": {
            "method": "maximize validation-set F1; ties prefer precision, lower false-positive rate, then fewer flags",
            "decision_rule": "anomaly if decision_score < decision_threshold",
            "decision_threshold": float(threshold),
            "calibration_partition": "validation",
        },
        "decision_threshold": float(threshold),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "Metrics describe only the supplied labeled dataset and split.",
            "Synthetic labels encode hand-authored patterns and are not real student ground truth.",
            "The validation-selected threshold may not transfer to a school population or a different exam.",
        ],
    }

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(FEATURE_COLUMNS),
        "decision_threshold": float(threshold),
        "metadata": metadata,
    }
    joblib.dump(artifact, output / "behavior_model.pkl")
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "evaluation_results.json").write_text(json.dumps({
        "validation": validation_metrics,
        "test": test_metrics,
        "threshold_calibration": metadata["threshold_calibration"],
        "dataset_kind": dataset_kind,
        "limitations": metadata["limitations"],
    }, indent=2), encoding="utf-8")
    pd.DataFrame(threshold_analysis).to_csv(output / "validation_threshold_analysis.csv", index=False)

    for name, partition in (("validation", validation), ("test", test)):
        exported = partition.loc[:, [*FEATURE_COLUMNS, LABEL_COLUMN]].copy()
        exported.insert(0, "source_index", partition.index.astype(int))
        exported.to_csv(output / f"behavior_{name}_split.csv", index=False)

    return {
        "model": model,
        "scaler": scaler,
        "threshold": float(threshold),
        "metadata": metadata,
        "threshold_analysis": threshold_analysis,
        "test_scores": test_scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="behavior_dataset.csv")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dataset-kind", choices=("synthetic", "external"), default="synthetic")
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()
    if args.n_estimators < 1:
        parser.error("--n-estimators must be positive")
    result = train_pipeline(
        args.dataset,
        output_directory=args.output_dir,
        seed=args.seed,
        dataset_kind=args.dataset_kind,
        n_estimators=args.n_estimators,
    )
    metadata = result["metadata"]
    print("Model trained. Threshold selected using validation data only.")
    print(f"Seed: {args.seed}; dataset kind: {args.dataset_kind}")
    print(f"Validation metrics: {json.dumps(metadata['validation_metrics'], sort_keys=True)}")
    print(f"Held-out test metrics: {json.dumps(metadata['test_metrics'], sort_keys=True)}")
    print(f"Artifacts written to: {Path(args.output_dir).resolve()}")
    if args.dataset_kind == "synthetic":
        print("These are synthetic-data diagnostics, not estimates of real-school performance.")


if __name__ == "__main__":
    main()
