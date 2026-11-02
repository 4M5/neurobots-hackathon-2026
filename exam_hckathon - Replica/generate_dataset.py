"""Generate a reproducible synthetic dataset for pipeline development only.

The labels describe generated patterns, not observed cheating. They are not
ground truth about how real students behave.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_normal_samples(n_samples=800, rng=None):
    """Generate synthetic samples from the hand-authored normal profile."""
    rng = np.random.default_rng(42) if rng is None else rng
    data = {
        "avg_key_interval": rng.uniform(0.1, 0.3, n_samples),
        "std_key_interval": rng.uniform(0.02, 0.08, n_samples),
        "typing_speed": rng.uniform(150, 300, n_samples),
        "backspace_rate": rng.uniform(0.02, 0.1, n_samples),
        "paste_count": rng.choice([0, 1], n_samples, p=[0.9, 0.1]),
        "tab_switch": rng.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]),
        "focus_loss": rng.choice([0, 1], n_samples, p=[0.85, 0.15]),
        "avg_idle": rng.uniform(2, 10, n_samples),
        "max_idle": rng.uniform(5, 30, n_samples),
        "edit_count": rng.integers(2, 9, n_samples),
        "time_to_first_key": rng.uniform(2, 15, n_samples),
        "answer_duration": rng.uniform(60, 300, n_samples),
    }
    frame = pd.DataFrame(data)
    frame["label"] = 0
    return frame


def generate_anomaly_samples(n_samples=200, rng=None):
    """Generate synthetic anomalies from hand-authored extreme patterns."""
    rng = np.random.default_rng(43) if rng is None else rng
    data = {
        "avg_key_interval": rng.uniform(0.01, 0.05, n_samples),
        "std_key_interval": rng.uniform(0.15, 0.4, n_samples),
        "typing_speed": np.where(
            rng.random(n_samples) > 0.5,
            rng.uniform(500, 1000, n_samples),
            rng.uniform(30, 80, n_samples),
        ),
        "backspace_rate": rng.uniform(0.0, 0.02, n_samples),
        "paste_count": rng.integers(3, 15, n_samples),
        "tab_switch": rng.integers(5, 25, n_samples),
        "focus_loss": rng.integers(3, 15, n_samples),
        "avg_idle": rng.uniform(20, 60, n_samples),
        "max_idle": rng.uniform(60, 180, n_samples),
        "edit_count": rng.integers(0, 2, n_samples),
        "time_to_first_key": np.where(
            rng.random(n_samples) > 0.5,
            rng.uniform(0.5, 2, n_samples),
            rng.uniform(30, 90, n_samples),
        ),
        "answer_duration": np.where(
            rng.random(n_samples) > 0.5,
            rng.uniform(10, 30, n_samples),
            rng.uniform(400, 600, n_samples),
        ),
    }
    frame = pd.DataFrame(data)
    frame["label"] = 1
    return frame


def generate_cheating_samples(n_samples=200, rng=None):
    """Backward-compatible alias for old imports; labels are synthetic only."""
    return generate_anomaly_samples(n_samples=n_samples, rng=rng)


def generate_dataset(normal_samples=800, anomaly_samples=200, seed=42):
    rng = np.random.default_rng(seed)
    normal = generate_normal_samples(normal_samples, rng)
    anomalies = generate_anomaly_samples(anomaly_samples, rng)
    return pd.concat([normal, anomalies], ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible synthetic samples")
    parser.add_argument("--normal-samples", type=int, default=800)
    parser.add_argument("--anomaly-samples", type=int, default=200)
    parser.add_argument("--output", default="behavior_dataset.csv")
    args = parser.parse_args()
    if args.normal_samples < 1 or args.anomaly_samples < 1:
        parser.error("sample counts must be positive")

    dataset = generate_dataset(args.normal_samples, args.anomaly_samples, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)
    print(f"Generated {len(dataset)} synthetic samples with seed {args.seed}.")
    print(f"Synthetic label counts: {dataset['label'].value_counts().sort_index().to_dict()}")
    print(f"Dataset written to: {output}")
    print("These generated labels are development examples, not real student ground truth.")


if __name__ == "__main__":
    main()
