
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import load_training_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATASET = "FD001"
THRESHOLD_QUANTILE = 0.025
RANDOM_STATE = 42

features = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = load_training_data(DATASET)

if df[features].isna().any().any():
    raise ValueError("Missing values found in anomaly detector features.")

# Split complete engines: 60% train, 20% validation, 20% final test.
split_a = GroupShuffleSplit(
    n_splits=1,
    test_size=0.4,
    random_state=42,
)

train_idx, remainder_idx = next(
    split_a.split(df, groups=df["unit"])
)

train_df = df.iloc[train_idx].copy()
remainder_df = df.iloc[remainder_idx].copy()

split_b = GroupShuffleSplit(
    n_splits=1,
    test_size=0.5,
    random_state=43,
)

validation_idx, final_idx = next(
    split_b.split(remainder_df, groups=remainder_df["unit"])
)

validation_df = remainder_df.iloc[validation_idx].copy()
final_test_df = remainder_df.iloc[final_idx].copy()

train_units = set(train_df["unit"])
validation_units = set(validation_df["unit"])
final_units = set(final_test_df["unit"])

assert train_units.isdisjoint(validation_units)
assert train_units.isdisjoint(final_units)
assert validation_units.isdisjoint(final_units)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "detector",
        IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
])

# Fit only on training engines.
pipeline.fit(train_df[features])

# Select the provisional threshold using validation scores only.
validation_scores = pipeline.decision_function(
    validation_df[features]
)

threshold = float(
    pd.Series(validation_scores).quantile(THRESHOLD_QUANTILE)
)

validation_flags = validation_scores <= threshold

print("Dataset:", DATASET)
print("Features:", len(features))
print("Training engines:", len(train_units))
print("Validation engines:", len(validation_units))
print("Reserved final-test engines:", len(final_units))
print("Training readings:", len(train_df))
print("Validation readings:", len(validation_df))
print("Reserved final-test readings:", len(final_test_df))
print(f"Threshold quantile: {THRESHOLD_QUANTILE:.3%}")
print(f"Frozen score threshold: {threshold:.8f}")
print(
    "Validation readings flagged:",
    f"{validation_flags.mean() * 100:.2f}%",
)

bundle = {
    "pipeline": pipeline,
    "features": features,
    "algorithm": "IsolationForest",
    "dataset": DATASET,
    "threshold": threshold,
    "threshold_quantile": THRESHOLD_QUANTILE,
    "threshold_rule": "decision_function_score <= threshold",
    "random_state": RANDOM_STATE,
    "split": {
        "training_units": sorted(train_units),
        "validation_units": sorted(validation_units),
        "final_test_units": sorted(final_units),
    },
}

output_path = MODEL_DIR / "sentinelx_anomaly_detector.joblib"
joblib.dump(bundle, output_path)

print("Saved detector and threshold:", output_path)
print(
    "\nThe final-test partition was reserved and was not used "
    "to fit the model or select the threshold."
)
print(
    "Flags indicate unusual readings, not verified faults. "
    "No ground-truth anomaly labels were used."
)
