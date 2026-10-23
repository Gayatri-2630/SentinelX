
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import load_training_data


ROOT = Path(__file__).resolve().parents[1]

df = load_training_data("FD001")

features = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

if df[features].isna().any().any():
    raise ValueError("Missing values found in anomaly detector features.")

# Split complete engine IDs: 60% train, 20% validation, 20% final test.
first_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.4,
    random_state=42,
)

train_idx, remainder_idx = next(
    first_split.split(df, groups=df["unit"])
)

train_df = df.iloc[train_idx].copy()
remainder_df = df.iloc[remainder_idx].copy()

second_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.5,
    random_state=43,
)

validation_idx, final_idx = next(
    second_split.split(
        remainder_df,
        groups=remainder_df["unit"],
    )
)

validation_df = remainder_df.iloc[validation_idx].copy()
final_test_df = remainder_df.iloc[final_idx].copy()

train_units = set(train_df["unit"])
validation_units = set(validation_df["unit"])
final_units = set(final_test_df["unit"])

assert train_units.isdisjoint(validation_units)
assert train_units.isdisjoint(final_units)
assert validation_units.isdisjoint(final_units)

# Fit the model on training engines only.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "detector",
        IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

pipeline.fit(train_df[features])

# Examine validation scores without selecting a threshold.
validation_df["anomaly_score"] = pipeline.decision_function(
    validation_df[features]
)

validation_df["cycle_stage"] = pd.cut(
    validation_df["cycle"],
    bins=[0, 50, 100, 150, 200, float("inf")],
    labels=["1-50", "51-100", "101-150", "151-200", "201+"],
)

validation_df["rul_stage"] = pd.cut(
    validation_df["RUL"],
    bins=[-1, 30, 60, 100, float("inf")],
    labels=["0-30", "31-60", "61-100", "101+"],
)

print("DATA PARTITION")
for name, part in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Final test (reserved)", final_test_df),
]:
    print(
        f"{name}: {part['unit'].nunique()} engines, "
        f"{len(part)} readings"
    )

print("\nValidation score distribution")
print(validation_df["anomaly_score"].describe().round(5).to_string())

print("\nValidation score quantiles")
print(
    validation_df["anomaly_score"]
    .quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    .round(5)
    .to_string()
)

print("\nScores by cycle stage")
print(
    validation_df.groupby("cycle_stage", observed=False)["anomaly_score"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(5)
    .to_string()
)

print("\nScores by RUL stage")
print(
    validation_df.groupby("rul_stage", observed=False)["anomaly_score"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(5)
    .to_string()
)

print("\nLowest-scoring validation engines")
engine_summary = validation_df.groupby("unit").agg(
    readings=("anomaly_score", "size"),
    mean_score=("anomaly_score", "mean"),
    median_score=("anomaly_score", "median"),
    minimum_score=("anomaly_score", "min"),
)

print(
    engine_summary.sort_values("mean_score")
    .head(10)
    .round(5)
    .to_string()
)

print(
    "\nInterpretation: lower scores indicate more unusual readings. "
    "These summaries describe score behavior, not verified fault accuracy."
)
print(
    "No threshold has been selected. The final-test engines remain "
    "reserved and were not used to fit or analyze the model."
)