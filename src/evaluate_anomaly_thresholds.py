
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from prepare_data import load_training_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentinelx_anomaly_detector.joblib"

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
features = bundle["features"]

df = load_training_data("FD001")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)
train_idx, test_idx = next(
    splitter.split(df, groups=df["unit"])
)

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx].copy()

assert set(train_df["unit"]).isdisjoint(set(test_df["unit"]))

# Higher scores are more normal; lower scores are more unusual.
scores = pipeline.decision_function(test_df[features])
test_df["anomaly_score"] = scores

# Examine several score quantiles, without retraining the model.
# These are experimental thresholds, not calibrated fault thresholds.
thresholds = [0.05, 0.10, 0.15, 0.20]

print("Held-out engines:", test_df["unit"].nunique())
print("Held-out readings:", len(test_df))

for fraction in thresholds:
    threshold = test_df["anomaly_score"].quantile(fraction)
    test_df["flagged"] = test_df["anomaly_score"] <= threshold

    print(f"\nLowest {fraction * 100:.0f}% of scores")
    print(f"Score threshold: {threshold:.6f}")
    print(
        "Flagged readings:",
        int(test_df["flagged"].sum()),
    )
    print(
        "Flagged percentage:",
        round(test_df["flagged"].mean() * 100, 2),
        "%",
    )

    test_df["rul_stage"] = pd.cut(
        test_df["RUL"],
        bins=[-1, 30, 60, 100, float("inf")],
        labels=["0-30", "31-60", "61-100", "101+"],
    )

    summary = test_df.groupby(
        "rul_stage", observed=False
    ).agg(
        readings=("flagged", "size"),
        flagged=("flagged", "sum"),
    )

    summary["flagged_percent"] = (
        summary["flagged"] / summary["readings"] * 100
    ).round(2)

    print(summary.to_string())

print(
    "\nNote: score-quantile thresholds control how many readings are "
    "flagged overall. They do not establish fault-detection accuracy."
)
