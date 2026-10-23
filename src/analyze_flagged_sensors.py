
from pathlib import Path

import joblib
import pandas as pd

from prepare_data import load_training_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentinelx_anomaly_detector.joblib"
OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
features = bundle["features"]
threshold = float(bundle["threshold"])
test_units = set(bundle["split"]["final_test_units"])

df = load_training_data(bundle["dataset"])
test_df = df[df["unit"].isin(test_units)].copy()

if test_df.empty:
    raise ValueError("No final-test readings found.")

# Apply the saved detector and frozen threshold.
test_df["anomaly_score"] = pipeline.decision_function(
    test_df[features]
)
test_df["is_anomaly"] = test_df["anomaly_score"] <= threshold

assert test_df["unit"].nunique() == len(test_units)

# Compare flagged readings with unflagged readings in the same
# broad RUL stage to reduce lifecycle-related confounding.
test_df["rul_stage"] = pd.cut(
    test_df["RUL"],
    bins=[-1, 30, 60, 100, float("inf")],
    labels=["0-30", "31-60", "61-100", "101+"],
)

sensor_features = [
    name for name in features if name.startswith("sensor_")
]

rows = []

for stage, stage_df in test_df.groupby("rul_stage", observed=True):
    flagged = stage_df[stage_df["is_anomaly"]]
    unflagged = stage_df[~stage_df["is_anomaly"]]

    if flagged.empty or unflagged.empty:
        continue

    for sensor in sensor_features:
        flagged_mean = flagged[sensor].mean()
        unflagged_mean = unflagged[sensor].mean()
        pooled_std = stage_df[sensor].std()

        difference = flagged_mean - unflagged_mean
        standardized_difference = (
            difference / pooled_std if pd.notna(pooled_std)
            and pooled_std > 0 else float("nan")
        )

        rows.append({
            "rul_stage": str(stage),
            "sensor": sensor,
            "flagged_readings": len(flagged),
            "unflagged_readings": len(unflagged),
            "flagged_mean": flagged_mean,
            "unflagged_mean": unflagged_mean,
            "mean_difference": difference,
            "standardized_difference": standardized_difference,
            "absolute_standardized_difference": abs(
                standardized_difference
            ) if pd.notna(standardized_difference) else float("nan"),
        })

comparison = pd.DataFrame(rows)

if comparison.empty:
    raise ValueError(
        "No RUL stage contained both flagged and unflagged readings."
    )

comparison = comparison.sort_values(
    ["rul_stage", "absolute_standardized_difference"],
    ascending=[True, False],
)

comparison.to_csv(
    OUTPUT_DIR / "flagged_sensor_comparison_by_rul.csv",
    index=False,
)

print("=== FLAGGED SENSOR COMPARISON ===")
print("Final-test engines:", test_df["unit"].nunique())
print("Final-test readings:", len(test_df))
print("Flagged readings:", int(test_df["is_anomaly"].sum()))
print(f"Frozen threshold: {threshold:.8f}")

for stage in comparison["rul_stage"].drop_duplicates():
    stage_results = comparison[
        comparison["rul_stage"] == stage
    ].head(5)

    print(f"\nTop sensor differences for RUL stage {stage}:")
    print(
        stage_results[
            [
                "sensor",
                "flagged_readings",
                "mean_difference",
                "standardized_difference",
            ]
        ].round(4).to_string(index=False)
    )

print("\nSaved:", OUTPUT_DIR / "flagged_sensor_comparison_by_rul.csv")
print(
    "\nInterpretation: these are descriptive group differences, "
    "not causal explanations or proof that individual sensors "
    "caused an anomaly flag."
)
