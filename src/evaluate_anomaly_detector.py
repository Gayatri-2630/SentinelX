from pathlib import Path

import joblib
import pandas as pd

from prepare_data import load_training_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentinelx_anomaly_detector.joblib"
OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the saved detector, feature list, threshold, and original split.
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
features = bundle["features"]
threshold = float(bundle["threshold"])
split = bundle["split"]

train_units = set(split["training_units"])
validation_units = set(split["validation_units"])
final_test_units = set(split["final_test_units"])

# Load the dataset and select only the previously reserved final-test engines.
df = load_training_data(bundle["dataset"])
test_df = df[df["unit"].isin(final_test_units)].copy()

if test_df.empty:
    raise ValueError("No final-test readings found.")

# Verify the partitions are separate and match the saved split.
assert train_units.isdisjoint(validation_units)
assert train_units.isdisjoint(final_test_units)
assert validation_units.isdisjoint(final_test_units)
assert set(test_df["unit"]) == final_test_units

# Apply the existing detector and frozen validation-derived threshold.
test_df["anomaly_score"] = pipeline.decision_function(
    test_df[features]
)
test_df["anomaly_flag"] = test_df["anomaly_score"] <= threshold

# Overall summary.
summary = pd.DataFrame([{
    "dataset": bundle["dataset"],
    "test_engines": test_df["unit"].nunique(),
    "test_readings": len(test_df),
    "frozen_threshold": threshold,
    "flagged_readings": int(test_df["anomaly_flag"].sum()),
    "flagged_percent": round(
        test_df["anomaly_flag"].mean() * 100, 2
    ),
    "engines_with_flags": int(
        test_df.groupby("unit")["anomaly_flag"].any().sum()
    ),
}])

print("\n=== FINAL-TEST SUMMARY ===")
print(summary.to_string(index=False))

# Cycle-stage analysis using the original cycle bins.
test_df["cycle_stage"] = pd.cut(
    test_df["cycle"],
    bins=[0, 50, 100, 150, 200, float("inf")],
    labels=["1-50", "51-100", "101-150", "151-200", "201+"],
)

stage_summary = test_df.groupby(
    "cycle_stage", observed=False
).agg(
    readings=("anomaly_flag", "size"),
    flagged=("anomaly_flag", "sum"),
    mean_score=("anomaly_score", "mean"),
)

stage_summary["flagged_percent"] = (
    stage_summary["flagged"] / stage_summary["readings"] * 100
).round(2)

print("\n=== FLAG RATE BY CYCLE STAGE ===")
print(stage_summary.to_string())

# Engine-level analysis.
engine_summary = test_df.groupby("unit").agg(
    readings=("anomaly_flag", "size"),
    flagged=("anomaly_flag", "sum"),
    last_cycle=("cycle", "max"),
    mean_score=("anomaly_score", "mean"),
    minimum_score=("anomaly_score", "min"),
)

engine_summary["flagged_percent"] = (
    engine_summary["flagged"] / engine_summary["readings"] * 100
).round(2)

engine_summary = engine_summary.sort_values(
    "flagged_percent", ascending=False
)

print("\n=== FLAG RATE BY ENGINE ===")
print(engine_summary.to_string())

# RUL-stage analysis. RUL is derived from the dataset's observed
# training trajectories; it is not an anomaly ground-truth label.
test_df["rul_stage"] = pd.cut(
    test_df["RUL"],
    bins=[-1, 30, 60, 100, float("inf")],
    labels=["0-30", "31-60", "61-100", "101+"],
)

rul_summary = test_df.groupby(
    "rul_stage", observed=False
).agg(
    readings=("anomaly_flag", "size"),
    flagged=("anomaly_flag", "sum"),
    mean_score=("anomaly_score", "mean"),
)

rul_summary["flagged_percent"] = (
    rul_summary["flagged"] / rul_summary["readings"] * 100
).round(2)

print("\n=== FLAG RATE BY RUL STAGE ===")
print(rul_summary.to_string())

print("\n=== FINAL-TEST RUL DISTRIBUTION ===")
print(test_df["RUL"].describe().to_string())

print("\nReadings with RUL <= 30:", int((test_df["RUL"] <= 30).sum()))

# Export results for the project report.
summary.to_csv(OUTPUT_DIR / "anomaly_test_summary.csv", index=False)
stage_summary.to_csv(
    OUTPUT_DIR / "anomaly_test_by_cycle_stage.csv"
)
engine_summary.to_csv(
    OUTPUT_DIR / "anomaly_test_by_engine.csv"
)
rul_summary.to_csv(
    OUTPUT_DIR / "anomaly_test_by_rul_stage.csv"
)

print("\nCSV reports saved in:", OUTPUT_DIR)
print("\nIMPORTANT:")
print("- The detector was not retrained.")
print("- The threshold was not recalibrated on final-test data.")
print("- Anomaly flags indicate unusual readings, not confirmed faults.")
print("- Without ground-truth anomaly labels, these are flag-burden")
print("  summaries, not precision, recall, or fault-detection accuracy.")
