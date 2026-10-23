
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
test_df = test_df.sort_values(["unit", "cycle"]).reset_index(drop=True)

if test_df.empty:
    raise ValueError("No final-test readings found.")

test_df["anomaly_score"] = pipeline.decision_function(
    test_df[features]
)
test_df["is_anomaly"] = test_df["anomaly_score"] <= threshold

# Identify consecutive flagged cycles separately within each engine.
previous_flag = test_df.groupby("unit")["is_anomaly"].shift(
    fill_value=False
)
previous_cycle = test_df.groupby("unit")["cycle"].shift()

continues_run = (
    test_df["is_anomaly"]
    & previous_flag
    & ((test_df["cycle"] - previous_cycle) == 1)
)

test_df["run_start"] = (
    test_df["is_anomaly"] & ~continues_run
)

test_df["run_id"] = (
    test_df.groupby("unit")["run_start"].cumsum()
)

flagged = test_df[test_df["is_anomaly"]].copy()

runs = (
    flagged.groupby(["unit", "run_id"])
    .agg(
        start_cycle=("cycle", "min"),
        end_cycle=("cycle", "max"),
        flagged_readings=("cycle", "size"),
        minimum_score=("anomaly_score", "min"),
        maximum_score=("anomaly_score", "max"),
    )
    .reset_index()
)

if not runs.empty:
    runs["run_length"] = (
        runs["end_cycle"] - runs["start_cycle"] + 1
    )

    # A run is consecutive only if all cycles between its endpoints
    # were flagged. Check the observed row count as well.
    runs = runs[
        runs["run_length"] == runs["flagged_readings"]
    ].copy()

# Engine-level persistence summary.
engine_summary = (
    test_df.groupby("unit")
    .agg(
        readings=("is_anomaly", "size"),
        flagged_readings=("is_anomaly", "sum"),
    )
    .reset_index()
)

engine_summary["flagged_percent"] = (
    engine_summary["flagged_readings"]
    / engine_summary["readings"] * 100
).round(2)

if runs.empty:
    run_summary = pd.DataFrame(columns=[
        "unit", "run_id", "start_cycle", "end_cycle",
        "flagged_readings", "minimum_score",
        "maximum_score", "run_length",
    ])
else:
    run_summary = runs.sort_values(
        ["run_length", "minimum_score"],
        ascending=[False, True],
    )

print("=== ANOMALY PERSISTENCE ===")
print("Final-test engines:", test_df["unit"].nunique())
print("Final-test readings:", len(test_df))
print("Flagged readings:", int(test_df["is_anomaly"].sum()))
print(f"Frozen threshold: {threshold:.8f}")
print("Consecutive flagged runs:", len(run_summary))

if not run_summary.empty:
    print("\nLongest consecutive runs:")
    print(run_summary.head(20).to_string(index=False))

    print("\nRun-length distribution:")
    print(
        run_summary["run_length"]
        .value_counts()
        .sort_index()
        .rename_axis("consecutive_cycles")
        .to_frame("number_of_runs")
        .to_string()
    )

print("\nEngine-level flag summary:")
print(
    engine_summary.sort_values(
        "flagged_percent", ascending=False
    ).to_string(index=False)
)

run_summary.to_csv(
    OUTPUT_DIR / "anomaly_consecutive_runs.csv", index=False
)
engine_summary.to_csv(
    OUTPUT_DIR / "anomaly_persistence_by_engine.csv", index=False
)

print("\nReports saved to:", OUTPUT_DIR)
print(
    "Interpretation: consecutive flags describe persistence, "
    "not confirmed faults or predictive lead time."
)
