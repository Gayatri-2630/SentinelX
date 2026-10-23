
from pathlib import Path

import joblib
import pandas as pd

from prepare_data import load_training_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentinelx_anomaly_detector.joblib"
OUTPUT_PATH = ROOT / "reports" / "anomaly_sensor_differences_by_rul.csv"

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
features = bundle["features"]
threshold = float(bundle["threshold"])
test_units = set(bundle["split"]["final_test_units"])

df = load_training_data(bundle["dataset"])
df = df[df["unit"].isin(test_units)].copy()

if df.empty:
    raise ValueError("No final-test data found.")

df["anomaly_score"] = pipeline.decision_function(df[features])
df["is_anomaly"] = df["anomaly_score"] <= threshold

# Keep the RUL groups consistent with the earlier analysis.
df["rul_stage"] = pd.cut(
    df["RUL"],
    bins=[-1, 30, 60, 100, float("inf")],
    labels=["0-30", "31-60", "61-100", "101+"],
)

results = []

for stage, group in df.groupby("rul_stage", observed=True):
    flagged = group[group["is_anomaly"]]
    normal = group[~group["is_anomaly"]]

    if flagged.empty or normal.empty:
        print(
            f"Skipping RUL {stage}: "
            f"{len(flagged)} flagged, {len(normal)} unflagged."
        )
        continue

    for feature in features:
        flagged_mean = flagged[feature].mean()
        normal_mean = normal[feature].mean()

        # Standardize the difference using the unflagged group's SD.
        normal_std = normal[feature].std(ddof=0)

        if pd.isna(normal_std) or normal_std == 0:
            standardized_difference = None
        else:
            standardized_difference = (
                flagged_mean - normal_mean
            ) / normal_std

        results.append({
            "rul_stage": str(stage),
            "feature": feature,
            "flagged_readings": len(flagged),
            "unflagged_readings": len(normal),
            "flagged_mean": flagged_mean,
            "unflagged_mean": normal_mean,
            "mean_difference": flagged_mean - normal_mean,
            "standardized_difference": standardized_difference,
        })

result_df = pd.DataFrame(results)

if result_df.empty:
    raise ValueError("No RUL stage had both flagged and unflagged readings.")

result_df["absolute_standardized_difference"] = (
    result_df["standardized_difference"].abs()
)

result_df = result_df.sort_values(
    ["rul_stage", "absolute_standardized_difference"],
    ascending=[True, False],
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT_PATH, index=False)

print("=== SENSOR DIFFERENCES WITHIN RUL STAGES ===")
print("Final-test engines:", df["unit"].nunique())
print("Final-test readings:", len(df))
print("Flagged readings:", int(df["is_anomaly"].sum()))
print(f"Frozen threshold: {threshold:.8f}")

for stage in ["0-30", "31-60", "61-100", "101+"]:
    stage_results = result_df[result_df["rul_stage"] == stage]

    if stage_results.empty:
        continue

    print(f"\n--- RUL {stage} ---")
    print(
        stage_results.head(8)[
            [
                "feature",
                "flagged_readings",
                "unflagged_readings",
                "standardized_difference",
            ]
        ].round(3).to_string(index=False)
    )

print(f"\nSaved: {OUTPUT_PATH.relative_to(ROOT)}")
print(
    "\nInterpretation: these are within-stage group differences, "
    "not causal explanations or proof of faults."
)
