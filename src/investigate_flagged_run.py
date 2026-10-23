
from pathlib import Path

import joblib
import pandas as pd

from prepare_data import load_training_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentinelx_anomaly_detector.joblib"

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
features = bundle["features"]
threshold = float(bundle["threshold"])
test_units = set(bundle["split"]["final_test_units"])

df = load_training_data(bundle["dataset"])
df = df[df["unit"].isin(test_units)].copy()

engine_id = 73
start_cycle = 206
end_cycle = 213

engine_df = df[df["unit"] == engine_id].copy()

if engine_df.empty:
    raise ValueError(f"Engine {engine_id} is not in the saved final-test split.")

engine_df["anomaly_score"] = pipeline.decision_function(
    engine_df[features]
)
engine_df["is_anomaly"] = (
    engine_df["anomaly_score"] <= threshold
)

# Inspect a small window around the flagged run.
window = engine_df[
    engine_df["cycle"].between(start_cycle - 3, end_cycle + 3)
].copy()

if window.empty:
    raise ValueError("No readings found in the requested cycle window.")

window["status"] = window["is_anomaly"].map({
    True: "FLAGGED",
    False: "not flagged",
})

display_columns = [
    "unit",
    "cycle",
    "RUL",
    "anomaly_score",
    "status",
] + features

print("=== ENGINE 73: CYCLES AROUND THE LONGEST FLAGGED RUN ===")
print(f"Frozen threshold: {threshold:.8f}")
print(window[display_columns].round(5).to_string(index=False))

# Compare flagged-run means with the three immediately preceding
# and three immediately following observed cycles.
run = engine_df[
    engine_df["cycle"].between(start_cycle, end_cycle)
]
nearby = engine_df[
    engine_df["cycle"].isin(
        list(range(start_cycle - 3, start_cycle))
        + list(range(end_cycle + 1, end_cycle + 4))
    )
]

if len(run) != end_cycle - start_cycle + 1:
    print(
        "\nWARNING: The requested run does not contain all eight "
        "cycle readings. Check the printed window."
    )

if not run.empty and not nearby.empty:
    comparison = pd.DataFrame({
        "run_mean": run[features].mean(),
        "nearby_mean": nearby[features].mean(),
    })
    comparison["difference"] = (
        comparison["run_mean"] - comparison["nearby_mean"]
    )

    print("\n=== FEATURE MEANS: FLAGGED RUN VS NEARBY CYCLES ===")
    print(comparison.round(5).to_string())

    comparison.to_csv(
        ROOT / "reports" / "engine73_run_feature_comparison.csv"
    )
    print(
        "\nSaved: reports/engine73_run_feature_comparison.csv"
    )

print(
    "\nInterpretation: these comparisons describe the observed "
    "readings; they do not prove causation or a confirmed engine fault."
)
