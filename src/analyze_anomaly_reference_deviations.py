
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
train_units = set(bundle["split"]["training_units"])
test_units = set(bundle["split"]["final_test_units"])

df = load_training_data(bundle["dataset"])

# Use only the saved training engines to define the reference.
train_df = df[df["unit"].isin(train_units)].copy()
test_df = df[df["unit"].isin(test_units)].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Training or final-test partition is empty.")

# Apply the unchanged detector and threshold.
test_df["anomaly_score"] = pipeline.decision_function(
    test_df[features]
)
test_df["is_anomaly"] = test_df["anomaly_score"] <= threshold

flagged = test_df[test_df["is_anomaly"]].copy()

if flagged.empty:
    raise ValueError("No flagged readings found in final-test data.")

# Use training-data means and standard deviations as the reference.
reference_mean = train_df[features].mean()
reference_std = train_df[features].std(ddof=0)

valid_features = reference_std[
    reference_std.notna() & (reference_std > 0)
].index.tolist()

# Standardized deviation: (reading - training mean) / training std.
standardized = (
    flagged[valid_features] - reference_mean[valid_features]
) / reference_std[valid_features]

# Summarize the magnitude and direction of deviations among flags.
summary = pd.DataFrame({
    "feature": valid_features,
    "training_mean": reference_mean[valid_features].values,
    "training_std": reference_std[valid_features].values,
    "mean_standardized_deviation": standardized.mean().values,
    "mean_absolute_standardized_deviation": standardized.abs().mean().values,
    "max_absolute_standardized_deviation": standardized.abs().max().values,
})

summary = summary.sort_values(
    "mean_absolute_standardized_deviation",
    ascending=False,
)

summary.to_csv(
    OUTPUT_DIR / "anomaly_reference_deviations.csv",
    index=False,
)

print("=== REFERENCE-DEVIATION ANALYSIS ===")
print("Training engines:", train_df["unit"].nunique())
print("Final-test engines:", test_df["unit"].nunique())
print("Final-test readings:", len(test_df))
print("Flagged readings:", len(flagged))
print(f"Frozen threshold: {threshold:.8f}")

print("\nFeatures with largest mean absolute standardized deviations:")
print(summary.head(10).round(4).to_string(index=False))

print("\nSaved: reports/anomaly_reference_deviations.csv")
print(
    "\nInterpretation: these are deviations from training-data "
    "statistics, not feature attributions or proof of causation."
)
