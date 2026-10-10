from pathlib import Path
import joblib
import pandas as pd

from prepare_data import load_training_data


ROOT = Path(__file__).resolve().parents[1]

bundle = joblib.load(
    ROOT / "models" / "sentinelx_lightgbm.joblib"
)

model = bundle["model"]
features = bundle["features"]

df = load_training_data("FD001")

# Pick an actual engine reading, rather than all-zero inputs.
row = df[
    (df["unit"] == 1) &
    (df["cycle"] == 100)
].iloc[0]

X = pd.DataFrame(
    [{feature: row[feature] for feature in features}]
)

prediction = float(model.predict(X)[0])
actual_rul = int(row["RUL"])

print("Engine:", int(row["unit"]))
print("Cycle:", int(row["cycle"]))
print("Actual RUL:", actual_rul, "cycles")
print("Predicted RUL:", round(prediction, 2), "cycles")
print(
    "Absolute error:",
    round(abs(prediction - actual_rul), 2),
    "cycles",
)
