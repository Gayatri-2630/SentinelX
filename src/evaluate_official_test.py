import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prepare_data import DATA_DIR, COLUMNS, load_training_data


# 1. Load the full training dataset with RUL labels
train_df = load_training_data("FD001")

# 2. Load official test trajectories
test_path = DATA_DIR / "test_FD001.txt"

test_df = pd.read_csv(
    test_path,
    sep=r"\s+",
    header=None,
    names=COLUMNS,
)

# 3. Load official RUL values
rul_path = DATA_DIR / "RUL_FD001.txt"

official_rul = pd.read_csv(
    rul_path,
    sep=r"\s+",
    header=None,
    names=["RUL"],
)["RUL"]

# 4. Select original features only
features = [
    col for col in train_df.columns
    if col.startswith("setting_")
    or (
        col.startswith("sensor_")
        and "_rollmean_" not in col
    )
]
features.append("cycle")

X_train = train_df[features]
y_train = train_df["RUL"]

# 5. Keep the final observed cycle for each test engine
test_df = test_df.sort_values(["unit", "cycle"])

last_cycles = (
    test_df.groupby("unit", sort=True)
    .tail(1)
    .sort_values("unit")
    .copy()
)

X_test = last_cycles[features]

# Check alignment with official labels
assert len(last_cycles) == len(official_rul), (
    "Test engine count and official RUL label count do not match"
)

assert last_cycles["unit"].is_unique, (
    "Expected exactly one prediction row per test engine"
)

# 6. Train the selected LightGBM configuration on all training engines
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=5,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# 7. Evaluate against official RUL labels
rmse = np.sqrt(
    mean_squared_error(official_rul, predictions)
)
mae = mean_absolute_error(official_rul, predictions)

print("Training engines:", train_df["unit"].nunique())
print("Official test engines:", len(last_cycles))
print("Predictions:", len(predictions))

print("\nOfficial FD001 test results")
print(f"RMSE: {rmse:.2f} cycles")
print(f"MAE:  {mae:.2f} cycles")

# 8. Save predictions for inspection
output = pd.DataFrame({
    "unit": last_cycles["unit"].to_numpy(),
    "actual_RUL": official_rul.to_numpy(),
    "predicted_RUL": predictions,
})

output["absolute_error"] = (
    output["actual_RUL"] - output["predicted_RUL"]
).abs()

output.to_csv(
    "models/fd001_test_predictions.csv",
    index=False,
)

print("\nSaved predictions to models/fd001_test_predictions.csv")
print("\nFirst 10 predictions:")
print(output.head(10).round(2).to_string(index=False))