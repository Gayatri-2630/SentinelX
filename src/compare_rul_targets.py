import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prepare_data import load_training_data


df = load_training_data("FD001")

# Keep the same engine-level split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, val_idx = next(
    splitter.split(df, groups=df["unit"])
)

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

# Use only original sensors, settings, and cycle
features = [
    col for col in df.columns
    if col.startswith("setting_")
    or (col.startswith("sensor_") and "_rollmean_" not in col)
]
features.append("cycle")

X_train = train_df[features]
X_val = val_df[features]

# Two target definitions
target_versions = {
    "Raw RUL": train_df["RUL"],
    "Capped RUL (125)": train_df["RUL"].clip(upper=125),
}

results = []

for name, y_train in target_versions.items():
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    # Evaluate both models against the same raw validation RUL.
    rmse = np.sqrt(
        mean_squared_error(val_df["RUL"], predictions)
    )
    mae = mean_absolute_error(val_df["RUL"], predictions)

    results.append({
        "Training target": name,
        "Validation RMSE": round(rmse, 2),
        "Validation MAE": round(mae, 2),
    })

print("Training engines:", train_df["unit"].nunique())
print("Validation engines:", val_df["unit"].nunique())
print("Shared engine IDs:",
      len(set(train_df["unit"]) & set(val_df["unit"])))

print("\n--- Target comparison ---")
print(pd.DataFrame(results).to_string(index=False))